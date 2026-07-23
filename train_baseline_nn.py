"""
===========================================================================
ThermoKP Baseline NN Training Script
Description: Trains a non-physics-informed direct kcat/K_m regressor
===========================================================================

Workflow:
1. Loads the same dataset from data/processed/tensors used by train.py.
2. Initializes BaselineNNModel (MultimodalEncoder + plain MLP head) and
   BaselineNNLoss.
3. Instantiates PINNTrainer with is_baseline=True, shared with train.py
   otherwise unmodified, so both scripts run the identical
   batch/schedule/optimizer/checkpointing pipeline (including the periodic
   and post-training detailed validation reports - see pinn_training.py).
   is_baseline=True only changes the checkpoint filenames `train` writes
   to (models/{best,final,checkpoint}_baseline_model.pth), so a baseline
   run never overwrites a PINN run's checkpoints on the same host.
4. Triggers the training loop.

Purely an ablation control: this model shares the exact same input tensors
and encoder as ThermoKPModel (train.py) but predicts log10(k_cat) and
log10(K_m) directly, with no Eyring thermodynamics layer and no
Briggs-Haldane K_m derivation - a plain supervised regressor over
pretrained protein/molecule embeddings with no physical constraints.
Comparing its validation loss against ThermoKPModel's isolates whether a
performance ceiling comes from the data/tensors/encoder (both models would
plateau similarly) or from the physics-constrained architecture (this
baseline would clear it easily).

Author: ThermoKP Team
License: MIT
"""

import argparse
import logging

import torch
import torch.nn as nn
from accelerate import PartialState
from accelerate.logging import get_logger

from src.data.models.dataset import EnzymeDataset
from src.encoders.multimodal_encoder import MultimodalEncoder
from src.training.pinn_training import PINNTrainer

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════
class BaselineNNModel(nn.Module):
    """
    Direct log-space kcat/K_m regressor sharing the PINN's encoder.

    Reuses `MultimodalEncoder` unmodified to produce the exact same
    ESM2/ChemBERTa-derived fused representation (`concat_rep`) the PINN's
    Eyring thermodynamics layer consumes, then feeds it into a plain MLP
    head that regresses log10(k_cat) and log10(K_m) directly - no Eyring
    layer, no Briggs-Haldane derivation, no k_1/k_reverse intermediate.

    `MultimodalEncoder.kinetics_head` still runs during the forward pass
    (it's baked into the shared encoder) but its k_1/k_reverse outputs
    are discarded here and never enter the loss, so it receives no
    gradient - harmless dead compute, not a second prediction pathway.

    Attributes
    ----------
    encoder : MultimodalEncoder
        Same protein/ligand/co-substrate encoder used by ThermoKPModel.
    baseline_head : torch.nn.Sequential
        Plain MLP mapping `concat_rep` to (log10_kcat, log10_km). Named
        baseline_head so it is treated as a network parameter, not physical.
    """

    def __init__(
        self,
        ligand_in_channels: int = 22,
        co_substrate_in_channels: int = 22,
        hidden_channels: int = 64,
        dropout: float = 0.3,
        head_dropout: float = 0.1,
        egnn_hidden_channels: int = 32,
        egnn_num_layers: int = 3,
        egnn_adapter_dim: int = 16,
    ) -> None:
        """
        Parameters
        ----------
        ligand_in_channels : int, optional
            Per-atom ligand feature width, by default 22 (see `ThermoKPModel`).
        co_substrate_in_channels : int, optional
            Per-atom co-substrate feature width, by default 22.
        hidden_channels : int, optional
            Hidden channel width for the shared encoder's 2D branch, by default 64.
        dropout : float, optional
            Encoder body dropout (both branches), by default 0.3.
        head_dropout : float, optional
            Dropout for this regression head (and the encoder's own,
            unused kinetics_head), by default 0.1.
        egnn_hidden_channels : int, optional
            Hidden channel width for the shared encoder's 3D EGNN branch, by default 32.
        egnn_num_layers : int, optional
            Number of 3D EGNN message-passing layers, by default 3.
        egnn_adapter_dim : int, optional
            Output width of the EGNN's ESM2-embedding adapter, by default 16.
        """
        super().__init__()
        self.encoder = MultimodalEncoder(
            ligand_in_channels=ligand_in_channels,
            co_substrate_in_channels=co_substrate_in_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
            head_dropout=head_dropout,
            egnn_hidden_channels=egnn_hidden_channels,
            egnn_num_layers=egnn_num_layers,
            egnn_adapter_dim=egnn_adapter_dim,
        )

        concat_dim = self.encoder.concat_dim
        hidden_dim = concat_dim // 2
        
        self.baseline_head = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, 2),  # log10_kcat, log10_km
        )

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        data : torch_geometric.data.HeteroData
            Batch of input molecular graphs, identical format to
            ThermoKPModel's input.

        Returns
        -------
        tuple of torch.Tensor
            (log10_kcat_pred, log10_km_pred), each shape (B,) - matches
            `batch.kcat`/`batch.km`'s batched shape (each per-graph target
            is a 1-element tensor, concatenated along dim 0 by PyG), and
            the (B,)-shaped convention `MultimodalEncoder`/`EyringArrheniusLayer`
            already use for k_1/k_reverse/k_cat.
        """
        _, _, concat_rep = self.encoder(data)
        out = self.baseline_head(concat_rep)
        return out[:, 0], out[:, 1]


# ═══════════════════════════════════════════════════════════════════════════
#  Loss
# ═══════════════════════════════════════════════════════════════════════════
class BaselineNNLoss(nn.Module):
    """
    Plain log-space RMSE loss for the baseline NN.

    Unlike `PINNMultiTaskLoss`, both targets are direct 1-degree-of-freedom
    regression outputs (no Briggs-Haldane compensation pathway), so no
    degrees-of-freedom weighting asymmetry applies here; `kcat_weight`
    defaults to 1.0 and only exists for parity with `PINNMultiTaskLoss`
    if the two need to be compared under matched weighting.

    Attributes
    ----------
    eps : float
        Must match `PINNMultiTaskLoss.eps` for comparable loss values. Also
        used inside each RMSE's `sqrt` for gradient stability, since the
        gradient of `sqrt(x)` diverges as the underlying MSE approaches zero.
    kcat_weight : float
        Weight applied to L_kcat in the combined total loss.
    mse_loss : torch.nn.MSELoss
        Mean squared error function; each RMSE reported to callers is
        `sqrt(mse_loss(...) + eps)`, computed in `forward`.
    """

    def __init__(self, eps: float = 1e-12, kcat_weight: float = 1.0) -> None:
        super().__init__()
        self.eps = eps
        self.kcat_weight = kcat_weight
        self.mse_loss = nn.MSELoss()

        logger.info("=======================================================================")
        logger.info(f"BaselineNNLoss initialized (kcat_weight={self.kcat_weight}).")
        logger.info("=======================================================================")

    def forward(
        self,
        predicted: tuple[torch.Tensor, torch.Tensor],
        target_params: tuple[torch.Tensor, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        predicted : tuple of torch.Tensor
            (log10_kcat_pred, log10_km_pred), each shape (B,).
        target_params : tuple of torch.Tensor
            (target_k_cat, target_K_m) in raw units (s^-1, M), each shape
            (B,) - matches `PINNTrainer`'s `(batch.kcat, batch.km)`.

        Returns
        -------
        dict of str to torch.Tensor
            'L_kcat', 'L_Km' (unweighted log-space RMSE), 'L_total' (the
            tensor to back-propagate: `kcat_weight * L_kcat + L_Km`), and
            the detached log10-space predictions/targets ('log_pred_kcat',
            'log_target_kcat', 'log_pred_Km', 'log_target_Km') callers can
            use for reporting-only regression metrics (e.g. R^2, p1mag).
        """
        pred_log_kcat, pred_log_km = predicted
        target_k_cat, target_K_m = target_params

        target_log_kcat = torch.log10(target_k_cat + self.eps)
        target_log_km = torch.log10(target_K_m + self.eps)

        L_kcat = torch.sqrt(self.mse_loss(pred_log_kcat, target_log_kcat) + self.eps)
        L_Km = torch.sqrt(self.mse_loss(pred_log_km, target_log_km) + self.eps)
        L_total = self.kcat_weight * L_kcat + L_Km

        return {
            "L_kcat": L_kcat,
            "L_Km": L_Km,
            "L_total": L_total,
            "log_pred_kcat": pred_log_kcat.detach(),
            "log_target_kcat": target_log_kcat.detach(),
            "log_pred_Km": pred_log_km.detach(),
            "log_target_Km": target_log_km.detach(),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    """
    Main entry point for the Baseline NN Training Script.
    Parses arguments, initializes the dataset and model, and starts training.
    """
    parser = argparse.ArgumentParser(description="ThermoKP Baseline NN Training Script")
    parser.add_argument("-b", "--batch_size", type=int, default=256,
                        help="Batch size for training. Defaults to 256.")
    parser.add_argument("-d", "--dropout", type=float, default=0.3,
                        help="Dropout probability for the MultimodalEncoder message-passing body. Defaults to 0.3.")
    parser.add_argument("--head_dropout", type=float, default=0.1,
                        help="Dropout probability for the regression head. Defaults to 0.1.")
    parser.add_argument("--hidden_channels", type=int, default=64,
                        help="Hidden channel width for the shared encoder's 2D branch. Defaults to 64.")
    parser.add_argument("--egnn_hidden_channels", type=int, default=32,
                        help="Hidden channel width for the shared encoder's 3D EGNN branch. Defaults to 32.")
    parser.add_argument("--egnn_num_layers", type=int, default=3,
                        help="Number of 3D EGNN message-passing layers. Defaults to 3.")
    parser.add_argument("--egnn_adapter_dim", type=int, default=16,
                        help="Output width of the EGNN's ESM2-embedding adapter. Defaults to 16.")
    parser.add_argument("--kcat_weight", type=float, default=1.0,
                        help="Weight applied to L_kcat when combining it with L_Km. Defaults to 1.0.")
    parser.add_argument("--lr_restart_patience", type=int, default=0,
                        help="Diagnostic: epochs of stalled validation k_cat loss before rewinding "
                             "the LR schedule. 0 (default) disables the feature.")
    parser.add_argument("--lr_restart_target_epoch", type=int, default=None,
                        help="Scheduler epoch to rewind to on a restart. Defaults to 0.")
    parser.add_argument("--lr_restart_max", type=int, default=1,
                        help="Maximum number of LR restarts permitted for the run. Defaults to 1.")
    parser.add_argument("--early_stop_slope_epsilon", type=float, default=1e-5,
                        help="Slope threshold for the auto-stop trigger. Defaults to 1e-5.")
    parser.add_argument("--early_stop_window_frac", type=float, default=0.2,
                        help="Fraction of max_epochs used as the val-loss slope lookback window. Defaults to 0.2.")
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="Decay rate for the EMA shadow model. Defaults to 0.999.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers for loading PyG tensors from disk. "
                             "PINNTrainer spins up 2 separate DataLoaders (train/val) with "
                             "pin_memory=True, each with this many workers, per process - under "
                             "multi-GPU accelerate DDP this multiplies by the rank count, so keep this "
                             "conservative on memory-constrained hosts. Defaults to 4.")
    parser.add_argument("--max_epochs", type=int, default=240,
                        help="Maximum number of training epochs. Defaults to 240.")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable Weights & Biases logging for this run (e.g. throwaway "
                             "diagnostic runs not meant to leave a record). Local checkpointing "
                             "to disk is unaffected.")
    args = parser.parse_args()

    PartialState()

    logger.info("=======================================================================")
    logger.info("Starting ThermoKP Baseline NN Training Script")
    logger.info("=======================================================================")

    # 1. Dataset
    dataset_dir = "data/processed/tensors"
    logger.info(f"Loading dataset from {dataset_dir}...")
    dataset = EnzymeDataset(dataset_dir)

    # 2. Model
    logger.info("Initializing Baseline NN Model...")
    model = BaselineNNModel(
        ligand_in_channels=22,
        co_substrate_in_channels=22,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        egnn_hidden_channels=args.egnn_hidden_channels,
        egnn_num_layers=args.egnn_num_layers,
        egnn_adapter_dim=args.egnn_adapter_dim,
    )

    # 3. Loss Function
    logger.info("Initializing Baseline NN Loss...")
    loss_fn = BaselineNNLoss(kcat_weight=args.kcat_weight)

    # 4. Trainer
    logger.info(f"Initializing PINN Trainer for Baseline NN with batch size {args.batch_size}...")
    trainer = PINNTrainer(
        model=model,
        dataset=dataset,
        loss_fn=loss_fn,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        ema_decay=args.ema_decay,
        num_workers=args.num_workers,
        lr_restart_patience=args.lr_restart_patience,
        lr_restart_target_epoch=args.lr_restart_target_epoch,
        lr_restart_max=args.lr_restart_max,
        early_stop_slope_epsilon=args.early_stop_slope_epsilon,
        early_stop_window_frac=args.early_stop_window_frac,
        use_wandb=not args.no_wandb,
        is_baseline=True,
    )

    # 5. Run Training
    logger.info("Starting Training Loop...")
    trainer.train()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

