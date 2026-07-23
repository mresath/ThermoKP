"""
===========================================================================
ThermoKP Master Training Script
Description: Orchestrates the training of the PINN model
===========================================================================

Workflow:
1. Loads the dataset from data/processed/tensors.
2. Initializes the ThermoKPModel (wrapper) and PINNMultiTaskLoss.
3. Instantiates the PINNTrainer.
4. Triggers the training loop, which prints an extended validation report
   (log-space RMSE, R^2, p1mag for both k_cat and K_m) every
   `PINNTrainer.DETAILED_VALIDATION_INTERVAL` epochs, and once more on the
   best Pareto checkpoint (or the final weights, if the Pareto gate never
   triggered) once training ends.

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
from src.physics.thermodynamics import EyringArrheniusLayer
from src.physics.loss import PINNMultiTaskLoss
from src.training.pinn_training import PINNTrainer

# ═══════════════════════════════════════════════════════════════════════════
#  Logging Configuration
# ═══════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Model Wrapper
# ═══════════════════════════════════════════════════════════════════════════
class ThermoKPModel(nn.Module):
    """
    Main model wrapper that joins the neural encoder and the physics-based thermodynamics layer.

    Attributes
    ----------
    encoder : MultimodalEncoder
        The underlying encoder fusing the sequence/2D-graph adapter with
        the 3D structural EGNN.
    thermo : EyringArrheniusLayer
        The thermodynamics layer to convert free energy to k_cat.
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
        Initializes the model wrapper.

        Parameters
        ----------
        ligand_in_channels : int, optional
            Number of raw structural input channels per ligand atom (see
            `featurize_ligand` in src/data/processors/generate_tensors.py):
            element, hybridization, formal charge, aromaticity, Gasteiger
            charge, ring membership, degree, hydrogen count, and an
            electrophilic/nucleophilic z-score pair = 22 dims. The
            whole-molecule ChemBERTa-2 embedding is a separate top-level
            per-graph tensor (data.ligand_embedding) fused in by
            MultimodalEncoder.mol_combine, not part of this per-atom tensor.
        co_substrate_in_channels : int, optional
            Number of raw structural input channels per co-substrate atom,
            same 22-dim layout as `ligand_in_channels`.
        hidden_channels : int, optional
            Number of hidden channels for the 2D sequence/D-MPNN branch.
        dropout : float, optional
            Dropout probability for the message-passing body (both the 2D
            and 3D branches).
        head_dropout : float, optional
            Dropout probability for the kinetics_head and energy_net
            regression heads, kept lower than the body dropout since these
            heads directly produce the scalar rates scored by the loss.
        egnn_hidden_channels : int, optional
            Hidden channel width for the 3D structural EGNN branch.
        egnn_num_layers : int, optional
            Number of EGNN message-passing layers.
        egnn_adapter_dim : int, optional
            Output width of the EGNN's ESM2-embedding adapter (see
            `StructuralEGNNEncoder.protein_pocket_adapter`).
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
        self.thermo = EyringArrheniusLayer(in_features=self.encoder.concat_dim, head_dropout=head_dropout)

    def forward(self, data, return_components: bool = False) -> tuple[torch.Tensor, ...]:
        """
        Forward pass.

        Parameters
        ----------
        data : torch_geometric.data.HeteroData
            The batch of input molecular graphs.
        return_components : bool, optional
            If True, also return the thermodynamics layer's intermediate
            `delta_G_ddagger` and `kappa` predictions, for callers (e.g. an
            interactive dashboard) wanting the model's actual internal
            activation-energy prediction rather than just k_cat. Defaults
            to False, preserving the 3-tuple contract used by the training
            loop and src/evaluation/evaluate_dataset.py.

        Returns
        -------
        tuple[torch.Tensor, ...]
            `(k_1, k_reverse, k_cat)`, or, if `return_components` is True,
            `(k_1, k_reverse, k_cat, delta_G_ddagger, kappa)`.
        """
        k_1, k_reverse, concat_rep = self.encoder(data)

        if hasattr(data, 'temperature'):
            T = data.temperature
        else:
            T = torch.full((k_1.size(0), 1), 298.15, device=k_1.device)

        if return_components:
            k_cat, delta_G_ddagger, kappa = self.thermo(concat_rep, T, return_components=True)
            return k_1, k_reverse, k_cat, delta_G_ddagger, kappa

        k_cat = self.thermo(concat_rep, T)
        return k_1, k_reverse, k_cat

# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    """
    Main entry point for the ThermoKP Master Training Script.
    Parses arguments, initializes datasets and models, and starts training.
    """

    parser = argparse.ArgumentParser(description="ThermoKP Master Training Script")
    parser.add_argument("-b", "--batch_size", type=int, default=256,
                        help="Batch size for training. Defaults to 256.")
    parser.add_argument("-d", "--dropout", type=float, default=0.3,
                        help="Dropout probability for the MultimodalEncoder message-passing body "
                             "(both the 2D and 3D branches). Defaults to 0.3.")
    parser.add_argument("--head_dropout", type=float, default=0.1,
                        help="Dropout probability for the kinetics_head/energy_net regression heads. Defaults to 0.1.")
    parser.add_argument("--hidden_channels", type=int, default=64,
                        help="Hidden channel width for the 2D sequence/D-MPNN branch and kinetics head. Defaults to 64.")
    parser.add_argument("--egnn_hidden_channels", type=int, default=32,
                        help="Hidden channel width for the 3D structural EGNN branch. Defaults to 32.")
    parser.add_argument("--egnn_num_layers", type=int, default=3,
                        help="Number of 3D structural EGNN message-passing layers. Defaults to 3.")
    parser.add_argument("--egnn_adapter_dim", type=int, default=16,
                        help="Output width of the EGNN's ESM2-embedding adapter. Defaults to 16.")
    parser.add_argument("--kcat_weight", type=float, default=2.5,
                        help="Weight applied to L_kcat when combining it with L_Km (weighted 1.0) into "
                             "the total loss. Counteracts K_m's 3-degrees-of-freedom compensation "
                             "advantage over k_cat's 2. Defaults to 2.5.")
    parser.add_argument("--kcat_only", action="store_true",
                        help="Diagnostic ablation: drop L_Km from the backpropagated total (L_kcat "
                             "is still logged). Isolates whether k_cat's validation plateau is caused "
                             "by L_Km's gradient competing for the shared trunk (via k_1/k_reverse) "
                             "rather than a representation ceiling. Also bypasses the km_tolerance "
                             "checkpoint gate, since k_1/k_reverse never train in this mode.")
    parser.add_argument("--lr_restart_patience", type=int, default=0,
                        help="Diagnostic: epochs of stalled validation k_cat loss before rewinding "
                             "the LR schedule, to test whether a plateau is an escapable "
                             "optimization artifact rather than a genuine ceiling. "
                             "0 (default) disables the feature.")
    parser.add_argument("--lr_restart_target_epoch", type=int, default=None,
                        help="Scheduler epoch to rewind to on a restart. Defaults to 0, replaying "
                             "the full schedule (warmup ramp back to peak LR, then rapid decay, "
                             "then cosine decay) rather than jumping straight to some intermediate "
                             "LR value.")
    parser.add_argument("--lr_restart_max", type=int, default=1,
                        help="Maximum number of LR restarts permitted for the run. Defaults to 1.")
    parser.add_argument("--early_stop_slope_epsilon", type=float, default=1e-5,
                        help="Slope threshold for the auto-stop trigger: fires once the val-loss "
                             "slope over the lookback window rises above -early_stop_slope_epsilon "
                             "(a plateau or a rising validation loss). Kept tight (default 1e-5) so "
                             "it only fires on a genuinely flat-or-rising slope, not one still "
                             "slowly improving.")
    parser.add_argument("--early_stop_window_frac", type=float, default=0.2,
                        help="Fraction of max_epochs used as the val-loss slope lookback window "
                             "for the auto-stop trigger. A longer window is less sensitive to a "
                             "few noisy epochs falsely reading as a plateau. Defaults to 0.2.")
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="Decay rate for the EMA shadow model used for validation and "
                             "checkpointing. Defaults to 0.999.")
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
    logger.info("Starting ThermoKP Master Training Script")
    logger.info("=======================================================================")

    # 1. Dataset
    dataset_dir = "data/processed/tensors"
    logger.info(f"Loading dataset from {dataset_dir}...")
    dataset = EnzymeDataset(dataset_dir)

    # 2. Model
    logger.info("Initializing ThermoKP Model...")
    model = ThermoKPModel(
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
    logger.info("Initializing PINN Multi-Task Loss...")
    loss_fn = PINNMultiTaskLoss(kcat_weight=args.kcat_weight, kcat_only=args.kcat_only)

    # 4. Trainer
    logger.info(f"Initializing PINN Trainer with batch size {args.batch_size}...")
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
    )
    
    # 5. Run Training
    logger.info("Starting Training Loop...")
    trainer.train()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
