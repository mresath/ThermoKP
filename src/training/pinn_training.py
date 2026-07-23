"""
===========================================================================
PINN Training Loop
Description: Physics-Informed Neural Network Training Loop Strategy
===========================================================================

Workflow:
1. Initializes the PyTorch Geometric DataLoader with an approximately
   90/10 train/validation split, grouped by (protein, substrate) reaction
   identity (see `train_val_split`).
2. Configures Weights & Biases for hyperparameter and loss tracking.
3. Sets up the AdamW optimizer, plus an EMA shadow model (`ema_model`)
   updated after every optimizer step.
4. Iterates through training epochs, minimizing the weighted total
   `kcat_weight * L_kcat + L_Km` (see src/physics/loss.py).
5. Validates against `ema_model` (not the live model) each epoch, and
   monitors its loss slope for the auto-stop trigger. Every
   `DETAILED_VALIDATION_INTERVAL` epochs, additionally computes and prints
   an extended report (log-space RMSE, R^2, p1mag) for both k_cat and K_m.
6. Performs Pareto checkpointing of `ema_model`'s weights based on
   physical tolerances.
7. Once training ends (either `max_epochs` is reached or the auto-stop
   trigger fires), re-runs the extended validation report against the
   Pareto checkpoint (`models/best_model.pth`) if one was ever saved, or
   the final `ema_model` weights otherwise.

Known Caveats:
- The extended validation report's R^2 and p1mag are computed over the
  full validation split, not per-batch, so per-batch predictions/targets
  are gathered via `accelerator.gather_for_metrics` (not the plain
  `accelerator.gather` used for the scalar losses) - the metrics functions
  need every validation record counted exactly once, whereas the scalar
  losses are already per-process means that plain averaging-across-gather
  tolerates.
- Validation and every saved checkpoint use `ema_model`, an exponential
  moving average of the live model's weights, updated once per optimizer
  step via `_update_ema`. The live weights' epoch-to-epoch trajectory is
  noisy enough (particularly for the exponentially-sensitive k_cat/Eyring
  pathway) that evaluating them directly can make real progress look like
  random fluctuation.
- Training runs AdamW for the entire schedule.
- The loss is pure data-fitting with no annealed physics-loss schedule,
  so the loss landscape is stable from epoch 0 - the auto-stop check
  fires as soon as `val_loss_history`'s slope-detection window fills and
  its slope is no longer meaningfully negative (a plateau or a rising
  validation loss), ending training early rather than running out the
  full `max_epochs` schedule for no further gain.
- The train/validation split (`train_val_split`) groups records by
  (`uniprot_id`, primary substrate) reaction identity, not by protein
  alone: the same protein can appear on both sides, but the same
  protein/substrate reaction never does, since a near-identical held-out
  record (same protein, same substrate, merely a different pH/
  temperature/mutation) would otherwise let the model partly memorize a
  validation reaction's kinetics instead of generalizing. Generalization
  to protein families the model has never seen at all is instead
  measured by the whole-enzyme benchmark holdout carved out before
  tensorization (`src/data/processors/clean_records.py`); grouping this
  split by protein alone (rather than by protein/substrate reaction)
  would leave some proteins entirely out of the training split,
  forfeiting labeled data the model could otherwise have learned from.

Author: ThermoKP Team
License: MIT
"""

import logging
import math
import collections
import copy
import random
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, WeightedRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs

# ═══════════════════════════════════════════════════════════════════════════
#  Logging Configuration
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
from accelerate.logging import get_logger
logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Train/Validation Split
# ═══════════════════════════════════════════════════════════════════════════

def train_val_split(
    dataset: Dataset, train_frac: float = 0.9, seed: int = 42
) -> Tuple[List[int], List[int], Dict[int, str]]:
    """Split dataset indices into train/validation, grouped by (protein, substrate) reaction identity.

    Novel-protein generalization is already measured separately by the
    whole-enzyme benchmark holdout carved out before tensorization (see
    `src/data/processors/clean_records.py`), so this split does not need
    to hold out whole proteins - the same `uniprot_id` may appear on both
    sides. It still must not leak the same reaction (the same enzyme
    assayed against the same primary substrate) across the split, or the
    model could partly memorize a validation reaction's kinetics from an
    almost-identical training record (same protein, same substrate,
    merely a different pH/temperature/mutation) instead of generalizing.
    Every record naming the same primary substrate produces a
    bit-identical `ligand_embedding` (`get_ligand_embedding` caches its
    ChemBERTa output keyed by the exact SMILES string), so that tensor
    doubles as a reliable substrate identity key without needing the raw
    SMILES/name to be stored on the record.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        The full dataset. Every record must expose `uniprot_id` and
        `ligand_embedding` attributes; a record missing either is treated
        as its own singleton group so it is never lumped in with
        unrelated records.
    train_frac : float, optional
        Target fraction of records (not reactions) in the training
        split, by default 0.9. The actual ratio only approximates this,
        since whole reaction groups (of varying record count) are
        assigned atomically rather than split.
    seed : int, optional
        Random seed for shuffling reaction groups, by default 42.

    Returns
    -------
    tuple of list of int, list of int, dict of int to str
        (train_indices, val_indices, idx_to_key) into `dataset`.
        `idx_to_key` maps every dataset index to its `uniprot_id` (or a
        per-index singleton key), independent of the finer-grained
        reaction grouping used for the split itself, so callers needing
        per-protein grouping (e.g. PINNTrainer's protein-frequency sample
        weights) don't need a second full `dataset.get()` pass over every
        index to re-derive it.
    """
    groups: Dict[Tuple[str, bytes], List[int]] = collections.defaultdict(list)
    idx_to_key: Dict[int, str] = {}
    for idx in range(len(dataset)):
        data = dataset.get(idx)
        uniprot_id = getattr(data, "uniprot_id", None)
        protein_key = uniprot_id if uniprot_id is not None else f"__no_uniprot_{idx}__"
        idx_to_key[idx] = protein_key

        ligand_embedding = getattr(data, "ligand_embedding", None)
        substrate_key = (
            ligand_embedding.numpy().tobytes()
            if ligand_embedding is not None
            else f"__no_ligand_{idx}__".encode("utf-8")
        )
        groups[(protein_key, substrate_key)].append(idx)

    group_keys = list(groups.keys())
    random.Random(seed).shuffle(group_keys)

    target_train_size = int(train_frac * len(dataset))

    train_indices: List[int] = []
    val_indices: List[int] = []
    for key in group_keys:
        if len(train_indices) < target_train_size:
            train_indices.extend(groups[key])
        else:
            val_indices.extend(groups[key])

    logger.info(
        "Reaction-grouped split: %d unique (protein, substrate) reactions -> "
        "%d train records / %d val records (no reaction appears in both splits)",
        len(group_keys), len(train_indices), len(val_indices),
    )
    return train_indices, val_indices, idx_to_key


# ═══════════════════════════════════════════════════════════════════════════
#  Trainer Class
# ═══════════════════════════════════════════════════════════════════════════

class PINNTrainer:
    """
    Handles the training loop for the ThermoKP model.

    Attributes
    ----------
    model : torch.nn.Module
        The PyTorch neural network model.
    dataset : torch_geometric.data.Dataset
        The full dataset containing sequence and 2D enzyme graphs.
    loss_fn : torch.nn.Module
        The multi-task loss function module.
    device : torch.device
        The compute device (CPU or GPU).
    batch_size : int
        Number of graphs per batch.
    max_epochs : int
        Total number of training epochs, by default 240.
    km_tolerance : float
        The physical tolerance threshold for K_m during checkpointing.
    early_stop_slope_epsilon : float
        The slope threshold for the auto-stop trigger: fires once the
        val-loss slope over `val_loss_history` rises above
        `-early_stop_slope_epsilon`. Kept small so the trigger only fires
        on a genuinely flat-or-rising slope rather than one still slowly
        improving, so training keeps running as long as it is still
        making real progress.
    early_stop_window_frac : float
        Fraction of `max_epochs` used as `val_loss_history`'s lookback
        window for the slope estimate. A longer window is less sensitive
        to a few noisy epochs falsely reading as a plateau, at the cost
        of a later-firing trigger.
    train_loader : DataLoader
        DataLoader for the training split. Draws with replacement via a
        `WeightedRandomSampler` weighted by inverse per-protein record
        count, so no single over-represented protein dominates gradient
        updates.
    val_loader : DataLoader
        DataLoader for the validation split.
    optimizer_adam : torch.optim.AdamW
        The training optimizer.
    adam_scheduler : torch.optim.lr_scheduler.LambdaLR
        Monotonic schedule (see `lr_lambda`): linear warmup over the first
        2% of epochs, rapid decay to `lr_moderate_factor` of peak through
        10% total, then cosine decay to `lr_min_factor` of peak.
    ema_decay : float
        Decay rate for `ema_model`'s exponential moving average update.
    ema_model : torch.nn.Module
        Shadow copy of `model`, updated via EMA after every optimizer
        step. Used for validation and checkpointing instead of the raw
        model, since the raw weights' epoch-to-epoch trajectory is noisy
        enough (particularly for the exponentially-sensitive k_cat
        pathway) that evaluating them directly makes a genuinely
        improving model look like it is fluctuating at random.
    val_loss_history : collections.deque
        History of validation total losses, used by the auto-stop trigger
        to detect a plateaued or rising slope.
    best_val_loss_kcat : float
        Best (lowest) recorded validation k_cat loss.
    DETAILED_VALIDATION_INTERVAL : int
        Class constant. Epoch interval (default 10) at which `train`
        additionally prints the extended validation report (log-space
        RMSE, R^2, p1mag) via `_log_detailed_report`.
    _pareto_checkpoint_saved : bool
        Whether a Pareto checkpoint (`models/best_model.pth`) has ever
        been saved this run. Set once the first checkpoint fires; used at
        the end of `train` to decide whether the post-training extended
        validation report should reload that checkpoint or fall back to
        the final `ema_model` weights (Pareto never triggered).
    lr_restart_patience : int
        Diagnostic: epochs of stalled validation k_cat loss before
        rewinding the LR schedule. 0 disables the feature.
        Distinguishes a genuine representation ceiling on k_cat from an
        optimization plateau the monotonic decay schedule can no longer
        escape - if a rewind measurably moves k_cat's validation loss,
        the plateau was reachable, not fundamental.
    lr_restart_target_epoch : int
        Scheduler epoch the LR is rewound to on a restart. Defaults to 0:
        replays the full schedule (warmup ramp back to peak, then rapid
        decay, then cosine decay) rather than landing directly on a
        specific LR value, so the jump is smoothed by the same warmup
        ramp the schedule already uses at epoch 0 instead of an abrupt
        discontinuity in the physical parameters' LR.
    lr_restart_max : int
        Maximum number of restarts permitted for the whole run.
    is_baseline : bool
        Whether this run trains `BaselineNNModel` (train_baseline_nn.py)
        rather than `ThermoKPModel`. Only affects the checkpoint filenames
        `train` writes to (`models/{best,final}_baseline_model.pth`
        instead of `models/best_model.pth`/`final_model.pth`), so a
        baseline run never clobbers a PINN run's checkpoints on the same
        host.
    """

    DETAILED_VALIDATION_INTERVAL = 10

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        loss_fn: nn.Module,
        batch_size: int = 128,
        max_epochs: int = 240,
        dropout: float = 0.3,
        head_dropout: float = 0.1,
        km_tolerance: float = 1.0,
        early_stop_slope_epsilon: float = 1e-5,
        early_stop_window_frac: float = 0.2,
        ema_decay: float = 0.999,
        num_workers: int = 0,
        lr_restart_patience: int = 0,
        lr_restart_target_epoch: Optional[int] = None,
        lr_restart_max: int = 1,
        use_wandb: bool = True,
        is_baseline: bool = False,
    ) -> None:
        """
        Initialize the PINNTrainer with the model, dataset, and configuration.

        Parameters
        ----------
        model : torch.nn.Module
            The ThermoKP neural network model.
        dataset : torch_geometric.data.Dataset
            The dataset containing the enzyme graphs.
        loss_fn : torch.nn.Module
            The multi-task loss module.
        batch_size : int, optional
            Batch size for DataLoader.
        max_epochs : int, optional
            Maximum number of epochs to train, by default 240.
        dropout : float, optional
            Dropout probability for the model's message-passing body, by default 0.3.
        head_dropout : float, optional
            Dropout probability for the model's regression heads, logged to
            WandB for traceability (the model itself is already configured
            with it), by default 0.1.
        km_tolerance : float, optional
            Tolerance for K_m to allow checkpointing, by default 1.0.
        early_stop_slope_epsilon : float, optional
            Threshold for convergence slope to trigger the auto-stop, by
            default 1e-5. Kept tight rather than loose - see
            `early_stop_slope_epsilon` attribute.
        early_stop_window_frac : float, optional
            Fraction of `max_epochs` used as the slope-detection lookback
            window, by default 0.2.
        ema_decay : float, optional
            Decay rate for the EMA shadow model, by default 0.999. Higher
            values smooth over more history at the cost of slower
            tracking of genuine improvement. See `ema_model` and
            `_update_ema`.
        lr_restart_patience : int, optional
            Diagnostic: epochs of stalled validation k_cat loss, by
            default 0 (disabled). See `lr_restart_patience` attribute.
        lr_restart_target_epoch : int or None, optional
            Scheduler epoch to rewind to on a restart, by default None
            (resolved to 0, replaying the full schedule from its warmup
            ramp, at construction time).
        lr_restart_max : int, optional
            Maximum restarts permitted for the run, by default 1.
        use_wandb : bool, optional
            Whether to log this run to Weights & Biases, by default True.
            Set False for diagnostic/throwaway runs that shouldn't leave a
            record. Local checkpointing to disk is unaffected.
        is_baseline : bool, optional
            Whether this run trains `BaselineNNModel` rather than
            `ThermoKPModel`, by default False. See `is_baseline` attribute.
        """
        self.use_wandb = use_wandb
        self.is_baseline = is_baseline

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=is_baseline)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

        logger.info("=======================================================================")
        logger.info("PINNTrainer initialized")
        logger.info("=======================================================================")

        self.device = self.accelerator.device
        logger.info(f"Using device: {self.device} via Accelerate")

        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)

        # EMA shadow model: validation/checkpointing read from this frozen
        # copy rather than the live model (see `_update_ema`).
        self.ema_decay = ema_decay
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        self.ema_model.eval()

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.km_tolerance = km_tolerance
        self.early_stop_slope_epsilon = early_stop_slope_epsilon
        self.early_stop_window_frac = early_stop_window_frac
        self.dropout = dropout
        self.head_dropout = head_dropout
        self.total_size = len(dataset)

        # Setup & Data Loading (reaction-grouped 90/10 split - see train_val_split)
        train_indices, val_indices, idx_to_key = train_val_split(dataset, train_frac=0.9, seed=44)
        self.train_size = len(train_indices)
        train_size = self.train_size
        val_size = len(val_indices)

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        logger.info(f"Dataset split: {train_size} train / {val_size} validation")

        # Per-record sample weight, inversely proportional to how many
        # training records share the same uniprot_id, so every protein
        # gets roughly equal expected exposure per epoch regardless of how
        # many records BRENDA/SABIO-RK happen to have for it.
        train_protein_keys = [idx_to_key[idx] for idx in train_indices]
        protein_counts = collections.Counter(train_protein_keys)
        sample_weights = [1.0 / protein_counts[key] for key in train_protein_keys]
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=train_size, replacement=True)

        self.train_loader = DataLoader(cast(Dataset, train_dataset), batch_size=self.batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
        self.val_loader = DataLoader(cast(Dataset, val_dataset), batch_size=self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        # Split parameters along two axes: physical role (kinetics_head/
        # thermo get a lower LR, since they directly produce the
        # exponentially-sensitive rate constants) and dimensionality
        # (1-D parameters - LayerNorm and bias terms - are excluded from
        # weight decay, since decaying a normalization scale/shift or bias
        # toward zero doesn't serve the same regularization purpose as
        # decaying an actual weight matrix).
        physical_decay, physical_no_decay = [], []
        network_decay, network_no_decay = [], []

        for name, param in model.named_parameters():
            is_physical = any(keyword in name.lower() for keyword in ["eyring", "arrhenius", "kinetics_head", "thermo"])
            is_no_decay = param.ndim < 2

            if is_physical:
                (physical_no_decay if is_no_decay else physical_decay).append(param)
            else:
                (network_no_decay if is_no_decay else network_decay).append(param)

            logger.info(
                f"{'Physical' if is_physical else 'Network'} parameter "
                f"({'no' if is_no_decay else 'with'} weight decay): {name}"
            )

        self.optimizer_adam = torch.optim.AdamW([
            {'name': 'network', 'params': network_decay, 'weight_decay': 5e-2},
            {'name': 'network', 'params': network_no_decay, 'weight_decay': 0.0},
            {'name': 'physical', 'params': physical_decay, 'weight_decay': 1e-4, 'lr': 1.5e-3},
            {'name': 'physical', 'params': physical_no_decay, 'weight_decay': 0.0, 'lr': 1.5e-3},
        ], lr=5e-3)

        # Monotonic decay, deliberately not a restart schedule (e.g.
        # CosineAnnealingWarmRestarts), to avoid reintroducing violent LR
        # jumps into the physical (kinetics_head/thermo) parameters.
        # Stored as attributes so train()'s wandb config logging always
        # reflects the schedule actually in effect.
        rapid_end = max(1, int(self.max_epochs * 0.10))
        self.rapid_end = rapid_end
        self.lr_moderate_factor = 0.2
        self.lr_min_factor = 0.0001
        warmup_epochs = max(1, int(self.max_epochs * 0.02))
        moderate_factor = self.lr_moderate_factor
        min_factor = self.lr_min_factor

        def lr_lambda(epoch: int) -> float:
            """
            Computes the learning rate multiplier for the current epoch.

            Parameters
            ----------
            epoch : int
                The current epoch number.

            Returns
            -------
            float
                The learning rate multiplier.
            """
            if epoch < warmup_epochs:
                # Linear warmup from 1% to 100% of peak LR to stabilize MultimodalEncoder initialization
                return 0.01 + 0.99 * (epoch / warmup_epochs)
            elif epoch < rapid_end:
                progress = (epoch - warmup_epochs) / max(1, rapid_end - warmup_epochs)
                return 1.0 - (1.0 - moderate_factor) * progress
            else:
                # Smooth cosine plateau decay down to ultra-low final LR
                progress = (epoch - rapid_end) / max(1, self.max_epochs - rapid_end)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return min_factor + (moderate_factor - min_factor) * cosine_decay

        self.adam_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_adam, lr_lambda
        )

        # Prepare everything with Accelerate for multi-GPU / TPU support
        (
            self.model,
            self.optimizer_adam,
            self.train_loader,
            self.val_loader,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer_adam,
            self.train_loader,
            self.val_loader,
        )

        self.val_loss_history: collections.deque = collections.deque(
            maxlen=max(1, int(self.max_epochs * self.early_stop_window_frac))
        )

        self.best_val_loss_kcat = float("inf")
        self._pareto_checkpoint_saved = False

        self.kcat_only = getattr(loss_fn, "kcat_only", False)

        # LR-restart diagnostic (opt-in; see `lr_restart_patience` attribute).
        self.lr_restart_patience = lr_restart_patience
        self.lr_restart_target_epoch = (
            lr_restart_target_epoch if lr_restart_target_epoch is not None else 0
        )
        self.lr_restart_max = lr_restart_max
        self.restarts_used = 0
        self._kcat_stagnation_epochs = 0
        self._best_kcat_for_restart = float("inf")

    def _update_ema(self) -> None:
        """
        Update `ema_model`'s weights toward the live model's current
        weights, in place.

        Must be called once per actual parameter update (i.e. once per
        `optimizer_adam.step()`), not per batch's forward pass. Non-
        floating-point buffers (e.g. the Eyring layer's integer/bool
        state, if any) are copied directly rather than averaged, since a
        running average is meaningless for them.

        Returns
        -------
        None
        """
        model_state = self.accelerator.unwrap_model(self.model).state_dict()
        ema_state = self.ema_model.state_dict()
        with torch.no_grad():
            for key, param in model_state.items():
                ema_param = ema_state[key]
                if param.dtype.is_floating_point:
                    ema_param.mul_(self.ema_decay).add_(param, alpha=1 - self.ema_decay)
                else:
                    ema_param.copy_(param)

    def _train_epoch(self) -> Dict[str, float]:
        """
        Run a single training epoch.

        Returns
        -------
        dict of str to float
            Averaged training losses for the epoch.
        """
        epoch_losses = {"total": 0.0, "kcat": 0.0, "km": 0.0}

        self.model.train()
        num_batches = len(self.train_loader)
        for batch_idx, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)

            self.optimizer_adam.zero_grad()

            predicted_rates = self.model(batch)
            target_params = (batch.kcat, batch.km)
            loss_dict = self.loss_fn(predicted_rates, target_params)

            L_total = loss_dict["L_total"]
            self.accelerator.backward(L_total)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer_adam.step()
            self._update_ema()

            L_total_gathered = cast(torch.Tensor, self.accelerator.gather(L_total)).mean()
            L_kcat_gathered = cast(torch.Tensor, self.accelerator.gather(loss_dict["L_kcat"])).mean()
            L_km_gathered = cast(torch.Tensor, self.accelerator.gather(loss_dict["L_Km"])).mean()

            epoch_losses["total"] += L_total_gathered.item()
            epoch_losses["kcat"] += L_kcat_gathered.item()
            epoch_losses["km"] += L_km_gathered.item()

            if batch_idx % max(1, num_batches // 5) == 0:
                logger.info(
                    f"[{batch_idx}/{num_batches}] Training Total Loss: {L_total.item():.4e}"
                )

        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    @staticmethod
    def _regression_metrics(log_pred: torch.Tensor, log_target: torch.Tensor) -> Dict[str, float]:
        """
        Compute log-space RMSE, R^2, and p1mag over a full set of predictions.

        Parameters
        ----------
        log_pred : torch.Tensor
            Predicted values in log10-space, shape (N,).
        log_target : torch.Tensor
            Ground-truth values in log10-space, shape (N,).

        Returns
        -------
        dict of str to float
            'rmse' (log10 units, equivalent to the corresponding training
            loss component), 'r2' (coefficient of determination against
            the target mean; `nan` if the target has zero variance), and
            'p1mag' (percent of predictions within one order of magnitude,
            i.e. `|log_pred - log_target| <= 1`, following CatPred).
        """
        residuals = log_pred - log_target
        ss_res = torch.sum(residuals ** 2)
        ss_tot = torch.sum((log_target - log_target.mean()) ** 2)
        r2 = (1.0 - ss_res / ss_tot).item() if ss_tot > 0 else float("nan")
        rmse = torch.sqrt(residuals.pow(2).mean()).item()
        p1mag = (residuals.abs() <= 1.0).float().mean().item() * 100.0
        return {"rmse": rmse, "r2": r2, "p1mag": p1mag}

    def _log_detailed_report(
        self,
        label: str,
        log_pred_kcat: torch.Tensor,
        log_target_kcat: torch.Tensor,
        log_pred_km: torch.Tensor,
        log_target_km: torch.Tensor,
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Print the extended validation analysis (log-space RMSE, R^2, p1mag)
        for both k_cat and K_m, on the main process only.

        Parameters
        ----------
        label : str
            Header label for the boxed report, e.g. an epoch tag or
            "Best Model (Pareto Checkpoint)".
        log_pred_kcat, log_target_kcat, log_pred_km, log_target_km : torch.Tensor
            Concatenated log10-space predictions/targets gathered across
            the full validation split (see `_validate_epoch`).

        Returns
        -------
        dict of str to dict, optional
            `{"kcat": kcat_metrics, "km": km_metrics}` (each an
            `_regression_metrics` dict) so the caller can also log these
            to Weights & Biases alongside the printed report, or None on
            a non-main process (nothing was computed/printed).
        """
        if not self.accelerator.is_main_process:
            return None

        kcat_metrics = self._regression_metrics(log_pred_kcat, log_target_kcat)
        km_metrics = self._regression_metrics(log_pred_km, log_target_km)

        logger.info("=======================================================================")
        logger.info(f"Detailed Validation Analysis - {label}")
        logger.info(f"{'target':>8} {'log-RMSE':>10} {'R^2':>8} {'p1mag':>8}")
        logger.info(
            f"{'k_cat':>8} {kcat_metrics['rmse']:>10.4f} {kcat_metrics['r2']:>8.4f} "
            f"{kcat_metrics['p1mag']:>7.2f}%"
        )
        logger.info(
            f"{'K_m':>8} {km_metrics['rmse']:>10.4f} {km_metrics['r2']:>8.4f} "
            f"{km_metrics['p1mag']:>7.2f}%"
        )
        logger.info("=======================================================================")

        return {"kcat": kcat_metrics, "km": km_metrics}

    def _validate_epoch(
        self,
        model: Optional[nn.Module] = None,
        detailed: bool = False,
        report_label: str = "Validation",
    ) -> Tuple[Dict[str, float], Optional[Dict[str, Dict[str, float]]]]:
        """
        Run a single validation epoch against the EMA shadow model.

        Uses `ema_model` rather than `model` so validation reflects a
        smoothed weight trajectory instead of the live, noisier one (see
        `_update_ema`); `ema_model` is already frozen and kept in eval mode.

        Parameters
        ----------
        model : torch.nn.Module, optional
            Model to validate against, by default None (resolves to
            `ema_model`). Overridden only by `train`'s post-training
            report, which re-validates the reloaded best checkpoint.
        detailed : bool, optional
            When True, additionally gathers every batch's log-space
            predictions/targets and prints the extended report (log-space
            RMSE, R^2, p1mag) via `_log_detailed_report`, by default False.
        report_label : str, optional
            Header label passed through to `_log_detailed_report` when
            `detailed` is set, by default "Validation".

        Returns
        -------
        Tuple[dict of str to float, dict of str to dict, optional]
            `(val_losses, detailed_metrics)`. `val_losses` are the
            averaged validation losses (unaffected by `detailed`).
            `detailed_metrics` is `_log_detailed_report`'s
            `{"kcat": ..., "km": ...}` return value when `detailed` is
            True (so the caller can also log it to Weights & Biases),
            else None.
        """
        eval_model = model if model is not None else self.ema_model
        val_losses = {"total": 0.0, "kcat": 0.0, "km": 0.0}
        num_batches = len(self.val_loader)

        log_pred_kcat_batches: List[torch.Tensor] = []
        log_target_kcat_batches: List[torch.Tensor] = []
        log_pred_km_batches: List[torch.Tensor] = []
        log_target_km_batches: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                predicted_rates = eval_model(batch)
                target_params = (batch.kcat, batch.km)
                loss_dict = self.loss_fn(predicted_rates, target_params)

                L_total = loss_dict["L_total"]

                l_total_gathered = cast(torch.Tensor, self.accelerator.gather(L_total)).mean()
                l_kcat_gathered = cast(torch.Tensor, self.accelerator.gather(loss_dict["L_kcat"])).mean()
                l_km_gathered = cast(torch.Tensor, self.accelerator.gather(loss_dict["L_Km"])).mean()

                val_losses["total"] += l_total_gathered.item()
                val_losses["kcat"] += l_kcat_gathered.item()
                val_losses["km"] += l_km_gathered.item()

                if detailed:
                    # gather_for_metrics (not the plain gather above) dedups
                    # any distributed-sampler padding, since every record
                    # must be counted exactly once for R^2/p1mag to be correct.
                    log_pred_kcat_batches.append(
                        cast(torch.Tensor, self.accelerator.gather_for_metrics(loss_dict["log_pred_kcat"])).cpu()
                    )
                    log_target_kcat_batches.append(
                        cast(torch.Tensor, self.accelerator.gather_for_metrics(loss_dict["log_target_kcat"])).cpu()
                    )
                    log_pred_km_batches.append(
                        cast(torch.Tensor, self.accelerator.gather_for_metrics(loss_dict["log_pred_Km"])).cpu()
                    )
                    log_target_km_batches.append(
                        cast(torch.Tensor, self.accelerator.gather_for_metrics(loss_dict["log_target_Km"])).cpu()
                    )

        for key in val_losses:
            val_losses[key] /= num_batches

        detailed_metrics = None
        if detailed:
            detailed_metrics = self._log_detailed_report(
                report_label,
                torch.cat(log_pred_kcat_batches),
                torch.cat(log_target_kcat_batches),
                torch.cat(log_pred_km_batches),
                torch.cat(log_target_km_batches),
            )

        return val_losses, detailed_metrics

    def train(self) -> None:
        """
        Execute the full training pipeline.
        """
        suffix = "_baseline" if self.is_baseline else ""
        target_path = f"models/best{suffix}_model.pth"
        final_path = f"models/final{suffix}_model.pth"

        # Weights & Biases Logging Initialization
        if self.accelerator.is_main_process and self.use_wandb:
            wandb.init(
                entity="ThermoKP",
                project="ThermoKP",
                config={
                    "batch_size": self.batch_size,
                    "max_epochs": self.max_epochs,
                    "dropout": self.dropout,
                    "head_dropout": self.head_dropout,
                    "ema_decay": self.ema_decay,
                    "kcat_weight": getattr(self.loss_fn, "kcat_weight", None),
                    "dataset_size": self.total_size,
                    "adam_lr_network_peak": 5e-3,
                    "adam_lr_physical_peak": 1.5e-3,
                    "adam_lr_network_moderate": 5e-3 * self.lr_moderate_factor,
                    "adam_lr_physical_moderate": 1.5e-3 * self.lr_moderate_factor,
                    "adam_lr_min_factor": self.lr_min_factor,
                    "km_tolerance": self.km_tolerance,
                    "early_stop_slope_epsilon": self.early_stop_slope_epsilon,
                    "early_stop_window_frac": self.early_stop_window_frac,
                    "kcat_only": self.kcat_only,
                    "lr_restart_patience": self.lr_restart_patience,
                    "lr_restart_target_epoch": self.lr_restart_target_epoch,
                    "lr_restart_max": self.lr_restart_max,
                },
            )

        logger.info("=======================================================================")
        logger.info("Starting PINN training loop")
        logger.info("=======================================================================")

        epoch = 0
        for epoch in range(self.max_epochs):
            train_losses = self._train_epoch()
            self.adam_scheduler.step()
            run_detailed_report = (epoch + 1) % self.DETAILED_VALIDATION_INTERVAL == 0
            val_losses, detailed_metrics = self._validate_epoch(
                detailed=run_detailed_report,
                report_label=f"[{epoch + 1}/{self.max_epochs}]",
            )

            if self.accelerator.is_main_process and self.use_wandb:
                wandb_metrics = {
                    "epoch": epoch,
                    # Each role spans 2 param groups (decay/no-decay) that
                    # always share the same LR, so lookup by name is safe.
                    "lr/network": next(g["lr"] for g in self.optimizer_adam.param_groups if g["name"] == "network"),
                    "lr/physical": next(g["lr"] for g in self.optimizer_adam.param_groups if g["name"] == "physical"),

                    "train/total_loss": train_losses["total"],
                    "train/loss_kcat": train_losses["kcat"],
                    "train/loss_km": train_losses["km"],

                    "val/total_loss": val_losses["total"],
                    "val/loss_kcat": val_losses["kcat"],
                    "val/loss_km": val_losses["km"],
                }
                # Only populated every DETAILED_VALIDATION_INTERVAL epochs
                # (see _log_detailed_report) - gathering full-split
                # predictions/targets every epoch would be wasteful.
                if detailed_metrics is not None:
                    wandb_metrics.update({
                        "val/kcat_rmse": detailed_metrics["kcat"]["rmse"],
                        "val/kcat_r2": detailed_metrics["kcat"]["r2"],
                        "val/kcat_p1mag": detailed_metrics["kcat"]["p1mag"],
                        "val/km_rmse": detailed_metrics["km"]["rmse"],
                        "val/km_r2": detailed_metrics["km"]["r2"],
                        "val/km_p1mag": detailed_metrics["km"]["p1mag"],
                    })
                wandb.log(wandb_metrics)

            logger.info(
                f"[{epoch + 1}/{self.max_epochs}] Val Total: {val_losses['total']:.4e} | "
                f"k_cat: {val_losses['kcat']:.4e} | K_m: {val_losses['km']:.4e}"
            )

            # Auto-stop: triggers once val loss slope is no longer
            # meaningfully negative (plateau or rising), not a fixed epoch.
            self.val_loss_history.append(val_losses["total"])

            # LR-restart diagnostic (opt-in via lr_restart_patience > 0):
            # rewinds the LR schedule on a stalled val k_cat loss, testing
            # whether the plateau is an escapable optimization artifact
            # rather than a genuine representation ceiling. Runs before the
            # auto-stop check below and clears val_loss_history on trigger,
            # so a restart gets a fresh window instead of an immediate stop.
            restart_triggered = False
            if self.lr_restart_patience > 0:
                if val_losses["kcat"] < self._best_kcat_for_restart - 1e-4:
                    self._best_kcat_for_restart = val_losses["kcat"]
                    self._kcat_stagnation_epochs = 0
                else:
                    self._kcat_stagnation_epochs += 1

                if (
                    self._kcat_stagnation_epochs >= self.lr_restart_patience
                    and self.restarts_used < self.lr_restart_max
                ):
                    self.adam_scheduler.last_epoch = self.lr_restart_target_epoch
                    self.restarts_used += 1
                    self._kcat_stagnation_epochs = 0
                    self.val_loss_history.clear()
                    restart_triggered = True

                    logger.info("=======================================================================")
                    logger.info(
                        f"[{epoch + 1}/{self.max_epochs}] LR-restart diagnostic triggered "
                        f"({self.restarts_used}/{self.lr_restart_max}): rewound scheduler to "
                        f"epoch {self.lr_restart_target_epoch} after {self.lr_restart_patience} "
                        f"stalled epochs (best val k_cat so far: {self._best_kcat_for_restart:.4e})."
                    )
                    logger.info("=======================================================================")
                    if self.accelerator.is_main_process and self.use_wandb:
                        wandb.log({"epoch": epoch, "diagnostic/lr_restart": self.restarts_used})

            auto_stop_triggered = False
            if not restart_triggered and len(self.val_loss_history) == self.val_loss_history.maxlen:
                y = np.array(self.val_loss_history)
                x = np.arange(len(y))
                slope, _ = np.polyfit(x, y, 1)

                if slope > -self.early_stop_slope_epsilon:
                    auto_stop_triggered = True
                    logger.info("=======================================================================")
                    logger.info(
                        f"[{epoch + 1}/{self.max_epochs}] Auto-stop triggered "
                        f"(val loss slope = {slope:.4e} over the last "
                        f"{len(self.val_loss_history)} epochs). Stopping training."
                    )
                    logger.info("=======================================================================")

            # Pareto Checkpointing
            # Save ONLY IF loss_kcat is all-time low AND loss_km < km_tolerance.
            # In kcat_only mode, k_1/k_reverse never receive gradient (see
            # PINNMultiTaskLoss.kcat_only), so gating on an untrained Km
            # pathway would block every checkpoint; km_tolerance is bypassed.
            km_ok = self.kcat_only or val_losses["km"] < self.km_tolerance
            if val_losses["kcat"] < self.best_val_loss_kcat and km_ok:
                self.best_val_loss_kcat = val_losses["kcat"]
                self._pareto_checkpoint_saved = True
                if self.accelerator.is_main_process:
                    state_dict = self.ema_model.state_dict()
                    torch.save(state_dict, target_path)
                    if self.use_wandb:
                        wandb.save(target_path)
                logger.info(
                    f"[{epoch + 1}/{self.max_epochs}] Pareto Checkpoint saved! "
                    f"New best k_cat loss: {self.best_val_loss_kcat:.4e}"
                )

            if auto_stop_triggered:
                break

        if self.accelerator.is_main_process:
            state_dict = self.ema_model.state_dict()
            torch.save(state_dict, final_path)
            if self.use_wandb:
                wandb.save(final_path)

        # Post-training extended report: on the Pareto checkpoint if one was
        # ever saved (the model actually meant for downstream use), or on the
        # just-saved final ema_model weights if the Pareto gate never fired.
        self.accelerator.wait_for_everyone()
        if self._pareto_checkpoint_saved:
            report_model = copy.deepcopy(self.ema_model)
            report_model.load_state_dict(torch.load(target_path, map_location=self.device, weights_only=True))
            report_model.eval()
            report_label = "Best Model (Pareto Checkpoint)"
        else:
            report_model = self.ema_model
            report_label = "Final Model (Pareto Never Triggered)"

        _, final_detailed_metrics = self._validate_epoch(model=report_model, detailed=True, report_label=report_label)

        if self.accelerator.is_main_process and self.use_wandb:
            if final_detailed_metrics is not None:
                wandb.log({
                    "final/kcat_rmse": final_detailed_metrics["kcat"]["rmse"],
                    "final/kcat_r2": final_detailed_metrics["kcat"]["r2"],
                    "final/kcat_p1mag": final_detailed_metrics["kcat"]["p1mag"],
                    "final/km_rmse": final_detailed_metrics["km"]["rmse"],
                    "final/km_r2": final_detailed_metrics["km"]["r2"],
                    "final/km_p1mag": final_detailed_metrics["km"]["p1mag"],
                })
            wandb.finish()

# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=======================================================================")
    logger.info("ThermoKP Training Script")
    logger.info("=======================================================================")
    logger.error("This script is intended to be imported. To run, instantiate PINNTrainer with your Model and Dataset.")
