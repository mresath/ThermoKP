"""
===========================================================================
PINN Multi-Task Loss Module
Description: Physics-Informed Neural Network Multi-Task Loss Computation
===========================================================================

Workflow:
1. Receives predicted micro-rates (k_1, k_reverse, k_cat) and empirical targets.
2. Computes log-space Root Mean Squared Error (RMSE) for k_cat.
3. Derives theoretical K_m via Briggs-Haldane and computes its log-space RMSE.
4. Combines both into a weighted total, `kcat_weight * L_kcat + L_Km`.

Known Caveats:
- Requires strict positivity for all kinetic parameters to prevent NaN in log10.
- K_m is algebraically derived from all three predicted rates (k_1,
  k_reverse, k_cat), so it has 3 degrees of freedom against 1 target,
  while k_cat is a direct function of only the Eyring layer's two outputs
  (Delta-G-double-dagger, kappa) against 1 target. Equal 1:1 weighting
  lets the optimizer shrink L_Km via k_1/k_reverse compensation without
  the k_cat pathway needing to improve at all; `kcat_weight` counteracts
  this by weighting L_kcat's contribution to the combined total more
  heavily than L_Km's.
- Physical plausibility is enforced as hard architectural constraints on
  the predicted rate constants themselves (diffusion limit on k_1,
  Eyring/TST speed limit on k_reverse and k_cat via delta_G >= 0), not as
  a soft physics-loss term here — see src/encoders/multimodal_encoder.py and
  src/physics/thermodynamics.py, and ARCHITECTURE.md Section 3. A QSSA
  algebraic residual is unsuitable for this purpose: ES_ss is defined by
  rearranging the exact steady-state balance such a residual would
  re-check, so it is analytically ~0 for any predicted rates and any
  reference concentration choice. An ODE trajectory loss is similarly
  unsuitable: its gradient vanishes below float32 precision at
  biologically plausible rate magnitudes, which under gradient-norm-based
  reweighting risks amplifying floating-point noise into the shared trunk.
- `eps` is a defensive guard against a literal log(0)/div-by-0, not a
  precision knob - it must stay well below the smallest physically
  plausible target value. K_m is measured in Molar concentration and can
  legitimately be nanomolar (1e-9-1e-8 M) for high-affinity enzymes. It
  is reused as the same guard inside each RMSE's `sqrt`, since the
  gradient of `sqrt(x)` diverges as the underlying MSE approaches zero.

Author: ThermoKP Team
License: MIT
"""

import logging
import torch
import torch.nn as nn
from accelerate.logging import get_logger

# ═══════════════════════════════════════════════════════════════════════════
#  Logging Configuration
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Classes
# ═══════════════════════════════════════════════════════════════════════════


class PINNMultiTaskLoss(nn.Module):
    """
    Multi-task loss function for the ThermoKP Physics-Informed Neural Network.

    Combines the log-space data-fitting losses for k_cat and the
    Briggs-Haldane-derived K_m. Rate-constant physical plausibility (the
    diffusion limit and the Eyring/TST speed limit) is enforced structurally
    upstream in the encoder and thermodynamics layers rather than as a loss
    term here.

    Attributes
    ----------
    eps : float
        Offset to prevent division by zero or log of zero. Kept well below
        the smallest physically plausible target value (see Known Caveats
        above) so it never distorts a genuine small measurement. Also used
        inside each RMSE's `sqrt` for gradient stability (see Known Caveats).
    kcat_weight : float
        Weight applied to L_kcat when combining it with L_Km into the
        total loss, counteracting the degrees-of-freedom asymmetry
        between the two targets (see Known Caveats above).
    kcat_only : bool
        Diagnostic ablation. When True, `L_total` excludes `L_Km` entirely
        (`L_kcat` is still computed and returned for logging/comparison).
        Isolates whether k_cat's own gradient path is being degraded by
        L_Km's backward pass through the shared trunk (via k_1/k_reverse),
        as opposed to a representation ceiling independent of Km.
    mse_loss : torch.nn.MSELoss
        Mean squared error function; each RMSE reported to callers is
        `sqrt(mse_loss(...) + eps)`, computed in `forward`.
    """

    def __init__(self, eps: float = 1e-12, kcat_weight: float = 2.0, kcat_only: bool = False) -> None:
        """
        Initialize the multi-task loss module.

        Parameters
        ----------
        eps : float, optional
            Small value to prevent division by zero or log of zero.
        kcat_weight : float, optional
            Weight applied to L_kcat.
        kcat_only : bool, optional
            Diagnostic ablation dropping L_Km.
        """
        super().__init__()
        self.eps = eps
        self.kcat_weight = kcat_weight
        self.kcat_only = kcat_only
        self.mse_loss = nn.MSELoss()

        logger.info("=======================================================================")
        logger.info(
            f"PINNMultiTaskLoss module initialized (kcat_weight={self.kcat_weight}, "
            f"kcat_only={self.kcat_only})."
        )
        logger.info("=======================================================================")

    def forward(
        self,
        predicted_rates: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target_params: tuple[torch.Tensor, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Compute the multi-task data-fitting loss components.

        Parameters
        ----------
        predicted_rates : tuple of torch.Tensor
            A tuple containing (k_1, k_reverse, k_cat).
            Units:
            - k_1: M^-1 s^-1 (forward binding rate)
            - k_reverse: s^-1 (complex dissociation rate)
            - k_cat: s^-1 (catalytic turnover rate)
        target_params : tuple of torch.Tensor
            A tuple containing empirical targets (target_k_cat, target_K_m).
            Units:
            - target_k_cat: s^-1
            - target_K_m: M

        Returns
        -------
        dict of str to torch.Tensor
            A dictionary containing 'L_kcat' and 'L_Km' (unweighted
            log-space RMSE, for logging/comparability), 'L_total' (the
            tensor callers should actually back-propagate: `kcat_weight *
            L_kcat + L_Km`, or just `kcat_weight * L_kcat` if `kcat_only`
            is set), and the detached log10-space predictions/targets
            ('log_pred_kcat', 'log_target_kcat', 'log_pred_Km',
            'log_target_Km') callers can use to compute additional
            reporting-only regression metrics (e.g. R^2, p1mag) without
            re-deriving the Briggs-Haldane K_m themselves.
        """
        k_1, k_reverse, k_cat = predicted_rates
        target_k_cat, target_K_m = target_params

        # 1. Log-Space Data Loss for k_cat ───────────────────────────────────
        pred_log_kcat = torch.log10(k_cat + self.eps)
        target_log_kcat = torch.log10(target_k_cat + self.eps)
        L_kcat = torch.sqrt(self.mse_loss(pred_log_kcat, target_log_kcat) + self.eps)

        # 2. Briggs-Haldane K_m Loss ──────────────────────────────────────────
        pred_K_m = (k_reverse + k_cat) / (k_1 + self.eps)
        pred_log_Km = torch.log10(pred_K_m + self.eps)
        target_log_Km = torch.log10(target_K_m + self.eps)
        L_Km = torch.sqrt(self.mse_loss(pred_log_Km, target_log_Km) + self.eps)

        L_total = self.kcat_weight * L_kcat if self.kcat_only else self.kcat_weight * L_kcat + L_Km

        return {
            "L_kcat": L_kcat,
            "L_Km": L_Km,
            "L_total": L_total,
            "log_pred_kcat": pred_log_kcat.detach(),
            "log_target_kcat": target_log_kcat.detach(),
            "log_pred_Km": pred_log_Km.detach(),
            "log_target_Km": target_log_Km.detach(),
        }
