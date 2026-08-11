"""
===========================================================================
Eyring-Arrhenius Thermodynamics Layer
Description: Thermodynamics calculations for catalytic rate constants
===========================================================================

Workflow:
1. Accepts a representation tensor and absolute temperature (`T`).
2. Predicts two quantities from the representation via `energy_net`: the
   activation free energy (`delta_G_ddagger`), passed through softplus so
   it is structurally non-negative, and the transmission coefficient
   (`kappa`), passed through sigmoid so it is bounded to (0, 1). Together
   these guarantee the Eyring/TST speed limit
   k_cat <= kappa*(k_B*T/h) <= k_B*T/h without any soft physics-loss term.
3. Calculates the catalytic rate constant (`k_cat`) using the Eyring-Polanyi equation.

Known Caveats:
- Computations use float64 internally for precision, returning the original tensor dtype.
- Only `clamp_min` is needed: since `softplus` guarantees
  `delta_G_ddagger >= 0`, the exponent `-delta_G/(R*T)` is always `<= 0`,
  so no upper clamp is required. `clamp_min` guards against float64
  underflow when delta_G_ddagger is very large (an extremely slow
  reaction). Set to -700.0 (float64 only underflows exp() to exactly 0.0
  around -744) rather than a tight bound, since torch.clamp has zero
  gradient outside its range - a tighter clamp would silently kill
  training signal for any example whose true delta_G_ddagger sits above
  the clamp, which a dataset containing catalytically-impaired mutant
  enzymes can plausibly need.

Author: ThermoKP Team
License: MIT
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, cast

# ═══════════════════════════════════════════════════════════════════════════
#  Logging Configuration
# ═══════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

# Physical constants in standard SI units
KB_SI = 1.380649e-23   # Boltzmann constant (J/K)
H_SI = 6.62607015e-34  # Planck constant (J s)
R_SI = 8.314462618     # Ideal Gas constant (J/(mol K))

# ═══════════════════════════════════════════════════════════════════════════
#  Thermodynamic Layers
# ═══════════════════════════════════════════════════════════════════════════

class EyringArrheniusLayer(nn.Module):
    """
    Differentiable layer converting activation free energy to catalytic rate.

    Calculates the turnover number (k_cat) based on the Eyring-Polanyi
    transition state theory. delta_G_ddagger is passed through softplus,
    which structurally guarantees delta_G_ddagger >= 0 and therefore the
    Eyring/TST speed limit k_cat <= kappa*(k_B*T/h) <= k_B*T/h — a hard
    architectural constraint rather than a soft physics-loss penalty.

    Attributes
    ----------
    k_B : torch.Tensor
        Boltzmann constant in J/K. Non-trainable float64 buffer.
    h : torch.Tensor
        Planck constant in J s. Non-trainable float64 buffer.
    R : torch.Tensor
        Ideal gas constant in J/(mol K). Non-trainable float64 buffer.
    energy_net : torch.nn.Sequential
        Predicts, per example, the activation free energy `delta_G_ddagger`
        (kcal/mol, pre-softplus) and the transmission coefficient `kappa`
        (dimensionless, pre-sigmoid) from the input representation.
    clamp_min : float
        Minimum value for the exponent argument, set just above float64's
        actual underflow floor (~-744) rather than tightly, since a
        tighter bound would zero the gradient (via torch.clamp) for any
        plausible slow-reaction example above it.
    head_dropout : float
        Dropout probability for the energy_net regression head. Kept lower
        than typical body dropout since energy_net directly predicts
        delta_G_ddagger, and the Eyring relation maps a ~1.36 kcal/mol error
        in that prediction to a full decade of error in k_cat.
    """

    def __init__(self, in_features: int, clamp_min: float = -700.0, head_dropout: float = 0.1) -> None:
        """
        Initializes the EyringArrheniusLayer with physical constants and an energy predictor network.

        Parameters
        ----------
        in_features : int
            The dimension of the input feature vector.
        clamp_min : float, optional
            Lower bound for the exponent argument -delta_G/(R*T), by
            default -700.0. float64 underflows exp() to exactly 0.0 only
            around -744 (ln of the smallest subnormal, ~4.9e-324), so this
            leaves a ~44-unit safety margin. Since torch.clamp has zero
            gradient outside its bounds, a tighter value would kill
            training signal for any example whose true delta_G_ddagger
            sits above it - plausible for slow/catalytically-impaired
            mutant enzymes, which this dataset includes.
        head_dropout : float, optional
            Dropout probability for energy_net, by default 0.1. Kept lower
            than the encoder body's dropout since this MLP directly predicts
            the activation free energy that the Eyring equation exponentiates
            into k_cat, where prediction noise costs fold-error directly.
        """
        super().__init__()

        # Deliberately narrow (a single hidden layer), mirroring
        # MultimodalEncoder.kinetics_head's own narrowness: even though
        # delta_G_ddagger is directly supervised (unlike k_1/k_reverse),
        # more capacity here still gives the optimizer more room to fit
        # training-set idiosyncrasies rather than the underlying physics.
        hidden_dim = in_features // 2
        self.energy_net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, 2)
        )
        # Initialize the output bias to ~15 kcal/mol so that, at T = 298.15 K,
        # the Eyring prefactor yields k_cat ~ 10^2 s^-1 — the biological median.
        # softplus(15.0) ~ 15.0000003, so this bias still lands at ~15 kcal/mol
        # post-softplus. Without this, random initialization gives delta_G ~ 0,
        # driving k_cat ~ 10^12 s^-1 and stalling gradients at initialization.
        with torch.no_grad():
            final_linear = self.energy_net[-1]
            assert isinstance(final_linear, nn.Linear)
            final_linear.bias[0] = 15.0  # delta_G ~ 15 kcal/mol (pre-softplus)
            final_linear.bias[1] = 2.0   # kappa ~ sigmoid(2.0) ≈ 0.88

        # Register physical constants as non-trainable buffers in single precision
        # to ensure compatibility with Apple Silicon (MPS) which does not support float64.
        # Single precision easily covers these values since 6.62e-34 >> 1.18e-38 (float32 min).
        self.register_buffer("k_B", torch.tensor(KB_SI, dtype=torch.float32))
        self.register_buffer("h", torch.tensor(H_SI, dtype=torch.float32))
        self.register_buffer("R", torch.tensor(R_SI, dtype=torch.float32))

        self.clamp_min = clamp_min

    def forward(
        self, x: torch.Tensor, T: torch.Tensor, return_components: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass computing k_cat from the representation and temperature.

        Parameters
        ----------
        x : torch.Tensor
            The input representation tensor (e.g., from a graph neural network).
        T : torch.Tensor
            Absolute temperature tensor in Kelvin (K).
        return_components : bool, optional
            If True, also return the intermediate `delta_G_ddagger` (kcal/mol)
            and `kappa` (dimensionless) the Eyring equation was evaluated
            with, for callers (e.g. an interactive dashboard) that want the
            model's actual internal activation-energy prediction rather than
            just the resulting rate constant. Defaults to False, preserving
            the plain-tensor return used by training/evaluation.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            Catalytic rate constant k_cat in s^-1, matching input dtype. If
            `return_components` is True, instead returns
            `(k_cat, delta_G_ddagger, kappa)`.
        """
        # Predict delta G double dagger (kcal/mol) and transmission coefficient (kappa).
        # softplus structurally guarantees delta_G_ddagger >= 0, which is the
        # TST condition for a physically valid transition state (a barrierless
        # process is the maximum rate; a negative barrier is unphysical).
        out = self.energy_net(x)
        delta_G_kcal = F.softplus(out[..., 0]).view(-1)
        kappa = torch.sigmoid(out[..., 1]).to(torch.float64).view(-1)


        delta_G_J = delta_G_kcal * 4184.0

        input_dtype = delta_G_J.dtype

        # Cast inputs to float64 for intermediate precision
        delta_G_fp64 = delta_G_J.to(torch.float64)
        T_fp64 = T.to(torch.float64).view(-1)

        # Pre-factor: kappa * (k_B * T) / h
        k_B_val = cast(torch.Tensor, self.k_B)
        h_val = cast(torch.Tensor, self.h)
        R_val = cast(torch.Tensor, self.R)
        
        pre_factor = kappa * (k_B_val * T_fp64) / h_val

        # Exponent argument: -delta_G / (R * T). Since delta_G >= 0, this is
        # always <= 0 — the exponent can no longer drive k_cat above the
        # Eyring/TST ceiling kappa*(k_B*T/h).
        exponent = -delta_G_fp64 / (R_val * T_fp64)

        # Clamp only guards against float64 underflow for very large delta_G
        # (extremely slow reactions); since delta_G_ddagger >= 0 (softplus),
        # the exponent is always <= 0.
        clamped_exponent = torch.clamp(exponent, min=self.clamp_min)

        # Compute turnover number k_cat
        k_cat = pre_factor * torch.exp(clamped_exponent)

        if return_components:
            return k_cat.to(input_dtype), delta_G_kcal, kappa
        return k_cat.to(input_dtype)

# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("==================================================")
    logger.info("       EyringArrheniusLayer Initialization        ")
    logger.info("==================================================")
    
    layer = EyringArrheniusLayer(in_features=64)
    
    # 298.15 K, ~50 kJ/mol
    T_test = torch.tensor([298.15], dtype=torch.float32)
    x_test = torch.randn(1, 64)
    k_cat_pred = layer(x_test, T_test)
    logger.info(f"Test computation successful: k_cat = {k_cat_pred.item():.4e} s^-1")
    
    logger.info("==================================================")
    logger.info("               Execution Summary                  ")
    logger.info("==================================================")
    logger.info("Target file initialized: src/physics/thermodynamics.py")


