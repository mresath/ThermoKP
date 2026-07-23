"""
===========================================================================
Dashboard Inference Helpers
Description: Thin wrappers around thermokp.py exposing derived physics
quantities and batch CSV execution for the Streamlit inference dashboard.
===========================================================================

Workflow:
1. predict_with_components wraps thermokp.py's build_enzyme_substrate_graph
   and a direct model forward pass, additionally deriving
   k_a = k_cat / K_m and, for the PINN model, k1/k-1/delta_G_ddagger/kappa
   via ThermoKPModel.forward's return_components flag.
2. run_batch iterates a CSV-derived DataFrame of queries through
   predict_with_components, capturing any ThermoKPError per row into an
   `error` column instead of aborting the whole run.

Known Caveats:
- Every prediction still incurs the mandatory 3D structural branch
  (AlphaFold/ESMFold fetch + P2Rank pocket prediction) thermokp.py already
  requires; batch mode does not skip, cache, or parallelize this per row.
- The baseline model has no Eyring layer, so k1/k_reverse/delta_G_ddagger/
  kappa are None for model_type="baseline".

Author: ThermoKP Team
License: MIT
"""

import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import thermokp  # noqa: E402

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Result Container
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class KineticsResult:
    """One query's prediction and all derived display quantities.

    Attributes
    ----------
    uniprot_id : str
        UniProt accession queried.
    mutation : str, optional
        Point-mutation code(s) queried, or None for wild-type.
    substrates : list of str
        Substrate list queried; the first entry is the primary substrate.
    ph : float
        Assay pH queried.
    temperature_celsius : float
        Assay temperature (Celsius) queried.
    model_type : str
        "pinn" or "baseline".
    k_cat : float
        Predicted turnover number, s^-1.
    k_m : float
        Predicted Michaelis constant, M.
    k_a : float
        Predicted catalytic efficiency k_cat / K_m, M^-1 s^-1.
    k1, k_reverse, delta_g_ddagger, kappa : float, optional
        PINN-only micro-rate/thermodynamic quantities; None for the
        baseline model.
    """

    uniprot_id: str
    mutation: Optional[str]
    substrates: List[str]
    ph: float
    temperature_celsius: float
    model_type: str
    k_cat: float
    k_m: float
    k_a: float
    k1: Optional[float] = None
    k_reverse: Optional[float] = None
    delta_g_ddagger: Optional[float] = None
    kappa: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════
#  Single-Query Prediction
# ═══════════════════════════════════════════════════════════════════════════
def predict_with_components(
    model: nn.Module,
    uniprot_id: str,
    mutation: Optional[str],
    substrates: List[str],
    ph: float,
    temperature_celsius: float,
    model_type: str = "pinn",
) -> KineticsResult:
    """Run one enzyme/substrate/condition query and derive all display quantities.

    Parameters
    ----------
    model : torch.nn.Module
        A loaded ThermoKPModel or BaselineNNModel (see thermokp.load_model).
    uniprot_id, mutation, substrates, ph, temperature_celsius
        See thermokp.build_enzyme_substrate_graph.
    model_type : str, optional
        "pinn" (default) or "baseline"; must match `model`'s actual type.

    Returns
    -------
    KineticsResult

    Raises
    ------
    thermokp.ThermoKPError
        See thermokp.build_enzyme_substrate_graph.
    """
    model_device = next(model.parameters()).device
    data = thermokp.build_enzyme_substrate_graph(uniprot_id, mutation, substrates, ph, temperature_celsius)
    data = data.to(model_device)

    with torch.no_grad():
        if model_type == "pinn":
            k_1, k_reverse, k_cat, delta_g_ddagger, kappa = model(data, return_components=True)
            k_cat_val = float(k_cat.view(-1).item())
            k_m_val = float(((k_reverse + k_cat) / (k_1 + thermokp.EPS)).view(-1).item())
            k1_val: Optional[float] = float(k_1.view(-1).item())
            k_reverse_val: Optional[float] = float(k_reverse.view(-1).item())
            delta_g_val: Optional[float] = float(delta_g_ddagger.view(-1).item())
            kappa_val: Optional[float] = float(kappa.view(-1).item())
        else:
            log_kcat, log_km = model(data)
            k_cat_val = float(10.0 ** log_kcat.view(-1).item())
            k_m_val = float(10.0 ** log_km.view(-1).item())
            k1_val = k_reverse_val = delta_g_val = kappa_val = None

    k_a_val = k_cat_val / (k_m_val + thermokp.EPS)

    return KineticsResult(
        uniprot_id=uniprot_id,
        mutation=mutation,
        substrates=substrates,
        ph=ph,
        temperature_celsius=temperature_celsius,
        model_type=model_type,
        k_cat=k_cat_val,
        k_m=k_m_val,
        k_a=k_a_val,
        k1=k1_val,
        k_reverse=k_reverse_val,
        delta_g_ddagger=delta_g_val,
        kappa=kappa_val,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Batch Prediction
# ═══════════════════════════════════════════════════════════════════════════
def run_batch(model: nn.Module, df: pd.DataFrame, model_type: str = "pinn") -> pd.DataFrame:
    """Run predict_with_components over each row of a batch CSV DataFrame.

    Expected columns: `uniprot_id`, `mutation` (blank for wild-type),
    `substrates` (``;``-separated, first entry primary), `ph`,
    `temperature`. Does not build or retain any structure-view data per
    row - see dashboard/structure_view.py for on-demand rendering.

    Parameters
    ----------
    model : torch.nn.Module
        A loaded ThermoKPModel or BaselineNNModel.
    df : pandas.DataFrame
        One row per query, with the columns above.
    model_type : str, optional
        "pinn" (default) or "baseline".

    Returns
    -------
    pandas.DataFrame
        One row per input row: every `KineticsResult` field, plus `error`
        (None on success, the `ThermoKPError` message otherwise).
    """
    records = []
    for row_num, (_, row) in enumerate(df.iterrows()):
        mutation = row.get("mutation")
        mutation = None if pd.isna(mutation) or not str(mutation).strip() else str(mutation).strip()
        substrates = [s.strip() for s in str(row["substrates"]).split(";") if s.strip()]

        logger.info(f"[{row_num + 1}/{len(df)}] {row['uniprot_id']} ({mutation or 'WT'})")
        try:
            result = predict_with_components(
                model,
                str(row["uniprot_id"]).strip(),
                mutation,
                substrates,
                float(row["ph"]),
                float(row["temperature"]),
                model_type=model_type,
            )
            record = asdict(result)
            record["error"] = None
        except thermokp.ThermoKPError as e:
            record = {
                "uniprot_id": row["uniprot_id"],
                "mutation": mutation,
                "substrates": substrates,
                "ph": row.get("ph"),
                "temperature_celsius": row.get("temperature"),
                "model_type": model_type,
                "k_cat": None,
                "k_m": None,
                "k_a": None,
                "k1": None,
                "k_reverse": None,
                "delta_g_ddagger": None,
                "kappa": None,
                "error": str(e),
            }
        records.append(record)

    return pd.DataFrame.from_records(records)
