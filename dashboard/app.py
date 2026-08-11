"""
===========================================================================
ThermoKP Streamlit Dashboard
Description: Interactive local dashboard for zero-shot kinetics inference,
demoing thermokp.py's single-query and batch prediction paths alongside an
interactive predicted-structure viewer.
===========================================================================

Workflow:
1. Sidebar: choose the model type (PINN/baseline) and an optional
   checkpoint path override; the resolved model is cached in session
   state and only reloaded when either selection changes.
2. Single Query tab: submit one enzyme/substrate/condition query, display
   k_cat/K_m/k_a (and, for the PINN, k1/k-1/delta_G_ddagger/kappa) as
   KaTeX-typeset cards, render the structure view immediately, and
   append the result to a running session-state history table.
3. Batch tab: upload a CSV of queries, display a numeric-only results
   table, and render the structure view on demand for a selected row.
4. Model loading and every prediction run inside an expandable
   dashboard/log_capture.run_with_log_feedback status box, streaming the
   underlying pipeline's own log lines (UniProt/embedding fetches,
   AlphaFold/P2Rank, batch row progress) live - most useful on a cold
   cache, where the first call can take a while with no other feedback.

Known Caveats:
- Structure rendering (P2Rank) is not cached, so each render - including
  a Streamlit script rerun - can take several seconds.

Author: ThermoKP Team
License: MIT
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
import rdkit.Chem as Chem
import streamlit as st
import streamlit.components.v1 as components
import py3Dmol

from src.data.processors import pretrained_embeddings
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.log_capture import run_with_log_feedback  # noqa: E402
from dashboard.structure_view import (  # noqa: E402
    LIGAND_COLOR,
    MUTATION_COLOR,
    POCKET_COLOR,
    PROTEIN_COLOR,
    build_structure_view,
)
from src.data.processors.dataset_validator import resolve_smiles  # noqa: E402
from src.data.utils.ligand_cleaner import canonicalize_and_filter_ligands  # noqa: E402
import thermokp  # noqa: E402
from thermokp import KineticsResult, predict_kinetics, run_batch  # noqa: E402

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="ThermoKP Dashboard", layout="wide")


# ═══════════════════════════════════════════════════════════════════════════
#  Model Loading
# ═══════════════════════════════════════════════════════════════════════════
def _get_model(model_type: str, checkpoint_path: str):
    """Load (or reuse a session-cached) model, reloading only on selection change."""
    cache_key = (model_type, checkpoint_path or None)
    if st.session_state.get("_model_cache_key") != cache_key:
        st.session_state["_model"] = run_with_log_feedback(
            f"Loading {model_type.upper()} checkpoint...",
            thermokp.load_model,
            checkpoint=checkpoint_path or None,
            model_type=model_type,
        )
        st.session_state["_model_cache_key"] = cache_key
    return st.session_state["_model"]


def _resolve_primary_mol(substrates):
    if not substrates:
        return None
    primary_smiles = resolve_smiles(substrates[0])
    if primary_smiles is None:
        return None
    return Chem.MolFromSmiles(primary_smiles)

def _clean_substrates_for_ui(raw_substrates):
    """Run ligand canonicalization before prediction."""
    if not raw_substrates:
        return []
    res = canonicalize_and_filter_ligands(raw_substrates[0], raw_substrates[1:])
    if res is None:
        return raw_substrates  # Fallback to raw if unparseable
    primary, cosubs = res
    return [primary] + cosubs


def _format_scientific(value: float, sig_figs: int = 3) -> str:
    """Format a float as LaTeX scientific notation, e.g. ``1.23 \times 10^{4}``."""
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.{sig_figs - 1}e}".split("e")
    return f"{mantissa} \\times 10^{{{int(exponent)}}}"


def _latex_metric(column, label_latex: str, value: Optional[float], unit: str = "", plain: bool = False) -> None:
    """Render one labeled quantity as a bordered card with KaTeX-typeset header and value.

    Parameters
    ----------
    column : streamlit.delta_generator.DeltaGenerator
        Streamlit column (from `st.columns`) to render the card into.
    label_latex : str
        LaTeX for the card's header (e.g. r"k_{cat}"), rendered as
        ``st.markdown(f"${label_latex}$")``.
    value : float, optional
        The quantity to display. Optional since the baseline model has no
        k1/k_reverse/delta_G_ddagger/kappa (see `KineticsResult`) - such a
        card renders "—".
    unit : str, optional
        LaTeX unit suffix appended to the value (e.g. r"\text{s}^{-1}").
    plain : bool, optional
        Render `value` as a plain decimal (``f"{value:.4g}"``) instead of
        LaTeX scientific notation - used for already-bounded quantities
        like kappa where scientific notation would be less readable.

    Returns
    -------
    None
    """
    with column.container(border=True):
        st.markdown(f"${label_latex}$")
        if value is None:
            st.latex(r"\text{---}")
            return
        body = f"{value:.4g}" if plain else _format_scientific(value)
        if unit:
            body += rf"\ {unit}"
        st.latex(body)


def _render_result_card(result: KineticsResult) -> None:
    st.markdown("##### Predicted Kinetics")
    cols = st.columns(3)
    _latex_metric(cols[0], r"k_{cat}", result.k_cat, unit=r"\text{s}^{-1}")
    _latex_metric(cols[1], r"K_m", result.k_m, unit=r"\text{M}")
    _latex_metric(cols[2], r"k_{cat}/K_m", result.k_a, unit=r"\text{M}^{-1}\text{s}^{-1}")

    if result.model_type in ("pinn", "no_mutant"):
        st.markdown("Physics-informed micro-rate quantities (from the Eyring/Briggs-Haldane layers)")
        cols = st.columns(4)
        _latex_metric(cols[0], r"k_1", result.k1)
        _latex_metric(cols[1], r"k_{-1}", result.k_reverse)
        _latex_metric(cols[2], r"\Delta G^\ddagger", result.delta_g_ddagger, unit=r"\text{kcal/mol}", plain=True)
        _latex_metric(cols[3], r"\kappa", result.kappa, plain=True)


_LEGEND_ITEMS = (
    (PROTEIN_COLOR, "Protein backbone"),
    (POCKET_COLOR, "Predicted binding pocket (P2Rank)"),
    (MUTATION_COLOR, "Mutated residue(s)"),
    (LIGAND_COLOR, "Substrate (illustrative placement, not a docked pose)"),
)


def _render_legend() -> None:
    swatches = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:1.25rem;">'
        f'<span style="width:0.85rem;height:0.85rem;border-radius:50%;background:{color};'
        f'display:inline-block;margin-right:0.4rem;"></span>{label}</span>'
        for color, label in _LEGEND_ITEMS
    )
    st.markdown(f'<div style="font-size:0.85rem;">{swatches}</div>', unsafe_allow_html=True)


def _render_structure(uniprot_id: str, mutation, substrates) -> None:
    primary_mol = _resolve_primary_mol(substrates)
    with st.spinner("Rendering predicted structure (AlphaFold/ESMFold + P2Rank)..."):
        html = build_structure_view(uniprot_id, mutation, primary_mol=primary_mol)
    with st.container(border=True):
        st.markdown("##### Predicted Structure")
        if html:
            _render_legend()
            components.html(html, height=520)
        else:
            st.info("No structure/pocket could be resolved for this enzyme.")


# ═══════════════════════════════════════════════════════════════════════════
#  Single Query Tab
# ═══════════════════════════════════════════════════════════════════════════
def _single_query_tab(model_type: str, checkpoint_path: str) -> None:
    if "history" not in st.session_state:
        st.session_state["history"] = []

    with st.form("single_query_form"):
        uniprot_id = st.text_input("UniProt ID (optional if Sequence/Structure files are given)")
        seq_file = st.text_input("Sequence File Path (optional)")
        struct_file = st.text_input("Structure File Path (optional, PDB format)")
        mutation = st.text_input("Mutation (optional)", help="e.g. N41D or N41D/N281D")
        substrates_raw = st.text_area("Substrates (one per line; first is primary)")
        ph = st.number_input("pH", value=7.0, min_value=0.0, max_value=14.0)
        temperature = st.number_input("Temperature (°C)", value=25.0)
        submitted = st.form_submit_button("Predict")

    if submitted:
        raw_sub_list = [s.strip() for s in substrates_raw.splitlines() if s.strip()]
        substrates = _clean_substrates_for_ui(raw_sub_list)
        mutation_code = mutation.strip() or None
        u_id = uniprot_id.strip() or None
        
        if not u_id and not (seq_file and struct_file):
            st.error("Must provide either UniProt ID or BOTH Sequence and Structure file paths.")
            return

        try:
            model = _get_model(model_type, checkpoint_path)
            result = run_with_log_feedback(
                "Predicting kinetics...",
                predict_kinetics,
                uniprot_id=u_id, 
                mutation=mutation_code, 
                substrates=substrates, 
                ph=ph, 
                temperature_celsius=temperature, 
                sequence_file=seq_file,
                structure_file=struct_file,
                model=model, 
                model_type=model_type,
            )
        except thermokp.ThermoKPError as e:
            st.error(str(e))
        else:
            st.session_state["history"].append(result)
            with st.container(border=True):
                _render_result_card(result)
            if u_id:
                _render_structure(u_id, mutation_code, substrates)
            else:
                st.info("Structure viewer is currently disabled for local PDB file uploads.")

    if st.session_state["history"]:
        st.divider()
        st.markdown("#### 🕓 Session History")
        history_df = pd.DataFrame([vars(r) for r in st.session_state["history"]])
        st.dataframe(history_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Batch Tab
# ═══════════════════════════════════════════════════════════════════════════
def _batch_tab(model_type: str, checkpoint_path: str) -> None:
    if "batch_results" not in st.session_state:
        st.session_state["batch_results"] = None

    st.markdown("Upload a bulk query file (CSV, JSON, or YAML).")
    st.markdown("Required keys: `uniprot_id` (or `sequence_file` + `structure_file`), `substrates` (semicolon-separated or list), `ph`, `temperature`. Optional: `mutation`.")
    uploaded = st.file_uploader("Upload batch file", type=["csv", "json", "yaml", "yml"])

    if uploaded is not None and st.button("Run batch"):
        ext = Path(uploaded.name).suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(uploaded)
        elif ext == ".json":
            df = pd.read_json(uploaded)
        elif ext in (".yaml", ".yml"):
            df = pd.DataFrame(yaml.safe_load(uploaded))
        else:
            st.error("Unsupported file format.")
            return

        model = _get_model(model_type, checkpoint_path)
        st.session_state["batch_results"] = run_with_log_feedback(
            f"Running {len(df)} queries...", run_batch, model, df, model_type=model_type
        )

    batch_results = st.session_state["batch_results"]
    if batch_results is None:
        return

    st.divider()
    st.markdown("#### 📋 Batch Results")
    st.dataframe(batch_results, use_container_width=True)

    # Let user download the results
    csv_data = batch_results.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv_data, file_name="batch_results.csv", mime="text/csv")
    
# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    # A cleaner, more formal UI setup
    st.title("ThermoKP Inference Dashboard")

    with st.sidebar:
        st.markdown("# ThermoKP")
        st.markdown("Physics-informed neural network for zero-shot enzyme kinetics predictions, generating biophysically constrained $k_{cat}$ and $K_m$ estimations.")
        st.divider()
        st.markdown("### Model")
        model_type = st.radio(
            "Model type", options=thermokp.MODEL_TYPES, horizontal=True, format_func=str.upper
        )
        checkpoint_path = st.text_input(
            "Checkpoint path override",
            value="",
            help="Leave blank to auto-resolve models/best_model.pth.",
        )
        st.divider()
        
        st.markdown("### Caches")
        cache_root_input = st.text_input("Cache Root Directory", value=str(PROJECT_ROOT / "data" / "cache"))
        cache_root = Path(cache_root_input)
        
        pretrained_embeddings.PDB_CACHE_DIR = cache_root / "pdbs"
        
        st.markdown(f"Embeddings are under `{cache_root.name}/` and ESM2 is under `~/.cache`.")
            
        st.divider()

        st.markdown("💡 **Tip:** Run the ligand sync scripts once before processing to cache SMILES strings for reliable name conversion.")

    single_tab, batch_tab = st.tabs(["Single Query", "Batch Processing"])
    with single_tab:
        _single_query_tab(model_type, checkpoint_path)
    with batch_tab:
        _batch_tab(model_type, checkpoint_path)


if __name__ == "__main__":
    main()
