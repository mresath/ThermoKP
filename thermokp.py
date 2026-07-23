"""
===========================================================================
ThermoKP Inference
Description: Zero-shot k_cat/K_m inference from a UniProt enzyme + substrates
===========================================================================

Workflow:
1. Fetch the UniProt precursor sequence and signal-peptide cleavage offset.
2. Apply point mutation(s), if any, producing the mature sequence and the
   mutation descriptor - identical to
   src/data/processors/generate_tensors.py's apply_mutations.
3. Resolve each substrate (database-style name or SMILES) to an RDKit
   molecule; the first is the primary substrate, the rest are combined as
   co-substrates, exactly like the training pipeline's co_substrates field.
4. Assemble the 2D HeteroData graph (protein sequence, ligand/co-substrate
   D-MPNN graphs, ChemBERTa/ESM2 embeddings), then augment it with the 3D
   structural branch (AlphaFold/ESMFold + P2Rank pocket, RDKit conformers).
5. Load (or reuse a caller-supplied) trained checkpoint and run a forward
   pass, deriving k_cat/K_m: the Eyring-Arrhenius layer and Briggs-Haldane
   relation for the physics-informed model, or a direct log10 regression
   for the non-physics baseline (train_baseline_nn.py).

Known Caveats:
- The assembled HeteroData graph is never persisted to disk (unlike
  src/data/processors/generate_tensors.py's data/processed/tensors/ cache)
  - it is rebuilt in memory for every call and discarded once the forward
  pass is done, so inference never contaminates the training tensor
  cache. The underlying ESM2/ChemBERTa/UniProt/AlphaFold sub-caches under
  data/cache/ are still reused, since those are keyed by uniprot_id/
  mutation/SMILES regardless of caller.
- The 3D structural branch is mandatory, not an optional fallback: every
  training tensor carries it, so a checkpoint has never seen a zeroed-out
  3D contribution at this stage. A failure anywhere in that branch (no
  AlphaFold/ESMFold structure, no local P2Rank install, conformer
  generation failure) raises ThermoKPError rather than degrading silently.
- Designed for a long-lived caller (a Streamlit dashboard,
  src/evaluation/evaluate_dataset.py): call load_model() once and pass its
  result into repeated predict_kinetics() calls, rather than
  re-resolving/reloading a checkpoint on every prediction. Only
  ThermoKPError is meant to be caught by an interactive caller; any other
  exception indicates a genuine bug rather than an anticipated bad input.

Author: ThermoKP Team
License: MIT
"""

import argparse
import asyncio
import enum
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import rdkit.Chem as Chem
import torch
import torch.nn as nn
from accelerate import PartialState
from accelerate.logging import get_logger
from torch_geometric.data import HeteroData

from src.data.processors.dataset_validator import resolve_smiles
from src.data.processors.generate_tensors import (
    apply_mutations,
    build_hetero_graph,
    combine_ligand_mols,
)
from src.data.processors.geometry_processor import augment_hetero_graph_with_3d
from src.data.processors.pretrained_embeddings import (
    ESM2_MAX_SEQ_LEN,
    fetch_uniprot_cleavage_offset,
    fetch_uniprot_sequence,
    get_catalytic_site_mask,
    get_protein_embedding,
)
from train import ThermoKPModel
from train_baseline_nn import BaselineNNModel

# ═══════════════════════════════════════════════════════════════════════════
#  Logging Configuration
# ═══════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

PartialState()
logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Constants & Configuration
# ═══════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_NAMES = ("best_model.pth", "final_model.pth")
DEFAULT_BASELINE_CHECKPOINT_NAMES = ("best_baseline_model.pth", "final_baseline_model.pth")
MODEL_TYPES = ("pinn", "baseline")
EPS = 1e-12


# ═══════════════════════════════════════════════════════════════════════════
#  Errors
# ═══════════════════════════════════════════════════════════════════════════
class ErrorCode(enum.IntEnum):
    """Every distinct failure mode thermokp.py can raise, each a stable numeric code."""

    UNIPROT_SEQUENCE_UNAVAILABLE = 1
    SEQUENCE_TOO_LONG = 2
    MALFORMED_MUTATION_CODE = 3
    MUTATION_RESIDUE_MISMATCH = 4
    EMPTY_SUBSTRATE_LIST = 5
    SUBSTRATE_UNRESOLVED = 6
    SUBSTRATE_PARSE_FAILED = 7
    CO_SUBSTRATE_UNRESOLVED = 8
    STRUCTURE_PIPELINE_FAILED = 9
    CHECKPOINT_NOT_FOUND = 10
    CHECKPOINT_LOAD_FAILED = 11
    INVALID_MODEL_TYPE = 12


class ThermoKPError(RuntimeError):
    """A ThermoKP inference failure with a stable, unique error code.

    Attributes
    ----------
    code : ErrorCode
        The specific failure mode, stable across releases so callers (a
        Streamlit dashboard, a batch evaluation script) can branch on it
        rather than parsing the message text.
    """

    def __init__(self, code: ErrorCode, message: str) -> None:
        self.code = code
        super().__init__(f"[TKP-{code.value:03d}] {message}")


# ═══════════════════════════════════════════════════════════════════════════
#  Model Loading
# ═══════════════════════════════════════════════════════════════════════════
def _resolve_checkpoint(explicit: Optional[str], model_type: str = "pinn") -> Path:
    """Resolve which checkpoint file to load.

    Mirrors the fallback chain used throughout src/evaluation/: an
    explicit path if given, else, depending on `model_type`,
    `DEFAULT_CHECKPOINT_NAMES` (models/best_model.pth, then
    final_model.pth) for "pinn", or `DEFAULT_BASELINE_CHECKPOINT_NAMES`
    (the same two, each with a `_baseline` suffix) for "baseline" - the
    two model types' checkpoints never share a filename, so a baseline
    training run can never overwrite (or be mistaken for) a PINN run's
    checkpoint.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise ThermoKPError(ErrorCode.CHECKPOINT_NOT_FOUND, f"Checkpoint not found: {path}")
        return path

    default_names = DEFAULT_BASELINE_CHECKPOINT_NAMES if model_type == "baseline" else DEFAULT_CHECKPOINT_NAMES
    for name in default_names:
        candidate = PROJECT_ROOT / "models" / name
        if candidate.exists():
            logger.info(f"No checkpoint given, using {candidate}")
            return candidate

    raise ThermoKPError(
        ErrorCode.CHECKPOINT_NOT_FOUND,
        f"No checkpoint given and none of models/{{{','.join(default_names)}}} exist.",
    )


def load_model(
    checkpoint: Optional[str] = None,
    model_type: str = "pinn",
    hidden_channels: int = 64,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Resolve a checkpoint and load it into a ThermoKPModel or BaselineNNModel.

    Intended to be called once by a long-lived caller (a Streamlit
    dashboard, src/evaluation/evaluate_dataset.py) and reused across many
    predict_log_kinetics/predict_kinetics calls, rather than re-resolving
    and re-loading a checkpoint from disk on every prediction.

    Parameters
    ----------
    checkpoint : str, optional
        Explicit checkpoint path. Defaults to trying, in order,
        models/best_model.pth, final_model.pth for `model_type="pinn"`,
        or best_baseline_model.pth, final_baseline_model.pth for
        `model_type="baseline"`.
    model_type : str, optional
        "pinn" (default) for ThermoKPModel, or "baseline" for
        BaselineNNModel (train_baseline_nn.py).
    hidden_channels : int, optional
        Must match the checkpoint's training run; not saved alongside the
        checkpoint (see train.py/diagnose_checkpoint.py). Defaults to 64.
    device : torch.device, optional
        Defaults to CUDA if available, else CPU.

    Returns
    -------
    torch.nn.Module
        The loaded model, moved to `device` and in eval mode.

    Raises
    ------
    ThermoKPError
        CHECKPOINT_NOT_FOUND, INVALID_MODEL_TYPE, or CHECKPOINT_LOAD_FAILED.
    """
    if model_type not in MODEL_TYPES:
        raise ThermoKPError(
            ErrorCode.INVALID_MODEL_TYPE,
            f"Unknown model_type {model_type!r}; expected one of {MODEL_TYPES}.",
        )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = _resolve_checkpoint(checkpoint, model_type=model_type)

    model: nn.Module
    if model_type == "pinn":
        model = ThermoKPModel(hidden_channels=hidden_channels)
    else:
        model = BaselineNNModel(hidden_channels=hidden_channels)

    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except (RuntimeError, KeyError) as e:
        raise ThermoKPError(
            ErrorCode.CHECKPOINT_LOAD_FAILED,
            f"Failed to load {checkpoint_path} into a {model_type!r} model "
            f"(hidden_channels={hidden_channels}): {e}",
        ) from e

    return model.to(device).eval()


# ═══════════════════════════════════════════════════════════════════════════
#  Featurization
# ═══════════════════════════════════════════════════════════════════════════
def build_enzyme_substrate_graph(
    uniprot_id: str,
    mutation: Optional[str],
    substrates: List[str],
    ph: float,
    temperature_celsius: float,
) -> HeteroData:
    """Assemble the model input graph for one (enzyme, substrates, conditions) query.

    Reproduces src/data/processors/generate_tensors.py's featurization
    path one-to-one: UniProt sequence fetch, Mutate-then-Chop mutation
    handling, ESM2/ChemBERTa embeddings, 2D D-MPNN graphs, and the 3D
    structural (AlphaFold/ESMFold + P2Rank) branch. The returned HeteroData
    is never written to disk.

    Parameters
    ----------
    uniprot_id : str
        UniProt accession of the enzyme.
    mutation : str, optional
        ``/``-separated point-mutation code(s) (e.g. "N41D" or
        "N41D/N281D"), or None/empty for wild-type.
    substrates : list of str
        Substrate list; the first entry is the primary substrate (the one
        K_m is reported against), the rest are combined as co-substrates.
        Each entry may be a database-style chemical name or a raw SMILES
        string.
    ph : float
        Assay pH.
    temperature_celsius : float
        Assay temperature in Celsius.

    Returns
    -------
    HeteroData
        The assembled 2D+3D graph, ready for a model forward pass.

    Raises
    ------
    ThermoKPError
        One of EMPTY_SUBSTRATE_LIST, UNIPROT_SEQUENCE_UNAVAILABLE,
        SEQUENCE_TOO_LONG, MALFORMED_MUTATION_CODE,
        MUTATION_RESIDUE_MISMATCH, SUBSTRATE_UNRESOLVED,
        SUBSTRATE_PARSE_FAILED, CO_SUBSTRATE_UNRESOLVED, or
        STRUCTURE_PIPELINE_FAILED.
    """
    if not substrates:
        raise ThermoKPError(
            ErrorCode.EMPTY_SUBSTRATE_LIST,
            "substrates must contain at least the primary substrate.",
        )

    full_sequence = fetch_uniprot_sequence(uniprot_id)
    if not full_sequence:
        raise ThermoKPError(
            ErrorCode.UNIPROT_SEQUENCE_UNAVAILABLE,
            f"Could not fetch a sequence for UniProt ID {uniprot_id!r}.",
        )

    offset = fetch_uniprot_cleavage_offset(uniprot_id)
    mature_len = len(full_sequence) - offset
    if mature_len > ESM2_MAX_SEQ_LEN:
        raise ThermoKPError(
            ErrorCode.SEQUENCE_TOO_LONG,
            f"{uniprot_id}'s mature sequence has {mature_len} residues, "
            f"exceeding ESM2's {ESM2_MAX_SEQ_LEN}-residue context window.",
        )

    catalytic_site_mask = get_catalytic_site_mask(uniprot_id, mature_len, offset=offset)

    mutation_code = mutation.strip() if isinstance(mutation, str) and mutation.strip() else None
    try:
        mature_sequence, embed_cache_key, mutation_features = apply_mutations(
            uniprot_id, full_sequence, mutation_code, offset, catalytic_site_mask
        )
    except ValueError as e:
        message = str(e)
        code = (
            ErrorCode.MALFORMED_MUTATION_CODE
            if message.startswith("Malformed mutation code")
            else ErrorCode.MUTATION_RESIDUE_MISMATCH
        )
        raise ThermoKPError(code, message) from e

    protein_embedding = get_protein_embedding(uniprot_id, mature_sequence, cache_key=embed_cache_key)

    primary_smiles = resolve_smiles(substrates[0])
    if primary_smiles is None:
        raise ThermoKPError(
            ErrorCode.SUBSTRATE_UNRESOLVED,
            f"Could not resolve primary substrate {substrates[0]!r} to a SMILES string.",
        )
    primary_mol = Chem.MolFromSmiles(primary_smiles)
    if primary_mol is None:
        raise ThermoKPError(
            ErrorCode.SUBSTRATE_PARSE_FAILED,
            f"RDKit failed to parse primary substrate SMILES: {primary_smiles!r}",
        )

    co_sub_mol = None
    if len(substrates) > 1:
        co_sub_smiles = []
        for entry in substrates[1:]:
            smiles = resolve_smiles(entry)
            if smiles is None:
                raise ThermoKPError(
                    ErrorCode.CO_SUBSTRATE_UNRESOLVED,
                    f"Could not resolve co-substrate {entry!r} to a SMILES string.",
                )
            co_sub_smiles.append(smiles)
        try:
            co_sub_mol = combine_ligand_mols(co_sub_smiles)
        except ValueError as e:
            raise ThermoKPError(ErrorCode.CO_SUBSTRATE_UNRESOLVED, str(e)) from e

    data = build_hetero_graph(
        uniprot_id=uniprot_id,
        mutation=mutation_code or "",
        sequence=mature_sequence,
        primary_mol=primary_mol,
        co_sub_mol=co_sub_mol,
        protein_embedding=protein_embedding,
        catalytic_site_mask=catalytic_site_mask,
        ph=ph,
        temperature_celsius=temperature_celsius,
        mutation_features=mutation_features,
    )

    try:
        data = asyncio.run(
            augment_hetero_graph_with_3d(
                data, uniprot_id, mature_sequence, offset,
                catalytic_site_mask, primary_mol, co_sub_mol,
            )
        )
    except ValueError as e:
        raise ThermoKPError(ErrorCode.STRUCTURE_PIPELINE_FAILED, str(e)) from e

    return data


# ═══════════════════════════════════════════════════════════════════════════
#  Prediction
# ═══════════════════════════════════════════════════════════════════════════
def predict_log_kinetics(
    model: nn.Module, data: HeteroData, model_type: str = "pinn"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run a forward pass and return (log10_kcat, log10_km), regardless of model type.

    Shared by predict_kinetics (single query) and
    src/evaluation/evaluate_dataset.py (batched tensors already on disk),
    so both dispatch on model_type identically rather than duplicating
    the PINN-vs-baseline branching.

    Parameters
    ----------
    model : torch.nn.Module
        A loaded ThermoKPModel or BaselineNNModel (see load_model), already
        in eval mode on the correct device.
    data : HeteroData
        A single graph or a PyG-batched collection of graphs.
    model_type : str, optional
        "pinn" (default) or "baseline"; must match `model`'s actual type.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (log10_kcat, log10_km), each shape (num_graphs,).
    """
    with torch.no_grad():
        if model_type == "pinn":
            k_1, k_reverse, k_cat = model(data)
            log_kcat = torch.log10(k_cat + EPS).view(-1)
            km = (k_reverse + k_cat) / (k_1 + EPS)
            log_km = torch.log10(km + EPS).view(-1)
        elif model_type == "baseline":
            log_kcat, log_km = model(data)
            log_kcat = log_kcat.view(-1)
            log_km = log_km.view(-1)
        else:
            raise ThermoKPError(
                ErrorCode.INVALID_MODEL_TYPE,
                f"Unknown model_type {model_type!r}; expected one of {MODEL_TYPES}.",
            )
    return log_kcat, log_km


def predict_kinetics(
    uniprot_id: str,
    mutation: Optional[str],
    substrates: List[str],
    ph: float,
    temperature_celsius: float,
    model: Optional[nn.Module] = None,
    checkpoint: Optional[str] = None,
    model_type: str = "pinn",
    hidden_channels: int = 64,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """Predict k_cat (s^-1) and K_m (M) for one enzyme/substrate/condition query.

    High-level convenience wrapper combining load_model,
    build_enzyme_substrate_graph, and predict_log_kinetics. Pass a
    pre-loaded `model` (from a prior load_model call) when making many
    predictions - e.g. from a Streamlit dashboard or
    src/evaluation/evaluate_dataset.py - to avoid re-loading the
    checkpoint from disk on every call.

    Parameters
    ----------
    uniprot_id : str
        UniProt accession of the enzyme.
    mutation : str, optional
        ``/``-separated point-mutation code(s), or None/empty for wild-type.
    substrates : list of str
        Substrate list; the first entry is the primary substrate (the one
        K_m is reported against), the rest are combined as co-substrates.
        Each entry may be a database-style chemical name or a SMILES string.
    ph : float
        Assay pH.
    temperature_celsius : float
        Assay temperature in Celsius.
    model : torch.nn.Module, optional
        A pre-loaded model from load_model. If omitted, one is loaded
        (and discarded after this call) using `checkpoint`/`model_type`/
        `hidden_channels`/`device`.
    checkpoint, model_type, hidden_channels, device
        Forwarded to load_model when `model` is not supplied.

    Returns
    -------
    Tuple[float, float]
        (k_cat, K_m) in raw units: s^-1 and M.

    Raises
    ------
    ThermoKPError
        See build_enzyme_substrate_graph and load_model.
    """
    if model is None:
        model = load_model(
            checkpoint=checkpoint, model_type=model_type, hidden_channels=hidden_channels, device=device
        )
    assert model is not None

    model_device = next(model.parameters()).device
    data = build_enzyme_substrate_graph(uniprot_id, mutation, substrates, ph, temperature_celsius)
    data = data.to(model_device)

    log_kcat, log_km = predict_log_kinetics(model, data, model_type=model_type)
    kcat = float(10.0 ** log_kcat.item())
    km = float(10.0 ** log_km.item())
    return kcat, km


# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="ThermoKP Zero-Shot Kinetics Inference")
    parser.add_argument("--uniprot_id", type=str, required=True, help="UniProt accession of the enzyme.")
    parser.add_argument("--mutation", type=str, default=None,
                        help="Point-mutation code(s), e.g. 'N41D' or 'N41D/N281D'. Omit for wild-type.")
    parser.add_argument("--substrates", type=str, nargs="+", required=True,
                        help="Substrate list; the first is the primary substrate (K_m target), "
                             "the rest are combined as co-substrates. Each may be a chemical "
                             "name or a SMILES string.")
    parser.add_argument("--ph", type=float, required=True, help="Assay pH.")
    parser.add_argument("--temperature", type=float, required=True, help="Assay temperature in Celsius.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint .pth file. Defaults to trying, in order, "
                             "models/best_model.pth, final_model.pth for --model_type pinn, "
                             "or best_baseline_model.pth, final_baseline_model.pth for "
                             "--model_type baseline.")
    parser.add_argument("--model_type", type=str, default="pinn", choices=MODEL_TYPES,
                        help="'pinn' (default, ThermoKPModel) or 'baseline' (BaselineNNModel).")
    parser.add_argument("--hidden_channels", type=int, default=64,
                        help="Must match the checkpoint's training run. Defaults to 64.")
    args = parser.parse_args()

    try:
        kcat, km = predict_kinetics(
            uniprot_id=args.uniprot_id,
            mutation=args.mutation,
            substrates=args.substrates,
            ph=args.ph,
            temperature_celsius=args.temperature,
            checkpoint=args.checkpoint,
            model_type=args.model_type,
            hidden_channels=args.hidden_channels,
        )
    except ThermoKPError as e:
        logger.error(str(e))
        sys.exit(e.code.value)

    logger.info("=======================================================================")
    logger.info(f"k_cat : {kcat:.6g} s^-1")
    logger.info(f"K_m   : {km:.6g} M")
    logger.info("=======================================================================")


if __name__ == "__main__":
    main()
