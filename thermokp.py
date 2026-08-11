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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
import yaml
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
MODEL_TYPES = ("pinn", "baseline", "no_mutant")
DEFAULT_CHECKPOINT_NAMES = ("models/best_model.pth", "models/final_model.pth")
DEFAULT_BASELINE_CHECKPOINT_NAMES = (
    "models/best_baseline_model.pth",
    "models/final_baseline_model.pth",
)
DEFAULT_NO_MUTATION_CHECKPOINT_NAMES = (
    "models/best_no_mutation_model.pth",
    "models/final_no_mutation_model.pth",
)
EPS = 1e-12


@dataclass
class KineticsResult:
    uniprot_id: Optional[str]
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
    error: Optional[str] = None

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
    
    if model_type == "baseline":
        default_names = DEFAULT_BASELINE_CHECKPOINT_NAMES
    elif model_type == "no_mutant":
        default_names = DEFAULT_NO_MUTATION_CHECKPOINT_NAMES
    else:
        default_names = DEFAULT_CHECKPOINT_NAMES
    for name in default_names:
        candidate = PROJECT_ROOT / name
        if candidate.exists():
            logger.info(f"No checkpoint given, using {candidate}")
            return candidate

    raise ThermoKPError(
        ErrorCode.CHECKPOINT_NOT_FOUND,
        f"No checkpoint given and none of {{{','.join(default_names)}}} exist.",
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
    elif model_type == "no_mutant":
        from train_no_mutation_nn import NoMutationNNModel
        model = NoMutationNNModel(hidden_channels=hidden_channels)
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
    uniprot_id: Optional[str],
    mutation: Optional[str],
    substrates: List[str],
    ph: float,
    temperature_celsius: float,
    sequence_file: Optional[str] = None,
    structure_file: Optional[str] = None,
) -> HeteroData:
    if not substrates:
        raise ThermoKPError(
            ErrorCode.EMPTY_SUBSTRATE_LIST,
            "substrates must contain at least the primary substrate.",
        )

    if uniprot_id:
        full_sequence = fetch_uniprot_sequence(uniprot_id)
        if not full_sequence:
            raise ThermoKPError(
                ErrorCode.UNIPROT_SEQUENCE_UNAVAILABLE,
                f"Could not fetch a sequence for UniProt ID {uniprot_id!r}.",
            )
        offset = fetch_uniprot_cleavage_offset(uniprot_id)
    else:
        if not sequence_file or not structure_file:
            raise ThermoKPError(
                ErrorCode.UNIPROT_SEQUENCE_UNAVAILABLE,
                "Must provide either uniprot_id or BOTH sequence_file and structure_file.",
            )
        with open(sequence_file, "r") as f:
            lines = f.readlines()
            full_sequence = "".join([l.strip() for l in lines if not l.startswith(">")])
        offset = 0

    mature_len = len(full_sequence) - offset
    if mature_len > ESM2_MAX_SEQ_LEN:
        raise ThermoKPError(
            ErrorCode.SEQUENCE_TOO_LONG,
            f"The mature sequence has {mature_len} residues, "
            f"exceeding ESM2's {ESM2_MAX_SEQ_LEN}-residue context window.",
        )

    catalytic_site_mask = get_catalytic_site_mask(uniprot_id or "UNKNOWN", mature_len, offset=offset)

    mutation_code = mutation.strip() if isinstance(mutation, str) and mutation.strip() else None
    try:
        mature_sequence, embed_cache_key, mutation_features = apply_mutations(
            uniprot_id or "UNKNOWN", full_sequence, mutation_code, offset, catalytic_site_mask
        )
    except ValueError as e:
        message = str(e)
        code = (
            ErrorCode.MALFORMED_MUTATION_CODE
            if message.startswith("Malformed mutation code")
            else ErrorCode.MUTATION_RESIDUE_MISMATCH
        )
        raise ThermoKPError(code, message) from e

    protein_embedding = get_protein_embedding(uniprot_id or "UNKNOWN", mature_sequence, cache_key=embed_cache_key)

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
        uniprot_id=uniprot_id or "UNKNOWN",
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

    pdb_path = Path(structure_file) if structure_file else None

    try:
        data = asyncio.run(
            augment_hetero_graph_with_3d(
                data, uniprot_id or "UNKNOWN", mature_sequence, offset,
                catalytic_site_mask, primary_mol, co_sub_mol, pdb_path=pdb_path
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
    src/evaluation/evaluate_dataset.py (batched DataLoader queries).

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
        if model_type in ("pinn", "no_mutant"):
            k_1, k_reverse, k_cat = model(data)
            log_kcat = torch.log10(k_cat + EPS).view(-1)

            # E_total derived identically to src.physics.loss.PINNMultiTaskLoss
            E_total = k_1.new_ones(k_1.size()) * 1.0  # 1.0 µM
            k_2 = k_cat
            K_m = (k_reverse + k_2) / (k_1 * E_total)
            log_km = torch.log10(K_m + EPS).view(-1)
            return log_kcat, log_km
            
        elif model_type == "baseline":
            log10_kcat, log10_km = model(data)
            return log10_kcat.view(-1), log10_km.view(-1)
        else:
            raise ThermoKPError(
                ErrorCode.INVALID_MODEL_TYPE,
                f"Unknown model_type {model_type!r}; expected one of {MODEL_TYPES}.",
            )


def predict_kinetics(
    uniprot_id: Optional[str],
    mutation: Optional[str],
    substrates: List[str],
    ph: float,
    temperature_celsius: float,
    sequence_file: Optional[str] = None,
    structure_file: Optional[str] = None,
    model: Optional[nn.Module] = None,
    checkpoint: Optional[str] = None,
    model_type: str = "pinn",
    hidden_channels: int = 64,
    device: Optional[torch.device] = None,
) -> KineticsResult:
    if model is None:
        model = load_model(
            checkpoint=checkpoint, model_type=model_type, hidden_channels=hidden_channels, device=device
        )
    assert model is not None

    model_device = next(model.parameters()).device
    data = build_enzyme_substrate_graph(
        uniprot_id, mutation, substrates, ph, temperature_celsius, sequence_file, structure_file
    )
    data = data.to(model_device)

    with torch.no_grad():
        if model_type in ("pinn", "no_mutant"):
            k_1, k_reverse, k_cat, delta_g_ddagger, kappa = model(data, return_components=True)
            k_cat_val = float(k_cat.view(-1).item())
            k_m_val = float(((k_reverse + k_cat) / (k_1 + EPS)).view(-1).item())
            k1_val = float(k_1.view(-1).item())
            k_reverse_val = float(k_reverse.view(-1).item())
            delta_g_val = float(delta_g_ddagger.view(-1).item())
            kappa_val = float(kappa.view(-1).item())
        else:
            log_kcat, log_km = predict_log_kinetics(model, data, model_type=model_type)
            k_cat_val = float(10.0 ** log_kcat.item())
            k_m_val = float(10.0 ** log_km.item())
            k1_val = k_reverse_val = delta_g_val = kappa_val = None

    k_a_val = k_cat_val / (k_m_val + EPS)

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

def run_batch(
    model: nn.Module, 
    df: pd.DataFrame, 
    model_type: str = "pinn",
    sequence_col: str = "sequence_file",
    structure_col: str = "structure_file"
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for row_num, (_, row) in enumerate(df.iterrows()):
        uniprot_id = row.get("uniprot_id")
        uniprot_id = None if pd.isna(uniprot_id) or not str(uniprot_id).strip() else str(uniprot_id).strip()
        
        mutation = row.get("mutation")
        mutation = None if pd.isna(mutation) or not str(mutation).strip() else str(mutation).strip()
        
        raw_subs = row.get("substrates", "")
        if isinstance(raw_subs, list):
            substrates = [str(s).strip() for s in raw_subs if str(s).strip()]
        else:
            substrates = [s.strip() for s in str(raw_subs).split(";") if s.strip()]
        seq_file = row.get(sequence_col)
        seq_file = None if pd.isna(seq_file) or not str(seq_file).strip() else str(seq_file).strip()
        
        struct_file = row.get(structure_col)
        struct_file = None if pd.isna(struct_file) or not str(struct_file).strip() else str(struct_file).strip()

        logger.info(f"[{row_num + 1}/{len(df)}] {uniprot_id or seq_file} ({mutation or 'WT'})")
        try:
            result = predict_kinetics(
                uniprot_id=uniprot_id,
                mutation=mutation,
                substrates=substrates,
                ph=float(row["ph"]),
                temperature_celsius=float(row["temperature"]),
                sequence_file=seq_file,
                structure_file=struct_file,
                model=model,
                model_type=model_type,
            )
            records.append(asdict(result))
        except ThermoKPError as e:
            record = {
                "uniprot_id": uniprot_id,
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



# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="ThermoKP Zero-Shot Kinetics Inference")
    parser.add_argument("--uniprot_id", type=str, default=None, help="UniProt accession of the enzyme.")
    parser.add_argument("--sequence_file", type=str, default=None, help="Path to sequence file (txt/fasta).")
    parser.add_argument("--structure_file", type=str, default=None, help="Path to structure file (pdb).")
    parser.add_argument("--mutation", type=str, default=None,
                        help="Point-mutation code(s), e.g. 'N41D' or 'N41D/N281D'. Omit for wild-type.")
    parser.add_argument("--substrates", type=str, nargs="*", default=[],
                        help="Substrate list; the first is the primary substrate (K_m target), "
                             "the rest are combined as co-substrates. Each may be a chemical "
                             "name or a SMILES string.")
    parser.add_argument("--ph", type=float, default=7.0, help="Assay pH.")
    parser.add_argument("--temperature", type=float, default=25.0, help="Assay temperature in Celsius.")
    
    parser.add_argument("--input_file", type=str, default=None, help="Path to input bulk file (csv/json/yaml).")
    parser.add_argument("--output_file", type=str, default=None, help="Path to output bulk file (csv/json/yaml).")
    
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint .pth file.")
    parser.add_argument("--model_type", type=str, default="pinn", choices=MODEL_TYPES,
                        help="'pinn' (default, ThermoKPModel) or 'baseline' (BaselineNNModel).")
    parser.add_argument("--hidden_channels", type=int, default=64,
                        help="Must match the checkpoint's training run. Defaults to 64.")
    args = parser.parse_args()

    model = load_model(
        checkpoint=args.checkpoint, model_type=args.model_type, hidden_channels=args.hidden_channels
    )

    if args.input_file:
        input_path = Path(args.input_file)
        if input_path.suffix == ".csv":
            df = pd.read_csv(input_path)
        elif input_path.suffix == ".json":
            df = pd.read_json(input_path)
        elif input_path.suffix in (".yaml", ".yml"):
            with open(input_path, "r") as f:
                df = pd.DataFrame(yaml.safe_load(f))
        else:
            logger.error("Input file must be .csv, .json, or .yaml")
            sys.exit(1)
            
        logger.info(f"Loaded {len(df)} records from {args.input_file}")
        results_df = run_batch(model, df, model_type=args.model_type)
        
        if args.output_file:
            out_path = Path(args.output_file)
            if out_path.suffix == ".csv":
                csv_df = results_df.copy()
                csv_df["substrates"] = csv_df["substrates"].apply(lambda x: ";".join(x) if isinstance(x, list) else x)
                csv_df.to_csv(out_path, index=False)
            elif out_path.suffix == ".json":
                results_df.to_json(out_path, orient="records", indent=2)
            elif out_path.suffix in (".yaml", ".yml"):
                with open(out_path, "w") as f:
                    yaml.dump(results_df.to_dict(orient="records"), f)
            else:
                logger.error("Output file must be .csv, .json, or .yaml")
                sys.exit(1)
            logger.info(f"Saved results to {args.output_file}")
        else:
            print(results_df.to_string())
    else:
        if not args.uniprot_id and not (args.sequence_file and args.structure_file):
            logger.error("Must provide either --uniprot_id or both --sequence_file and --structure_file")
            sys.exit(1)
        if not args.substrates:
            logger.error("Must provide --substrates when running single query")
            sys.exit(1)
            
        try:
            result = predict_kinetics(
                uniprot_id=args.uniprot_id,
                mutation=args.mutation,
                substrates=args.substrates,
                ph=args.ph,
                temperature_celsius=args.temperature,
                sequence_file=args.sequence_file,
                structure_file=args.structure_file,
                model=model,
                model_type=args.model_type,
            )
        except ThermoKPError as e:
            logger.error(str(e))
            sys.exit(e.code.value)

        logger.info("=======================================================================")
        logger.info(f"k_cat : {result.k_cat:.6g} s^-1")
        logger.info(f"K_m   : {result.k_m:.6g} M")
        logger.info(f"k_a   : {result.k_a:.6g} M^-1 s^-1")
        if result.k1 is not None:
            logger.info(f"k1    : {result.k1:.6g}")
            logger.info(f"k_-1  : {result.k_reverse:.6g}")
            logger.info(f"deltaG: {result.delta_g_ddagger:.6g}")
            logger.info(f"kappa : {result.kappa:.6g}")
        logger.info("=======================================================================")


if __name__ == "__main__":
    main()
