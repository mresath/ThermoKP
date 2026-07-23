"""
===========================================================================
Evaluate Dataset
Description: Regression metrics for a trained checkpoint over held-out or
             training data
===========================================================================

Workflow:
0. With no --split given, runs both 'val' and 'benchmark' as two separate
   evaluations (each summarized, logged, and saved under its own key) -
   a quick, local sanity check against data the model trained against
   the split of, followed by the slower true zero-shot benchmark. Pass
   --split explicitly to run only one.
1. --split benchmark: loads `benchmark_parameters` from
   data/thermokp_database.db and runs thermokp.py's live inference
   pipeline (UniProt/SMILES fetch, ESM2/ChemBERTa, 3D structural branch)
   per row, since this whole-enzyme holdout
   (src/data/processors/clean_records.py) was never tensorized.
2. --split {train,val,all}: loads cached PyG tensors from
   data/processed/tensors/ via EnzymeDataset, optionally reconstructing
   the reaction-grouped 90/10 train/val split
   (src/training/pinn_training.train_val_split; "all" is the full
   training set, train and validation combined), and runs the model
   directly - no live featurization, no network calls.
3. Computes log-space RMSE, R^2, and p1mag (formulas identical to
   PINNTrainer._regression_metrics) overall, per percentile-of-error bin
   (quartiles plus finer 90/95/99th-percentile tail bins), and split into
   wild-type-only/mutant-only subsections, for both k_cat and K_m -
   including the full wild-type/mutant log-space absolute-error
   distributions (not just their summary statistics), for later plotting
   as a density histogram analogous to images/generators/
   dataset_composition.py's kcat/K_m value-distribution panel.
4. Merges this run's metrics and per-record predictions into the single
   shared results store data/results/eval.json, keyed by
   "{split}_{model_type}" - every evaluation ever run stays queryable from
   one file (parity plots, error-distribution plots) rather than being
   scattered across many small per-run files.

Known Caveats:
- The benchmark path calls thermokp.build_enzyme_substrate_graph per row
  and is therefore far slower than the tensor-based paths (live UniProt/
  AlphaFold/ESM2/ChemBERTa calls per enzyme, though disk-cached after the
  first call); a row that raises thermokp.ThermoKPError (e.g. an
  unattainable structure) is logged and excluded rather than aborting the
  whole run. Since the same row fails identically on every future run,
  it is also appended to failed_benchmark.txt and purged from
  benchmark_parameters, so it is not re-attempted and re-skipped on
  every subsequent evaluation.
- --split train/val reconstructs the split from data/processed/tensors/'s
  current file listing; it is only correct if that listing hasn't changed
  since the checkpoint's training run (see train_val_split).
- Only --hidden_channels needs to match the checkpoint's run; no other
  run config is saved alongside a checkpoint (see thermokp.load_model).

Author: ThermoKP Team
License: MIT
"""

import argparse
import json
import logging
import math
import os
import re
import sqlite3
import sys
import uuid
from pathlib import Path
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import pandas as pd
import torch
import subprocess
from torch.utils.data import Subset
from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from accelerate import PartialState  # noqa: E402
from accelerate.logging import get_logger  # noqa: E402
import thermokp  # noqa: E402
from src.data.models.dataset import EnzymeDataset  # noqa: E402
from src.training.pinn_training import train_val_split  # noqa: E402
from Bio import Align  # noqa: E402
from src.data.processors.pretrained_embeddings import (  # noqa: E402
    fetch_uniprot_sequence,
    fetch_uniprot_cleavage_offset
)
from src.data.processors.generate_tensors import MUTATION_CODE_PATTERN  # noqa: E402

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
#  Constants & Configuration
# ═══════════════════════════════════════════════════════════════════════════
DB_PATH = PROJECT_ROOT / "data" / "thermokp_database.db"
FAILED_BENCHMARK_LOG_PATH = PROJECT_ROOT / "failed_benchmark.txt"
ENTRY_FILENAME_RE = re.compile(r"^entry(\d+)_")

# Percentile-of-|error| bins: quartiles, refined into finer tail slices
# ("also include 90 etc") so a small tail of badly-mispredicted records
# can be told apart from uniformly-spread error.
PERCENTILE_BINS: List[Tuple[int, int]] = [
    (0, 25), (25, 50), (50, 75), (75, 90), (90, 95), (95, 99), (99, 100)
]


# ═══════════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════════
def _regression_metrics(log_pred: torch.Tensor, log_target: torch.Tensor) -> Dict[str, float]:
    """Log-space RMSE, R^2, and p1mag over a set of predictions.

    Mirrors `PINNTrainer._regression_metrics`
    (src/training/pinn_training.py) exactly, so these numbers are
    directly comparable to the training loop's own periodic validation
    reports.

    Parameters
    ----------
    log_pred, log_target : torch.Tensor
        log10-space predictions/targets, same shape.

    Returns
    -------
    dict of str to float
        'rmse', 'r2' (nan if `log_target` has zero variance), 'p1mag'
        (percent within one order of magnitude), and 'n' (record count).
    """
    residuals = log_pred - log_target
    ss_res = torch.sum(residuals ** 2)
    ss_tot = torch.sum((log_target - log_target.mean()) ** 2)
    r2 = (1.0 - ss_res / ss_tot).item() if ss_tot > 0 else float("nan")
    rmse = torch.sqrt(residuals.pow(2).mean()).item()
    p1mag = (residuals.abs() <= 1.0).float().mean().item() * 100.0
    return {"rmse": rmse, "r2": r2, "p1mag": p1mag, "n": log_pred.numel()}


def _binned_rmse(log_pred: torch.Tensor, log_target: torch.Tensor) -> Dict[str, Dict[str, float]]:
    """RMSE within each percentile-of-|error| bin (quartiles, plus finer tail bins).

    Records are sorted by `|log_pred - log_target|` and split into
    percentile ranges of that error (PERCENTILE_BINS), then RMSE is
    computed within each range - showing whether error is spread evenly
    or concentrated in a small tail, rather than a single set-wide number.

    Parameters
    ----------
    log_pred, log_target : torch.Tensor
        log10-space predictions/targets, same shape.

    Returns
    -------
    dict of str to dict
        One entry per `PERCENTILE_BINS` range, keyed `"p{lo}-{hi}"`, each
        holding `'rmse'` (`nan` if the bin is empty) and `'n'` (record
        count in that bin).
    """
    residuals = log_pred - log_target
    abs_residuals = residuals.abs()
    n = abs_residuals.numel()
    order = torch.argsort(abs_residuals)

    result: Dict[str, Dict[str, float]] = {}
    for lo, hi in PERCENTILE_BINS:
        lo_idx = round(lo / 100 * n)
        hi_idx = round(hi / 100 * n)
        idx = order[lo_idx:hi_idx]
        label = f"p{lo}-{hi}"
        if idx.numel() == 0:
            result[label] = {"rmse": float("nan"), "n": 0}
            continue
        bin_residuals = residuals[idx]
        result[label] = {"rmse": torch.sqrt(bin_residuals.pow(2).mean()).item(), "n": idx.numel()}
    return result


def summarize_parameter(
    records: List[Dict[str, object]], log_pred_key: str, log_target_key: str
) -> Dict[str, object]:
    """Overall + percentile-binned + wild-type/mutant metrics for one parameter.

    Also includes the full wild-type/mutant log-space absolute-error
    distributions (`wild_type_abs_log_errors`/`mutant_abs_log_errors`),
    not just their summary statistics - the raw arrays a later figure can
    plot as an overlaid density histogram, analogous to
    images/generators/dataset_composition.py's kcat/K_m value-distribution
    panel (there split by wild-type/mutant status; here split the same
    way, but over |log_pred - log_target| instead of the raw value).

    Parameters
    ----------
    records : list of dict
        Per-record dicts, each expected to hold `log_pred_key`,
        `log_target_key`, and (optionally) a truthy `"mutation"` entry
        marking a mutant record.
    log_pred_key : str
        Key into each record for the log10-space prediction.
    log_target_key : str
        Key into each record for the log10-space ground truth.

    Returns
    -------
    dict of str to object
        `'overall'` (`_regression_metrics`), `'percentile_bins'`
        (`_binned_rmse`), `'wild_type_abs_log_errors'` /
        `'mutant_abs_log_errors'` (raw per-record `|log_pred -
        log_target|` lists), and `'wild_type'` / `'mutant'`
        (`_regression_metrics`, present only when that subset is
        non-empty). Non-finite predictions/targets are dropped before
        any statistic is computed; if none remain, `'overall'` is
        `nan`-filled with `n=0` and the rest are empty.
    """
    log_pred = torch.tensor([cast(float, r.get(log_pred_key, float('nan'))) for r in records], dtype=torch.float64)
    log_target = torch.tensor([cast(float, r.get(log_target_key, float('nan'))) for r in records], dtype=torch.float64)
    is_mutant = torch.tensor([bool(r.get("mutation")) for r in records], dtype=torch.bool)
    
    valid_mask = ~(torch.isnan(log_pred) | torch.isnan(log_target) | torch.isinf(log_pred) | torch.isinf(log_target))
    log_pred = log_pred[valid_mask]
    log_target = log_target[valid_mask]
    is_mutant = is_mutant[valid_mask]

    if log_pred.numel() == 0:
        return {
            "overall": {"rmse": float("nan"), "r2": float("nan"), "p1mag": 0.0, "n": 0},
            "percentile_bins": {},
            "wild_type_abs_log_errors": [],
            "mutant_abs_log_errors": [],
        }

    abs_errors = (log_pred - log_target).abs()

    summary: Dict[str, object] = {
        "overall": _regression_metrics(log_pred, log_target),
        "percentile_bins": _binned_rmse(log_pred, log_target),
        "wild_type_abs_log_errors": abs_errors[~is_mutant].tolist(),
        "mutant_abs_log_errors": abs_errors[is_mutant].tolist(),
    }
    if bool(is_mutant.any()):
        summary["mutant"] = _regression_metrics(log_pred[is_mutant], log_target[is_mutant])
    if bool((~is_mutant).any()):
        summary["wild_type"] = _regression_metrics(log_pred[~is_mutant], log_target[~is_mutant])
    return summary


# ═══════════════════════════════════════════════════════════════════════════
#  Record Collection
# ═══════════════════════════════════════════════════════════════════════════
def _parse_entry_id(filename_stem: str) -> Optional[int]:
    """Extract the numeric entry_id from a tensor filename stem (`entry{id}_{uniprot_id}`)."""
    match = ENTRY_FILENAME_RE.match(filename_stem)
    return int(match.group(1)) if match else None


def _load_benchmark_records(
    model: torch.nn.Module,
    model_type: str,
    device: torch.device,
    limit: Optional[int] = None,
    benchmark_cache_dir: Optional[Path] = None,
) -> List[Dict[str, object]]:
    """Evaluate every row of `benchmark_parameters` via thermokp.py's live inference pipeline.

    Builds each row's enzyme-substrate graph on demand (UniProt/SMILES
    fetch, ESM2/ChemBERTa embedding, 3D structural branch) via
    `thermokp.build_enzyme_substrate_graph`, caching the resulting tensor
    under `benchmark_cache_dir` (keyed by `entry{entry_id}_{uniprot_id}`)
    so a repeated evaluation run skips the live featurization entirely. A
    row that raises `thermokp.ThermoKPError` is logged and excluded
    rather than aborting the run, and (on the main process) purged from
    `benchmark_parameters` via `_purge_failed_benchmark_entries` so it is
    not re-attempted on a future run.

    Parameters
    ----------
    model : torch.nn.Module
        The model to run inference with.
    model_type : str
        Passed through to `thermokp.predict_log_kinetics` to select the
        model-specific prediction path (e.g. "pinn" vs "baseline").
    device : torch.device
        Device the built graph tensors are moved to before inference.
    limit : int, optional
        Caps the number of benchmark rows evaluated (applied as a SQL
        `LIMIT`), by default None (evaluate every row).
    benchmark_cache_dir : Path, optional
        Directory for the per-row cached input tensors. Required (via an
        internal assertion) whenever a row is not already cached, since a
        cache miss must know where to write the newly built tensor.

    Returns
    -------
    list of dict
        One dict per successfully evaluated row: `entry_id`,
        `uniprot_id`, `mutation`, `kcat_true`/`km_true` (km in Molar),
        `log_kcat_true`/`log_km_true`, `log_kcat_pred`/`log_km_pred`, and
        `kcat_pred`/`km_pred`. Rows that raised `ThermoKPError` are
        omitted rather than represented with placeholder values.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM benchmark_parameters ORDER BY entry_id ASC"
    if limit is not None:
        query += f" LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()

    total = len(df)
    records: List[Dict[str, object]] = []
    failed_entries: List[Tuple[int, str]] = []
    skipped = 0

    for i, row in df.iterrows():
        uniprot_id = row["uniprot_id"]
        logger.info(f"[{cast(int, i) + 1}/{total}] Evaluating {uniprot_id} (benchmark entry {row['entry_id']})...")

        mutation_code = row["mutation"] if isinstance(row["mutation"], str) and row["mutation"].strip() else None
        substrates = [row["measured_substrate"]]
        co_substrates = row.get("co_substrates")
        if isinstance(co_substrates, str) and co_substrates.strip():
            substrates.extend(s.strip() for s in co_substrates.split(";") if s.strip())

        try:
            cache_file = None
            if benchmark_cache_dir is not None:
                cache_file = benchmark_cache_dir / f"entry{row['entry_id']}_{uniprot_id}.pt"
            
            data = None
            if cache_file is not None and cache_file.exists():
                data = torch.load(cache_file, weights_only=False)
            
            if data is None:
                assert benchmark_cache_dir is not None
                data = thermokp.build_enzyme_substrate_graph(
                    uniprot_id, mutation_code, substrates, float(row["ph"]), float(row["temperature"])
                )
                if cache_file is not None:
                    benchmark_cache_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(data, cache_file)

            data = data.to(device)
            log_kcat_pred, log_km_pred = thermokp.predict_log_kinetics(model, data, model_type=model_type)
        except thermokp.ThermoKPError as e:
            logger.warning(f"Skipping benchmark entry {row['entry_id']} ({uniprot_id}): {e}")
            entry_id = int(row["entry_id"])
            failed_entries.append(
                (entry_id, f"entry_id={entry_id} uniprot_id={uniprot_id} mutation={mutation_code or ''!r}: {e}")
            )
            skipped += 1
            continue

        kcat_true = float(row["kcat"])
        km_true_m = float(row["km"]) * 1e-3
        log_kcat_pred_v = log_kcat_pred.item()
        log_km_pred_v = log_km_pred.item()
        records.append({
            "entry_id": int(row["entry_id"]),
            "uniprot_id": uniprot_id,
            "mutation": mutation_code or "",
            "kcat_true": kcat_true,
            "km_true": km_true_m,
            "log_kcat_true": math.log10(kcat_true + thermokp.EPS),
            "log_km_true": math.log10(km_true_m + thermokp.EPS),
            "log_kcat_pred": log_kcat_pred_v,
            "log_km_pred": log_km_pred_v,
            "kcat_pred": 10.0 ** log_kcat_pred_v,
            "km_pred": 10.0 ** log_km_pred_v,
        })

    if skipped:
        logger.warning(f"Skipped {skipped}/{total} benchmark entries due to inference errors.")
    if failed_entries and PartialState().is_main_process:
        _purge_failed_benchmark_entries(failed_entries)

    return records


def _purge_failed_benchmark_entries(failed_entries: List[Tuple[int, str]]) -> None:
    """Append failed benchmark rows to failed_benchmark.txt and delete them from benchmark_parameters.

    A row that raises thermokp.ThermoKPError fails identically on every
    future run (the failure stems from the row's data - an unresolvable
    UniProt sequence, an unparseable substrate - not from run-to-run
    variance), so once recorded here it is purged rather than
    re-attempted and re-skipped on every subsequent evaluation.

    Parameters
    ----------
    failed_entries : list of (int, str)
        Each entry's `entry_id` paired with its log line.
    """
    with FAILED_BENCHMARK_LOG_PATH.open("a", encoding="utf-8") as f:
        for _, line in failed_entries:
            f.write(line + "\n")

    entry_ids = [entry_id for entry_id, _ in failed_entries]
    conn = sqlite3.connect(DB_PATH)
    placeholders = ",".join("?" * len(entry_ids))
    conn.execute(f"DELETE FROM benchmark_parameters WHERE entry_id IN ({placeholders})", entry_ids)
    conn.commit()
    conn.close()

    logger.warning(
        f"Purged {len(entry_ids)} permanently-failing entries from benchmark_parameters "
        f"(see {FAILED_BENCHMARK_LOG_PATH})."
    )


@lru_cache(maxsize=None)
def get_mature_sequence(uniprot_id: str, mutation_code: Optional[str]) -> Optional[str]:
    """Retrieve the mature, mutated sequence for a given UniProt ID and mutation code.

    Fetches the full (signal-peptide-including) UniProt sequence and its
    cleavage offset, applies every point mutation in `mutation_code` to
    the full-length sequence, then strips the signal peptide by slicing
    from `offset` - mutation positions in this dataset's convention are
    1-indexed and given against whichever of the full-length or mature
    sequence a source database happened to number against, so each
    mutation is resolved by first trying the position directly against
    the full sequence and, on a residue mismatch, retrying at
    `position + offset` (the same position, but against the mature
    numbering) before giving up.

    Parameters
    ----------
    uniprot_id : str
        UniProt accession to fetch the sequence for.
    mutation_code : str, optional
        Slash-separated mutation codes (e.g. `"A123V/D456E"`), each
        matched by `MUTATION_CODE_PATTERN`. None or empty for a
        wild-type sequence.

    Returns
    -------
    str, optional
        The mature (signal-peptide-stripped), mutated sequence, or None
        if the UniProt sequence could not be fetched, or if any mutation
        code is malformed or its expected wild-type residue does not
        match the sequence at either candidate position.
    """
    full_sequence = fetch_uniprot_sequence(uniprot_id)
    if not full_sequence:
        return None
    offset = fetch_uniprot_cleavage_offset(uniprot_id)
    mutated_full_sequence = full_sequence

    if mutation_code:
        for m_code in mutation_code.split("/"):
            match = MUTATION_CODE_PATTERN.match(m_code)
            if not match:
                logger.warning(f"Malformed mutation code {m_code!r} for {uniprot_id}")
                return None
            wt_res, pos_str, mut_res = match.groups()
            seq_idx = int(pos_str) - 1

            if 0 <= seq_idx < len(full_sequence) and full_sequence[seq_idx] == wt_res:
                mut_idx = seq_idx
            elif 0 <= seq_idx + offset < len(full_sequence) and full_sequence[seq_idx + offset] == wt_res:
                mut_idx = seq_idx + offset
            else:
                logger.warning(f"Mutation {m_code} mismatch for {uniprot_id}")
                return None
            mutated_full_sequence = mutated_full_sequence[:mut_idx] + mut_res + mutated_full_sequence[mut_idx + 1:]

    return mutated_full_sequence[offset:]


def _get_training_sequences(model_type: str) -> Dict[str, str]:
    """Retrieve mature sequences from a single model's training set.

    For internal models, queries `clean_parameters` excluding `benchmark_parameters`.
    For external models, reads `data/external/<model>/train_sequences.json` if present,
    otherwise falls through to the ThermoKP training split.

    Parameters
    ----------
    model_type : str
        "pinn"/"baseline" for the ThermoKP training split, or an
        external model name (e.g. "dlkcat") for that model's own
        training set.

    Returns
    -------
    dict of str to str
        Maps a `uniprot_id` (or `"{uniprot_id}_{mutation}"` when
        mutated) to its mature sequence.
    """
    if model_type not in ["pinn", "baseline"]:
        ext_train_path = PROJECT_ROOT / "data" / "external" / model_type / "train_sequences.json"
        if ext_train_path.exists():
            with open(ext_train_path, "r", encoding="utf-8") as f:
                return json.load(f)
        # Fall through to the shared ThermoKP training split below.

    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT c.uniprot_id, c.mutation 
    FROM clean_parameters c 
    LEFT JOIN benchmark_parameters b 
    ON c.entry_id = b.entry_id 
    WHERE b.entry_id IS NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    train_seqs: Dict[str, str] = {}
    for _, row in df.iterrows():
        uid = str(row["uniprot_id"])
        mut = row["mutation"]
        mut_str = str(mut)
        mut = "" if (mut is None or mut_str == "nan") else mut_str
        key = f"{uid}_{mut}" if mut else uid
        if key not in train_seqs:
            seq = get_mature_sequence(uid, mut)
            if seq:
                train_seqs[key] = seq
    return train_seqs


def _get_union_training_sequences() -> Dict[str, str]:
    """Return the union of all model training sequences.

    Merges the ThermoKP training split with every available external model
    training set found under `data/external/*/train_sequences.json`. Used for
    the cross-model seq-identity filter so the benchmark subset is consistent
    across all models.

    Returns
    -------
    dict of str to str
        Maps a `uniprot_id` (or `"{uniprot_id}_{mutation}"` when
        mutated) to its mature sequence, across every model's training
        set combined.
    """
    union: Dict[str, str] = _get_training_sequences("pinn")
    external_models = ["dlkcat", "unikp", "catpred"]
    for m in external_models:
        ext_path = PROJECT_ROOT / "data" / "external" / m / "train_sequences.json"
        if ext_path.exists():
            with open(ext_path, "r", encoding="utf-8") as f:
                union.update(json.load(f))
    return union


def _calculate_max_seq_ids(
    records: List[Dict[str, object]],
    train_seqs_getter: Callable[[], Dict[str, str]],
    cache_path: Path,
    record_key: str = "max_seq_id",
) -> None:
    """Calculate and attach per-record maximum sequence identity against a training set.

    Parameters
    ----------
    records : list of dict
        Each record must contain ``uniprot_id`` and optionally ``mutation``.
        ``record_key`` is written in-place.
    train_seqs_getter : callable
        Zero-argument callable returning ``{key: sequence}`` for the reference set.
        Called lazily only when a cache miss is encountered.
    cache_path : Path
        JSON file used to persist computed identities across runs.
    record_key : str
        Key written into each record dict (default: ``max_seq_id``).
    """
    cache: Dict[str, float] = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

    train_seqs: Optional[Dict[str, str]] = None
    aligner: Optional[Align.PairwiseAligner] = None
    cache_dirty = False

    for r in records:
        uid = str(r["uniprot_id"])
        mut = r.get("mutation", "")
        mut_str = str(mut)
        mut = "" if (mut is None or mut_str == "nan") else mut_str
        key = f"{uid}_{mut}" if mut else uid

        if key in cache:
            r[record_key] = cache[key]
            continue

        if train_seqs is None:
            train_seqs = train_seqs_getter()
            aligner = Align.PairwiseAligner()
            aligner.mode = "global"

        test_seq = get_mature_sequence(uid, mut)
        if not test_seq:
            r[record_key] = 1.0  # Conservatively assume identical if unavailable
            continue

        max_id = 0.0
        for tr_key, tr_seq in train_seqs.items():
            if tr_key == key:
                continue
            assert aligner is not None
            score = aligner.score(test_seq, tr_seq)
            identity = score / max(len(test_seq), len(tr_seq))
            if identity > max_id:
                max_id = identity

        cache[key] = max_id
        r[record_key] = max_id
        cache_dirty = True

    if cache_dirty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

_EXTERNAL_INPUTS_EXPORTED = False

def _load_external_records(model_type: str, split: str) -> List[Dict[str, Any]]:
    """Load pre-computed external-model predictions from CSV and merge in ground truth.

    Reads `data/external_predictions/{model_type}_{split}_predictions.csv`,
    running the export script (`export_benchmark_csv.py`) and then the
    external model's own `run_inference.py` (under its own `.venv`) to
    generate it if missing. Each record's `entry_id` is then joined
    against `benchmark_parameters` to attach ground-truth
    `kcat_true`/`km_true` and `log_kcat_true`/`log_km_true`, so the
    returned records are directly comparable to the internal models'
    record format.

    Parameters
    ----------
    model_type : str
        External model name (e.g. "dlkcat", "unikp", "catpred").
    split : str
        Split name the predictions were generated for (only "benchmark"
        is currently produced by the external inference scripts).

    Returns
    -------
    list of dict
        One dict per row of the predictions CSV, with ground-truth
        columns merged in by `entry_id`. Empty if the CSV still could
        not be produced (logged as an error rather than raised).
    """
    global _EXTERNAL_INPUTS_EXPORTED
    csv_path = PROJECT_ROOT / "data" / "external_predictions" / f"{model_type}_{split}_predictions.csv"
    if not csv_path.exists():
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.warning(f"Prediction file {csv_path} not found. Running inference for {model_type}...")
            if not _EXTERNAL_INPUTS_EXPORTED:
                subprocess.run(["uv", "run", "python", "src/evaluation/export_benchmark_csv.py"], cwd=str(PROJECT_ROOT), check=True)
                _EXTERNAL_INPUTS_EXPORTED = True
            
            venv_python = PROJECT_ROOT / "external" / model_type / ".venv" / "bin" / "python"
            inference_script = PROJECT_ROOT / "external" / model_type / "run_inference.py"
            if venv_python.exists() and inference_script.exists():
                subprocess.run([str(venv_python), str(inference_script)], cwd=str(PROJECT_ROOT), check=True)
            else:
                logger.error(f"Cannot run {model_type}: missing venv or run_inference.py")
                
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        if not csv_path.exists():
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                logger.error(f"Prediction file {csv_path} still not found after running external models.")
            return []
    
    df = pd.read_csv(csv_path)
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        r = cast(Dict[str, Any], row.to_dict())
        r["entry_id"] = int(cast(float, r["entry_id"]))
        mut = r.get("mutation", "")
        mut_str = str(mut)
        r["mutation"] = "" if pd.isna(mut) or mut is None or mut_str == "nan" else mut_str  # type: ignore
        records.append(r)

    import sqlite3
    db_path = PROJECT_ROOT / "data" / "thermokp_database.db"
    conn = sqlite3.connect(db_path)
    truth_df = pd.read_sql_query("SELECT entry_id, uniprot_id, mutation, kcat as kcat_true, km as km_true FROM benchmark_parameters", conn)
    import numpy as np
    truth_df["log_kcat_true"] = np.log(truth_df["kcat_true"].astype(float))
    truth_df["log_km_true"] = np.log(truth_df["km_true"].astype(float))
    conn.close()
    
    truth_map = cast(Dict[int, Dict[str, Any]], truth_df.set_index("entry_id").to_dict("index"))
    for r in records:
        eid = r["entry_id"]
        if eid in truth_map:
            r.update(truth_map[eid])

    return records


def _load_tensor_records(
    model: torch.nn.Module,
    model_type: str,
    device: torch.device,
    data_dir: str,
    split: str,
    batch_size: int,
    num_workers: int,
) -> List[Dict[str, object]]:
    """Evaluate cached tensors under `data_dir` for --split train/val/all.

    Parameters
    ----------
    model : torch.nn.Module
        The model to run inference with.
    model_type : str
        Passed through to `thermokp.predict_log_kinetics` to select the
        model-specific prediction path.
    device : torch.device
        Device batches are moved to before inference.
    data_dir : str
        Directory of cached PyG tensors, loaded via `EnzymeDataset`.
    split : str
        "train", "val", or "all". For "train"/"val", the split is
        reconstructed via `train_val_split` with the same `train_frac`
        and `seed` used at training time; "all" evaluates every tensor.
    batch_size : int
        Inference batch size.
    num_workers : int
        DataLoader worker count.

    Returns
    -------
    list of dict
        One dict per record: `entry_id` (parsed from the tensor
        filename), `uniprot_id`, `mutation`, `kcat_true`/`km_true`,
        `log_kcat_true`/`log_km_true`, `log_kcat_pred`/`log_km_pred`, and
        `kcat_pred`/`km_pred`.
    """
    dataset = EnzymeDataset(data_dir)

    if split == "all":
        indices = list(range(len(dataset)))
    else:
        train_indices, val_indices, _ = train_val_split(dataset, train_frac=0.9, seed=44)
        indices = train_indices if split == "train" else val_indices

    subset = Subset(dataset, indices)
    loader = DataLoader(cast(PyGDataset, subset), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    records: List[Dict[str, object]] = []
    running_idx = 0
    for batch in loader:
        batch = batch.to(device)
        log_kcat_pred, log_km_pred = thermokp.predict_log_kinetics(model, batch, model_type=model_type)

        log_kcat_true = torch.log10(batch.kcat + thermokp.EPS).view(-1)
        log_km_true = torch.log10(batch.km + thermokp.EPS).view(-1)

        uniprot_ids = batch.uniprot_id if isinstance(batch.uniprot_id, list) else [batch.uniprot_id]
        mutations = batch.mutation if isinstance(batch.mutation, list) else [batch.mutation]

        for i in range(log_kcat_pred.size(0)):
            idx = indices[running_idx]
            log_kcat_pred_v = log_kcat_pred[i].item()
            log_km_pred_v = log_km_pred[i].item()
            records.append({
                "entry_id": _parse_entry_id(dataset.files[idx].stem),
                "uniprot_id": uniprot_ids[i],
                "mutation": mutations[i] or "",
                "kcat_true": batch.kcat[i].item(),
                "km_true": batch.km[i].item(),
                "log_kcat_true": log_kcat_true[i].item(),
                "log_km_true": log_km_true[i].item(),
                "log_kcat_pred": log_kcat_pred_v,
                "log_km_pred": log_km_pred_v,
                "kcat_pred": 10.0 ** log_kcat_pred_v,
                "km_pred": 10.0 ** log_km_pred_v,
            })
            running_idx += 1

    logger.info(f"Loaded {len(records)} records from the '{split}' split.")
    return records


# ═══════════════════════════════════════════════════════════════════════════
#  Reporting
# ═══════════════════════════════════════════════════════════════════════════
def _log_summary(summary: Dict[str, object]) -> None:
    """Log a boxed overall log-RMSE/R^2/p1mag report for k_cat and K_m.

    Parameters
    ----------
    summary : dict of str to object
        A `_evaluate_split`-built summary dict, expected to hold `split`,
        `model_type`, `n_records`, and `kcat`/`km` entries each with an
        `'overall'` `_regression_metrics` dict.

    Returns
    -------
    None
    """
    kcat_overall = cast(Dict[str, float], cast(Dict[str, object], summary["kcat"])["overall"])
    km_overall = cast(Dict[str, float], cast(Dict[str, object], summary["km"])["overall"])

    logger.info("=======================================================================")
    logger.info(f"  ThermoKP Dataset Evaluation - split={summary['split']}, model_type={summary['model_type']}")
    logger.info(f"  Records evaluated: {summary['n_records']}")
    logger.info("=======================================================================")
    logger.info(f"{'target':>8} {'log-RMSE':>10} {'R^2':>8} {'p1mag':>8} {'n':>8}")
    logger.info(
        f"{'k_cat':>8} {kcat_overall['rmse']:>10.4f} {kcat_overall['r2']:>8.4f} "
        f"{kcat_overall['p1mag']:>7.2f}% {kcat_overall['n']:>8}"
    )
    logger.info(
        f"{'K_m':>8} {km_overall['rmse']:>10.4f} {km_overall['r2']:>8.4f} "
        f"{km_overall['p1mag']:>7.2f}% {km_overall['n']:>8}"
    )
    logger.info("=======================================================================")


def _save_results(
    output_path: Path, run_key: str, summary: Dict[str, object], records: List[Dict[str, object]]
) -> None:
    """Merge this run's metrics + per-record predictions into the shared results store.

    Reads any existing `output_path` first so other runs' entries
    (different splits, different model types) are preserved rather than
    overwritten - the file accumulates one entry per `run_key`
    ("{split}_{model_type}"), keeping every evaluation ever recorded
    queryable from one place. Written atomically (temp file + rename) so
    an interrupted write never corrupts a store that already holds prior
    runs' results.

    Parameters
    ----------
    output_path : Path
        JSON file to merge this run's results into (created if absent).
    run_key : str
        Key this run is stored under, conventionally
        `"{split}_{model_type}"`.
    summary : dict of str to object
        This run's metrics dict (as built by `_evaluate_split`).
    records : list of dict
        This run's per-record predictions, stored alongside `summary`
        under the `"records"` key.

    Returns
    -------
    None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, object] = {}
    if output_path.exists():
        try:
            all_results = json.loads(output_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning(f"Failed to read existing {output_path}; starting a fresh results store.")

    all_results[run_key] = {**summary, "records": records}

    tmp_path = output_path.parent / f".tmp-{uuid.uuid4().hex}.json"
    tmp_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    os.replace(tmp_path, output_path)


# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_SPLITS = ("val", "benchmark")


def _evaluate_split(
    model: Optional[torch.nn.Module],
    model_type: str,
    device: torch.device,
    split: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    limit: Optional[int],
    output_path: Path,
    benchmark_cache_dir: Optional[Path] = None,
) -> bool:
    """Collect records, summarize, log, and save results for one split.

    Parameters
    ----------
    model : torch.nn.Module, optional
        The model to run inference with. Required (asserted) unless
        `model_type` is an external model, which instead loads
        pre-computed predictions.
    model_type : str
        "pinn", "baseline", or an external model name ("dlkcat", "unikp",
        "catpred").
    device : torch.device
        Device batches/graphs are moved to before inference.
    split : str
        "benchmark", "train", "val", or "all".
    data_dir : str
        Tensor directory for `split` in `{"train", "val", "all"}`.
    batch_size : int
        Inference batch size for `split` in `{"train", "val", "all"}`.
    num_workers : int
        DataLoader worker count for `split` in `{"train", "val", "all"}`.
    limit : int, optional
        Caps the number of rows evaluated for `split == "benchmark"`.
    output_path : Path
        Combined results store passed through to `_save_results`.
    benchmark_cache_dir : Path, optional
        Per-row input tensor cache directory for `split == "benchmark"`.

    Returns
    -------
    bool
        True if any records were evaluated (and results were saved), False
        if this split produced none (logged as an error, not raised, so a
        multi-split run can still report the splits that did succeed).
    """
    if model_type in ["dlkcat", "unikp", "catpred"]:
        records = _load_external_records(model_type, split)
        if not records:
            logger.warning(f"No records found for external model {model_type} in split {split}. Skipping...")
            return False
    elif split == "benchmark":
        assert model is not None, "Model must not be None for internal model types"
        records = _load_benchmark_records(model, model_type, device, limit=limit, benchmark_cache_dir=benchmark_cache_dir)
    else:
        assert model is not None, "Model must not be None for internal model types"
        records = _load_tensor_records(model, model_type, device, data_dir, split, batch_size, num_workers)

    if not records:
        logger.error(f"No records were evaluated for split={split!r}; skipping.")
        return False

    summary: Dict[str, object] = {
        "split": split,
        "model_type": model_type,
        "n_records": len(records),
        "kcat": summarize_parameter(records, "log_kcat_pred", "log_kcat_true"),
        "km": summarize_parameter(records, "log_km_pred", "log_km_true"),
    }

    if split == "benchmark":
        # Model-specific seq-id cutoffs (against this model's own training set).
        model_cache = PROJECT_ROOT / "data" / "cache" / f"benchmark_seq_ids_{model_type}.json"
        _calculate_max_seq_ids(
            records,
            train_seqs_getter=lambda mt=model_type: _get_training_sequences(mt),
            cache_path=model_cache,
            record_key="max_seq_id",
        )
        summary["seq_id_cutoffs"] = {}
        for cutoff in [0.99, 0.80, 0.60, 0.40]:
            subset = [r for r in records if cast(float, r.get("max_seq_id", 1.0)) <= cutoff]
            if not subset:
                continue
            summary["seq_id_cutoffs"][f"seq_id_<={int(cutoff*100)}"] = {
                "n_records": len(subset),
                "kcat": summarize_parameter(subset, "log_kcat_pred", "log_kcat_true"),
                "km": summarize_parameter(subset, "log_km_pred", "log_km_true"),
            }

        # Union seq-id cutoffs (against all model training sets combined).
        # Used by the cross-model comparison figure so all models are filtered
        # against a single consistent benchmark subset.
        union_cache = PROJECT_ROOT / "data" / "cache" / "benchmark_seq_ids_union.json"
        _calculate_max_seq_ids(
            records,
            train_seqs_getter=_get_union_training_sequences,
            cache_path=union_cache,
            record_key="union_max_seq_id",
        )
        summary["seq_id_union_cutoffs"] = {}
        for cutoff in [0.99, 0.80, 0.60, 0.40]:
            subset = [r for r in records if cast(float, r.get("union_max_seq_id", 1.0)) <= cutoff]
            if not subset:
                continue
            summary["seq_id_union_cutoffs"][f"seq_id_<={int(cutoff*100)}"] = {
                "n_records": len(subset),
                "kcat": summarize_parameter(subset, "log_kcat_pred", "log_kcat_true"),
                "km": summarize_parameter(subset, "log_km_pred", "log_km_true"),
            }

    _log_summary(summary)
    run_key = f"{split}_{model_type}"
    _save_results(output_path, run_key, summary, records)
    logger.info(f"Saved combined results to {output_path} (key={run_key!r})")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="ThermoKP Dataset Evaluation")
    parser.add_argument(
        "--split", type=str, default=None, choices=["benchmark", "train", "val", "all"],
        help="Which split to evaluate. Defaults to running BOTH 'val' and 'benchmark' as two "
             "separate evaluations (each saved under its own key) if omitted. 'benchmark': the "
             "withheld benchmark_parameters table, evaluated via thermokp.py's live inference "
             "pipeline. 'train'/'val': the cached training tensors' reconstructed split. "
             "'all': the full training set (train + validation combined).",
    )
    parser.add_argument("--model_type", type=str, default="all", choices=["pinn", "baseline", "thermokp", "dlkcat", "unikp", "catpred", "all"],
                        help="'pinn', 'baseline', 'thermokp' (both pinn and baseline), external models, or 'all' (default).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint .pth file. Defaults to trying, in order, "
                             "models/best_model.pth, final_model.pth for --model_type pinn, "
                             "or best_baseline_model.pth, final_baseline_model.pth for "
                             "--model_type baseline.")
    parser.add_argument("--hidden_channels", type=int, default=64,
                        help="Must match the checkpoint's training run. Defaults to 64.")
    parser.add_argument("--data_dir", type=str, default="data/processed/tensors",
                        help="Tensor directory for --split train/val/all. Must match the checkpoint's training run.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Inference batch size for --split train/val/all. Defaults to 256.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker count for --split train/val/all. Defaults to 4.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of benchmark rows evaluated (--split benchmark only).")
    parser.add_argument("--output_path", type=str, default="data/results/eval.json",
                        help="Combined results store. This run's metrics/records are merged in "
                             "under a '{split}_{model_type}' key, preserving other runs' entries.")
    parser.add_argument("--clean", action="store_true",
                        help="Delete the benchmark tensor cache and any cached external prediction "
                             "CSVs before running, forcing a full re-inference.")
    args = parser.parse_args()

    # This module's logger and train_val_split's both use accelerate's
    # get_logger (main-process-only by default under multi-GPU launch),
    # which requires an initialized PartialState.
    PartialState()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.model_type == "all":
        model_types_to_run = ["pinn", "baseline", "dlkcat", "unikp", "catpred"]
    elif args.model_type == "thermokp" or args.model_type == "both":
        model_types_to_run = ["pinn", "baseline"]
    else:
        model_types_to_run = [args.model_type]
        
    splits_to_run = [args.split] if args.split is not None else list(DEFAULT_SPLITS)
    output_path = Path(args.output_path)
    benchmark_cache_dir = PROJECT_ROOT / "data" / "processed" / "benchmark"

    if args.clean:
        import shutil

        def _delete_file(path: Path) -> None:
            if path.exists():
                path.unlink()
                logger.info(f"Deleted {path}")

        def _delete_dir(path: Path) -> None:
            if path.exists():
                shutil.rmtree(path)
                logger.info(f"Deleted directory {path}")

        cache_dir = PROJECT_ROOT / "data" / "cache"

        # Benchmark tensor cache (processed input tensors for internal models)
        _delete_dir(benchmark_cache_dir)

        # External prediction CSVs (re-inference required for external models)
        ext_preds_dir = PROJECT_ROOT / "data" / "external_predictions"
        if ext_preds_dir.exists():
            deleted = list(ext_preds_dir.glob("*.csv"))
            for f in deleted:
                f.unlink()
            if deleted:
                logger.info(f"Deleted {len(deleted)} external prediction CSV(s) from {ext_preds_dir}")

        # Seq-id caches (model-specific and union; re-computed on next benchmark run)
        for seq_cache in cache_dir.glob("benchmark_seq_ids*.json"):
            _delete_file(seq_cache)

        # Accumulated evaluation results store
        _delete_file(Path(args.output_path))

        # NOTE: data/cache/pdbs/ and data/cache/sequences/ are intentionally
        # preserved — they are static biological data whose re-fetching is slow
        # and produces identical results.

    any_success = False
    for model_type in model_types_to_run:
        logger.info(f"Preparing evaluation for {model_type}...")
        
        model = None
        if model_type in ["pinn", "baseline"]:
            logger.info(f"Loading {model_type} model (hidden_channels={args.hidden_channels})...")
            model = thermokp.load_model(
                checkpoint=args.checkpoint, model_type=model_type, hidden_channels=args.hidden_channels, device=device
            )

        for split in splits_to_run:
            if _evaluate_split(
                model, model_type, device, split,
                args.data_dir, args.batch_size, args.num_workers, args.limit, output_path,
                benchmark_cache_dir=benchmark_cache_dir,
            ):
                any_success = True

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    if not any_success:
        logger.error("No records were evaluated for any requested split; nothing to report.")
        sys.exit(1)


if __name__ == "__main__":
    main()
