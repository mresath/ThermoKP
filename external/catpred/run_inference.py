"""
===========================================================================
CatPred Inference Wrapper
Phase: Benchmarking Cross-Model Comparison
Description: Executes CatPred predictions against the ThermoKP benchmark dataset.
===========================================================================

Workflow:
1. Connect to the ThermoKP database and retrieve benchmark sequences.
2. Locate existing AlphaFold/ESMFold PDB files from ThermoKP cache.
3. Write standard input CSV (`SMILES`, `sequence`, `pdbpath`) for CatPred.
4. Execute CatPred pipeline in an isolated environment.
5. Export predicted kcat values to the central external_predictions directory.

Known Caveats:
- Requires previously cached PDB files in `data/cache/pdbs/`.
- Executed via bash using the isolated CatPred python environment.

Author: ThermoKP Team
License: MIT
"""

import os
import sys
import math
import sqlite3
import logging
import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
from typing import Optional

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
#  Path Setup
# ═══════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATPRED_DIR = PROJECT_ROOT / "external" / "catpred" / "src"

sys.path.insert(0, str(PROJECT_ROOT))

# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    logger.info("===========================================================================")
    logger.info("Starting CatPred Inference Preparation")
    logger.info("===========================================================================")

    inputs_path = PROJECT_ROOT / "data" / "external_predictions" / "benchmark_inputs.csv"
    if not inputs_path.exists():
        logger.error(f"Input file not found at {inputs_path}. Please run export_benchmark_csv.py first.")
        return
        
    df = pd.read_csv(inputs_path)

    input_records = []
    skipped = 0
    pdb_cache_dir = PROJECT_ROOT / "data" / "cache" / "pdbs"

    total_rows = len(df)
    
    for i, (idx, row) in enumerate(df.iterrows()):
        entry_id = row["entry_id"]
        seq = row["sequence"]
        smiles = row["smiles"]
        pdb_path = row["pdbpath"]

        if isinstance(seq, str) and isinstance(smiles, str) and smiles != 'None' and "." not in smiles and isinstance(pdb_path, str):
            input_records.append({
                "entry_id": entry_id,
                "SMILES": smiles,
                "sequence": seq,
                "pdbpath": pdb_path
            })
        else:
            skipped += 1
            input_records.append({
                "entry_id": entry_id,
                "SMILES": None,
                "sequence": None,
                "pdbpath": None
            })
            
    out_dir = PROJECT_ROOT / "data" / "external_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # CatPred complains if multiple different sequences map to the exact same PDB path string.
    # We create a temporary directory of symlinks so each sequence gets a unique PDB path!
    catpred_symlink_dir = out_dir / "catpred_pdbs"
    catpred_symlink_dir.mkdir(exist_ok=True)
    
    for r in input_records:
        if r["pdbpath"] is not None:
            original_pdb = Path(r["pdbpath"])
            symlink_pdb = catpred_symlink_dir / f"{r['entry_id']}.pdb"
            if not symlink_pdb.exists() and original_pdb.exists():
                symlink_pdb.symlink_to(original_pdb.absolute())
            r["pdbpath"] = str(symlink_pdb.absolute())
    
    # Save input CSV for CatPred
    catpred_input_csv = out_dir / "catpred_input.csv"
    valid_df = pd.DataFrame([r for r in input_records if r["SMILES"] is not None])
    valid_df.to_csv(catpred_input_csv, index=False)

    logger.info(f"Generated input CSV with {len(valid_df)} valid records ({skipped} skipped).")

    # Execute CatPred Demo script using CatPred's logic
    catpred_venv_python = PROJECT_ROOT / "external" / "catpred" / ".venv" / "bin" / "python"
    demo_script = CATPRED_DIR / "demo_run.py"

    # Set pythonpath to include catpred src
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CATPRED_DIR)

    # Prepend the venv/bin to PATH so internally spawned `python` calls use CatPred's python
    if catpred_venv_python.parent.exists():
        env["PATH"] = f"{str(catpred_venv_python.parent)}:{env.get('PATH', '')}"

    if not catpred_venv_python.exists():
        logger.error(f"CatPred venv not found at {catpred_venv_python}. Please install.")
        return

    # Auto-detect CUDA
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except ImportError:
        use_gpu = False

    if use_gpu:
        logger.info("CUDA detected. CatPred inference will use GPU.")
    else:
        logger.info("No GPU detected. CatPred inference will run on CPU (this may take up to 20 minutes for 500+ records).")

    def _run_parameter(parameter: str) -> Optional[pd.Series]:
        """Run CatPred for a single kinetic parameter and return the prediction column."""
        base_checkpoint_dir = CATPRED_DIR.parent / "data" / "pretrained" / "production" / parameter
        if not base_checkpoint_dir.exists():
            base_checkpoint_dir = CATPRED_DIR.parent / "CatPred-DB" / "data" / "pretrained" / "production" / parameter
        if not base_checkpoint_dir.exists():
            logger.warning(f"No checkpoint directory found for parameter '{parameter}'. Skipping.")
            return None

        cmd = [
            str(catpred_venv_python), str(demo_script),
            "--parameter", parameter,
            "--input_file", str(catpred_input_csv),
            "--checkpoint_dir", str(base_checkpoint_dir),
        ]
        if use_gpu:
            cmd.append("--use_gpu")

        logger.info(f"Running CatPred prediction pipeline for {parameter}...")
        try:
            subprocess.run(cmd, cwd=str(CATPRED_DIR), env=env, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"CatPred {parameter} inference failed with exit code {e.returncode}.")
            return None

        # CatPred service appends "_input" then "_output" to the stem
        raw_output = CATPRED_DIR.parent / "results" / f"{catpred_input_csv.stem}_input_output.csv"
        if not raw_output.exists():
            logger.error(f"Could not locate CatPred {parameter} output file at {raw_output}.")
            return None

        preds_df = pd.read_csv(raw_output)
        unit_map = {"kcat": "s^(-1)", "km": "mM"}
        col = f"Prediction_({unit_map[parameter]})"
        if col not in preds_df.columns:
            logger.error(f"Expected column '{col}' not found in CatPred {parameter} output. Columns: {list(preds_df.columns)}")
            return None

        return preds_df[col].reset_index(drop=True)

    kcat_preds = _run_parameter("kcat")
    km_preds = _run_parameter("km")

    logger.info("Parsing CatPred predictions...")

    if kcat_preds is not None:
        valid_df = valid_df.reset_index(drop=True)
        valid_df["catpred_kcat"] = kcat_preds
    if km_preds is not None:
        valid_df = valid_df.reset_index(drop=True)
        valid_df["catpred_km"] = km_preds

    results = []
    for row in input_records:
        entry_id = row["entry_id"]
        if row["SMILES"] is None:
            log_kcat_pred = np.nan
            kcat_pred = np.nan
            log_km_pred = np.nan
            km_pred = np.nan
        else:
            matched = valid_df[valid_df["entry_id"] == entry_id]
            if matched.empty:
                log_kcat_pred, kcat_pred, log_km_pred, km_pred = np.nan, np.nan, np.nan, np.nan
            else:
                row_match = matched.iloc[0]
                kcat_pred = row_match["catpred_kcat"] if "catpred_kcat" in valid_df.columns else np.nan
                log_kcat_pred = math.log(kcat_pred) if not (math.isnan(kcat_pred) or kcat_pred <= 0) else np.nan
                km_pred = row_match["catpred_km"] if "catpred_km" in valid_df.columns else np.nan
                log_km_pred = math.log(km_pred) if not (math.isnan(km_pred) or km_pred <= 0) else np.nan

        results.append({
            "entry_id": entry_id,
            "log_kcat_pred": log_kcat_pred,
            "log_km_pred": log_km_pred,
            "kcat_pred": kcat_pred,
            "km_pred": km_pred,
        })

    out_path = out_dir / "catpred_benchmark_predictions.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)

    logger.info("===========================================================================")
    logger.info(f"Inference complete. Results exported to {out_path}")
    logger.info("===========================================================================")

if __name__ == "__main__":
    main()
