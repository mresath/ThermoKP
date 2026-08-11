"""
===========================================================================
DLKcat Inference Wrapper
Phase: Benchmarking Cross-Model Comparison
Description: Executes DLKcat predictions against the ThermoKP benchmark dataset.
===========================================================================

Workflow:
1. Connect to the ThermoKP database and retrieve benchmark sequences.
2. Initialize DLKcat's KcatPrediction model and load pre-trained weights.
3. Process mature sequences and canonical SMILES into DLKcat input vectors.
4. Export predicted kcat values to the central external_predictions directory.

Known Caveats:
- DLKcat only predicts kcat, hence Km predictions are populated with NaNs.
- Original DLKcat outputs are log base 2; this script converts them to natural logarithm.

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
import torch
from pathlib import Path

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
#  Path & Environment Setup
# ═══════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DLKCAT_CODE = PROJECT_ROOT / "external" / "dlkcat" / "src" / "DeeplearningApproach" / "Code"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DLKCAT_CODE))
sys.path.insert(0, str(DLKCAT_CODE / "example"))

# Change directory so DLKcat's prediction_for_input relative paths work
_original_cwd = os.getcwd()
os.chdir(DLKCAT_CODE / "example")

import model as dlkcat_model  # type: ignore
from prediction_for_input import split_sequence, create_atoms, create_ijbonddict, extract_fingerprints, create_adjacency, Predictor  # type: ignore
import prediction_for_input  # type: ignore

os.chdir(_original_cwd)

from rdkit import Chem

# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    logger.info("===========================================================================")
    logger.info("Starting DLKcat Inference")
    logger.info("===========================================================================")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load DLKcat dictionaries
    data_dir = DLKCAT_CODE.parent / "Data" / "input"
    fingerprint_dict = dlkcat_model.load_pickle(str(data_dir / "fingerprint_dict.pickle"))
    atom_dict = dlkcat_model.load_pickle(str(data_dir / "atom_dict.pickle"))
    bond_dict = dlkcat_model.load_pickle(str(data_dir / "bond_dict.pickle"))
    word_dict = dlkcat_model.load_pickle(str(data_dir / "sequence_dict.pickle"))
    
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    Kcat_model = dlkcat_model.KcatPrediction(device, n_fingerprint, n_word, 2*10, 3, 11, 3, 3).to(device)
    model_path = DLKCAT_CODE.parent / "Results" / "output" / "all--radius2--ngram3--dim20--layer_gnn3--window11--layer_cnn3--layer_output3--lr1e-3--lr_decay0.5--decay_interval10--weight_decay1e-6--iteration50"
    Kcat_model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True))
    predictor = Predictor(Kcat_model)

    inputs_path = PROJECT_ROOT / "data" / "external_predictions" / "benchmark_inputs.csv"
    if not inputs_path.exists():
        logger.error(f"Input file not found at {inputs_path}. Please run export_benchmark_csv.py first.")
        return
        
    df = pd.read_csv(inputs_path)

    results = []
    
    # Needs to match the global dicts inside prediction_for_input
    prediction_for_input.fingerprint_dict = fingerprint_dict
    prediction_for_input.atom_dict = atom_dict
    prediction_for_input.bond_dict = bond_dict
    prediction_for_input.word_dict = word_dict

    total_rows = len(df)
    
    for i, (idx, row) in enumerate(df.iterrows()):
        entry_id = row["entry_id"]
        seq = row["sequence"]
        smiles = row["smiles"]

        log_kcat_pred = np.nan
        kcat_pred = np.nan

        if isinstance(seq, str) and isinstance(smiles, str) and smiles != 'None' and "." not in smiles:
            try:
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                atoms = create_atoms(mol)
                i_jbond_dict = create_ijbonddict(mol)
                fingerprints = extract_fingerprints(atoms, i_jbond_dict, 2)
                adjacency = create_adjacency(mol)
                words = split_sequence(seq, 3)

                fingerprints = torch.LongTensor(fingerprints).to(device)
                adjacency = torch.FloatTensor(adjacency).to(device)
                words = torch.LongTensor(words).to(device)

                inputs = [fingerprints, adjacency, words]
                prediction = predictor.predict(inputs)
                
                log2_kcat = prediction.item()
                kcat_pred = math.pow(2, log2_kcat)
                log_kcat_pred = math.log(kcat_pred)

            except Exception as e:
                logger.error(f"Error predicting entry {entry_id}: {e}")
        
        results.append({
            "entry_id": entry_id,
            "log_kcat_pred": log_kcat_pred,
            "log_km_pred": np.nan,
            "kcat_pred": kcat_pred,
            "km_pred": np.nan
        })
        
        if (i + 1) % 10 == 0 or (i + 1) == total_rows:
            logger.info(f"[{i + 1}/{total_rows}] Predictions computed")

    out_dir = PROJECT_ROOT / "data" / "external_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dlkcat_benchmark_predictions.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    
    logger.info("===========================================================================")
    logger.info(f"Inference complete. {total_rows} rows processed.")
    logger.info(f"Results exported to {out_path}")
    logger.info("===========================================================================")

if __name__ == "__main__":
    main()
