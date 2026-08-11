"""
===========================================================================
Export Benchmark Inputs
Description: Extracts sequences, SMILES, and PDB paths for all benchmark entries.
===========================================================================

Workflow:
1. Connects to the benchmark SQLite database.
2. Uses dataset validation logic to resolve UniProt IDs, mutations, and SMILES.
3. Exports a unified `benchmark_inputs.csv` to be read by external models.

Known Caveats:
- Must be executed in the main ThermoKP environment.

Author: ThermoKP Team
License: MIT
"""

import sys
import logging
import sqlite3
import pandas as pd
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
#  Path & Imports
# ═══════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate_dataset import get_mature_sequence
from src.data.processors.dataset_validator import get_smiles

def main() -> None:
    logger.info("===========================================================================")
    logger.info("Exporting Benchmark Inputs for External Models")
    logger.info("===========================================================================")

    db_path = PROJECT_ROOT / "data" / "thermokp_database.db"
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT entry_id, uniprot_id, mutation, measured_substrate FROM benchmark_parameters ORDER BY entry_id ASC", conn)
    conn.close()

    records = []
    pdb_cache_dir = PROJECT_ROOT / "data" / "cache" / "pdbs"

    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        entry_id = row["entry_id"]
        uniprot_id = row["uniprot_id"]
        mutation = row["mutation"] if isinstance(row["mutation"], str) and row["mutation"].strip() else None
        substrate = row["measured_substrate"]

        seq = get_mature_sequence(uniprot_id, mutation)
        smiles = get_smiles(substrate)
        pdb_path = pdb_cache_dir / f"{uniprot_id}.pdb"

        records.append({
            "entry_id": entry_id,
            "uniprot_id": uniprot_id,
            "sequence": seq,
            "smiles": smiles,
            "pdbpath": str(pdb_path.absolute()) if pdb_path.exists() else None
        })

        if (i + 1) % 100 == 0 or (i + 1) == total:
            logger.info(f"[{i + 1}/{total}] Processed inputs")

    out_dir = PROJECT_ROOT / "data" / "external_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_inputs.csv"
    
    pd.DataFrame(records).to_csv(out_path, index=False)
    logger.info(f"Exported {len(records)} records to {out_path}")

if __name__ == "__main__":
    main()
