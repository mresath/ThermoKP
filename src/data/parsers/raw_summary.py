"""
===========================================================================
Raw Ingestion Summary Generator
Description: Generates a summary text of the raw ingestion phase.
===========================================================================

Workflow:
1. Connect to `data/thermokp_database.db`.
2. Count `raw_parameters` rows contributed by each source (BRENDA, SABIO-RK).
3. Log the per-source and total raw row counts.

Known Caveats:
- Assumes `raw_parameters` already exists (populated by brenda_parser.py
  and sabio_rk_parser.py); it is not created here.

Author: ThermoKP Team
License: MIT
"""

import logging
import pathlib
import sqlite3

# ═══════════════════════════════════════════════════════════════════════════
#  Logging configuration
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH: pathlib.Path = _PROJECT_ROOT / "data" / "thermokp_database.db"


def main():
    """Log a summary of per-source row counts in `raw_parameters`.

    Returns
    -------
    None
    """
    logger.info("==========================================================================")
    logger.info("==                       Raw Ingestion Summary                          ==")
    logger.info("==========================================================================")

    try:
        conn = sqlite3.connect(DEFAULT_DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM raw_parameters WHERE source_db='BRENDA'")
        brenda_rows = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM raw_parameters WHERE source_db='SABIO-RK'")
        sabio_rows = cursor.fetchone()[0]

        conn.close()

        logger.info(f"BRENDA raw rows             : {brenda_rows}")
        logger.info(f"SABIO-RK raw rows           : {sabio_rows}")
        logger.info(f"Total raw rows              : {brenda_rows + sabio_rows}")

    except Exception as e:
        logger.error(f"Error generating summary: {e}")


if __name__ == "__main__":
    main()
