"""
===========================================================================
Fetch EC Numbers
Description: ThermoKP EC Master Index Generator
===========================================================================

Workflow:
1. Connect to `https://ftp.expasy.org/databases/enzyme/enzyme.dat` via HTTPS.
2. Parse the file line-by-line, extracting strings from lines that begin with `ID   `.
3. Filter out any malformed strings and structure them into a flat JSON list.
4. Write the list to `data/enzyme_targets.json`.

Author: ThermoKP Team
License: MIT
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import urllib.request
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
#  Logging configuration
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

EXPASY_URL: str = "https://ftp.expasy.org/databases/enzyme/enzyme.dat"

_PROJECT_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[3]
OUTPUT_FILE: pathlib.Path = _PROJECT_ROOT / "data" / "enzyme_targets.json"

# ═══════════════════════════════════════════════════════════════════════════
#  Fetcher Logic
# ═══════════════════════════════════════════════════════════════════════════

def _fetch_enzyme_dat(url: str = EXPASY_URL) -> Optional[str]:
    """Download the ENZYME database text file.
    
    Parameters
    ----------
    url : str, default: EXPASY_URL
        The URL to the ExPASy enzyme.dat file.
        
    Returns
    -------
    str or None
        The complete text content of the database, or None if the request fails.
    """
    logger.info("Downloading ExPASy ENZYME database from %s", url)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        logger.error("Failed to download ExPASy data: %s", e)
        return None

def _parse_ec_numbers(dat_text: str) -> list[str]:
    """Parse EC numbers from the raw ENZYME database string.
    
    Parameters
    ----------
    dat_text : str
        The full text of `enzyme.dat`.
        
    Returns
    -------
    list[str]
        A list of extracted EC numbers.
    """
    ec_numbers: list[str] = []
    
    for line in dat_text.splitlines():
        if line.startswith("ID   "):
            # Format: "ID   1.1.1.1"
            parts = line.split()
            if len(parts) >= 2:
                ec = parts[1].strip()
                if ec.count(".") == 3:
                    ec_numbers.append(ec)
                    
    logger.info("Successfully extracted %d unique EC numbers.", len(ec_numbers))
    return ec_numbers

# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Execute the EC extraction pipeline."""
    logger.info("============================================================")
    logger.info("ThermoKP — ExPASy EC Master Index Generator")
    logger.info("============================================================")

    dat_text = _fetch_enzyme_dat()
    if not dat_text:
        return
        
    ec_numbers = _parse_ec_numbers(dat_text)
    if not ec_numbers:
        logger.error("No EC numbers were found in the file.")
        return
        
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(ec_numbers, f, indent=4)
        
    logger.info("==========================================================================")
    logger.info("==                        EC Extraction Summary                         ==")
    logger.info("==========================================================================")
    logger.info(f"Targets saved             : {len(ec_numbers)}")
    logger.info(f"Output file               : {os.path.basename(OUTPUT_FILE)}")
    logger.info("==========================================================================")

if __name__ == "__main__":
    main()
