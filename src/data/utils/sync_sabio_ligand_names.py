"""
===========================================================================
Sync SABIO-RK Ligand Names
Description: Bulk-import SABIO-RK's own ligand name -> structure data into
             the shared SMILES cache
===========================================================================

Workflow:
1. Query SABIO-RK's REST API for all compounds.
2. Extract the compound names and their associated ChEBI (or KEGG) IDs.
3. Resolve each ID to a canonical SMILES via ChEBI/KEGG APIs.
4. Merge {lowercased name: SMILES} into the shared cache (src/data/smiles_cache.py).

Author: ThermoKP Team
License: MIT
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.data.utils.smiles_cache import merge_entries
from rdkit import Chem

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

SABIO_COMPOUNDS_URL: str = "https://sabiork.h-its.org/export-api/sabio/compound"
REQUEST_TIMEOUT_S: int = 60
REQUEST_DELAY_S: float = 0.2

CHEBI_ENTITY_URL_TEMPLATE: str = (
    "https://www.ebi.ac.uk/chebi/backend/api/public/compound/{chebi_id}"
)
KEGG_MOL_URL_TEMPLATE: str = "https://rest.kegg.jp/get/cpd:{kegg_id}/mol"

# ═══════════════════════════════════════════════════════════════════════════
#  Fetching
# ═══════════════════════════════════════════════════════════════════════════

def _fetch_sabio_compounds() -> list[tuple[str, Optional[str], Optional[str]]]:
    """Fetch all compounds from SABIO-RK API and extract Name, ChEBI, KEGG.

    Returns
    -------
    list[tuple[str, Optional[str], Optional[str]]]
        List of (name, chebi_id, kegg_id).
    """
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    compounds = []
    page = 1
    total_pages = 1

    try:
        while page <= total_pages:
            response = session.get(
                f"{SABIO_COMPOUNDS_URL}?page={page}&pageSize=1000",
                headers={"Accept": "application/json"},
                timeout=REQUEST_TIMEOUT_S,
            )
            if response.status_code != 200:
                logger.warning("SABIO-RK returned HTTP %d on page %d", response.status_code, page)
                break
                
            data = response.json()
            
            if page == 1:
                meta = data.get("meta", {})
                total_pages = meta.get("total_pages", 1)
                
            for comp in data.get("data", []):
                name = comp.get("name")
                if not name:
                    continue
                    
                chebi = None
                kegg = None
                
                # Extract ChEBI and KEGG from external_links
                for link in comp.get("external_links", []):
                    ext_name = link.get("ext_name")
                    ext_id = link.get("ext_id")
                    if ext_name == "chebi" and not chebi:
                        chebi = ext_id
                    elif ext_name == "kegg_compound" and not kegg:
                        kegg = ext_id
                        
                compounds.append((name.strip(), chebi, kegg))
            
            page += 1
            time.sleep(REQUEST_DELAY_S)
            
        return compounds
    except Exception as e:
        logger.error("Failed to fetch SABIO-RK compounds: %s", e)
        return compounds


def _resolve_chebi_smiles(chebi_id: str, session: requests.Session) -> Optional[str]:
    """Fetch a canonical SMILES for a ChEBI ID via ChEBI's REST API.

    Parameters
    ----------
    chebi_id : str
        Bare ChEBI numeric ID (no "CHEBI:" prefix).
    session : requests.Session
        The requests session object to use.

    Returns
    -------
    str or None
        The SMILES string, or None on any failure.
    """
    url = CHEBI_ENTITY_URL_TEMPLATE.format(chebi_id=chebi_id)
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT_S)
        if response.status_code != 200:
            return None
        smiles = response.json().get("default_structure", {}).get("smiles")
        return smiles.strip() if smiles else None
    except Exception as e:
        logger.debug("Error fetching ChEBI:%s: %s", chebi_id, e)
    return None


def _resolve_kegg_smiles(kegg_id: str, session: requests.Session) -> Optional[str]:
    """Fetch a canonical SMILES for a KEGG compound ID via its MOL file.

    Parameters
    ----------
    kegg_id : str
        Bare KEGG compound ID (e.g. ``"C00022"``).
    session : requests.Session
        The requests session object to use.

    Returns
    -------
    str or None
        The SMILES string (converted from the KEGG MOL block via RDKit), or
        None on any failure.
    """
    url = KEGG_MOL_URL_TEMPLATE.format(kegg_id=kegg_id)
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT_S)
        if response.status_code != 200 or not response.text.strip():
            return None
        mol = Chem.MolFromMolBlock(response.text)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception as e:
        logger.debug("Error fetching KEGG:%s: %s", kegg_id, e)
    return None


def _resolve_to_smiles(chebi_id: Optional[str], kegg_id: Optional[str], session: requests.Session) -> Optional[str]:
    """Resolve ChEBI or KEGG ID to canonical SMILES using REST APIs.
    
    Parameters
    ----------
    chebi_id : str or None
        The ChEBI numeric ID.
    kegg_id : str or None
        The KEGG compound ID.
    session : requests.Session
        The requests session object to use.

    Returns
    -------
    str or None
        The resolved SMILES string, or None if neither ID resolves.
    """
    smiles = _resolve_chebi_smiles(chebi_id, session) if chebi_id else None
    if smiles is None and kegg_id:
        smiles = _resolve_kegg_smiles(kegg_id, session)
    return smiles

# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def _sync() -> dict[str, str]:
    """Run the SABIO-RK compound sync.

    Returns
    -------
    dict[str, str]
        {lowercased chemical name: SMILES}
    """
    collected: dict[str, str] = {}
    parse_failures = 0

    logger.info("Fetching compounds from SABIO-RK...")
    compounds = _fetch_sabio_compounds()
    logger.info("Found %d compounds in SABIO-RK.", len(compounds))
    
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    for idx, (name, chebi, kegg) in enumerate(compounds, 1):
        key = name.lower()
        if key in collected:
            continue
            
        if not chebi and not kegg:
            continue
            
        smiles = _resolve_to_smiles(chebi, kegg, session)
        if smiles is None:
            parse_failures += 1
            continue
            
        collected[key] = smiles

        if idx % 500 == 0 or idx == len(compounds):
            logger.info(
                "[%d/%d] compounds processed - %d unique names resolved (%d API parse failures)",
                idx, len(compounds), len(collected), parse_failures,
            )
        time.sleep(REQUEST_DELAY_S)

    logger.info(
        "Sync complete: %d compounds processed, %d unique names resolved, %d API parse failures.",
        len(compounds), len(collected), parse_failures,
    )
    return collected


def main() -> None:
    """Execute the SABIO-RK ligand-name sync and merge results into the shared cache."""
    logger.info("============================================================")
    logger.info("ThermoKP — SABIO-RK Ligand Name Sync")
    logger.info("============================================================")

    collected = _sync()

    added = merge_entries(collected, overwrite=False)

    logger.info("==========================================================================")
    logger.info("==                        Ligand Name Sync Summary                      ==")
    logger.info("==========================================================================")
    logger.info(f"New names merged          : {added}")
    logger.info(f"Already cached (skipped)  : {len(collected) - added}")
    logger.info("==========================================================================")


if __name__ == "__main__":
    main()
