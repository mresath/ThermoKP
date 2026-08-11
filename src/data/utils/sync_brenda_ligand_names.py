"""
===========================================================================
Sync BRENDA Ligand Names
Description: Bulk-import BRENDA's own ligand name -> structure data into
             the shared SMILES cache
===========================================================================

Workflow:
1. Query BRENDA's public SPARQL endpoint (part of its 2026 knowledge-graph
   release) for every (name, InChI) pair across its ~220k curated
   ChemicalCompound entities, paginated by compound-ID range.
2. Convert each InChI to a canonical SMILES via RDKit.
3. Merge {lowercased name: SMILES} into the cache shared with
   sabio_rk_parser.py / dataset_validator.py (src/data/smiles_cache.py),
   without overwriting any name that's already cached from a source
   already trusted (e.g. ChEBI via SABIO parsing).

Why this exists: `dataset_validator.py`'s chemical-resolution tiers
(PubChem, OPSIN, CACTUS, the local peptide-notation builder) fail on
~30% of substrate names not previously seen - largely obscure natural
products, glycosides, and nucleoside derivatives that simply aren't
indexed under these exact strings elsewhere. Many of these names are
literally the strings BRENDA itself curated (they came from
`brenda.json` in the first place), so BRENDA's own ligand database is
the single most direct source for resolving them.

Known gotchas (endpoint-specific behavior this script works around):
- Case-insensitive matching (`FILTER(LCASE(STR(?name)) = ...)`) forces a
  full unindexed scan and is far too slow for per-name runtime queries.
  This script sidesteps the problem
  entirely by pulling every name in bulk, once, and doing the
  case-insensitive matching locally (a plain Python dict keyed by
  `.lower()`) - no per-name network round trip is ever needed again.
- Plain `OFFSET`-based pagination intermittently returns malformed JSON
  (a stray leading comma) on this endpoint when unordered. ID-range
  `FILTER` pagination (`?s >= <compound/N> && ?s < <compound/N+chunk>`)
  is fast and reliably well-formed - used here instead.

Author: ThermoKP Team
License: MIT
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests
import rdkit.Chem as Chem
from rdkit import rdBase

from src.data.utils.smiles_cache import merge_entries

# RDKit's InChI parser logs directly to stderr via its own C++ logger,
# bypassing the `logging` module entirely - a large fraction of BRENDA's
# ~972k InChI strings fail to parse (expected; already counted and reported
# via the "InChI parse failures" summary line below), and each failure
# prints several raw, mostly-blank "ERROR:" lines. Silencing it here keeps
# this script's actual output readable without losing any information -
# the failure count is still tracked in `_inchi_to_smiles`'s caller.
rdBase.DisableLog("rdApp.*")

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

BRENDA_SPARQL_URL: str = "https://sparql.dsmz.de/api/brenda"
# Highest observed compound ID at the time this was written is ~281,908
# against ~219,715 total ChemicalCompound entities (i.e. IDs are sparse,
# not contiguous). A generous upper bound is used since out-of-range
# chunks simply return zero rows quickly rather than erroring.
COMPOUND_ID_UPPER_BOUND: int = 300_000
CHUNK_SIZE: int = 3_000
REQUEST_TIMEOUT_S: int = 60
# Politeness delay between chunk requests - this is a shared institutional
# endpoint, not a dedicated API; there's no documented rate limit, but a
# ~100 request bulk _sync costs us nothing to space out slightly.
REQUEST_DELAY_S: float = 0.2

_QUERY_TEMPLATE = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ns: <https://purl.dsmz.de/schema/>
SELECT ?name ?inchi WHERE {{
  ?s rdf:type ns:ChemicalCompound .
  FILTER(?s >= <https://purl.dsmz.de/brenda/compound/{lo}> && ?s < <https://purl.dsmz.de/brenda/compound/{hi}>)
  {{ ?s rdfs:label ?name }} UNION {{ ?s ns:hasSynonym ?name }} UNION {{ ?s ns:hasSystematicName ?name }}
  ?s ns:hasStructure ?struct .
  ?struct ns:hasInChI ?inchi .
}}
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Fetching
# ═══════════════════════════════════════════════════════════════════════════

def _fetch_chunk(lo: int, hi: int) -> list[tuple[str, str]]:
    """Fetch (name, InChI) pairs for compound IDs in [lo, hi).

    Parameters
    ----------
    lo : int
        Inclusive lower bound of the compound-ID range.
    hi : int
        Exclusive upper bound of the compound-ID range.

    Returns
    -------
    list[tuple[str, str]]
        (name, InChI) pairs found in this range. Empty on any request or
        parse failure - a single bad chunk should not abort the whole _sync.
    """
    query = _QUERY_TEMPLATE.format(lo=lo, hi=hi)
    try:
        response = requests.get(
            BRENDA_SPARQL_URL,
            params={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=REQUEST_TIMEOUT_S,
        )
        if response.status_code != 200:
            logger.warning("Chunk [%d, %d) returned HTTP %d", lo, hi, response.status_code)
            return []
        data = response.json()
        return [
            (b["name"]["value"], b["inchi"]["value"])
            for b in data.get("results", {}).get("bindings", [])
        ]
    except Exception as e:
        logger.warning("Chunk [%d, %d) failed: %s", lo, hi, e)
        return []


def _inchi_to_smiles(inchi: str) -> Optional[str]:
    """Convert an InChI string to a canonical SMILES via RDKit.

    Parameters
    ----------
    inchi : str
        The InChI string.

    Returns
    -------
    str or None
        The canonical SMILES, or None if RDKit couldn't parse it.
    """
    try:
        mol = Chem.MolFromInchi(inchi, sanitize=True, logLevel=None)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def _sync(id_upper_bound: int = COMPOUND_ID_UPPER_BOUND, chunk_size: int = CHUNK_SIZE) -> dict[str, str]:
    """Run the full paginated _sync and return the collected name->SMILES map.

    Parameters
    ----------
    id_upper_bound : int, default: COMPOUND_ID_UPPER_BOUND
        Exclusive upper bound of the compound-ID range to scan.
    chunk_size : int, default: CHUNK_SIZE
        Number of compound IDs to request per SPARQL query.

    Returns
    -------
    dict[str, str]
        {lowercased chemical name: SMILES}, deduplicated (first InChI seen
        for a given name wins if RDKit can parse it; later duplicates for
        the same name are skipped without re-parsing).
    """
    collected: dict[str, str] = {}
    total_rows = 0
    parse_failures = 0

    chunk_starts = list(range(1, id_upper_bound, chunk_size))
    for idx, lo in enumerate(chunk_starts, 1):
        hi = lo + chunk_size
        rows = _fetch_chunk(lo, hi)
        total_rows += len(rows)
        for name, inchi in rows:
            key = name.lower()
            if key in collected:
                continue
            smiles = _inchi_to_smiles(inchi)
            if smiles is None:
                parse_failures += 1
                continue
            collected[key] = smiles

        if idx % 10 == 0 or idx == len(chunk_starts):
            logger.info(
                "[%d/%d] compound IDs up to %d - %d unique names so far (%d rows seen, %d InChI parse failures)",
                idx, len(chunk_starts), hi, len(collected), total_rows, parse_failures,
            )
        time.sleep(REQUEST_DELAY_S)

    logger.info(
        "Sync complete: %d rows seen, %d unique names resolved, %d InChI parse failures.",
        total_rows, len(collected), parse_failures,
    )
    return collected


def main() -> None:
    """Execute the BRENDA ligand-name _sync and merge results into the shared cache."""
    logger.info("============================================================")
    logger.info("ThermoKP — BRENDA Ligand Name Sync")
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
