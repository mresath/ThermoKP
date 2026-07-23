"""
===========================================================================
Parser Common
Description: Shared helpers used by both brenda_parser.py and sabio_rk_parser.py
===========================================================================

Neither BRENDA nor SABIO-RK parsing "owns" these pieces - both parsers need
a UniProt lookup and both need to classify mutation notation the same way,
so they live here rather than one parser importing from the other (which
would make one parser an implicit dependency of the other for no reason
tied to what either actually does).

- `fetch_uniprot_id`: BRENDA uses it directly when a record's `organism`
  field yields no accession; SABIO-RK uses it as a fallback when a
  species' RDF annotation carries no UniProt cross-reference.
- The mutation-classification pattern: BRENDA's `_parse_mutation`
  (free-text commentary) and SABIO-RK's `_parse_sabio_mutation`
  (structured enzyme-species names) both extract one-or-more point
  mutations from different source text.

Author: ThermoKP Team
License: MIT
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  UniProt Lookup
# ═══════════════════════════════════════════════════════════════════════════


def fetch_uniprot_id(ec_number: str, organism: str) -> Optional[str]:
    """Fetch a unique reviewed UniProt ID for a given EC and organism.

    Parameters
    ----------
    ec_number : str
        The Enzyme Commission number.
    organism : str
        The organism name.

    Returns
    -------
    Optional[str]
        The primary UniProt accession string if found, otherwise None.
    """
    query = f'ec:{ec_number} AND organism_name:"{organism}" AND reviewed:true'
    url = f"https://rest.uniprot.org/uniprotkb/search?query={urllib.parse.quote(query)}&format=json&fields=accession"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))
        results = data.get("results", [])
        if len(results) == 1:
            return results[0]["primaryAccession"]
    except Exception as e:
        logger.debug("Error querying UniProt for EC %s / %s: %s", ec_number, organism, e)
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Point-Mutation Classification
# ═══════════════════════════════════════════════════════════════════════════

# Keywords that flag a record as a point mutant rather than wild-type.
MUTATION_KEYWORDS: list[str] = ["mutant", "variant"]

# Single-letter amino acid codes, used to parse mutation notation like "W95L".
_AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"

_MUTATION_PATTERN = re.compile(
    rf"\b([{_AA_LETTERS}])(\d{{1,4}})([{_AA_LETTERS}])\b"
)

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

_MUTATION_PATTERN_3LETTER = re.compile(
    r"\b(" + "|".join(THREE_TO_ONE) + r")(\d{1,4})(" + "|".join(THREE_TO_ONE) + r")\b",
    re.IGNORECASE
)

def find_point_mutations(text: str) -> list[tuple[str, str, str]]:
    """Extract one or more point mutations from unstructured text.

    Handles both 1-letter (e.g. W95L) and 3-letter (e.g. Trp95Leu) notation.
    Multiple occurrences of the same mutation are deduplicated.

    Parameters
    ----------
    text : str
        Unstructured source text to search (e.g. BRENDA commentary or a
        SABIO-RK enzyme-species name).

    Returns
    -------
    list[tuple[str, str, str]]
        List of (wt_res, pos, mut_res) tuples in first-seen order, normalized
        to 1-letter codes.
    """
    seen = set()
    mutations = []
    
    # 1-letter codes
    for match in _MUTATION_PATTERN.findall(text):
        if match not in seen:
            seen.add(match)
            mutations.append(match)
            
    # 3-letter codes
    for match in _MUTATION_PATTERN_3LETTER.findall(text):
        wt_res_3, pos, mut_res_3 = match
        wt_res = THREE_TO_ONE.get(wt_res_3.upper())
        mut_res = THREE_TO_ONE.get(mut_res_3.upper())
        if wt_res and mut_res:
            mut_tuple = (wt_res, pos, mut_res)
            if mut_tuple not in seen:
                seen.add(mut_tuple)
                mutations.append(mut_tuple)
                
    return mutations
