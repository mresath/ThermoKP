"""
===========================================================================
Dataset Validator
Description: Database Validation Pre-requisite
===========================================================================

Workflow:
1. Connect to the SQLite database and load `clean_parameters` into a pandas DataFrame.
2. Thermodynamic Sanity Check: Flag rows where k_cat / K_m > 10^9 (diffusion limit) and drop them.
3. Chemical Feasibility Dry Run: Resolve substrates to SMILES via (in order)
   a manual override dict, a disk cache shared with sabio_rk_parser.py
   (`data/cache/smiles_cache.json` - populated from ChEBI/KEGG cross-references
   there, and from every other tier here, so a name is never re-resolved
   over the network by a later run of the parser, the validator, or the
   geometry pipeline), a local peptide-notation builder for protease-assay
   substrates like "benzyloxycarbonyl-ala-gly-leu-ala" (`_resolve_peptide_smiles`
   - no network call, PubChem/OPSIN/CACTUS don't recognize this shorthand no
   matter how many times it's retried), a skip for names already
   known-unresolvable from a prior full validator run (`failed_chemicals.txt`
   - delete it to force a retry), then PubChem, OPSIN, and NIH CACTUS; as a
   last resort, retry the network tiers with a leading stereodescriptor
   stripped (logged to `destereo_fallback_used.txt` since this discards
   stereochemistry). Parse with RDKit, log failures to `failed_chemicals.txt`,
   and drop associated rows.
4. Structure Availability Check: Send HEAD requests to the AlphaFold EBI
   endpoint; for any 404, fall back to predicting a structure via the ESM
   Atlas folding API (`pretrained_embeddings.fetch_esmfold_pdb`), caching
   a hit to the same directory geometry_processor.py reads from. Log entries
   with no structure via either source to `failed_structures.txt` and drop them.
5. Sequence Length Check: Fetch each unique UniProt sequence and drop entries
   whose length exceeds ESM2's usable context window (too long to embed).
   Failures are logged to `failed_sequences.txt`.
6. Update `clean_parameters` with the validated rows and print a terminal report.

Known Caveats:
- PubChem API has strict rate limits. A 0.2s delay is implemented.
- AlphaFold/ESMFold validation is performed via parallel requests.
- The UniProt REST FASTA endpoint has no documented rate limit but is
  queried sequentially per unique UniProt ID to stay conservative.
- The ESMFold Atlas API's practical sequence-length limit (~400 residues)
  is well below ESM2's (1022), so it cannot rescue every AlphaFold miss.

Author: ThermoKP Team
License: MIT
"""

import threading
import concurrent.futures
import logging
from typing import Optional
import pathlib
import re
import sqlite3
import time
import urllib.parse

import pandas as pd
import requests
import rdkit.Chem as Chem
from rdkit import rdBase
from functools import lru_cache

from src.data.processors.pretrained_embeddings import (
    ESM2_MAX_SEQ_LEN,
    PDB_CACHE_DIR,
    check_alphafold_structure_exists,
    fetch_esmfold_pdb,
    fetch_uniprot_sequence,
    fetch_uniprot_cleavage_offset,
)
from src.data.utils.smiles_cache import load_smiles_cache, save_smiles_cache_entry

MUTATION_CODE_PATTERN = re.compile(r"^([A-Z])(\d+)([A-Z])$")


# ═══════════════════════════════════════════════════════════════════════════
#  Logging Configuration
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

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
DB_PATH = _PROJECT_ROOT / "data" / "thermokp_database.db"
FAILED_CHEMICALS_FILE = _PROJECT_ROOT / "data" / "failed_chemicals.txt"
FAILED_STRUCTURES_FILE = _PROJECT_ROOT / "data" / "failed_structures.txt"
FAILED_SEQUENCES_FILE = _PROJECT_ROOT / "data" / "failed_sequences.txt"
DESTEREO_FALLBACK_FILE = _PROJECT_ROOT / "data" / "destereo_fallback_used.txt"

DIFFUSION_LIMIT = 1e9  # M^-1 s^-1

# Matches one or more leading parenthesized stereodescriptor groups, e.g.
# "(r)-", "(r,s)-", "(1r,4s)-", or an optical-rotation sign "(-)-"/"(+)-".
# Deliberately anchored to the start of the string so embedded substituent
# parentheses elsewhere in the name (e.g. "(6-methoxy-2-naphthyl)") are
# never touched.
_STEREO_PREFIX_PATTERN = re.compile(r"^(\([0-9]{0,3}[rsRS](,[0-9]{0,3}[rsRS])*\)-|\([+-]\)-)+")


# ═══════════════════════════════════════════════════════════════════════════
#  Peptide-Notation Substrate Parser
# ═══════════════════════════════════════════════════════════════════════════
#
# Protease/peptidase assay substrates are frequently named as a hyphenated
# chain of 3-letter amino acid codes (e.g. "ala-gly-leu-ala"), often flanked
# by an N-terminal protecting group (Z-/Boc-/Ac-/Bz-/Suc-/H-) and/or a
# C-terminal chromogenic/fluorogenic leaving group (-NH2, -pNA, -AMC, a pNP
# ester, -OMe, -OEt). PubChem/OPSIN/CACTUS don't recognize this shorthand
# since it isn't IUPAC nomenclature, so these fail every existing tier no
# matter how many times they're retried - building the structure directly
# from known fragments resolves them locally with no network call at all.
#
# Every fragment below was manually verified against RDKit's own
# `Chem.MolFromSequence` L-amino-acid builder or a known reference molecular
# formula/weight before use - a silently wrong structure is worse than a
# resolution failure, so these must be re-verified the same way if edited.
# D-amino acids are deliberately NOT supported: naively swapping the SMILES
# stereo-bond marker does not reliably yield the correct D-enantiomer
# (confirmed mismatch on D-Ala vs RDKit's own D-form builder), so any
# "d-"-prefixed residue causes this parser to bail out entirely rather than
# risk producing a silently mirror-imaged structure.
_PEPTIDE_SIDECHAINS: dict[str, str] = {
    "ala": "C", "val": "C(C)C", "leu": "CC(C)C", "ile": "[C@@H](C)CC",
    "phe": "Cc1ccccc1", "trp": "Cc1c[nH]c2ccccc12", "met": "CCSC",
    "ser": "CO", "thr": "[C@H](O)C", "cys": "CS", "tyr": "Cc1ccc(O)cc1",
    "asn": "CC(N)=O", "gln": "CCC(N)=O", "asp": "CC(=O)O", "glu": "CCC(=O)O",
    "lys": "CCCCN", "arg": "CCCNC(=N)N", "his": "Cc1c[nH]cn1",
}
_PEPTIDE_RESIDUE_CODES = frozenset(_PEPTIDE_SIDECHAINS) | {"gly", "pro"}

# N-terminal protecting groups: a SMILES prefix ending exactly where the
# first residue's backbone N continues the chain (plain string concatenation
# forms the amide/carbamate bond, no explicit bond syntax needed). Keyed by
# the token as it appears after name_clean.split("-") - i.e. without the
# trailing hyphen itself.
_PEPTIDE_NTERM_CAPS: dict[str, str] = {
    "benzyloxycarbonyl": "O=C(OCc1ccccc1)",
    "carbobenzoxy": "O=C(OCc1ccccc1)",  # synonym for benzyloxycarbonyl/Z
    "z": "O=C(OCc1ccccc1)",
    "boc": "O=C(OC(C)(C)C)",
    "acetyl": "CC(=O)",
    "ac": "CC(=O)",
    "benzoyl": "O=C(c1ccccc1)",
    "bz": "O=C(c1ccccc1)",
    "succinyl": "O=C(CCC(=O)O)",
    "suc": "O=C(CCC(=O)O)",
    "h": "",  # explicit "free amine" marker, equivalent to no prefix at all
}

# C-terminal leaving/reporter groups: a SMILES suffix appended directly after
# the last residue's "C(=O)", completing it into an amide (attaches via N)
# or an ester (attaches via O). Longer keys are matched first (see
# _resolve_peptide_smiles) since some groups are themselves hyphenated.
_PEPTIDE_CTERM_CAPS: dict[str, str] = {
    "nh2": "N",
    "p-nitroanilide": "Nc1ccc([N+](=O)[O-])cc1",
    "4-nitroanilide": "Nc1ccc([N+](=O)[O-])cc1",
    "pna": "Nc1ccc([N+](=O)[O-])cc1",
    "7-amino-4-methylcoumarin": "Nc1ccc2c(c1)oc(=O)cc2C",
    "4-methylcoumarin-7-ylamide": "Nc1ccc2c(c1)oc(=O)cc2C",
    "4-methylcoumaryl-7-amide": "Nc1ccc2c(c1)oc(=O)cc2C",  # spelling variant
    "amc": "Nc1ccc2c(c1)oc(=O)cc2C",
    "p-nitrophenyl ester": "Oc1ccc([N+](=O)[O-])cc1",
    "pnp": "Oc1ccc([N+](=O)[O-])cc1",
    "ome": "OC",
    "oet": "OCC",
}
# Sorted longest-first so a multi-token cap (e.g. "p-nitroanilide") is tried
# before a shorter one that could otherwise match a trailing substring of it.
_PEPTIDE_CTERM_CAPS_BY_LENGTH = sorted(
    _PEPTIDE_CTERM_CAPS, key=lambda k: k.count("-"), reverse=True
)


def _resolve_peptide_smiles(name_clean: str) -> Optional[str]:
    """Build a SMILES string for a hyphen-separated peptide-notation substrate.

    Recognizes a chain of standard 3-letter amino acid codes, optionally
    flanked by a known N-terminal protecting group and/or C-terminal
    leaving group (see the module-level caps above). Bails out (returns
    None, leaving the network tiers to try instead) the moment any token
    doesn't match a known residue or cap - including any "d-"-prefixed
    (D-form) residue - rather than guess.

    Parameters
    ----------
    name_clean : str
        The already-stripped, lowercased chemical name.

    Returns
    -------
    str or None
        A SMILES string, or None if the name doesn't fit this pattern.
    """
    tokens = name_clean.split("-")

    n_cap = ""
    if tokens and tokens[0] in _PEPTIDE_NTERM_CAPS:
        n_cap = _PEPTIDE_NTERM_CAPS[tokens[0]]
        tokens = tokens[1:]

    c_cap = "O"  # Default: free carboxylic acid.
    for cap_key in _PEPTIDE_CTERM_CAPS_BY_LENGTH:
        cap_tokens = cap_key.split("-")
        if len(tokens) > len(cap_tokens) and tokens[-len(cap_tokens):] == cap_tokens:
            c_cap = _PEPTIDE_CTERM_CAPS[cap_key]
            tokens = tokens[: -len(cap_tokens)]
            break

    if not tokens:
        return None

    backbone: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "d":
            return None  # D-form residue - deliberately unsupported, see module docstring.
        if tok not in _PEPTIDE_RESIDUE_CODES:
            return None  # Unrecognized token - not a peptide this parser can build.
        backbone.append(tok)
        i += 1

    parts = [n_cap]
    for code in backbone:
        if code == "gly":
            parts.append("NCC(=O)")
        elif code == "pro":
            parts.append("N1CCC[C@H]1C(=O)")
        else:
            parts.append(f"N[C@@H]({_PEPTIDE_SIDECHAINS[code]})C(=O)")
    parts.append(c_cap)
    smiles = "".join(parts)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return smiles


# ═══════════════════════════════════════════════════════════════════════════
#  Validation Functions
# ═══════════════════════════════════════════════════════════════════════════

MANUAL_SMILES_OVERRIDES = {
    "juvenile hormone iii": r"COC(=O)/C=C(\C)/CCC=C(C)/CCC1OC1(C)C",
    "juvenile hormone iii acid": r"C/C(=C\CCC(=CC(=O)O)C)CCC1OC1(C)C",
    "saturated juvenile hormone iii": r"C/C(=C\CC/C(=C/C(=O)OC)/C)/CC[C@@H]1C(O1)(C)C",
    "juvenile hormone iii acid bisepoxide": r"CC(C)(O1)[C@H]1CC[C@]2([C@@H](O2)CC/C(C)=C/C(O)=O)C",
    "thio-nad+": r"NC1=NC=NC2=C1N=CN2[C@H]3[C@H](O)[C@H](O)[C@@H](COP(OP(OC[C@@H](O[C@H]([C@@H]4O)[N+]5=CC(C(N)=S)=CC=C5)[C@H]4O)([O-])=O)(O)=O)O3",
    "ubiquinone": r"CC1=C(C(=O)C(=C(C1=O)OC)OC)C/C=C(\C)/CC/C=C(\C)/CC/C=C(\C)/CC/C=C(\C)/CC/C=C(\C)/CC/C=C(\C)/CC/C=C(\C)/CC/C=C(\C)/CC/C=C(\C)/CC/C=C(\C)C"
}

_API_LOCK = threading.Lock()
_LAST_REQ = 0.0

def _safe_get(url: str, max_retries: int = 3) -> Optional[requests.Response]:
    """
    Safely executes a GET request with retries.

    Parameters
    ----------
    url : str
        The target URL.
    max_retries : int, optional
        Maximum number of retry attempts.

    Returns
    -------
    Optional[requests.Response]
        The response object if successful, else None.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response
            elif response.status_code in (429, 503, 504):
                time.sleep(1.0 * (2 ** attempt))
                continue
            else:
                return response
        except Exception:
            if attempt == max_retries - 1:
                return None
            time.sleep(1.0 * (2 ** attempt))
    return None

def _normalize_opsin_stereo(name_clean: str) -> str:
    """Upper-case stereodescriptors/D-L tags so OPSIN's strict parser accepts them.

    Parameters
    ----------
    name_clean : str
        The chemical name, already stripped.

    Returns
    -------
    str
        The name with stereodescriptors normalized for OPSIN.
    """
    opsin_name = re.sub(r"([0-9',]+)([rs])\b", lambda m: f"{m.group(1)}{m.group(2).upper()}", name_clean)
    opsin_name = opsin_name.replace("(r)", "(R)").replace("(s)", "(S)")
    opsin_name = opsin_name.replace("(r,", "(R,").replace(",r)", ",R)")
    opsin_name = opsin_name.replace("(s,", "(S,").replace(",s)", ",S)")
    opsin_name = re.sub(r"\bd-", "D-", opsin_name)
    opsin_name = re.sub(r"\bl-", "L-", opsin_name)
    return opsin_name


def _try_resolve_tiers(name_clean: str) -> Optional[str]:
    """Try PubChem, then OPSIN, then CACTUS for a single name variant.

    Callers are responsible for rate-limiting (see `_API_LOCK`/`_LAST_REQ`
    in `get_smiles`) - this only tries the three network tiers in order.

    Parameters
    ----------
    name_clean : str
        The chemical name variant to resolve.

    Returns
    -------
    str or None
        The SMILES string, or None if all three tiers failed.
    """
    encoded_name = urllib.parse.quote(name_clean)

    # Tier: PubChem PUG REST
    url_pubchem = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/property/IsomericSMILES,CanonicalSMILES/JSON"
    response = _safe_get(url_pubchem)
    if response and response.status_code == 200:
        try:
            data = response.json()
            props = data["PropertyTable"]["Properties"][0]
            smiles = (props.get("CanonicalSMILES") or
                      props.get("IsomericSMILES") or
                      props.get("SMILES") or
                      props.get("ConnectivitySMILES"))
            if smiles:
                return smiles
        except Exception:
            pass

    # Tier: OPSIN API (Robust IUPAC Parsing)
    opsin_name = _normalize_opsin_stereo(name_clean)
    url_opsin = f"https://opsin.ch.cam.ac.uk/opsin/{urllib.parse.quote(opsin_name)}.smi"
    response = _safe_get(url_opsin)
    if response and response.status_code == 200:
        smiles = response.text.strip()
        if smiles:
            return smiles

    # Tier: NIH CACTUS API
    url_cactus = f"https://cactus.nci.nih.gov/chemical/structure/{encoded_name}/smiles"
    response = _safe_get(url_cactus)
    if response and response.status_code == 200:
        smiles = response.text.strip()
        if smiles:
            return smiles

    return None


_known_failed_chemicals: Optional[set[str]] = None


def _load_known_failed_chemicals() -> set[str]:
    """Load (and memoize) names already confirmed unresolvable by a prior run.

    Consults FAILED_CHEMICALS_FILE (written by main() at the end of a full
    validator run) so a repeat run skips straight to returning None for
    these names instead of re-attempting every network tier - delete
    FAILED_CHEMICALS_FILE to force a fresh attempt for all names (e.g.
    after adding a manual override or fixing a name-cleaning bug).

    Returns
    -------
    set of str
        Names known to fail as of the last full validator run, empty if
        the file doesn't exist yet.
    """
    global _known_failed_chemicals
    if _known_failed_chemicals is None:
        if FAILED_CHEMICALS_FILE.exists():
            _known_failed_chemicals = {
                line.strip()
                for line in FAILED_CHEMICALS_FILE.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        else:
            _known_failed_chemicals = set()
    return _known_failed_chemicals


@lru_cache(maxsize=None)
def get_smiles(name: str) -> Optional[str]:
    """Fetch canonical SMILES from APIs using a multi-tiered approach.

    Tier 1: Manual Override Dictionary
    Tier 2: Shared disk cache (ChEBI/KEGG-resolved by sabio_rk_parser.py, or
            resolved by any tier below on a previous call/run/process)
    Tier 3: Local peptide-notation builder (see `_resolve_peptide_smiles`) -
            no network call, so this always gets a chance even if a prior
            run's failed_chemicals.txt already marked the name unresolvable.
    Tier 4: Skip entirely if `name` is in a prior run's failed_chemicals.txt
    Tier 5: PubChem PUG REST API
    Tier 6: OPSIN API
    Tier 7: NIH CACTUS API
    Tier 8: Retry tiers 5-7 with a leading stereodescriptor stripped (e.g.
            "(r,s)-(-)-ephedrine" -> "ephedrine"), only if every prior tier
            failed. This discards real stereochemistry information, so it
            is logged and recorded in DESTEREO_FALLBACK_FILE for audit
            rather than applied silently.

    Any resolution reached via tiers 3, 5-8 is written back into the Tier 2
    disk cache before returning, so a name is never re-resolved over the
    network again by this or any later process (dataset_validator.py,
    geometry_pipeline.py, or a future sabio_rk_parser.py run) - `lru_cache`
    alone only helps within a single process's lifetime.

    Parameters
    ----------
    name : str
        The chemical name.

    Returns
    -------
    str or None
        The SMILES string, or None if not found.
    """
    global _LAST_REQ
    if not name:
        return None

    name_clean = name.strip()

    # Tier 1: Manual Override
    if name_clean in MANUAL_SMILES_OVERRIDES:
        return MANUAL_SMILES_OVERRIDES[name_clean]

    # Tier 2: shared disk cache
    cached = load_smiles_cache().get(name_clean.lower())
    if cached:
        return cached

    # Tier 3: local peptide-notation builder (pure computation, no network)
    peptide_smiles = _resolve_peptide_smiles(name_clean.lower())
    if peptide_smiles:
        save_smiles_cache_entry(name_clean.lower(), peptide_smiles)
        return peptide_smiles

    # Tier 4: known-unresolvable from a prior full validator run
    if name_clean in _load_known_failed_chemicals():
        return None

    with _API_LOCK:
        now = time.time()
        if now - _LAST_REQ < 0.25:
            time.sleep(0.25 - (now - _LAST_REQ))

        smiles = _try_resolve_tiers(name_clean)

        if not smiles:
            stripped = _STEREO_PREFIX_PATTERN.sub("", name_clean)
            if stripped != name_clean and stripped:
                smiles = _try_resolve_tiers(stripped)
                if smiles:
                    logger.warning(
                        "Resolved '%s' only after stripping its stereodescriptor "
                        "prefix (as '%s') - stereochemistry is not represented "
                        "in the resulting SMILES.", name_clean, stripped,
                    )
                    with open(DESTEREO_FALLBACK_FILE, "a") as f:
                        f.write(f"{name_clean} -> {stripped}\n")

        _LAST_REQ = time.time()

    if smiles:
        save_smiles_cache_entry(name_clean.lower(), smiles)

    return smiles


def resolve_smiles(entry: str) -> Optional[str]:
    """Resolve a chemical entry that may be either a name or a raw SMILES string.

    Tries `entry` directly as a SMILES string first (RDKit parse); falls
    back to `get_smiles` name resolution otherwise. Lets callers accept
    either a database-style chemical name or a literal SMILES string
    without needing to know in advance which one they were given.

    Parameters
    ----------
    entry : str
        A chemical name or a SMILES string.

    Returns
    -------
    str or None
        `entry` itself if it already parses as valid SMILES, the name-
        resolved SMILES string otherwise, or None if neither succeeds.
    """
    if entry:
        with rdBase.BlockLogs():
            is_smiles = Chem.MolFromSmiles(entry) is not None
        if is_smiles:
            return entry
    return get_smiles(entry)


def _fetch_and_parse_chemical(name: str) -> bool:
    """Fetch and parse a chemical to ensure valid RDKit handling.

    Parameters
    ----------
    name : str
        The chemical name.

    Returns
    -------
    bool
        True if the chemical is successfully parsed by RDKit, False otherwise.
    """
    smiles = get_smiles(name)
    if not smiles:
        return False

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumHeavyAtoms() > 100:
            return False
        return True
    except Exception:
        return False

def _validate_chemicals(chemical_names: set[str]) -> set[str]:
    """Validate a set of chemical names via PubChem and RDKit.

    Runs concurrently (mirrors validation's thread pool) - this
    is pure I/O (network calls, or instant cache/peptide-builder hits), so
    threading is a natural fit. `get_smiles`'s own `_API_LOCK`/`_LAST_REQ`
    pacing already serializes and rate-limits the actual network tiers
    across all threads, so concurrent callers can't overwhelm PubChem/OPSIN/
    CACTUS even though many threads call `_fetch_and_parse_chemical` at once.

    Parameters
    ----------
    chemical_names : set of str
        Unique set of chemical names.

    Returns
    -------
    set of str
        Set of chemical names that failed validation.
    """
    failed_chemicals = set()
    total = len(chemical_names)

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        future_to_name = {
            executor.submit(_fetch_and_parse_chemical, name): name
            for name in chemical_names
        }
        completed = 0
        for future in concurrent.futures.as_completed(future_to_name):
            completed += 1
            name = future_to_name[future]
            if future.result():
                logger.info(f"[{completed}/{total}] Successfully validated chemical: {name}")
            else:
                logger.warning(f"[{completed}/{total}] Failed to validate chemical: {name}")
                failed_chemicals.add(name)

    return failed_chemicals





def _validate_single_structure(uid: str) -> tuple[str, bool]:
    """Check structure availability for one UniProt ID, AlphaFold first, ESMFold fallback.

    A hit from either source is cached to `PDB_CACHE_DIR` so `geometry_processor.py`
    reuses it at tensor-generation time instead of re-downloading/re-folding.

    Parameters
    ----------
    uid : str
        UniProt accession to check.

    Returns
    -------
    tuple[str, bool]
        ``(uid, available)`` where ``available`` is ``True`` if a structure
        was found or successfully cached via AlphaFold or ESMFold.
    """
    if (PDB_CACHE_DIR / f"{uid}.pdb").exists():
        return uid, True

    if check_alphafold_structure_exists(uid):
        return uid, True

    sequence = fetch_uniprot_sequence(uid)
    if sequence is None:
        return uid, False

    return uid, fetch_esmfold_pdb(uid, sequence, PDB_CACHE_DIR) is not None


def _validate_structures(uniprot_ids: set[str]) -> set[str]:
    """Check AlphaFold/ESMFold structure availability for a set of UniProt IDs.

    Parameters
    ----------
    uniprot_ids : set of str
        UniProt accessions to check, deduplicated across the dataset.

    Returns
    -------
    set of str
        UniProt IDs with no structure available via either source.
    """
    failed_uids = set()
    total = len(uniprot_ids)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_uid = {executor.submit(_validate_single_structure, uid): uid for uid in uniprot_ids}
        completed = 0
        for future in concurrent.futures.as_completed(future_to_uid):
            completed += 1
            uid, available = future.result()
            if available:
                logger.info(f"[{completed}/{total}] Structure available for: {uid}")
            else:
                logger.warning(f"[{completed}/{total}] No AlphaFold/ESMFold structure for: {uid}")
                failed_uids.add(uid)

    return failed_uids


def _validate_single_protein(uid: str, mutations: set[Optional[str]]) -> tuple[str, Optional[str], set[Optional[str]], Optional[str]]:
    """Fetch a protein's sequence and cleavage offset, and validate all its mutations.

    Parameters
    ----------
    uid : str
        UniProt accession for the protein.
    mutations : set of str or None
        Mutation codes (e.g. ``"A123B"``) reported against this protein;
        ``None`` denotes a wild-type record.

    Returns
    -------
    tuple[str, str or None, set of str or None, str or None]
        ``(uid, sequence, failed_mutations, reason)``. ``sequence`` is
        ``None`` if the UniProt fetch failed. ``failed_mutations`` is the
        subset of `mutations` that could not be validated against the
        sequence (accounting for the signal-peptide cleavage offset).
        ``reason`` summarizes why validation failed, or ``None`` on success.
    """
    sequence = fetch_uniprot_sequence(uid)
    if sequence is None:
        return uid, sequence, mutations, "Failed to fetch UniProt sequence"  # All fail if fetch fails

    offset = fetch_uniprot_cleavage_offset(uid)
    mature_len = len(sequence) - offset
    if mature_len > ESM2_MAX_SEQ_LEN:
        return uid, sequence, mutations, f"Mature sequence length {mature_len} exceeds ESM2 limit of {ESM2_MAX_SEQ_LEN}"

    failed_muts: set[Optional[str]] = set()
    fail_reasons = []

    for mut in mutations:
        if not mut or not mut.strip():
            continue
        mut_str = mut.strip()
        match = MUTATION_CODE_PATTERN.match(mut_str)
        if not match:
            fail_reasons.append(f"Mutation code '{mut_str}' does not match expected pattern (e.g. 'A123B')")
            failed_muts.add(mut)
            continue
            
        wt_res, pos_str, _ = match.groups()
        seq_idx = int(pos_str) - 1
        
        # Stage 1: Try raw match
        if 0 <= seq_idx < len(sequence) and sequence[seq_idx] == wt_res:
            continue
            
        # Stage 2: Try offset match
        shifted_idx = seq_idx + offset
        if 0 <= shifted_idx < len(sequence) and sequence[shifted_idx] == wt_res:
            continue
            
        raw_found = sequence[seq_idx] if 0 <= seq_idx < len(sequence) else "out of range"
        shifted_found = sequence[shifted_idx] if 0 <= shifted_idx < len(sequence) else "out of range"
        fail_reasons.append(f"WT residue '{wt_res}' at raw index {seq_idx} (found '{raw_found}' or '{shifted_found}' if offset) does not match sequence")
        failed_muts.add(mut)

    reason_str = "; ".join(fail_reasons) if fail_reasons else None
    return uid, sequence, failed_muts, reason_str


def _validate_sequences(pairs: set[tuple[str, Optional[str]]]) -> set[tuple[str, Optional[str]]]:
    """Validate that sequences are fetchable, within ESM2's window, and mutations align.

    Parameters
    ----------
    pairs : set of tuple(str, str or None)
        (uniprot_id, mutation) pairs to validate, grouped internally by
        `uniprot_id` so each protein's sequence is fetched only once.

    Returns
    -------
    set of tuple(str, str or None)
        Set of (uniprot_id, mutation) pairs that failed validation.
    """
    failed_pairs = set()
    
    uid_to_muts = {}
    for uid, mut in pairs:
        uid_to_muts.setdefault(uid, set()).add(mut)
        
    total = len(uid_to_muts)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_uid = {
            executor.submit(_validate_single_protein, uid, muts): uid
            for uid, muts in uid_to_muts.items()
        }
        completed = 0
        for future in concurrent.futures.as_completed(future_to_uid):
            completed += 1
            uid, sequence, failed_muts, reason = future.result()
            if sequence is None:
                logger.warning(f"[{completed}/{total}] Failed to fetch UniProt sequence for: {uid}")
            elif failed_muts:
                if len(failed_muts) == len(uid_to_muts[uid]):
                    logger.warning(f"[{completed}/{total}] All records failed validation for: {uid} (Reason: {reason})")
                else:
                    logger.warning(f"[{completed}/{total}] {len(failed_muts)}/{len(uid_to_muts[uid])} records failed validation for: {uid} (Reason: {reason})")
            else:
                logger.info(f"[{completed}/{total}] Successfully validated sequences for: {uid}")
                
            for mut in failed_muts:
                failed_pairs.add((uid, mut))

    return failed_pairs


# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Execute the dataset validation pipeline.

    Returns
    -------
    None
    """
    logger.info("==========================================================================")
    logger.info("==                ThermoKP Database Validation Pipeline                    ==")
    logger.info("==========================================================================")

    if not DB_PATH.exists():
        logger.error(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    
    try:
        df_result = pd.read_sql_query("SELECT * FROM clean_parameters", conn)
        if not isinstance(df_result, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame")
        df: pd.DataFrame = df_result
    except Exception as exc:
        logger.error(f"Failed to read 'clean_parameters' table: {exc}")
        conn.close()
        return

    initial_count = len(df)
    logger.info(f"Loaded {initial_count} records from clean_parameters.")

    # -----------------------------------------------------------------------
    # 1. Thermodynamic Sanity Check
    # -----------------------------------------------------------------------
    # K_m is provided in mM, converted to M by multiplying by 1e-3.
    # Catalytic efficiency = k_cat / K_m (M^-1 s^-1)
    df["catalytic_efficiency"] = df["kcat"] / (df["km"] * 1e-3)
    
    mask_thermo = df["catalytic_efficiency"] <= DIFFUSION_LIMIT
    dropped_thermo = initial_count - mask_thermo.sum()
    df = df[mask_thermo].copy()
    logger.info(f"Dropped {dropped_thermo} records due to thermodynamic infeasibility." )
    
    # -----------------------------------------------------------------------
    # 2. Chemical Feasibility Dry Run
    # -----------------------------------------------------------------------
    all_chemicals = set()
    for _, row in df.iterrows():  
        sub_val = row.get("measured_substrate")
        if isinstance(sub_val, str) and sub_val.strip():
            all_chemicals.add(sub_val.strip())
        co_val = row.get("co_substrates")
        if isinstance(co_val, str) and co_val.strip():
            for sub in co_val.split("; "):
                all_chemicals.add(sub.strip())

    failed_chemicals = _validate_chemicals(all_chemicals)
    
    with open(FAILED_CHEMICALS_FILE, "w") as f:
        for ch in sorted(failed_chemicals):
            f.write(f"{ch}\n")
            
    def contains_failed_chemical(row: pd.Series) -> bool:
        """
        Checks if a record contains any failed chemical.

        Parameters
        ----------
        row : pd.Series
            The dataframe row representing a kinetic record.

        Returns
        -------
        bool
            True if it contains a failed chemical, False otherwise.
        """
        sub_val = row.get("measured_substrate")
        if isinstance(sub_val, str) and sub_val.strip():
            if sub_val.strip() in failed_chemicals:
                return True
        co_val = row.get("co_substrates")
        if isinstance(co_val, str) and co_val.strip():
            for sub in co_val.split("; "):
                if sub.strip() in failed_chemicals:
                    return True
        return False

    mask_chem = ~df.apply(contains_failed_chemical, axis=1)
    dropped_chem = len(df) - mask_chem.sum()
    df = df[mask_chem].copy()

    # -----------------------------------------------------------------------
    # 3. Structure Availability Check
    # -----------------------------------------------------------------------
    all_uniprot_ids = {
        uid.strip() for uid in df["uniprot_id"] if isinstance(uid, str) and uid.strip()
    }
    failed_structures = _validate_structures(all_uniprot_ids)

    with open(FAILED_STRUCTURES_FILE, "w") as f:
        for uid in sorted(failed_structures):
            f.write(f"{uid}\n")

    mask_structure = ~df["uniprot_id"].isin(list(failed_structures))
    dropped_structure = len(df) - mask_structure.sum()
    df = df[mask_structure].copy()

    # -----------------------------------------------------------------------
    # 4. ESM2 Sequence and Mutation Check
    # -----------------------------------------------------------------------
    pairs = set()
    for _, row in df.iterrows():  
        uid_val = row.get("uniprot_id")
        if isinstance(uid_val, str) and uid_val.strip():
            uid = uid_val.strip()
            mut_val = row.get("mutation")
            mut = mut_val.strip() if isinstance(mut_val, str) else None  
            pairs.add((uid, mut))
            
    failed_pairs = _validate_sequences(pairs)

    with open(FAILED_SEQUENCES_FILE, "w") as f:
        for uid, mut in sorted(list(failed_pairs), key=lambda x: str(x[0])):
            f.write(f"{uid}\t{mut}\n")

    def is_valid_seq(row: pd.Series) -> bool:
        """Checks if a record's (uniprot_id, mutation) pair passed sequence validation."""
        uid_val = row.get("uniprot_id")
        if not isinstance(uid_val, str) or not uid_val.strip():
            return False
        uid = uid_val.strip()
        mut_val = row.get("mutation")
        mut = mut_val.strip() if isinstance(mut_val, str) else None  
        return (uid, mut) not in failed_pairs

    mask_seq = df.apply(is_valid_seq, axis=1)  
    dropped_seq = len(df) - mask_seq.sum()
    df = df[mask_seq].copy()

    # -----------------------------------------------------------------------
    # 5. Execution and Reporting
    # -----------------------------------------------------------------------
    # Drop the temporary column before saving
    df = df.drop(columns=["catalytic_efficiency"])  
    
    df.to_sql("clean_parameters", conn, if_exists="replace", index=False)
    conn.close()

    final_count = len(df)
    
    logger.info("==========================================================================")
    logger.info("==                        Validation Summary                            ==")
    logger.info("==========================================================================")
    logger.info(f"Initial row count         : {initial_count}")
    logger.info(f"Dropped (Thermodynamics)  : {dropped_thermo}")
    logger.info(f"Dropped (Chemicals)       : {dropped_chem}")
    logger.info(f"Dropped (Structures)      : {dropped_structure}")
    logger.info(f"Dropped (Sequence)        : {dropped_seq}")
    logger.info(f"Final safe row count      : {final_count}")
    logger.info("==========================================================================")


if __name__ == "__main__":
    main()
