"""
===========================================================================
SABIO-RK Parser
Description: SABIO-RK Database Ingestion for ThermoKP
===========================================================================

Workflow:
1. GET `/sabioRestWebServices/searchKineticLaws/sbml` with a query string of the form `ECNumber:"<ec>"` to retrieve an SBML document containing kinetic laws, parameters, and experimental conditions.
2. Parse the XML (namespace-agnostic for SBML L2V4 and L3V1) to extract: Entry ID, EC number, Organism, Substrate name(s), k_cat, K_m, Temperature, and pH.
3. For each species, also scan its RDF annotation for a ChEBI (or, failing
   that, KEGG compound) cross-reference and resolve+cache a SMILES for it
   via `_cache_species_smiles` - a more reliable substrate-identity source
   than the free-text name lookup `dataset_validator.py` falls back to.
4. If a species carries no UniProt cross-reference but an organism name was
   recovered, fall back to `brenda_parser.fetch_uniprot_id(ec, organism)`
   rather than dropping the record.
5. Unlike BRENDA, SABIO-RK carries mutant/variant identity directly in the
   enzyme modifier species' `name` (e.g. "...(Enzyme) mutant C75A"), not in
   free-text commentary. `_parse_sabio_mutation` parses point mutations.
   Wild-type is kept as-is, point substitutions are kept and labelled, and
   anything else (deletion, chimera) is dropped rather than silently
   merged into the wild-type replicate group.
6. Insert each valid record into the `raw_parameters` table via Python's built-in `sqlite3` module.

Known Caveats:
- SABIO-RK responses are often sparse: many kinetic laws lack one or more of the target fields. The parser silently skips entries that contain neither k_cat nor K_m, and logs warnings for partially populated records.
- Temperature values reported in Kelvin (>200) are converted to °C for consistency with BRENDA.
- The API can be slow for broadly-scoped queries. Constrain queries with organism filters or use the `max_results` parameter when possible.

`--full` run resumption:
- A `--full` run (the full ~8,400-EC target list) can take hours; `python -m
  src.data.parsers.sabio_rk_parser --full` alone always starts clean - it first
  deletes any existing `SABIO-RK` rows from `raw_parameters` and clears any
  stale checkpoint, then processes every EC number.
- After each EC finishes (data or not), its number is written to
  `SABIO_CHECKPOINT_FILE`. On a full, uninterrupted completion, that file
  is removed.
- If the run is interrupted (killed, network outage, machine restart), the
  checkpoint file is left behind. Re-run with `python -m
  src.data.parsers.sabio_rk_parser --full --continue` to resume immediately after
  the last completed EC instead of restarting from scratch - the existing
  rows already inserted are left untouched.

Author: ThermoKP Team
License: MIT

References
----------
- Wittig et al. (2012), Nucleic Acids Res. 40(D1), D790-D796.
- SABIO-RK REST docs:
  http://sabiork.h-its.org/layouts/content/docuRESTfulWeb/manual.gsp

Author : ThermoKP Team
License: MIT
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import sqlite3
import sys
import time
import uuid
import hashlib
import yaml
import xml.etree.ElementTree as ET
from typing import Optional

import requests
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import _CalcMolWt

from src.data.models.models import KineticRecord
from src.data.utils.ligand_cleaner import canonicalize_and_filter_ligands
from src.data.processors.dataset_validator import get_smiles
from src.data.processors.dataset_validator import get_smiles
from src.data.processors.pretrained_embeddings import fetch_uniprot_sequence

from src.data.parsers.parser_common import (
    fetch_uniprot_id,
    MUTATION_KEYWORDS,
    find_point_mutations,
)
from src.data.utils.smiles_cache import load_smiles_cache, save_smiles_cache_entry

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

SABIO_RK_BASE_URL: str = (
    "http://sabiork.h-its.org/sabioRestWebServices/searchKineticLaws/sbml"
)

# SBO term identifiers used by SABIO-RK for kinetic parameter types.
SBO_KM: str = "SBO:0000027"
SBO_KCAT: str = "SBO:0000025"
SBO_VMAX: str = "SBO:0000186"
SBO_E0_1: str = "SBO:0000505"
SBO_E0_2: str = "SBO:0000196"

# Project-relative path to the local database.
_PROJECT_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH: pathlib.Path = _PROJECT_ROOT / "data" / "thermokp_database.db"
TARGETS_FILE: pathlib.Path = _PROJECT_ROOT / "data" / "enzyme_targets.json"

# Records the last EC number a `--full` run finished processing, so an
# interrupted multi-hour run (network hang, manual kill, machine restart)
# can resume with `--continue` instead of starting over. Removed on a
# successful full completion - its presence means "there is an interrupted
# run to resume from"; its absence means "the last run finished cleanly".
SABIO_CHECKPOINT_FILE: pathlib.Path = _PROJECT_ROOT / "data" / "raw" / "sabio_checkpoint.txt"

CHEBI_ENTITY_URL_TEMPLATE: str = (
    "https://www.ebi.ac.uk/chebi/backend/api/public/compound/{chebi_id}"
)

# Default set of well-characterised enzymes for the test batch.
# Queries are constrained by organism to keep response sizes manageable
# and avoid server timeouts on very large EC-number datasets.
DEFAULT_QUERIES: list[dict[str, str]] = [
    {
        "ec": "4.2.1.11",
        "organism": "Saccharomyces cerevisiae",
        "label": "enolase (yeast)",
    },
    {
        "ec": "2.7.1.40",
        "organism": "Homo sapiens",
        "label": "pyruvate kinase (human)",
    },
    {
        "ec": "1.1.1.27",
        "organism": "Homo sapiens",
        "label": "lactate dehydrogenase (human)",
    },
]

# Seconds to wait after a query that returned actual data (queries that
# returned nothing already skip this delay entirely - see _run_batch). Lowered
# from 2.0s: this is pure inter-request politeness pacing, not related to
# the server's own response-generation time (which dominates for
# data-heavy ECs and dwarfs this delay regardless of its value).
_REQUEST_DELAY_S: float = 1.0
_REQUEST_TIMEOUT_S: int = 180

# Shared session so repeated requests reuse the same TCP/TLS connection
# instead of paying a fresh handshake every time - a real, if modest,
# saving across thousands of sequential requests in a --full run.
_session: requests.Session = requests.Session()


# ---------------------------------------------------------------------------
# Data containers have been moved to src/data/models.py


# ═══════════════════════════════════════════════════════════════════════════
#  API interaction
# ═══════════════════════════════════════════════════════════════════════════

def _build_query_string(ec_number: str, organism: Optional[str] = None) -> str:
    """Construct a SABIO-RK query string.

    Parameters
    ----------
    ec_number : str
        Enzyme Commission number, e.g. ``"4.2.1.11"``.
    organism : str or None
        Optional organism filter (e.g. ``"Homo sapiens"``).

    Returns
    -------
    str
        Formatted query string for the ``q`` parameter.
    """
    query = f"ECNumber:{ec_number}"
    if organism:
        query += f" AND Organism:\"{organism}\""
    # The parser drops any kinetic law lacking both kcat and Km anyway (see
    # _parse_sbml_response), so asking the server to pre-filter to laws
    # reporting at least one of the two is a lossless reduction in response
    # size/generation time - verified against EC 1.1.1.1 (SABIO's biggest
    # single EC): ~18% faster, ~17% smaller, zero change in parsed records.
    query += ' AND (Parametertype:"kcat" OR Parametertype:"Km")'
    return query


def _query_sabio_rk(
    ec_number: str,
    organism: Optional[str] = None,
) -> Optional[str]:
    """Send a GET request to SABIO-RK for kinetic laws.

    Parameters
    ----------
    ec_number : str
        Enzyme Commission number, e.g. ``"4.2.1.11"``.
    organism : str or None
        Optional organism filter.

    Returns
    -------
    str or None
        Raw SBML/XML response text, or ``None`` on failure.
    """
    query_string: str = _build_query_string(ec_number, organism)
    logger.info("Querying SABIO-RK  |  %s", query_string)

    try:
        response: requests.Response = _session.get(
            SABIO_RK_BASE_URL,
            params={"q": query_string},
            timeout=_REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error("Request timed out for EC %s", ec_number)
        return None
    except requests.exceptions.HTTPError as exc:
        logger.error(
            "HTTP %s for EC %s: %s",
            exc.response.status_code if exc.response is not None else "???",
            ec_number,
            exc,
        )
        return None
    except requests.exceptions.ConnectionError as exc:
        logger.error("Connection error for EC %s: %s", ec_number, exc)
        return None
    except requests.exceptions.RequestException as exc:
        logger.error("Unexpected request error for EC %s: %s", ec_number, exc)
        return None

    if not response.text.strip() or response.text.strip().startswith("No results found"):
        logger.info("No records found in SABIO-RK for EC %s", ec_number)
        return None

    logger.info(
        "Received %d bytes for EC %s", len(response.text), ec_number
    )
    return response.text


# ═══════════════════════════════════════════════════════════════════════════
#  XML / SBML parsing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _detect_sbml_namespace(root: ET.Element) -> str:
    """Extract the SBML namespace URI from the root ``<sbml>`` element.

    SABIO-RK may return SBML Level 2 Version 4 or Level 3 Version 1;
    the namespace differs between them.

    Parameters
    ----------
    root : ET.Element
        Parsed root element of the SBML document.

    Returns
    -------
    str
        Namespace URI (e.g.
        ``"http://www.sbml.org/sbml/level3/version1/core"``),
        or empty string if no namespace is detected.
    """
    match: Optional[re.Match[str]] = re.match(r"\{(.+)\}", root.tag)
    return match.group(1) if match else ""


def _ns_find(
    element: ET.Element,
    path: str,
    ns: dict[str, str],
) -> Optional[ET.Element]:
    """Namespace-aware ``find`` with fallback to bare tags.

    Parameters
    ----------
    element : ET.Element
        Parent element.
    path : str
        XPath using the ``s:`` prefix for the SBML namespace.
    ns : dict[str, str]
        Namespace mapping (``{"s": "<uri>"}``).

    Returns
    -------
    ET.Element or None
    """
    result: Optional[ET.Element] = element.find(path, ns)
    if result is None:
        bare_path: str = path.replace("s:", "")
        result = element.find(bare_path)
    return result


def _ns_findall(
    element: ET.Element,
    path: str,
    ns: dict[str, str],
) -> list[ET.Element]:
    """Namespace-aware ``findall`` with fallback to bare tags.

    Parameters
    ----------
    element : ET.Element
        Parent element.
    path : str
        XPath using the ``s:`` prefix for the SBML namespace.
    ns : dict[str, str]
        Namespace mapping.

    Returns
    -------
    list[ET.Element]
    """
    result: list[ET.Element] = element.findall(path, ns)
    if not result:
        bare_path: str = path.replace("s:", "")
        result = element.findall(bare_path)
    return result


def _local_tag(element: ET.Element) -> str:
    """Strip the namespace from an element's tag.

    Parameters
    ----------
    element : ET.Element

    Returns
    -------
    str
        Local tag name (e.g. ``"organism"`` from
        ``"{http://sabiork.h-its.org/}organism"``).
    """
    return element.tag.rpartition("}")[-1] if "}" in element.tag else element.tag


def _extract_local_parameter_value(
    kinetic_law_element: ET.Element,
    sbo_term: str,
    unit_definitions: Optional[dict[str, str]] = None,
) -> tuple[Optional[float], Optional[str], Optional[str]]:
    """Extract a numeric parameter value, its units, and its associatedSpecies from a ``<kineticLaw>``.

    Searches ``<localParameter>`` (or ``<parameter>``) children for one
    whose ``sboTerm`` attribute matches *sbo_term* and returns its
    ``value`` attribute cast to float, its ``units`` string, and
    ``associatedSpecies``.

    Parameters
    ----------
    kinetic_law_element : ET.Element
        The ``<kineticLaw>`` child of a ``<reaction>``.
    sbo_term : str
        SBO term identifier (e.g. ``"SBO:0000025"`` for k_cat).
    unit_definitions : dict[str, str], optional
        Map of ``<unitDefinition>`` ``id`` -> human-readable ``name``
        (e.g. ``"swedgeone"`` -> ``"s^(-1)"``). SABIO-RK's local parameter
        ``units`` attribute is a sanitized SBML SId (must match
        ``[a-zA-Z_][a-zA-Z0-9_]*``, so ``/``, ``^``, ``-`` etc. get mangled
        into words like "div"/"wedge"), NOT the literal unit string -
        callers must resolve it through the model's unitDefinition list to
        get something the unit-matching logic downstream can recognise.

    Returns
    -------
    tuple[Optional[float], Optional[str], Optional[str]]
        (Parameter value, units string, associatedSpecies ID).
    """
    for param in kinetic_law_element.iter():
        tag: str = _local_tag(param)
        if tag in ("localParameter", "parameter"):
            if param.get("sboTerm") == sbo_term:
                raw_value: Optional[str] = param.get("value")
                units: Optional[str] = param.get("units")
                if units and unit_definitions and units in unit_definitions:
                    units = unit_definitions[units]
                # SBML uses either species or associatedSpecies in various versions/flavours
                associated_species: Optional[str] = param.get("associatedSpecies") or param.get("species")
                
                # SABIO-RK often lacks the above attributes but embeds the species ID in the parameter ID
                if not associated_species:
                    pid = param.get("id", "")
                    if pid.startswith("Km_") or pid.startswith("Ks_"):
                        associated_species = pid[3:]

                if raw_value is not None:
                    try:
                        return float(raw_value), units, associated_species
                    except ValueError:
                        logger.debug(
                            "Non-numeric value '%s' for %s",
                            raw_value,
                            sbo_term,
                        )
    return None, None, None


def _normalise_temperature(raw: Optional[float]) -> Optional[float]:
    """Convert temperature to °C if it appears to be in Kelvin.

    Heuristic: values > 200 are assumed Kelvin.

    Parameters
    ----------
    raw : float or None
        Temperature value as reported in the SBML annotation.

    Returns
    -------
    float or None
        Temperature in °C.
    """
    if raw is None:
        return None
    if raw > 200.0:
        return round(raw - 273.15, 2)
    return round(raw, 2)


def _extract_annotation_data(
    annotation_element: Optional[ET.Element],
) -> tuple[Optional[int], Optional[str], Optional[float], Optional[float]]:
    """Extract kinetic-law ID, organism, temperature, pH from annotations.

    SABIO-RK uses two annotation formats depending on the SBML version
    and query type:

    *Format A* (organism-constrained queries, older exports):
        ``<sabiork:kineticlaw id="11">``  — ID as attribute
        ``<sabiork:startConditions>``     — temperature, pH
        ``<sabiork:experimentalConditions>`` — organism

    *Format B* (unconstrained queries, newer exports):
        ``<sabiork:kineticLawID>56568</sabiork:kineticLawID>`` — text
        Conditions may be absent or placed inside ``<kineticLaw>``
        annotation.

    Parameters
    ----------
    annotation_element : ET.Element or None
        The ``<annotation>`` child of a ``<reaction>`` **or** of a
        ``<kineticLaw>``.

    Returns
    -------
    tuple[int | None, str | None, float | None, float | None]
        ``(kinetic_law_id, organism, temperature_celsius, ph)``.
    """
    kl_id: Optional[int] = None
    organism: Optional[str] = None
    temperature: Optional[float] = None
    ph: Optional[float] = None

    if annotation_element is None:
        return kl_id, organism, temperature, ph

    for elem in annotation_element.iter():
        tag: str = _local_tag(elem)

        # Format A: <sabiork:kineticlaw id="11">
        if tag == "kineticlaw":
            raw_id: Optional[str] = elem.get("id")
            if raw_id is not None:
                try:
                    kl_id = int(raw_id)
                except ValueError:
                    pass

        # Format B: <sabiork:kineticLawID>56568</sabiork:kineticLawID>
        elif tag == "kineticLawID" and elem.text:
            try:
                kl_id = int(elem.text.strip())
            except ValueError:
                pass

        # Organism
        elif tag == "organism" and elem.text:
            organism = elem.text.strip()

        # Temperature (°C or K)
        elif tag in ("temperature", "startValueTemperature"):
            raw_val: Optional[str] = elem.text
            if raw_val is None:
                raw_val = elem.get("value")
            if raw_val is not None and raw_val.strip():
                try:
                    temperature = float(raw_val.strip())
                except ValueError:
                    pass

        # pH
        elif tag in ("pH", "ph", "startValuepH"):
            raw_val = elem.text
            if raw_val is None:
                raw_val = elem.get("value")
            if raw_val is not None and raw_val.strip():
                try:
                    ph = float(raw_val.strip())
                except ValueError:
                    pass

    temperature = _normalise_temperature(temperature)
    return kl_id, organism, temperature, ph


# ═══════════════════════════════════════════════════════════════════════════
#  Main parser
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
#  ChEBI/KEGG-backed SMILES cache
# ═══════════════════════════════════════════════════════════════════════════
KEGG_MOL_URL_TEMPLATE: str = "https://rest.kegg.jp/get/cpd:{kegg_id}/mol"


def _resolve_chebi_smiles(chebi_id: str) -> Optional[str]:
    """Fetch a canonical SMILES for a ChEBI ID via ChEBI's REST API.

    Parameters
    ----------
    chebi_id : str
        Bare ChEBI numeric ID (no "CHEBI:" prefix).

    Returns
    -------
    str or None
        The SMILES string, or None on any failure.
    """
    url = CHEBI_ENTITY_URL_TEMPLATE.format(chebi_id=chebi_id)
    try:
        response = _session.get(url, timeout=15)
        if response.status_code != 200:
            return None
        smiles = response.json().get("default_structure", {}).get("smiles")
        return smiles.strip() if smiles else None
    except Exception as e:
        logger.debug("Error fetching ChEBI:%s: %s", chebi_id, e)
    return None


def _resolve_kegg_smiles(kegg_id: str) -> Optional[str]:
    """Fetch a canonical SMILES for a KEGG compound ID via its MOL file.

    Parameters
    ----------
    kegg_id : str
        Bare KEGG compound ID (e.g. ``"C00022"``).

    Returns
    -------
    str or None
        The SMILES string (converted from the KEGG MOL block via RDKit), or
        None on any failure.
    """
    url = KEGG_MOL_URL_TEMPLATE.format(kegg_id=kegg_id)
    try:
        response = _session.get(url, timeout=15)
        if response.status_code != 200 or not response.text.strip():
            return None
        mol = Chem.MolFromMolBlock(response.text)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception as e:
        logger.debug("Error fetching KEGG:%s: %s", kegg_id, e)
    return None


def _cache_species_smiles(name: str, chebi_id: Optional[str], kegg_id: Optional[str]) -> None:
    """Resolve and cache a SMILES for `name` via ChEBI (preferred) or KEGG.

    No-op if `name` is already cached or neither ID is available - this
    keeps re-parsing the same EC/organism combination cheap on reruns.

    Parameters
    ----------
    name : str
        Lowercased species name, used as the cache key.
    chebi_id : str or None
        Bare ChEBI ID cross-referenced for this species, if any.
    kegg_id : str or None
        Bare KEGG compound ID cross-referenced for this species, if any.
    """
    if name in load_smiles_cache():
        return
    smiles = _resolve_chebi_smiles(chebi_id) if chebi_id else None
    if smiles is None and kegg_id:
        smiles = _resolve_kegg_smiles(kegg_id)
    if smiles:
        save_smiles_cache_entry(name, smiles)


def _resolve_species_molar_mass(species_name: Optional[str]) -> Optional[float]:
    """Compute a species' molar mass (g/mol) from its cached SMILES.

    Reuses the {name: SMILES} disk cache populated by
    `_cache_species_smiles` - no extra network call is needed as long as the
    species' ChEBI/KEGG cross-reference was already resolved earlier in the
    same parse.

    Parameters
    ----------
    species_name : str or None
        Lowercased species name (cache key).

    Returns
    -------
    float or None
        Molar mass in g/mol, or None if no cached SMILES exists for this
        species or it fails to parse.
    """
    if not species_name:
        return None
    smiles = load_smiles_cache().get(species_name)
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _CalcMolWt(mol)


def _parse_sabio_mutation(enzyme_name: str) -> tuple[bool, Optional[str]]:
    """Classify a SABIO-RK enzyme modifier species name as wild-type, a
    point mutant, or unsalvageable.

    Unlike BRENDA, SABIO-RK encodes variant identity directly in the
    modifier species' ``name`` (e.g. ``"chorismate mutase(Enzyme) mutant
    C75A"``, ``"trypsin(Enzyme) mutant DeltaI16V17/H40F trypsinogen"``,
    ``"...(Enzyme) wildtype"``). Without this parsing, a mutagenesis
    screening study - wild-type plus a dozen point mutants/deletions/
    chimeras with deliberately different kinetics - collapses into a single
    `clean_records.py` replicate group and gets entirely rejected by the
    intra-group variance filter, dragging the good wild-type rows down with
    the mutants. Parsing this out lets wild-type and each
    salvageable point mutant form their own correctly-separated groups.

    Parameters
    ----------
    enzyme_name : str
        The enzyme modifier species' raw (mixed-case) ``name`` attribute.

    Returns
    -------
    tuple[bool, str | None]
        ``(keep, mutation_code)``. ``keep=False`` means the record must be
        dropped entirely. ``keep=True`` with ``mutation_code=None`` means
        wild-type. ``keep=True`` with a code (e.g. ``"C75A"`` or ``"C75A/D80G"``)
        means one or more point mutations were identified.
    """
    lower = enzyme_name.lower()
    # "Delta"-prefixed residue codes denote deletions (e.g. "DeltaI16V17").
    if "delta" in lower or "Δ" in enzyme_name:
        return False, None
    if not any(kw in lower for kw in MUTATION_KEYWORDS):
        return True, None  # Wild-type.

    matches = find_point_mutations(enzyme_name)
    if len(matches) > 0:
        mutation_code = "/".join(f"{wt}{pos}{mut}" for wt, pos, mut in matches)
        return True, mutation_code

    return False, None


def _parse_sbml_response(
    sbml_text: str,
    ec_number: str,
    target_organism: Optional[str] = None
) -> list[KineticRecord]:
    """Parse an SBML/XML response from SABIO-RK into `KineticRecord` objects.

    Parameters
    ----------
    sbml_text : str
        The raw XML/SBML string.
    ec_number : str
        The EC number that was queried.
    target_organism : str, optional
        The organism string that was used in the query, to be used as a
        fallback if the SABIO-RK API omitted the organism tag.

    Returns
    -------
    list[KineticRecord]
        Parsed records.  Entries lacking both k_cat and K_m are dropped.
    """
    records: list[KineticRecord] = []
    # Per-call cache so multiple reactions sharing the same (EC, organism)
    # only trigger one UniProt REST fallback lookup each.
    _uniprot_fallback_cache: dict[tuple[str, str], Optional[str]] = {}

    try:
        root: ET.Element = ET.fromstring(sbml_text)
    except ET.ParseError as exc:
        logger.error("XML parse error for EC %s: %s", ec_number, exc)
        return records

    # Auto-detect SBML namespace (supports L2V4 and L3V1).
    sbml_ns_uri: str = _detect_sbml_namespace(root)
    ns: dict[str, str] = {"s": sbml_ns_uri} if sbml_ns_uri else {}

    # Locate the <model> element.
    model: Optional[ET.Element] = _ns_find(root, "s:model", ns)
    if model is None:
        logger.warning("No <model> found in SBML.")
        return records

    # SABIO-RK's local parameter `units` attribute is a sanitized SBML SId,
    # not the literal unit string (SIds can't contain "/", "^", "-", etc., so
    # e.g. "s^(-1)" becomes "swedgeone" and "mg/ml" becomes "mgdivml"). The
    # actual human-readable unit lives in the matching <unitDefinition>'s
    # `name` attribute - build that lookup once per document.
    unit_definitions: dict[str, str] = {}
    for udef in _ns_findall(model, ".//s:listOfUnitDefinitions/s:unitDefinition", ns):
        udef_id = udef.get("id")
        udef_name = udef.get("name")
        if udef_id:
            unit_definitions[udef_id] = udef_name if udef_name else udef_id

    # Build a lookup for species to UniProt IDs, and separately resolve each
    # species' chemical identity (ChEBI, falling back to KEGG compound) to a
    # SMILES via the disk cache - far more reliable than the later name-string
    # lookup in dataset_validator.py's get_smiles(), since it's sourced from
    # an authoritative chemical database ID rather than free-text matching.
    # The same cache doubles as a molar-mass source (see
    # `_resolve_species_molar_mass`) for converting mass-concentration Km
    # units (e.g. mg/mL) to mM.
    species_uniprot_map = {}
    species_name_map: dict[str, str] = {}
    # Original (mixed-case) names, needed for `_parse_sabio_mutation` since
    # the point-mutation regex is case-sensitive (e.g. "C75A").
    species_original_name_map: dict[str, str] = {}
    rdf_ns = {"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"}
    for species in _ns_findall(model, ".//s:listOfSpecies/s:species", ns):
        species_id = species.get("id")
        species_name = species.get("name")
        if species_id and species_name:
            species_name_map[species_id] = species_name.lower()
            species_original_name_map[species_id] = species_name
        chebi_id = None
        kegg_id = None
        for li in species.findall(".//rdf:li", rdf_ns):
            resource = li.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource")
            if not resource:
                continue
            if "identifiers.org/uniprot/" in resource and species_id not in species_uniprot_map:
                species_uniprot_map[species_id] = resource.split("/")[-1]
            elif "identifiers.org/chebi/CHEBI:" in resource and chebi_id is None:
                chebi_id = resource.rsplit("CHEBI:", 1)[-1]
            elif "identifiers.org/kegg.compound/" in resource and kegg_id is None:
                kegg_id = resource.rsplit("/", 1)[-1]
        if species_name and (chebi_id or kegg_id):
            _cache_species_smiles(species_name.lower(), chebi_id, kegg_id)

    # Iterate over <reaction> elements (each maps to one kinetic law).
    reactions: list[ET.Element] = _ns_findall(
        model, ".//s:listOfReactions/s:reaction", ns
    )

    for reaction in reactions:
        # ----- Annotation: kinetic-law ID, organism, conditions -----
        annotation: Optional[ET.Element] = _ns_find(
            reaction, "s:annotation", ns
        )
        kl_id, organism, temperature, ph = _extract_annotation_data(
            annotation
        )

        # kinetic-law ID fallback will happen later if still None
        # ----- Substrates -----
        reactant_refs: list[ET.Element] = _ns_findall(
            reaction, ".//s:listOfReactants/s:speciesReference", ns
        )
        # Extract all reactant names
        all_reactants = []
        for ref in reactant_refs:
            sp_id = ref.get("species")
            if sp_id:
                for sp in _ns_findall(model, ".//s:listOfSpecies/s:species", ns):
                    if sp.get("id") == sp_id:
                        sp_name = sp.get("name")
                        if sp_name:
                            all_reactants.append(sp_name.lower())
                        break

        # ----- UniProt ID & mutation status -----
        uniprot_id: Optional[str] = None
        enzyme_name: Optional[str] = None
        for modifier in _ns_findall(reaction, ".//s:listOfModifiers/s:modifierSpeciesReference", ns):
            species_ref = modifier.get("species")
            if species_ref and species_ref in species_uniprot_map:
                uniprot_id = species_uniprot_map[species_ref]
                enzyme_name = species_original_name_map.get(species_ref)
                break

        # Reject unsalvageable mutants (multi-mutant, deletion, chimera) and
        # relabel conservative single point mutants - see
        # `_parse_sabio_mutation` for why this matters.
        mutation_code: Optional[str] = None
        if enzyme_name:
            keep_mutation, mutation_code = _parse_sabio_mutation(enzyme_name)
            if not keep_mutation:
                continue

        # ----- Kinetic parameters & Law Type Filter -----
        kl_element: Optional[ET.Element] = _ns_find(
            reaction, "s:kineticLaw", ns
        )
        
        # Skip non-Michaelis-Menten rate laws and any containing inhibition constants (Ki, Kd)
        if kl_element is not None:
            kl_sbo = kl_element.get("sboTerm")
            # If the law has a specific SBO term, it must be 0000029 (Henri-MM) or 0000028 (MM)
            if kl_sbo and kl_sbo not in ("SBO:0000029", "SBO:0000028"):
                logger.debug("Skipping non-MM rate law: %s", kl_sbo)
                continue
                
            # Double check there are no inhibition parameters hidden inside
            # We permit SBO:0000282 for dissociation constants (Kd).
            inhibition_found = False
            for param in _ns_findall(kl_element, ".//s:listOfLocalParameters/s:localParameter", ns):
                if param.get("sboTerm") in ("SBO:0000262", "SBO:0000558"):
                    inhibition_found = True
                    break
            if inhibition_found:
                logger.debug("Skipping rate law with inhibition parameter")
                continue

        kcat: Optional[float] = None
        km: Optional[float] = None
        measured_substrate: Optional[str] = None

        if kl_element is not None:
            kcat, kcat_unit, _ = _extract_local_parameter_value(kl_element, SBO_KCAT, unit_definitions)
            km, km_unit, assoc_species = _extract_local_parameter_value(kl_element, SBO_KM, unit_definitions)
            
            # Handle kcat units (ensure s^-1)
            if kcat is not None:
                if kcat_unit:
                    kcat_u = kcat_unit.lower().replace(" ", "").replace("*", "")
                    if kcat_u in ("s^(-1)", "s-1", "1/s", "s^-1", "per_second", "1/sec", "sec^-1", "sec-1"):
                        pass
                    elif kcat_u in ("min^(-1)", "min-1", "1/min", "min^-1", "per_minute"):
                        kcat /= 60.0
                    elif kcat_u in ("h^(-1)", "h-1", "1/h", "h^-1", "hr^-1", "per_hour"):
                        kcat /= 3600.0
                    elif kcat_u in ("ms^(-1)", "ms-1", "1/ms", "ms^-1"):
                        kcat *= 1000.0
                    else:
                        logger.warning("Unrecognized kcat unit '%s'; dropping kcat rather than assuming s^-1", kcat_unit)
                        kcat = None
                else:
                    logger.warning("Dropping kcat for EC %s: no unit info to validate against", ec_number)
                    kcat = None

            # Handle Km units (ensure mM)
            if km is not None:
                if km_unit:
                    km_u = km_unit.lower().replace(" ", "").replace("*", "")
                    
                    # Molar concentrations
                    if km_u in ("m", "molar", "mol/l", "moll^(-1)", "mol/dm^3", "moldm^-3", "mol/dm3"):
                        km *= 1000.0
                    elif km_u in ("mm", "mmol/l", "mmoll^(-1)", "millimolar", "mmol/dm^3", "mmoldm^-3"):
                        pass
                    elif km_u in ("microm", "um", "µm", "micro m", "micromolar", "umol/l", "µmol/l", "umoll^(-1)", "µmoll^(-1)", "umol/dm^3", "µmol/dm^3"):
                        km /= 1000.0
                    elif km_u in ("nanom", "nm", "nano m", "nanomolar", "nmol/l", "nmoll^(-1)", "nmol/dm^3"):
                        km /= 1e6
                    elif km_u in ("picom", "pm", "pico m", "picomolar", "pmol/l", "pmoll^(-1)", "pmol/dm^3"):
                        km /= 1e9

                    # Molality / mass-based concentrations (assuming 1 g/mL density)
                    elif km_u in ("mol/g", "molg^(-1)"):
                        km *= 1e6
                    elif km_u in ("mmol/g", "mmolg^(-1)", "mol/kg", "molkg^(-1)"):
                        km *= 1000.0
                    elif km_u in ("umol/g", "µmol/g", "umolg^(-1)", "µmolg^(-1)", "mmol/kg", "mmolkg^(-1)"):
                        pass
                    elif km_u in ("nmol/g", "nmolg^(-1)", "umol/kg", "µmol/kg", "umolkg^(-1)", "µmolkg^(-1)"):
                        km /= 1000.0

                    # Gas Pressures
                    elif km_u in ("atm", "pa", "kpa", "mpa", "bar", "mbar"):
                        # Use ideal gas law: C(mM) = P(atm) * 1000 / (R * T(K))
                        t_k = (temperature + 273.15) if temperature is not None else 298.15
                        R = 0.08206 # L atm / (K mol)
                        
                        p_atm = km
                        if km_u == "pa":
                            p_atm = km / 101325.0
                        elif km_u == "kpa":
                            p_atm = km * 1000.0 / 101325.0
                        elif km_u == "mpa":
                            p_atm = km * 1e6 / 101325.0
                        elif km_u == "bar":
                            p_atm = km * 0.986923
                        elif km_u == "mbar":
                            p_atm = (km / 1000.0) * 0.986923
                            
                        km = p_atm * 1000.0 / (R * t_k)
                        logger.debug("Converted gas pressure to %s mM", km)

                    # Mass concentrations
                    elif km_u in ("mg/ml", "mg/l", "g/l", "ug/ml", "µg/ml", "microg/ml", "g/ml", "ng/ml"):
                        # Mass-concentration units require the substrate's molar
                        # mass to convert to mM; recover it from the SMILES
                        # already cached for this species (via ChEBI/KEGG) rather
                        # than assuming a value or dropping outright.
                        molar_mass = _resolve_species_molar_mass(
                            species_name_map.get(assoc_species) if assoc_species else None
                        )
                        if molar_mass:
                            # Normalise to g/L first (1 mg/mL == 1 g/L numerically).
                            if km_u in ("mg/l", "ug/ml", "µg/ml", "microg/ml"):
                                g_per_l = km / 1000.0
                            elif km_u == "g/ml":
                                g_per_l = km * 1000.0
                            elif km_u == "ng/ml":
                                g_per_l = km / 1e6
                            else: # mg/ml, g/l
                                g_per_l = km
                            km = (g_per_l / molar_mass) * 1000.0
                            logger.debug(
                                "Converted mass-concentration Km to %.4g mM using molar mass %.2f g/mol",
                                km, molar_mass,
                            )
                        else:
                            logger.warning("Dropping Km due to unsupported units (mass concentration without molar mass)")
                            km = None
                    else:
                        logger.warning("Dropping Km due to unsupported units: %s", km_unit)
                        km = None
                else:
                    logger.warning("Dropping km for EC %s: no unit info to validate against", ec_number)
                    km = None

            # Resolve measured_substrate from assoc_species ID
            if assoc_species:
                for sp in _ns_findall(model, ".//s:listOfSpecies/s:species", ns):
                    if sp.get("id") == assoc_species:
                        name = sp.get("name")
                        if name:
                            measured_substrate = name.lower()
                        break
            
            if kcat is None:
                vmax, vmax_unit, _ = _extract_local_parameter_value(kl_element, SBO_VMAX, unit_definitions)
                e0, e0_unit, _ = _extract_local_parameter_value(kl_element, SBO_E0_1, unit_definitions)
                if e0 is None:
                    e0, e0_unit, _ = _extract_local_parameter_value(kl_element, SBO_E0_2, unit_definitions)
                
                if vmax is not None and vmax_unit:
                    vu = vmax_unit.lower().replace(" ", "").replace("*", "")
                    
                    # Specific activity handling (convert Vmax directly to kcat if it is given per mass of enzyme)
                    multiplier_to_mol_per_s_per_g = None
                    if vu in ("mol/(s.g)", "mols^(-1)g^(-1)", "mol/s/g", "mols-1g-1", "mol/(sg)", "mol/sg"):
                        multiplier_to_mol_per_s_per_g = 1.0
                    elif vu in ("mol/(min.g)", "molmin^(-1)g^(-1)", "mol/min/g", "mol/(ming)"):
                        multiplier_to_mol_per_s_per_g = 1.0 / 60.0
                    elif vu in ("mmol/(s.g)", "mmols^(-1)g^(-1)", "mmol/s/g", "mmol/(sg)"):
                        multiplier_to_mol_per_s_per_g = 1e-3
                    elif vu in ("mmol/(min.g)", "mmolmin^(-1)g^(-1)", "mmol/min/g", "mmol/(ming)"):
                        multiplier_to_mol_per_s_per_g = 1e-3 / 60.0
                    elif vu in ("µmol/(min.mg)", "umol/(min.mg)", "umol/min/mg", "µmol/min/mg", "umolmin^(-1)mg^(-1)", "µmolmin^(-1)mg^(-1)", "u/mg", "units/mg", "umol/(minmg)", "µmol/(minmg)"):
                        multiplier_to_mol_per_s_per_g = 1e-6 / (60.0 * 1e-3)
                    elif vu in ("mmol/(min.mg)", "mmol/min/mg", "mmolmin^(-1)mg^(-1)", "mmol/(minmg)"):
                        multiplier_to_mol_per_s_per_g = 1e-3 / (60.0 * 1e-3)
                    elif vu in ("µmol/(s.mg)", "umol/(s.mg)", "umol/s/mg", "µmol/s/mg", "umols^(-1)mg^(-1)", "µmols^(-1)mg^(-1)", "umol/(smg)", "µmol/(smg)"):
                        multiplier_to_mol_per_s_per_g = 1e-6 / 1e-3
                    elif vu in ("mol/(s.mg)", "mols^(-1)mg^(-1)", "mol/s/mg", "mol/(smg)"):
                        multiplier_to_mol_per_s_per_g = 1.0 / 1e-3
                    elif vu in ("u/g", "units/g", "µmol/(min.g)", "umol/(min.g)", "umol/min/g", "µmol/min/g", "umol/(ming)", "µmol/(ming)"):
                        multiplier_to_mol_per_s_per_g = 1e-6 / 60.0

                    if multiplier_to_mol_per_s_per_g is not None and uniprot_id:
                        # Extract enzyme sequence to estimate MW
                        seq = fetch_uniprot_sequence(uniprot_id)
                        if seq:
                            mw_enzyme = len(seq) * 110.0 # Standard rough approximation of protein MW (g/mol)
                            # kcat (1/s) = Vmax (mol/s/g) * MW (g/mol)
                            kcat = round(vmax * multiplier_to_mol_per_s_per_g * mw_enzyme, 4)
                            logger.debug("Computed kcat=%s from specific activity Vmax=%s and enzyme MW ~%.2f", kcat, vmax, mw_enzyme)

                    # Fallback to standard Vmax / E0 calculation
                    if kcat is None and e0 is not None and e0_unit:
                        eu = e0_unit.lower().replace(" ", "").replace("*", "")
                        # Simple checks: does Vmax unit contain E0 unit?
                        if vu.startswith(eu) or (eu in vu and ("s^(-1)" in vu or "s-1" in vu or "min^(-1)" in vu or "min-1" in vu)):
                            try:
                                kcat_raw = vmax / e0
                                if "min^(-1)" in vu or "min-1" in vu or "/min" in vu:
                                    kcat_raw /= 60.0
                                elif "h^(-1)" in vu or "h-1" in vu or "/h" in vu:
                                    kcat_raw /= 3600.0
                                kcat = round(kcat_raw, 4)
                                logger.debug("Computed kcat=%s from Vmax=%s and [E]0=%s", kcat, vmax, e0)
                            except ZeroDivisionError:
                                pass

            # Format B: kinetic law ID may be in the kineticLaw's own
            # annotation rather than the reaction's annotation.
            if kl_id is None:
                kl_annotation: Optional[ET.Element] = _ns_find(
                    kl_element, "s:annotation", ns
                )
                kl_id_b, org_b, temp_b, ph_b = _extract_annotation_data(
                    kl_annotation
                )
                if kl_id_b is not None:
                    kl_id = kl_id_b
                if organism is None and org_b is not None:
                    organism = org_b
                if temperature is None and temp_b is not None:
                    temperature = temp_b
                if ph is None and ph_b is not None:
                    ph = ph_b

            # Fallback: extract ID from metaid (e.g. "META_KL_56568").
            if kl_id is None:
                metaid: Optional[str] = kl_element.get("metaid", "")
                if metaid and metaid.startswith("META_KL_"):
                    try:
                        kl_id = int(metaid.replace("META_KL_", ""))
                    except ValueError:
                        pass

        # Fallback: extract ID from the reaction's ``id`` attribute.
        if kl_id is None:
            reaction_id: Optional[str] = reaction.get("id")
            if reaction_id is not None:
                digits: str = "".join(filter(str.isdigit, reaction_id))
                if digits:
                    kl_id = int(digits)

        # Skip entries without an entry ID.
        if kl_id is None:
            continue

        org_final = organism or target_organism

        # Strict null constraint: Drop any record missing a key kinetic parameter
        if (kcat is None or km is None or temperature is None or
            ph is None or not measured_substrate):
            continue

        if measured_substrate not in all_reactants:
            # If the km is reported for something that isn't a reactant, drop to prevent target ambiguity
            continue

        co_subs_list = [r for r in all_reactants if r != measured_substrate]

        canon = canonicalize_and_filter_ligands(measured_substrate, co_subs_list)
        if not canon:
            logger.debug("Dropped SABIO record due to blacklisted substrate/cofactor: %s, %s", measured_substrate, co_subs_list)
            continue

        std_sub, std_cos = canon
        measured_substrate = std_sub
        co_substrates = "; ".join(std_cos)

        # If the species RDF didn't carry a UniProt cross-reference but an
        # organism name was recovered, fall back to the same EC+organism
        # UniProt search brenda_parser.py uses - otherwise this record is
        # guaranteed to be dropped later by clean_records.py's
        # dropna(subset=["uniprot_id"]) with no chance to recover it.
        if uniprot_id is None and org_final:
            cache_key = (ec_number, org_final)
            if cache_key not in _uniprot_fallback_cache:
                _uniprot_fallback_cache[cache_key] = fetch_uniprot_id(ec_number, org_final)
                time.sleep(0.1)  # Be polite to the UniProt REST API.
            uniprot_id = _uniprot_fallback_cache[cache_key]

        # A record with no resolvable UniProt ID can never survive
        # clean_records.py's aggregation step, so there's no point keeping it.
        if not uniprot_id:
            continue

        records.append(
            KineticRecord(
                entry_id=kl_id,
                source_db="SABIO-RK",
                ec_number=ec_number,
                uniprot_id=uniprot_id,
                measured_substrate=measured_substrate,
                co_substrates=co_substrates,
                kcat=kcat,
                km=km,
                temperature=temperature,
                ph=ph,
                mutation=mutation_code,
            )
        )

    logger.info(
        "Parsed %d valid kinetic records for EC %s", len(records), ec_number
    )
    return records


# ═══════════════════════════════════════════════════════════════════════════
#  Database operations
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_tables(conn: sqlite3.Connection) -> None:
    """Create the ``raw_parameters`` and table if absent.

    This is a safety net in case the tables were not pre-initialised via
    MCP or an external migration tool.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open database connection.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_parameters (
            entry_id     INTEGER PRIMARY KEY,
            source_db    TEXT,
            ec_number    TEXT NOT NULL,
            uniprot_id   TEXT,
            measured_substrate TEXT NOT NULL,
            co_substrates TEXT,
            kcat         REAL,
            km           REAL,
            temperature  REAL,
            ph           REAL,
            mutation     TEXT
        )
        """
    )
    # CREATE TABLE IF NOT EXISTS does not add columns to a database file built
    # under an older schema, so the ALTER TABLE below guarantees the mutation
    # column exists regardless of the file's schema version.
    cursor = conn.cursor()
    existing_cols = {row[1] for row in cursor.execute("PRAGMA table_info(raw_parameters)").fetchall()}
    if "mutation" not in existing_cols:
        conn.execute("ALTER TABLE raw_parameters ADD COLUMN mutation TEXT")

    conn.commit()


def _insert_records(
    records: list[KineticRecord],
    db_path: pathlib.Path = DEFAULT_DB_PATH,
) -> int:
    """Insert parsed kinetic records into the local SQLite database.

    Uses ``INSERT OR REPLACE`` to allow idempotent re-runs without
    duplicate-key errors.

    Parameters
    ----------
    records : list[KineticRecord]
        Validated records to persist.
    db_path : pathlib.Path
        Path to the SQLite database file.

    Returns
    -------
    int
        Number of rows successfully inserted (or replaced).
    """
    if not records:
        logger.warning("No records to insert.")
        return 0

    conn: sqlite3.Connection = sqlite3.connect(str(db_path))
    try:
        _ensure_tables(conn)

        cursor: sqlite3.Cursor = conn.cursor()
        inserted: int = 0
        try:
            cursor.executemany(
                """
                INSERT OR REPLACE INTO raw_parameters 
                (entry_id, source_db, ec_number, uniprot_id, measured_substrate, co_substrates, kcat, km, temperature, ph, mutation)
            VALUES 
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.entry_id,
                        r.source_db,
                        r.ec_number,
                        r.uniprot_id,
                        r.measured_substrate,
                        r.co_substrates,
                        r.kcat,
                        r.km,

                        r.temperature,
                        r.ph,
                        r.mutation,
                    )
                    for r in records
                ],
            )
            inserted = cursor.rowcount
        except sqlite3.Error as exc:
            logger.error("Failed to batch insert records: %s", exc)

        conn.commit()
        logger.info(
            "Inserted %d / %d records into %s",
            inserted,
            len(records),
            db_path.name,
        )
        return inserted
    finally:
        conn.close()


def _delete_source_rows(source_db: str, db_path: pathlib.Path = DEFAULT_DB_PATH) -> int:
    """Delete all ``raw_parameters`` rows for a given ``source_db``.

    Used to give a fresh ``--full`` run (one not resuming via
    ``--continue``) a clean slate for its own source, without touching
    rows inserted by the other parser.

    Parameters
    ----------
    source_db : str
        The ``source_db`` value to delete (e.g. ``"SABIO-RK"``).
    db_path : pathlib.Path, default: DEFAULT_DB_PATH

    Returns
    -------
    int
        Number of rows deleted (0 if the table doesn't exist yet).
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("DELETE FROM raw_parameters WHERE source_db = ?", (source_db,))
        conn.commit()
        return cursor.rowcount
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def _read_checkpoint(path: pathlib.Path) -> Optional[str]:
    """Read the last successfully-processed EC number from a checkpoint file.

    Parameters
    ----------
    path : pathlib.Path

    Returns
    -------
    str or None
        The checkpointed EC number, or None if no checkpoint exists.
    """
    if not path.exists():
        return None
    ec = path.read_text(encoding="utf-8").strip()
    return ec or None


def _write_checkpoint(path: pathlib.Path, ec_number: str) -> None:
    """Atomically persist the last successfully-processed EC number.

    Parameters
    ----------
    path : pathlib.Path
    ec_number : str
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".tmp-{uuid.uuid4().hex}.txt"
    tmp_path.write_text(ec_number, encoding="utf-8")
    os.replace(tmp_path, path)


def _clear_checkpoint(path: pathlib.Path) -> None:
    """Remove a checkpoint file, if present - signals a clean completion.

    Parameters
    ----------
    path : pathlib.Path
    """
    path.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Orchestration
# ═══════════════════════════════════════════════════════════════════════════

def _run_batch(
    queries: list[dict[str, str]] | None = None,
    db_path: pathlib.Path = DEFAULT_DB_PATH,
    checkpoint_path: Optional[pathlib.Path] = None,
) -> int:
    """Fetch, parse, and store kinetic data for a batch of queries.

    Parameters
    ----------
    queries : list[dict[str, str]] or None
        Each dict must contain ``"ec"`` and optionally ``"organism"``
        and ``"label"``.  Defaults to ``DEFAULT_QUERIES``.
    db_path : pathlib.Path
        Path to the local SQLite database.
    checkpoint_path : pathlib.Path, optional
        If given, the EC number is written here (via `_write_checkpoint`)
        after each query finishes processing (whether or not it returned
        data) - lets a `--full` run resume after an interruption instead
        of restarting from scratch. Not cleared here; the caller clears it
        on full completion.

    Returns
    -------
    int
        Total number of rows inserted across all queries.
    """
    if queries is None:
        queries = DEFAULT_QUERIES

    total_inserted: int = 0

    for idx, entry in enumerate(queries):
        ec: str = entry["ec"]
        organism: Optional[str] = entry.get("organism")
        label: str = entry.get("label", ec)

        logger.info(
            "[%d/%d] %s (EC %s)", idx + 1, len(queries), label, ec
        )

        sbml_text: Optional[str] = _query_sabio_rk(ec, organism)
        if sbml_text is not None:
            records: list[KineticRecord] = _parse_sbml_response(sbml_text, ec, organism)
            inserted: int = _insert_records(records, db_path)
            total_inserted += inserted

        if checkpoint_path is not None:
            _write_checkpoint(checkpoint_path, ec)

        # Polite delay between requests to avoid overloading the server.
        if idx < len(queries) - 1 and sbml_text is not None:
            time.sleep(_REQUEST_DELAY_S)

    return total_inserted


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("ThermoKP — SABIO-RK Kinetic Data Ingestion")
    logger.info("=" * 60)
    
    queries = DEFAULT_QUERIES
    checkpoint_path: Optional[pathlib.Path] = None
    if "--full" in sys.argv:
        if not TARGETS_FILE.exists():
            logger.error("Master index not found. Run fetch_ec_numbers.py first.")
            sys.exit(1)
        with open(TARGETS_FILE, "r") as f:
            ec_list = json.load(f)
        queries = [{"ec": ec, "label": ec} for ec in ec_list]
        logger.info("Loaded %d EC targets from %s", len(queries), TARGETS_FILE.name)

        checkpoint_path = SABIO_CHECKPOINT_FILE
        if "--continue" in sys.argv:
            last_ec = _read_checkpoint(checkpoint_path)
            if last_ec is None:
                logger.info("No checkpoint found; starting from the beginning.")
            else:
                resume_idx = next(
                    (i for i, q in enumerate(queries) if q["ec"] == last_ec), None
                )
                if resume_idx is None:
                    logger.warning(
                        "Checkpointed EC %s not found in target list; starting from the beginning.",
                        last_ec,
                    )
                else:
                    queries = queries[resume_idx + 1:]
                    logger.info(
                        "Resuming after EC %s (%d EC targets remaining).",
                        last_ec, len(queries),
                    )
        else:
            deleted = _delete_source_rows("SABIO-RK", DEFAULT_DB_PATH)
            logger.info("Fresh run: deleted %d existing SABIO-RK rows from raw_parameters.", deleted)
            _clear_checkpoint(checkpoint_path)

    total: int = _run_batch(queries=queries, checkpoint_path=checkpoint_path)

    if checkpoint_path is not None:
        _clear_checkpoint(checkpoint_path)

    logger.info("=" * 74)
    logger.info("==                        SABIO-RK Parser Summary                       ==")
    logger.info("=" * 74)
    logger.info(f"Total rows inserted       : {total}")
    logger.info(f"Database                  : {DEFAULT_DB_PATH}")
    logger.info("=" * 74)
