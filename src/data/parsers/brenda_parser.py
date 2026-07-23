"""
===========================================================================
BRENDA Parser
Description: BRENDA Database Ingestion for ThermoKP
===========================================================================

Workflow:
1. Load `data/raw/brenda.json`.
2. Extract the `"protein"` dictionary to map PR IDs to organism names and UniProt IDs.
3. Parse `km_value` and `turnover_number`, extracting substrate, temperature and pH
   from the unstructured `comment` string, falling back to the enzyme's curated
   `ph_optimum`/`temperature_optimum` fields (per protein) when the entry's own
   commentary doesn't state them explicitly.
4. Merge Km and kcat entries when their PR ID, substrate, resolved (temperature, pH),
   and mutation status all align - matching resolved numeric condition rather than
   the raw commentary string, since two entries describing the same real assay are
   often phrased slightly differently.
5. Classify each entry as wild-type, an unsalvageable modified construct (fusion,
   truncation, chemical modification, etc. - dropped), or a point mutation
   (kept, tagged with its mutation code) via `_parse_mutation`.
6. Insert each valid record into the `raw_parameters` table via Python's built-in `sqlite3` module.

Known Caveats:
- BRENDA records are highly fragmented. We store them as separate records unless
  they match on organism + substrate + resolved condition + mutation status.
- The `ph_optimum`/`temperature_optimum` fallback is an approximation: it reports
  the enzyme's curated optimum condition, not necessarily the exact condition used
  in that specific assay, when the assay's own commentary is silent on it.
- Mutation handling recovers point mutations parsed directly from the commentary string
  or curated protein_variants list. Unsalvageable constructs are dropped.

`--full` run resumption:
- `python -m src.data.parsers.brenda_parser --full` alone always starts clean - it
  first deletes any existing `BRENDA` rows from `raw_parameters` and clears
  any stale checkpoint, then processes every EC number.
- After each EC finishes, its number is written to `BRENDA_CHECKPOINT_FILE`.
  On a full, uninterrupted completion, that file is removed.
- If the run is interrupted, the checkpoint file is left behind. Re-run
  with `--full --continue` to resume immediately after the last completed
  EC instead of restarting from scratch - existing rows are left untouched.
  Mirrors sabio_rk_parser.py's identical mechanism.

Author: ThermoKP Team
License: MIT
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sqlite3
import re
import sys
import time
import hashlib
import yaml
import uuid
import zlib
from typing import Any, Optional

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

_PROJECT_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH: pathlib.Path = _PROJECT_ROOT / "data" / "thermokp_database.db"
BRENDA_JSON_PATH: pathlib.Path = _PROJECT_ROOT / "data" / "raw" / "brenda.json"
TARGETS_FILE: pathlib.Path = _PROJECT_ROOT / "data" / "enzyme_targets.json"


# Records the last EC number a `--full` run finished processing, so an
# interrupted run can resume with `--continue` instead of starting over.
# Removed on a successful full completion - see sabio_rk_parser.py's
# identical mechanism (SABIO_CHECKPOINT_FILE) for the SABIO-RK side.
BRENDA_CHECKPOINT_FILE: pathlib.Path = _PROJECT_ROOT / "data" / "raw" / "brenda_checkpoint.txt"

BRENDA_ID_OFFSET: int = 10000000

# Keywords indicating a modified enzyme construct that cannot be salvaged by
# a simple point-mutation relabel (fusion proteins, truncations, chemical
# modification, immobilization, etc.) and must always be dropped.
HARD_REJECT_KEYWORDS: list[str] = [
    "cleaved", "recombinant", "modified", "deletion",
    "chimera", "synthetic", "artificial", "immobilized", "bound",
    "fusion", "truncated", "tagged", "his-tag",
    "presence", "enzyme in"
]
# Keywords that flag a record as a point mutant rather than wild-type. Records
# matching only these (no HARD_REJECT_KEYWORDS) are kept if and only if a
# single conservative point-mutation code can be parsed from the commentary
# (see `_parse_conservative_mutation`); otherwise they are dropped too.
# MODIFIED_KEYWORDS combines HARD_REJECT_KEYWORDS with the shared
# MUTATION_KEYWORDS from parser_common.py (also used by sabio_rk_parser.py's
# _parse_sabio_mutation) for callers that need the full combined list.
MODIFIED_KEYWORDS: list[str] = HARD_REJECT_KEYWORDS + MUTATION_KEYWORDS


def _parse_mutation(commentary: str, pr_id_variants: Optional[str] = None) -> tuple[bool, Optional[str]]:
    """Classify a BRENDA commentary as wild-type, a point mutant, or unsalvageable.

    Parameters
    ----------
    commentary : str
        The unstructured commentary string from BRENDA.
    pr_id_variants : str, optional
        The curated protein_variant string for this protein ID, if any.

    Returns
    -------
    tuple[bool, str | None]
        ``(keep, mutation_code)``. ``keep=False`` means the record must be
        dropped entirely. ``keep=True`` with ``mutation_code=None`` means
        wild-type. ``keep=True`` with a code (e.g. ``"W95L"`` or ``"W95L/V100A"``)
        means one or more point mutations were identified.
    """
    lower = commentary.lower()
    if any(kw in lower for kw in HARD_REJECT_KEYWORDS):
        return False, None
    if not any(kw in lower for kw in MUTATION_KEYWORDS):
        return True, None  # Wild-type.

    matches = find_point_mutations(commentary)
    if len(matches) > 0:
        mutation_code = "/".join(f"{wt}{pos}{mut}" for wt, pos, mut in matches)
        return True, mutation_code

    # Fallback to curated variants if regex found nothing
    if pr_id_variants:
        matches = find_point_mutations(pr_id_variants)
        if len(matches) > 0:
            mutation_code = "/".join(f"{wt}{pos}{mut}" for wt, pos, mut in matches)
            return True, mutation_code

    return False, None

# A unified regex filter (see ligand_cleaner.py) handles substrate/cofactor exclusion.
# Default set of well-characterised enzymes for the test batch.
DEFAULT_QUERIES: list[dict[str, str]] = [
    {"ec": "4.2.1.11", "label": "enolase"},
    {"ec": "2.7.1.40", "label": "pyruvate kinase"},
    {"ec": "1.1.1.27", "label": "lactate dehydrogenase"},
]

# ═══════════════════════════════════════════════════════════════════════════
#  Data Parsing & Extraction
# ═══════════════════════════════════════════════════════════════════════════

def _extract_conditions(commentary: str) -> tuple[Optional[float], Optional[float]]:
    """Extract temperature and pH from BRENDA commentary text.

    Parameters
    ----------
    commentary : str
        The unstructured commentary string from BRENDA.

    Returns
    -------
    tuple[float | None, float | None]
        A tuple of (temperature, ph) extracted from the commentary, each
        None if absent or outside a physically plausible range (rejecting a
        malformed/incidental regex hit is preferred over accepting a
        nonsensical value).
    """
    if not commentary:
        return None, None

    ph = None
    temp = None

    ph_match = re.search(r'pH[\s\-]*=?\s*([0-9]+\.?[0-9]*)', commentary, re.IGNORECASE)
    if ph_match:
        try:
            candidate = float(ph_match.group(1))
            if 0.0 <= candidate <= 14.0:
                ph = candidate
        except ValueError:
            pass

    # Celsius: a number directly (only whitespace between) followed by an
    # optional degree marker and a literal "C".
    temp_c_match = re.search(
        r'([0-9]+\.?[0-9]*)\s*(?:°|\?|deg\.?|degrees?)?\s*C\b', commentary, re.IGNORECASE
    )
    if temp_c_match:
        try:
            candidate = float(temp_c_match.group(1))
            if -30.0 <= candidate <= 150.0:
                temp = candidate
        except ValueError:
            pass

    # Kelvin fallback (e.g. "298 K"), tried only when no Celsius value was
    # found - BRENDA occasionally reports assay temperature in Kelvin
    # instead. Only tried second (rather than always) to avoid misreading an
    # incidental "... K" elsewhere in the same comment (e.g. "10 mM KCl" has
    # no bare number-then-"K", but stay conservative regardless).
    if temp is None:
        temp_k_match = re.search(r'([0-9]+\.?[0-9]*)\s*K\b', commentary)
        if temp_k_match:
            try:
                candidate = float(temp_k_match.group(1))
                if 220.0 <= candidate <= 420.0:
                    temp = round(candidate - 273.15, 2)
            except ValueError:
                pass
            
    return temp, ph

def _parse_value_string(val_str: str) -> tuple[Optional[float], str]:
    """Parse a value string into a numeric value and substrate name.
    
    Parameters
    ----------
    val_str : str
        A string formatted like '0.029 {2-phospho-D-glycerate}'.
        
    Returns
    -------
    tuple[float | None, str]
        A tuple containing the numeric value and the parsed substrate string.
    """
    match = re.search(r'([0-9.]+)\s*\{([^}]+)\}', val_str)
    if match:
        try:
            return float(match.group(1)), match.group(2).lower().strip()
        except ValueError:
            pass
    
    # Check if there is only a number
    try:
        if '{' not in val_str:
            return float(val_str.strip()), ""
    except ValueError:
        pass
        
    return None, ""

def _parse_reaction_equation(reaction_str: str) -> list[str]:
    """
    Parse a reaction string 'A + B = C + D' into a list of reactants.

    Parameters
    ----------
    reaction_str : str
        The reaction equation string.

    Returns
    -------
    list[str]
        A list of reactant names.
    """
    parts = reaction_str.split(' = ')
    if not parts:
        return []
    reactants_part = parts[0]
    return [r.strip().lower() for r in reactants_part.split(' + ') if r.strip()]

def _parse_brenda_data(ec_number: str, data: dict[str, Any]) -> list[KineticRecord]:
    """Parse Km and kcat data for a specific enzyme from the JSON payload.
    
    Parameters
    ----------
    ec_number : str
        The Enzyme Commission number (e.g., "4.2.1.11").
    data : dict[str, Any]
        The parsed BRENDA JSON 'data' dictionary.
        
    Returns
    -------
    list[KineticRecord]
        A list of parsed and validated kinetic records.
    """
    logger.info("Querying BRENDA JSON | ECNumber:%s", ec_number)
    
    ec_data = data.get(ec_number)
    if not ec_data:
        logger.warning("EC %s not found in BRENDA JSON.", ec_number)
        return []

    # Map PR IDs to (organism, uniprot_id)
    pr_map: dict[str, tuple[str, Optional[str]]] = {}
    for pr_id, p_info in ec_data.get("protein", {}).items():
        organism = p_info.get("organism", "")
        accessions = p_info.get("accessions", [])
        uniprot_id = accessions[0] if accessions else None
        pr_map[str(pr_id)] = (organism, uniprot_id)

    # Per-protein curated optimum conditions, used as a fallback when a given
    # km_value/turnover_number entry's own commentary doesn't state pH/temp
    # explicitly. First reported value per pr_id is used when BRENDA lists
    # more than one (from different references).
    #
    # BRENDA uses "-999" as a "not a single reportable value" sentinel in
    # these fields (e.g. "3 charge variant forms with pIs of 5.6, 5.7, 5.9",
    # or "assay carried out at room temperature") - bounds-check exactly like
    # _extract_conditions() does for commentary-derived values, or this
    # sentinel silently becomes a literal -999 pH/temperature downstream.
    ph_opt_map: dict[str, float] = {}
    for item in ec_data.get("ph_optimum", []):
        try:
            val = float(item.get("value", ""))
        except (TypeError, ValueError):
            continue
        if not (0.0 <= val <= 14.0):
            continue
        for pr_id in item.get("proteins", []):
            ph_opt_map.setdefault(str(pr_id), val)

    temp_opt_map: dict[str, float] = {}
    for item in ec_data.get("temperature_optimum", []):
        try:
            val = float(item.get("value", ""))
        except (TypeError, ValueError):
            continue
        if not (-30.0 <= val <= 150.0):
            continue
        for pr_id in item.get("proteins", []):
            temp_opt_map.setdefault(str(pr_id), val)

    variants_map: dict[str, str] = {}
    for item in ec_data.get("protein_variants", []):
        val = item.get("value", "").strip()
        if val:
            for pr_id in item.get("proteins", []):
                variants_map[str(pr_id)] = val

    kcat_km_map: dict[str, list[dict[str, Any]]] = {}
    for item in ec_data.get("kcat_km_value", []):
        val_str = item.get("value", "")
        kkm_val, substrate = _parse_value_string(val_str)
        if kkm_val is None or kkm_val <= 0 or not substrate:
            continue
        temp, ph = _extract_conditions(item.get("comment", ""))
        for pr_id in item.get("proteins", []):
            pr_id = str(pr_id)
            kcat_km_map.setdefault(pr_id, []).append({
                "substrate": substrate,
                "value": kkm_val,
                "temperature": temp,
                "ph": ph,
                "commentary": item.get("comment", "")
            })

    records: list[KineticRecord] = []
    temp_records: list[dict[str, Any]] = []
    # zlib.crc32 (unlike the builtin hash()) is stable across processes, so
    # reruns mint the same entry_id for the same logical record instead of
    # accumulating stale duplicates via INSERT OR REPLACE.
    record_id_counter = BRENDA_ID_OFFSET + zlib.crc32(ec_number.encode("utf-8")) % 1000000

    # Extract all reaction equations for this EC
    reactions_list = ec_data.get("reaction", [])
    possible_reactions = [_parse_reaction_equation(r.get("value", "")) for r in reactions_list if "value" in r]

    # 1. Process Km records
    for item in ec_data.get("km_value", []):
        commentary = item.get("comment", "")

        val_str = item.get("value", "")
        km_val, substrate = _parse_value_string(val_str)
        if km_val is None or km_val <= 0 or not substrate:
            continue

        raw_temp, raw_ph = _extract_conditions(commentary)

        for pr_id in item.get("proteins", []):
            pr_id = str(pr_id)
            org, uniprot = pr_map.get(pr_id, ("", None))
            if not org:
                continue

            keep, mutation = _parse_mutation(commentary, pr_id_variants=variants_map.get(pr_id))
            if not keep:
                continue

            temp = raw_temp if raw_temp is not None else temp_opt_map.get(pr_id)
            ph = raw_ph if raw_ph is not None else ph_opt_map.get(pr_id)
            if temp is None or ph is None:
                continue

            record_id_counter += 1
            temp_records.append({
                "entry_id": record_id_counter,
                "ec_number": ec_number,
                "pr_id": pr_id,
                "organism": org,
                "uniprot_id": uniprot,
                "measured_substrate": substrate,
                "kcat": None,
                "vmax": None, # Will store Vmax temporarily if we found it instead of kcat (though this is for Km)
                "km": km_val,

                "temperature": temp,
                "ph": ph,
                "commentary": commentary,
                "mutation": mutation,
            })

    # 2. Process kcat records
    for item in ec_data.get("turnover_number", []):
        commentary = item.get("comment", "")

        val_str = item.get("value", "")
        kcat_val, substrate = _parse_value_string(val_str)
        if kcat_val is None or kcat_val <= 0 or not substrate:
            continue

        raw_temp, raw_ph = _extract_conditions(commentary)

        for pr_id in item.get("proteins", []):
            pr_id = str(pr_id)
            org, uniprot = pr_map.get(pr_id, ("", None))
            if not org:
                continue

            keep, mutation = _parse_mutation(commentary, pr_id_variants=variants_map.get(pr_id))
            if not keep:
                continue

            temp = raw_temp if raw_temp is not None else temp_opt_map.get(pr_id)
            ph = raw_ph if raw_ph is not None else ph_opt_map.get(pr_id)
            if temp is None or ph is None:
                continue

            # Pairing requires the same protein, substrate, physical condition
            # (temperature + pH), and mutation status (wild-type only pairs
            # with wild-type; a given mutation code only pairs with itself) -
            # matching the resolved numeric condition rather than requiring
            # the exact same commentary string, since two entries reporting
            # the same real assay are often phrased slightly differently.
            matched = False
            for rec in temp_records:
                if (rec["pr_id"] == pr_id and
                    rec["measured_substrate"] == substrate and
                    rec["temperature"] == temp and
                    rec["ph"] == ph and
                    rec["mutation"] == mutation and
                    rec["kcat"] is None):

                    rec["kcat"] = kcat_val
                    matched = True
                    break

            if not matched:
                record_id_counter += 1
                temp_records.append({
                    "entry_id": record_id_counter,
                    "ec_number": ec_number,
                    "pr_id": pr_id,
                    "organism": org,
                    "uniprot_id": uniprot,
                    "measured_substrate": substrate,
                    "kcat": kcat_val,
                    "km": None,

                    "temperature": temp,
                    "ph": ph,
                    "commentary": commentary,
                    "mutation": mutation,
                })



    # Cross-check kcat/km against curated kcat_km_value if available
    valid_temp_records = []
    for d in temp_records:
        if d["kcat"] is not None and d["km"] is not None:
            derived_kcat_km = d["kcat"] / d["km"]
            curated_kcat_kms = kcat_km_map.get(d["pr_id"], [])
            matched_curated = False
            disagrees_by_10x = False
            curated_val = None
            for curated in curated_kcat_kms:
                if curated["substrate"] == d["measured_substrate"]:
                    # Check condition match if provided
                    t_match = (curated["temperature"] is None) or (curated["temperature"] == d["temperature"])
                    ph_match = (curated["ph"] is None) or (curated["ph"] == d["ph"])
                    if t_match and ph_match:
                        matched_curated = True
                        curated_val = curated["value"]
                        ratio = derived_kcat_km / curated_val
                        if ratio > 10.0 or ratio < 0.1:
                            disagrees_by_10x = True
                        break
            if matched_curated and disagrees_by_10x and curated_val is not None:
                logger.debug("Dropped entry %s: derived kcat/km (%f) disagrees >10x with curated (%f)", d["entry_id"], derived_kcat_km, curated_val)
                continue
        valid_temp_records.append(d)

    # Convert holding dicts to KineticRecord objects
    uniprot_cache = {}
    for d in valid_temp_records:
        if (d["kcat"] is None or d["km"] is None or 
            d["temperature"] is None or d["ph"] is None or 
            not d["organism"] or not d["measured_substrate"]):
            continue

        # Resolve UniProt ID
        uniprot_id = d["uniprot_id"]
        if not uniprot_id:
            cache_key = (d["ec_number"], d["organism"])
            if cache_key not in uniprot_cache:
                uniprot_cache[cache_key] = fetch_uniprot_id(d["ec_number"], d["organism"])
                time.sleep(0.1) # Be polite
            uniprot_id = uniprot_cache[cache_key]

        if not uniprot_id:
            continue

        # Substrate matching logic
        measured_sub = d["measured_substrate"]
        co_subs = []
        valid_reaction_found = False

        if not possible_reactions:
            # No reaction info, we cannot be sure if it's multi-substrate or not, so we assume single-substrate
            valid_reaction_found = True
        else:
            for react_list in possible_reactions:
                # Direct match
                if measured_sub in react_list:
                    co_subs = [r for r in react_list if r != measured_sub]
                    valid_reaction_found = True
                    break
                # Substring match (e.g. BRENDA might list "ATP" while reaction says "atp")
                else:
                    for r in react_list:
                        if measured_sub in r or r in measured_sub:
                            co_subs = [x for x in react_list if x != r]
                            valid_reaction_found = True
                            break
                if valid_reaction_found:
                    break

        if not valid_reaction_found and len(possible_reactions) > 0:
            # Could not resolve measured substrate to any reaction, drop it
            continue
        canon = canonicalize_and_filter_ligands(measured_sub, co_subs)
        if not canon:
            logger.debug("Dropped record due to blacklisted substrate/cofactor: %s, %s", measured_sub, co_subs)
            continue
            
        std_sub, std_cos = canon
        std_co_substrates = "; ".join(std_cos)

        records.append(KineticRecord(
            entry_id=d["entry_id"],
            source_db="BRENDA",
            ec_number=d["ec_number"],
            uniprot_id=uniprot_id,
            measured_substrate=std_sub,
            co_substrates=std_co_substrates,
            kcat=d["kcat"],
            km=d["km"],

            temperature=d["temperature"],
            ph=d["ph"],
            mutation=d["mutation"],
        ))

    logger.info("Parsed %d valid kinetic records for EC %s", len(records), ec_number)
    return records


# ═══════════════════════════════════════════════════════════════════════════
#  Database operations
# ═══════════════════════════════════════════════════════════════════════════

def _save_to_database(records: list[KineticRecord], db_path: pathlib.Path = DEFAULT_DB_PATH) -> None:
    """Insert or replace parsed kinetic records into the SQLite database.
    
    Parameters
    ----------
    records : list[KineticRecord]
        A list of parsed KineticRecord instances to insert.
    db_path : pathlib.Path, default: DEFAULT_DB_PATH
        The file path to the SQLite database.
        
    Returns
    -------
    None
    """
    if not records:
        return

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_parameters (
            entry_id INTEGER PRIMARY KEY,
            source_db TEXT,
            ec_number TEXT NOT NULL,
            uniprot_id TEXT,
            measured_substrate TEXT NOT NULL,
            co_substrates TEXT,
            kcat REAL,
            km REAL,

            temperature REAL,
            ph REAL,
            mutation TEXT
        )
        """
    )
    # CREATE TABLE IF NOT EXISTS does not add columns to a database file built
    # under an older schema, so the ALTER TABLE below guarantees the mutation
    # column exists regardless of the file's schema version.
    existing_cols = {row[1] for row in cursor.execute("PRAGMA table_info(raw_parameters)").fetchall()}
    if "mutation" not in existing_cols:
        cursor.execute("ALTER TABLE raw_parameters ADD COLUMN mutation TEXT")


    inserted = 0
    try:
        cursor.executemany(
            """
            INSERT OR REPLACE INTO raw_parameters
            (entry_id, source_db, ec_number, uniprot_id, measured_substrate, co_substrates, kcat, km, temperature, ph, mutation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        logger.error("Failed to insert records: %s", exc)

    conn.commit()
    conn.close()
    
    logger.info("Inserted %d / %d records into %s", inserted, len(records), os.path.basename(db_path))


def _delete_source_rows(source_db: str, db_path: pathlib.Path = DEFAULT_DB_PATH) -> int:
    """Delete all ``raw_parameters`` rows for a given ``source_db``.

    Used to give a fresh ``--full`` run (one not resuming via
    ``--continue``) a clean slate for its own source, without touching
    rows inserted by the other parser.

    Parameters
    ----------
    source_db : str
        The ``source_db`` value to delete (e.g. ``"BRENDA"``).
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
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Execute the BRENDA ingestion pipeline from local JSON.
    
    Parameters
    ----------
    None
        
    Returns
    -------
    None
    """
    logger.info("============================================================")
    logger.info("ThermoKP — BRENDA Kinetic Data Ingestion")
    logger.info("============================================================")

    if not BRENDA_JSON_PATH.exists():
        logger.error("BRENDA JSON file not found at %s", BRENDA_JSON_PATH)
        sys.exit(1)

    logger.info("Loading BRENDA JSON into memory...")
    try:
        with open(BRENDA_JSON_PATH, "r") as f:
            full_json = json.load(f)
        brenda_data = full_json.get("data", {})
        logger.info("Loaded %d EC entries from JSON.", len(brenda_data))
    except Exception as e:
        logger.error("Failed to load BRENDA JSON: %s", e)
        sys.exit(1)

    total_inserted = 0
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

        checkpoint_path = BRENDA_CHECKPOINT_FILE
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
            deleted = _delete_source_rows("BRENDA", DEFAULT_DB_PATH)
            logger.info("Fresh run: deleted %d existing BRENDA rows from raw_parameters.", deleted)
            _clear_checkpoint(checkpoint_path)

    for idx, entry in enumerate(queries):
        ec: str = entry["ec"]
        label: str = entry.get("label", ec)

        logger.info(
            "[%d/%d] %s (EC %s)", idx + 1, len(queries), label, ec
        )

        records = _parse_brenda_data(ec, brenda_data)
        _save_to_database(records, DEFAULT_DB_PATH)
        total_inserted += len(records)

        if checkpoint_path is not None:
            _write_checkpoint(checkpoint_path, ec)

    if checkpoint_path is not None:
        _clear_checkpoint(checkpoint_path)

    logger.info("==========================================================================")
    logger.info("==                        BRENDA Parser Summary                         ==")
    logger.info("==========================================================================")
    logger.info(f"Total rows inserted       : {total_inserted}")
    logger.info(f"Database                  : {os.path.abspath(DEFAULT_DB_PATH)}")
    logger.info("==========================================================================")


if __name__ == "__main__":
    main()
