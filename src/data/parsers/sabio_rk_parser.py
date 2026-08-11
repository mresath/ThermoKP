"""
===========================================================================
SABIO-RK Parser (JSON API)
Description: SABIO-RK Database Ingestion for ThermoKP
===========================================================================

Workflow:
1. GET `/export-api/sabio/kinlaw-entry/json` with a query string of the form `ECNumber:"<ec>"` and `(Parametertype:"kcat" OR Parametertype:"Km")` to retrieve paginated JSON.
2. Parse the JSON to extract: Entry ID, EC number, Organism, Substrate name(s), k_cat, K_m, Temperature, and pH.
3. Cache SMILES strings natively provided by the JSON response for reactants, skipping slow third-party REST lookups.
4. If a species carries no UniProt cross-reference but an organism name was recovered, fall back to `parser_common.fetch_uniprot_id(ec, organism)`.
5. Parse mutant identity from the enzyme description and separate wildtypes from point mutants, rejecting unsalvageable complex mutants.
6. Insert each valid record into the `raw_parameters` table.

Author: ThermoKP Team
License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sqlite3
import sys
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import _CalcMolWt

from src.data.models.models import KineticRecord
from src.data.utils.ligand_cleaner import canonicalize_and_filter_ligands
from src.data.parsers.parser_common import (
    fetch_uniprot_id,
    MUTATION_KEYWORDS,
    find_point_mutations,
)
from src.data.parsers.parser_common import fetch_uniprot_id
from src.data.processors.pretrained_embeddings import fetch_uniprot_sequence
from src.data.utils.smiles_cache import load_smiles_cache, save_smiles_cache_entry

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

SABIO_RK_JSON_URL: str = "https://sabiork.h-its.org/export-api/sabio/kinlaw-entry/json"

_PROJECT_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH: pathlib.Path = _PROJECT_ROOT / "data" / "thermokp_database.db"
TARGETS_FILE: pathlib.Path = _PROJECT_ROOT / "data" / "enzyme_targets.json"

SABIO_CHECKPOINT_FILE: pathlib.Path = _PROJECT_ROOT / "data" / "raw" / "sabio_checkpoint.txt"

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

# Time to sleep between successful API calls
_REQUEST_DELAY_S: float = 2.0


# ═══════════════════════════════════════════════════════════════════════════
#  Data Parsing & Extraction
# ═══════════════════════════════════════════════════════════════════════════

_session: requests.Session = requests.Session()
_retry = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
_adapter = HTTPAdapter(max_retries=_retry)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)

def _get_api_session() -> requests.Session:
    return _session


def _query_sabio_rk(ec_number: str, target_organism: Optional[str] = None) -> list[dict]:
    """Query the SABIO-RK JSON API and return all matching entries.
    
    Handles pagination automatically. Sleeps `_REQUEST_DELAY_S` between pages.
    
    Parameters
    ----------
    ec_number : str
        The Enzyme Commission number to query (e.g., "1.1.1.1").
    target_organism : str, optional
        The specific organism to filter results by.

    Returns
    -------
    list[dict]
        A list of JSON dictionary entries retrieved from SABIO-RK.
    """
    query_parts = [f"ECNumber:{ec_number}", '(Parametertype:"kcat" OR Parametertype:"Km")']
    if target_organism:
        query_parts.append(f'Organism:"{target_organism}"')
    
    query = " AND ".join(query_parts)
    logger.info("Querying SABIO-RK  |  %s", query)

    page = 1
    page_size = 1000
    all_entries = []

    while True:
        params = {
            "q": query,
            "page": page,
            "pageSize": page_size,
        }
        try:
            response = _session.get(SABIO_RK_JSON_URL, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            entries = data.get("data", [])
            all_entries.extend(entries)
            
            meta = data.get("meta", {})
            total_pages = meta.get("total_pages", 1)
            
            logger.info("Received %d entries for EC %s (page %d/%d)", len(entries), ec_number, page, total_pages)
            
            if page >= total_pages:
                break
                
            page += 1
            time.sleep(_REQUEST_DELAY_S)
        except requests.exceptions.RequestException as e:
            logger.error("Request failed for EC %s (page %d): %s", ec_number, page, e)
            break

    return all_entries


def _parse_sabio_mutation(enzyme_name: str) -> tuple[bool, Optional[str]]:
    """Parse mutant identity from an enzyme description string.

    Parameters
    ----------
    enzyme_name : str
        The descriptive name of the enzyme from SABIO-RK.

    Returns
    -------
    tuple[bool, str | None]
        ``(keep, mutation_code)``. ``keep=False`` means the record must be
        dropped entirely (e.g. for deletions). ``keep=True`` with 
        ``mutation_code=None`` means wild-type. ``keep=True`` with a code
        means one or more point mutations were successfully extracted.
    """
    lower = enzyme_name.lower()
    if "delta" in lower or "Δ" in enzyme_name:
        return False, None
    if not any(kw in lower for kw in MUTATION_KEYWORDS):
        return True, None

    matches = find_point_mutations(enzyme_name)
    if len(matches) > 0:
        mutation_code = "/".join(f"{wt}{pos}{mut}" for wt, pos, mut in matches)
        return True, mutation_code

    return False, None


def _parse_json_response(entries: list[dict], ec_number: str, target_organism: Optional[str] = None) -> list[KineticRecord]:
    """Parse SABIO-RK JSON entries into KineticRecord objects.

    Parameters
    ----------
    entries : list[dict]
        The raw JSON entries retrieved from the API.
    ec_number : str
        The Enzyme Commission number corresponding to the entries.
    target_organism : str, optional
        The fallback organism name to use if not provided in the entry.

    Returns
    -------
    list[KineticRecord]
        A list of populated KineticRecord instances.
    """
    records: list[KineticRecord] = []
    
    smiles_cache = load_smiles_cache()
    
    # UniProt ID lookup cache per EC/Organism to avoid redundant web lookups
    _uniprot_fallback_cache: dict[tuple[str, str], Optional[str]] = {}

    for entry in entries:
        try:
            entry_id = entry.get("id") or 0
            
            # 1. Uniprot ID
            uniprot_id = None
            for kl in entry.get("external_links", {}).get("kinlaw_entry", []):
                if kl.get("key") == "UniProtKB_AC":
                    uniprot_id = kl.get("value")
                    break
                    
            # 2. Organism
            organism = entry.get("general", {}).get("organism", {}).get("name")
            if not organism:
                organism = target_organism
                
            # Fallback lookup if UniProt ID is missing
            if not uniprot_id and organism:
                cache_key = (ec_number, organism)
                if cache_key not in _uniprot_fallback_cache:
                    logger.debug("Falling back to UniProt search for %s (%s)", ec_number, organism)
                    _uniprot_fallback_cache[cache_key] = fetch_uniprot_id(ec_number, organism)
                uniprot_id = _uniprot_fallback_cache[cache_key]
            
            # 3. Conditions
            temp_raw = entry.get("experimental_conditions", {}).get("envvar_temperature", {}).get("start_value")
            ph_raw = entry.get("experimental_conditions", {}).get("envvar_ph", {}).get("start_value")
            
            temp = float(temp_raw) if temp_raw is not None else None
            ph = float(ph_raw) if ph_raw is not None else None
            
            if temp is not None and temp > 200.0:
                temp = temp - 273.15
            
            # 4. Mutation
            enzyme_name = entry.get("enzyme_description", {}).get("enzyme_name") or ""
            mutant_spec = entry.get("enzyme_description", {}).get("mutant_spec")
            if mutant_spec and mutant_spec.lower() != "wildtype":
                enzyme_name += " " + mutant_spec
                
            mutation_code = None
            if enzyme_name:
                keep, mut = _parse_sabio_mutation(enzyme_name)
                if not keep:
                    continue
                mutation_code = mut
                
            # 5. Reactants & SMILES Caching
            all_reactants = {} # id (int) -> name (str)
            compounds_data = entry.get("external_links", {}).get("compound", [])
            
            compound_smiles = {}
            for c in compounds_data:
                if c.get("key") == "SMILES":
                    compound_smiles[c.get("id")] = c.get("value")
                    
            for sp in entry.get("reaction", {}).get("species", []):
                if sp.get("role") == "Substrate":
                    comp = sp.get("compound", {})
                    cid = comp.get("id")
                    cname = comp.get("name")
                    if cid is not None and cname:
                        cname = cname.lower()
                        all_reactants[cid] = cname
                        if cid in compound_smiles and cname not in smiles_cache:
                            save_smiles_cache_entry(cname, compound_smiles[cid])
                            smiles_cache[cname] = compound_smiles[cid]
                            
            # 6. Parameters
            # 6. Parameters
            kcat = None
            vmax = None
            vu = ""
            e0 = None
            eu = ""
            km_values = {} # species_name (str) -> km
            kcat_km_values = {} # species_name (str) -> kcat/Km
            
            for param in entry.get("kineticlaw", {}).get("parameter", []):
                p_type = param.get("parameter_type", {}).get("name")
                if p_type == "kcat":
                    val = param.get("n_start_value")
                    unit = param.get("unit", {}).get("n_name")
                    if val is not None and unit in ("s^(-1)", "1/s", "s-1", "s^-1"):
                        kcat = round(float(val), 4)
                    elif val is not None:
                        logger.warning("Dropping kcat due to unexpected normalized unit: %s", unit)
                
                elif p_type in ("Vmax", "specific enz. activity"):
                    val = param.get("start_value")
                    unit = param.get("unit", {}).get("name", "")
                    if val is not None:
                        vu_clean = unit.lower().replace(" ", "").replace("*", "")
                        multiplier = None
                        if vu_clean in ("mol/(s.g)", "mols^(-1)g^(-1)", "mol/s/g", "mols-1g-1", "mol/(sg)", "mol/sg"):
                            multiplier = 1.0
                        elif vu_clean in ("mol/(min.g)", "molmin^(-1)g^(-1)", "mol/min/g", "mol/(ming)"):
                            multiplier = 1.0 / 60.0
                        elif vu_clean in ("mmol/(s.g)", "mmols^(-1)g^(-1)", "mmol/s/g", "mmol/(sg)"):
                            multiplier = 1e-3
                        elif vu_clean in ("mmol/(min.g)", "mmolmin^(-1)g^(-1)", "mmol/min/g", "mmol/(ming)"):
                            multiplier = 1e-3 / 60.0
                        elif vu_clean in ("µmol/(min.mg)", "umol/(min.mg)", "umol/min/mg", "µmol/min/mg", "umolmin^(-1)mg^(-1)", "µmolmin^(-1)mg^(-1)", "u/mg", "units/mg", "umol/(minmg)", "µmol/(minmg)"):
                            multiplier = 1e-6 / (60.0 * 1e-3)
                        elif vu_clean in ("mmol/(min.mg)", "mmol/min/mg", "mmolmin^(-1)mg^(-1)", "mmol/(minmg)"):
                            multiplier = 1e-3 / (60.0 * 1e-3)
                        elif vu_clean in ("µmol/(s.mg)", "umol/(s.mg)", "umol/s/mg", "µmol/s/mg", "umols^(-1)mg^(-1)", "µmols^(-1)mg^(-1)", "umol/(smg)", "µmol/(smg)"):
                            multiplier = 1e-6 / 1e-3
                        elif vu_clean in ("mol/(s.mg)", "mols^(-1)mg^(-1)", "mol/s/mg", "mol/(smg)"):
                            multiplier = 1.0 / 1e-3
                        elif vu_clean in ("u/g", "units/g", "µmol/(min.g)", "umol/(min.g)", "umol/min/g", "µmol/min/g", "umol/(ming)", "µmol/(ming)"):
                            multiplier = 1e-6 / 60.0

                        if multiplier is not None and uniprot_id:
                            seq = fetch_uniprot_sequence(uniprot_id)
                            if seq:
                                mw_enzyme = len(seq) * 110.0
                                kcat = round(float(val) * multiplier * mw_enzyme, 4)
                        else:
                            vmax = float(val)
                            vu = vu_clean
                
                elif p_type == "concentration":
                    sp_key = param.get("species", {}).get("species_key", "")
                    if "Catalyst" in sp_key or "Enzyme" in sp_key:
                        val = param.get("start_value")
                        unit = param.get("unit", {}).get("name", "")
                        if val is not None:
                            e0 = float(val)
                            eu = unit.lower().replace(" ", "").replace("*", "")

                elif p_type == "Km":
                    val = param.get("n_start_value")
                    unit = param.get("unit", {}).get("n_name")
                    sp_key = param.get("species", {}).get("species_key")
                    
                    sp_name = None
                    if sp_key:
                        parts = sp_key.split(" | ")
                        if len(parts) >= 2:
                            sp_name = parts[1].lower()
                    elif len(all_reactants) == 1:
                        sp_name = list(all_reactants.values())[0]

                    if sp_name:
                        if val is not None and unit in ("M", "mol/l", "mol/L", "Molar", "molar"):
                            km_values[sp_name] = round(float(val) * 1000.0, 4)
                        elif param.get("start_value") is not None:
                            raw_val = float(param.get("start_value"))
                            raw_unit = param.get("unit", {}).get("name", "").lower().replace(" ", "")
                            g_per_l = None
                            if raw_unit in ("mg/l", "ug/ml", "µg/ml", "microg/ml"):
                                g_per_l = raw_val / 1000.0
                            elif raw_unit == "g/ml":
                                g_per_l = raw_val * 1000.0
                            elif raw_unit == "ng/ml":
                                g_per_l = raw_val / 1e6
                            elif raw_unit in ("mg/ml", "g/l"):
                                g_per_l = raw_val
                            
                            if g_per_l is not None:
                                smiles = smiles_cache.get(sp_name)
                                if smiles:
                                    mol = Chem.MolFromSmiles(smiles)
                                    if mol:
                                        molar_mass = _CalcMolWt(mol)
                                        km_values[sp_name] = round((g_per_l / molar_mass) * 1000.0, 4)
                            else:
                                if val is not None:
                                    logger.warning("Dropping Km due to unexpected normalized unit: %s", unit)

                elif p_type == "kcat/Km":
                    val = param.get("n_start_value")
                    unit = param.get("unit", {}).get("n_name")
                    sp_key = param.get("species", {}).get("species_key")
                    sp_name = None
                    if sp_key:
                        parts = sp_key.split(" | ")
                        if len(parts) >= 2:
                            sp_name = parts[1].lower()
                    elif len(all_reactants) == 1:
                        sp_name = list(all_reactants.values())[0]

                    if sp_name and val is not None and unit in ("M^(-1)*s^(-1)", "M-1s-1", "1/(M*s)"):
                        kcat_km_values[sp_name] = float(val)

            if kcat is None and vmax is not None and e0 is not None and e0 > 0:
                if vu.startswith(eu) or (eu in vu and ("s^(-1)" in vu or "s-1" in vu or "min^(-1)" in vu or "min-1" in vu)):
                    kcat_raw = vmax / e0
                    if "min^(-1)" in vu or "min-1" in vu or "/min" in vu:
                        kcat_raw /= 60.0
                    elif "h^(-1)" in vu or "h-1" in vu or "/h" in vu:
                        kcat_raw /= 3600.0
                    kcat = round(kcat_raw, 4)

            if kcat is None or temp is None or ph is None or not km_values:
                continue
                
            # Create a record for each primary Km
            for sub_name, km in km_values.items():
                if sub_name not in all_reactants.values():
                    logger.debug("Dropped Km for %s, not in reactants %s", sub_name, all_reactants.values())
                    continue
                
                # Cross-check kcat/Km
                kcat_km_val = kcat_km_values.get(sub_name)
                if kcat_km_val is not None:
                    km_molar = km / 1000.0
                    calc_val = kcat / km_molar if km_molar > 0 else float('inf')
                    if min(calc_val, kcat_km_val) > 0:
                        ratio = max(calc_val, kcat_km_val) / min(calc_val, kcat_km_val)
                        if ratio > 2.0:
                            logger.debug("Dropped record due to kcat/Km cross-check failure for %s", sub_name)
                            continue

                co_subs = [name for name in all_reactants.values() if name != sub_name]
                
                canon = canonicalize_and_filter_ligands(sub_name, co_subs)
                if not canon:
                    logger.debug("Dropped canon for %s", sub_name)
                    continue
                    
                records.append(KineticRecord(
                    entry_id=entry_id,
                    source_db="SABIO-RK",
                    ec_number=ec_number,
                    uniprot_id=uniprot_id or "",
                    measured_substrate=canon[0],
                    co_substrates="; ".join(canon[1]) if canon[1] else "",
                    kcat=kcat,
                    km=km,
                    temperature=temp,
                    ph=ph,
                    mutation=mutation_code
                ))
        except Exception as e:
            logger.error("Error parsing JSON entry: %s", e)
            
    return records


# ═══════════════════════════════════════════════════════════════════════════
#  Database operations
# ═══════════════════════════════════════════════════════════════════════════

def _insert_records(records: list[KineticRecord], db_path: pathlib.Path) -> int:
    """Insert parsed KineticRecord objects into the database.

    Parameters
    ----------
    records : list[KineticRecord]
        The list of records to insert.
    db_path : pathlib.Path
        The path to the SQLite database file.

    Returns
    -------
    int
        The number of records successfully inserted.
    """
    if not records:
        return 0

    inserted_count = 0
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            cursor = conn.cursor()
            for r in records:
                cursor.execute(
                    """
                    INSERT INTO raw_parameters (
                        source_db,
                        ec_number,
                        uniprot_id,
                        measured_substrate,
                        co_substrates,
                        kcat,
                        km,
                        temperature,
                        ph,
                        mutation
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
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
                    ),
                )
                inserted_count += 1
            conn.commit()
    except sqlite3.Error as e:
        logger.error("Database error during insert: %s", e)
    return inserted_count


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def _run_batch(queries: list[dict[str, str]], db_path: pathlib.Path) -> None:
    """Run ingestion for a specific batch of queries.

    Parameters
    ----------
    queries : list[dict[str, str]]
        A list of dictionaries specifying the "ec" and optional "organism" queries.
    db_path : pathlib.Path
        The path to the SQLite database file.
    """
    for idx, q in enumerate(queries):
        ec: str = q["ec"]
        organism: Optional[str] = q.get("organism")
        label: str = q.get("label", ec)
        
        logger.info("[%d/%d] %s (EC %s)", idx + 1, len(queries), label, ec)
        
        entries = _query_sabio_rk(ec, organism)
        if not entries:
            logger.info("No records found in SABIO-RK for EC %s", ec)
        else:
            records = _parse_json_response(entries, ec, organism)
            if records:
                inserted = _insert_records(records, db_path)
                logger.info("Saved %d records to DB for EC %s", inserted, ec)
            else:
                logger.info("Parsed 0 valid kinetic records for EC %s", ec)
        
        if idx < len(queries) - 1:
            time.sleep(_REQUEST_DELAY_S)


def run_full(db_path: pathlib.Path, resume: bool = False) -> None:
    """Run full SABIO-RK data ingestion across all targets.

    Parameters
    ----------
    db_path : pathlib.Path
        The path to the SQLite database file.
    resume : bool
        If True, resume processing from the last checkpointed EC number.
    """
    if not TARGETS_FILE.exists():
        logger.error("Target list not found: %s", TARGETS_FILE)
        sys.exit(1)

    try:
        with open(TARGETS_FILE, "r", encoding="utf-8") as f:
            targets_data = json.load(f)
    except Exception as e:
        logger.error("Failed to read targets file %s: %s", TARGETS_FILE, e)
        sys.exit(1)

    all_ecs = targets_data
    if not all_ecs:
        logger.error("No valid EC numbers found in %s", TARGETS_FILE)
        sys.exit(1)

    logger.info("Loaded %d EC targets from %s", len(all_ecs), TARGETS_FILE.name)

    start_idx = 0
    if resume:
        if SABIO_CHECKPOINT_FILE.exists():
            try:
                with open(SABIO_CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                    last_ec = f.read().strip()
                if last_ec in all_ecs:
                    start_idx = all_ecs.index(last_ec) + 1
                    logger.info("Resuming run after EC %s (starting at %d/%d)", last_ec, start_idx + 1, len(all_ecs))
                else:
                    logger.warning("Checkpoint EC %s not in target list, ignoring checkpoint.", last_ec)
            except Exception as e:
                logger.warning("Failed to read checkpoint file: %s", e)
        else:
            logger.info("--continue passed but no checkpoint file found; starting from beginning.")

    if start_idx == 0:
        try:
            with sqlite3.connect(db_path, timeout=10) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM raw_parameters WHERE source_db = 'SABIO-RK'")
                deleted = cursor.rowcount
                conn.commit()
            logger.info("Fresh run: deleted %d existing SABIO-RK rows from raw_parameters.", deleted)
        except sqlite3.Error as e:
            logger.error("Failed to clear existing SABIO-RK data: %s", e)
            sys.exit(1)
            
        if SABIO_CHECKPOINT_FILE.exists():
            try:
                SABIO_CHECKPOINT_FILE.unlink()
            except OSError as e:
                logger.warning("Failed to remove old checkpoint file: %s", e)

    SABIO_CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)

    for i in range(start_idx, len(all_ecs)):
        ec = all_ecs[i]
        logger.info("[%d/%d] %s (EC %s)", i + 1, len(all_ecs), ec, ec)
        
        entries = _query_sabio_rk(ec, None)
        if entries:
            records = _parse_json_response(entries, ec, None)
            if records:
                _insert_records(records, db_path)
                logger.info("Saved %d records to DB for EC %s", len(records), ec)
            else:
                logger.info("Parsed 0 valid kinetic records for EC %s", ec)
        else:
            logger.info("No records found in SABIO-RK for EC %s", ec)

        try:
            with open(SABIO_CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                f.write(ec)
        except OSError as e:
            logger.warning("Failed to write checkpoint for EC %s: %s", ec, e)
            
        if i < len(all_ecs) - 1:
            time.sleep(_REQUEST_DELAY_S)

    if SABIO_CHECKPOINT_FILE.exists():
        try:
            SABIO_CHECKPOINT_FILE.unlink()
            logger.info("Run completed successfully; removed checkpoint file.")
        except OSError as e:
            logger.warning("Failed to remove checkpoint file upon completion: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="SABIO-RK Database Ingestion")
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help="Path to the SQLite database.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run against all EC targets from enzyme_targets.json.",
    )
    parser.add_argument(
        "--continue",
        dest="resume",
        action="store_true",
        help="Resume a --full run from the checkpoint file without clearing existing SABIO-RK rows.",
    )

    args = parser.parse_args()
    db_path = pathlib.Path(args.db).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ThermoKP — SABIO-RK Kinetic Data Ingestion (JSON API)")
    logger.info("=" * 60)

    if args.full:
        run_full(db_path, resume=args.resume)
    else:
        _run_batch(DEFAULT_QUERIES, db_path)


if __name__ == "__main__":
    main()
