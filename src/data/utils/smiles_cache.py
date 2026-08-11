"""
===========================================================================
SMILES Cache
Description: Shared on-disk {chemical name: SMILES} cache
===========================================================================

Both sabio_rk_parser.py (populating it from a species' ChEBI/KEGG
cross-reference at parse time) and dataset_validator.py (populating it from
every other resolution tier - manual override, the local peptide-notation
builder, PubChem, OPSIN, CACTUS) read and write the exact same on-disk
cache through this module, so there is exactly one in-memory cache and one
atomic-write implementation shared by both, instead of each file
maintaining its own copy that could silently drift out of sync.

Author: ThermoKP Team
License: MIT
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import threading
import uuid
from typing import Optional

import rdkit.Chem as Chem

logger = logging.getLogger(__name__)

# Guards read-modify-write access to `_cache` and the on-disk file. Needed
# now that dataset_validator.py resolves chemicals from a thread pool -
# without this, two threads writing different entries at the same moment
# could race on the same temp-file-then-rename, silently dropping one.
# Must be reentrant (RLock, not Lock): both save_smiles_cache_entry and
# merge_entries hold this lock and then call load_smiles_cache(), which
# also acquires it on a cold cache (_cache is still None) - a plain Lock
# would self-deadlock the first time a save/merge happens before any load
# in that process.
_CACHE_LOCK = threading.RLock()

_PROJECT_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[3]

# Populated from ChEBI/KEGG cross-references (sabio_rk_parser.py) and from
# every SMILES-resolution tier (dataset_validator.py) - a name resolved once,
# by any process, never triggers a repeat network round trip in a later run
# of the parser, the validator, or the geometry pipeline.
SMILES_CACHE_FILE: pathlib.Path = _PROJECT_ROOT / "data" / "cache" / "smiles_cache.json"

_cache: Optional[dict[str, str]] = None


def _is_parseable(smiles: str) -> bool:
    """Check that RDKit can actually parse a SMILES string before caching it.

    Some chemical-name resolvers (e.g. OPSIN/CACTUS) can return SMILES using
    a non-RDKit aromaticity convention - aromatic ring nitrogens written as
    bare lowercase `n` where the structure actually needs an explicit
    `[nH]` for valence/kekulization to work (e.g. "thymine" as
    `Cc1cnc(=O)nc1=O`, which RDKit rejects with "Can't kekulize mol"). Once
    such a string is cached, `get_smiles()`'s cache tier would return it
    forever with no later tier getting a chance to produce a working
    structure, so caching only happens if this check passes.

    Parameters
    ----------
    smiles : str
        The SMILES string to validate.

    Returns
    -------
    bool
        True if RDKit can parse it, False otherwise.
    """
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def load_smiles_cache() -> dict[str, str]:
    """Load (and memoize) the on-disk {lowercased chemical name: SMILES} cache.

    Returns
    -------
    dict[str, str]
        The cache contents, empty if the file doesn't exist yet.
    """
    global _cache
    if _cache is None:
        with _CACHE_LOCK:
            if _cache is None:  # Re-check: another thread may have loaded it first.
                loaded: dict[str, str] = {}
                if SMILES_CACHE_FILE.exists():
                    try:
                        loaded = json.loads(SMILES_CACHE_FILE.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        logger.warning("Failed to read %s; starting a fresh cache.", SMILES_CACHE_FILE)
                _cache = loaded
    return _cache


def save_smiles_cache_entry(name: str, smiles: str) -> None:
    """Persist one {name: smiles} entry to the on-disk cache atomically.

    Silently no-ops if `smiles` isn't RDKit-parseable (see `_is_parseable`)
    rather than caching a value that would permanently block this name
    from ever resolving correctly.

    Parameters
    ----------
    name : str
        Lowercased chemical name, used as the cache key.
    smiles : str
        The resolved SMILES string.
    """
    if not _is_parseable(smiles):
        logger.warning("Refusing to cache unparseable SMILES for %r: %s", name, smiles)
        return
    with _CACHE_LOCK:
        cache = load_smiles_cache()
        cache[name] = smiles
        SMILES_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = SMILES_CACHE_FILE.parent / f".tmp-{uuid.uuid4().hex}.json"
        tmp_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, SMILES_CACHE_FILE)


def merge_entries(entries: dict[str, str], overwrite: bool = False) -> int:
    """Merge many {name: smiles} entries into the on-disk cache in one write.

    Used by bulk-import scripts (e.g. sync_brenda_ligand_names.py) where
    calling `save_smiles_cache_entry` once per entry would mean one
    atomic-rename disk write per name - fine for a handful of live
    resolutions, far too slow for hundreds of thousands of entries from a
    bulk source.

    Parameters
    ----------
    entries : dict[str, str]
        {lowercased chemical name: SMILES} entries to merge in.
    overwrite : bool, default: False
        If False (the default), an existing cache entry for a name is left
        untouched - a name already resolved by a trusted source (e.g.
        ChEBI via SABIO parsing) keeps that value rather than being
        replaced by a bulk-import value for the same name.

    Returns
    -------
    int
        Number of entries actually added or overwritten (entries that
        fail RDKit parseability - see `_is_parseable` - are silently
        skipped, not counted).
    """
    with _CACHE_LOCK:
        cache = load_smiles_cache()
        added = 0
        skipped_unparseable = 0
        for name, smiles in entries.items():
            if not overwrite and name in cache:
                continue
            if not _is_parseable(smiles):
                skipped_unparseable += 1
                continue
            cache[name] = smiles
            added += 1
        if skipped_unparseable:
            logger.warning("Skipped %d unparseable SMILES during merge.", skipped_unparseable)
        if added:
            SMILES_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = SMILES_CACHE_FILE.parent / f".tmp-{uuid.uuid4().hex}.json"
            tmp_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp_path, SMILES_CACHE_FILE)
        return added
