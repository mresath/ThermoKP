"""
===========================================================================
Pretrained Sequence Embeddings
Description: ESM2 protein and ChemBERTa-2 ligand embedding extraction
===========================================================================

Workflow:
1. Fetch the full-length UniProt sequence for a protein via the UniProt
   REST FASTA endpoint.
2. Run ESM2-650M once per unique UniProt sequence to obtain per-residue
   embeddings, stripping the <cls>/<eos> special-token positions so the
   output aligns 1:1 with the input sequence.
3. Fetch UniProt's curated "Active site" / "Binding site" sequence
   features via the UniProt REST JSON endpoint and build a per-residue
   catalytic-site mask, aligned 1:1 with the same sequence.
4. For point-mutant records, score the substitution with ESM2's own
   masked-LM head (masked-marginal log-odds) as a pooling-independent
   mutation-effect signal, baked into the record as a top-level scalar
   rather than relying on the per-residue embedding surviving pooling.
5. Run ChemBERTa-2 once per unique canonical SMILES string and mean-pool
   its token embeddings (via the attention mask) into a single whole-
   molecule embedding.
6. Cache the embeddings, the catalytic-site mask, and the mutation
   log-odds scores to disk, keyed by UniProt ID (features and per-residue
   embeddings), UniProt ID + residue position (log-odds), or a SHA-256
   hash of the SMILES string (ligand embeddings), so repeated
   enzymes/ligands/positions across kinetic records are only
   fetched/embedded/scored once.

Known Caveats:
- Both pretrained models are used strictly as frozen feature extractors
  (no_grad, eval mode); only the downstream adapter layers in
  src/encoders/multimodal_encoder.py are trainable. The raw embeddings, the
  catalytic-site mask, and the mutation log-odds score produced here are
  baked directly into each HeteroData tensor at data-generation time (see
  src/data/processors/generate_tensors.py).
- ESM2 is loaded via `AutoModelForMaskedLM` rather than the plain
  `AutoModel`, so one loaded checkpoint serves both per-residue
  embeddings (`output_hidden_states=True`) and the masked-LM logits the
  mutation log-odds score needs, rather than holding two copies of a
  650M-parameter model in memory.
- ESM2-650M's positional embeddings cap the usable context at 1022
  residues (1026 max_position_embeddings minus the <cls>/<eos> tokens).
  Sequences beyond this length must be dropped upstream by
  src/data/processors/dataset_validator.py; get_protein_embedding raises rather than
  silently truncating, since truncation would desynchronize the
  residue-index-to-embedding-row mapping used by geometry_processor.py.
- UniProt's Active site/Binding site annotations are curated and often
  sparse or absent for poorly-characterized enzymes; `get_catalytic_site_mask`
  degrades to an all-zero mask in that case rather than raising, since
  it is a supplementary signal rather than a hard data requirement.
- The UniProt REST endpoints and the HuggingFace Hub (for first-time
  checkpoint downloads) both require network access.

Author: ThermoKP Team
License: MIT
"""

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple, cast

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import requests
import torch
import transformers
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
import huggingface_hub.utils.logging as hf_hub_logging
import transformers.utils.logging as hf_utils_logging

transformers.logging.set_verbosity_error()
hf_utils_logging.disable_progress_bar()
hf_hub_logging.set_verbosity_error()
logging.getLogger("urllib3").setLevel(logging.WARNING)

torch.set_num_threads(1)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ═══════════════════════════════════════════════════════════════════════════
#  Logging Configuration
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Constants & Configuration
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EMBEDDING_CACHE_ROOT = PROJECT_ROOT / "data" / "cache"
UNIPROT_SEQUENCE_CACHE_DIR = EMBEDDING_CACHE_ROOT / "uniprot_sequences"
UNIPROT_FEATURE_CACHE_DIR = EMBEDDING_CACHE_ROOT / "uniprot_features"
PROTEIN_EMBEDDING_CACHE_DIR = EMBEDDING_CACHE_ROOT / "protein_embeddings"
LIGAND_EMBEDDING_CACHE_DIR = EMBEDDING_CACHE_ROOT / "ligand_embeddings"
CATALYTIC_SITE_CACHE_DIR = EMBEDDING_CACHE_ROOT / "catalytic_site_masks"
MUTATION_LOG_ODDS_CACHE_DIR = EMBEDDING_CACHE_ROOT / "mutation_log_odds"
PDB_CACHE_DIR = EMBEDDING_CACHE_ROOT / "pdbs"

UNIPROT_FASTA_URL_TEMPLATE = "https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
UNIPROT_JSON_URL_TEMPLATE = "https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
UNIPARC_SEARCH_URL_TEMPLATE = "https://rest.uniprot.org/uniparc/search?query={uniprot_id}&format=json"

AFDB_URL_TEMPLATE: str = "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb"

ESMFOLD_API_URL: str = "https://api.esmatlas.com/foldSequence/v1/pdb/"
ESMFOLD_MAX_SEQ_LEN: int = 400

# ═══════════════════════════════════════════════════════════════════════════
#  External API Rate Limiting
# ═══════════════════════════════════════════════════════════════════════════

class _RateLimiter:
    """Thread-safe "at most one request every `min_interval_s`" pacer, with an optional concurrency cap.

    Attributes
    ----------
    _min_interval_s : float
        Minimum number of seconds required between the start of two
        consecutive requests.
    _lock : threading.Lock
        Guards read-modify-write access to `_last_request` across threads.
    _last_request : float
        `time.time()` timestamp of the most recently permitted request.
    _semaphore : threading.Semaphore or None
        Caps the number of requests in flight concurrently when
        `max_concurrent` is given; None if no concurrency cap is configured.
    """

    def __init__(self, min_interval_s: float, max_concurrent: Optional[int] = None):
        self._min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._last_request = 0.0
        self._semaphore = threading.Semaphore(max_concurrent) if max_concurrent else None

    def acquire(self) -> None:
        """Block until pacing (and, if configured, concurrency) allows a request."""
        if self._semaphore is not None:
            self._semaphore.acquire()
        with self._lock:
            wait = self._min_interval_s - (time.time() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.time()

    def release(self) -> None:
        """Release the concurrency slot acquired via `acquire` (no-op if unconfigured)."""
        if self._semaphore is not None:
            self._semaphore.release()

    def __enter__(self) -> "_RateLimiter":
        self.acquire()
        return self

    def __exit__(self, *_exc_info) -> None:
        self.release()


uniprot_limiter = _RateLimiter(min_interval_s=0.15)

alphafold_limiter = _RateLimiter(min_interval_s=0.1)
esmfold_limiter = _RateLimiter(min_interval_s=1.0, max_concurrent=2)

_uniprot_concurrency = threading.Semaphore(3)


def _uniprot_get(url: str, max_retries: int = 3) -> Optional[requests.Response]:
    """Rate-limited, concurrency-capped GET with retries, so transient
    429/5xx/timeouts under bulk fetching don't get misread as a
    permanently-missing accession.
    """
    for attempt in range(max_retries):
        try:
            with uniprot_limiter, _uniprot_concurrency:
                response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response
            if response.status_code in (429, 503, 504) and attempt < max_retries - 1:
                time.sleep(1.0 * (2 ** attempt))
                continue
            return response
        except requests.exceptions.RequestException:
            if attempt == max_retries - 1:
                return None
            time.sleep(1.0 * (2 ** attempt))
    return None

# UniProt sequence-feature `type` values kept as catalytic-site signal,
# mapped to their column index in the mask returned by
# `get_catalytic_site_mask` (0 = active site, 1 = binding site).
CATALYTIC_FEATURE_TYPES: dict[str, int] = {
    "Active site": 0,
    "Binding site": 1,
}
NUM_CATALYTIC_FEATURE_TYPES: int = len(CATALYTIC_FEATURE_TYPES)

# ESM2-650M: 1280-dim per-residue embeddings, 33 transformer layers.
ESM2_MODEL_NAME: str = "facebook/esm2_t33_650M_UR50D"
ESM2_EMBED_DIM: int = 1280
ESM2_MAX_SEQ_LEN: int = 1022  # max_position_embeddings (1026) - <cls>/<eos>

# ChemBERTa-2 (DeepChem, masked-LM checkpoint): 384-dim SMILES token embeddings.
CHEMBERTA_MODEL_NAME: str = "DeepChem/ChemBERTa-77M-MLM"
CHEMBERTA_EMBED_DIM: int = 384
CHEMBERTA_MAX_SEQ_LEN: int = 512  # safety margin under max_position_embeddings (515)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_device(device: torch.device) -> None:
    """Set the target device for pretrained models. Used by multi-GPU/TPU accelerate deployments.

    Parameters
    ----------
    device : torch.device
        Device that subsequently loaded/used ESM2 and ChemBERTa-2 models
        run on.
    """
    global _DEVICE
    _DEVICE = device
    logger.info(f"PretrainedEmbedder device set to {_DEVICE}")

# ═══════════════════════════════════════════════════════════════════════════
#  Lazy Model Loading
# ═══════════════════════════════════════════════════════════════════════════

_esm2_tokenizer: Optional[PreTrainedTokenizerBase] = None
_esm2_model: Optional[PreTrainedModel] = None
_chemberta_tokenizer: Optional[PreTrainedTokenizerBase] = None
_chemberta_model: Optional[PreTrainedModel] = None
_esm2_lock = threading.Lock()
_chemberta_lock = threading.Lock()


def _get_esm2() -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Lazily load and cache the ESM2 tokenizer and model.

    Loaded via `AutoModelForMaskedLM` (not the plain `AutoModel`) so the
    same model instance serves both per-residue embeddings (via
    `output_hidden_states=True` - see `get_protein_embedding`) and the
    masked-LM logits used to score point-mutation effects (see
    `get_mutation_log_odds`), rather than holding two separate 650M-
    parameter checkpoints in memory.

    Returns
    -------
    tuple of (PreTrainedTokenizerBase, PreTrainedModel)
        The ESM2 tokenizer and model, moved to `_DEVICE` and in eval mode.
    """
    global _esm2_tokenizer, _esm2_model
    if _esm2_model is None:
        with _esm2_lock:
            if _esm2_model is None:
                logger.info(f"Loading ESM2 checkpoint '{ESM2_MODEL_NAME}'...")
                _esm2_tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)
                # pyrefly: ignore [not-callable]
                model = AutoModelForMaskedLM.from_pretrained(ESM2_MODEL_NAME).to(_DEVICE)
                model.eval()
                assert model.config.hidden_size == ESM2_EMBED_DIM, (
                    f"ESM2 checkpoint '{ESM2_MODEL_NAME}' hidden_size "
                    f"({model.config.hidden_size}) does not match the "
                    f"expected ESM2_EMBED_DIM ({ESM2_EMBED_DIM}); the "
                    f"adapter layers in src/encoders/multimodal_encoder.py "
                    f"must be updated."
                )
                _esm2_model = model
    assert _esm2_tokenizer is not None and _esm2_model is not None
    return _esm2_tokenizer, _esm2_model


def _get_chemberta() -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Lazily load and cache the ChemBERTa-2 tokenizer and model.

    Returns
    -------
    tuple of (PreTrainedTokenizerBase, PreTrainedModel)
        The ChemBERTa-2 tokenizer and model, moved to `_DEVICE` and in eval mode.
    """
    global _chemberta_tokenizer, _chemberta_model
    if _chemberta_model is None:
        with _chemberta_lock:
            if _chemberta_model is None:
                logger.info(f"Loading ChemBERTa-2 checkpoint '{CHEMBERTA_MODEL_NAME}'...")
                _chemberta_tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL_NAME)
                # pyrefly: ignore [not-callable]
                model = AutoModel.from_pretrained(CHEMBERTA_MODEL_NAME).to(_DEVICE)
                model.eval()
                assert model.config.hidden_size == CHEMBERTA_EMBED_DIM, (
                    f"ChemBERTa checkpoint '{CHEMBERTA_MODEL_NAME}' "
                    f"hidden_size ({model.config.hidden_size}) does not "
                    f"match the expected CHEMBERTA_EMBED_DIM "
                    f"({CHEMBERTA_EMBED_DIM}); the adapter layers in "
                    f"src/encoders/multimodal_encoder.py must be updated."
                )
                _chemberta_model = model
    assert _chemberta_tokenizer is not None and _chemberta_model is not None
    return _chemberta_tokenizer, _chemberta_model


# ═══════════════════════════════════════════════════════════════════════════
#  UniProt Sequence Fetching
# ═══════════════════════════════════════════════════════════════════════════

def _fetch_uniparc_fallback_sequence(uniprot_id: str) -> Optional[str]:
    """Look up a sequence in UniParc for accessions no longer resolvable via uniprotkb/.

    Secondary/demerged/obsolete UniProtKB accessions (e.g. Q988H5) 404 against
    the uniprotkb REST endpoint but their sequence is still archived in
    UniParc under a cross-reference. Matches are confirmed against
    `uniProtKBAccessions` before use, since the UniParc search query is
    unfielded free text.

    Parameters
    ----------
    uniprot_id : str
        UniProt accession to look up in UniParc.

    Returns
    -------
    str or None
        The archived sequence if a confirmed cross-reference is found,
        otherwise None.
    """
    url = UNIPARC_SEARCH_URL_TEMPLATE.format(uniprot_id=uniprot_id)
    response = _uniprot_get(url)
    if response is None or response.status_code != 200:
        return None
    try:
        payload = response.json()
    except Exception:
        return None

    for entry in payload.get("results", []):
        accessions = (acc.split(".")[0] for acc in entry.get("uniProtKBAccessions", []))
        if uniprot_id in accessions:
            sequence = entry.get("sequence", {}).get("value")
            if sequence:
                return sequence
    return None


def fetch_uniprot_sequence(uniprot_id: str) -> Optional[str]:
    """Fetch (or load cached) the full-length canonical sequence for a UniProt ID.

    Falls back to UniParc when the accession no longer resolves directly
    against uniprotkb/ (e.g. it was merged/demerged into another entry).
    A successful fetch is cached to `UNIPROT_SEQUENCE_CACHE_DIR` as a plain
    text file, keyed by accession, so repeated records for the same protein
    across the dataset never re-trigger a network fetch.

    Parameters
    ----------
    uniprot_id : str
        UniProt accession to fetch the sequence for.

    Returns
    -------
    str or None
        The full-length precursor sequence (one-letter amino-acid codes),
        or None if it could not be resolved via either the UniProtKB FASTA
        endpoint or the UniParc fallback.
    """
    cache_path = UNIPROT_SEQUENCE_CACHE_DIR / f"{uniprot_id}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8") or None

    url = UNIPROT_FASTA_URL_TEMPLATE.format(uniprot_id=uniprot_id)
    sequence = None
    response = _uniprot_get(url)
    if response is not None and response.status_code == 200:
        lines = response.text.strip().splitlines()
        sequence = "".join(line.strip() for line in lines if not line.startswith(">")) or None

    if sequence is None:
        sequence = _fetch_uniparc_fallback_sequence(uniprot_id)

    if not sequence:
        return None

    UNIPROT_SEQUENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = UNIPROT_SEQUENCE_CACHE_DIR / f".tmp-{uuid.uuid4().hex}.txt"
    tmp_path.write_text(sequence, encoding="utf-8")
    os.replace(tmp_path, cache_path)
    return sequence


def _fetch_uniprot_features_json(uniprot_id: str) -> dict:
    """Fetch (or load cached) the raw JSON features payload for a UniProt entry.

    Parameters
    ----------
    uniprot_id : str
        UniProt accession to fetch the JSON features payload for.

    Returns
    -------
    dict
        The parsed UniProt REST JSON response (containing, among other
        keys, a `"features"` list), or an empty dict if the entry could
        not be fetched or parsed.
    """
    cache_path = UNIPROT_FEATURE_CACHE_DIR / f"{uniprot_id}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    url = UNIPROT_JSON_URL_TEMPLATE.format(uniprot_id=uniprot_id)
    response = _uniprot_get(url)
    if response is None or response.status_code != 200:
        return {}
    try:
        payload = response.json()
    except Exception:
        return {}

    UNIPROT_FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = UNIPROT_FEATURE_CACHE_DIR / f".tmp-{uuid.uuid4().hex}.json"
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp_path, cache_path)
    return payload


def _fetch_uniprot_features(uniprot_id: str) -> list[dict]:
    """Fetch curated Active site / Binding site sequence features for a UniProt entry.

    Parameters
    ----------
    uniprot_id : str
        UniProt accession to fetch catalytic-site features for.

    Returns
    -------
    list of dict
        One dict per retained feature, each with keys ``"type"`` (one of
        `CATALYTIC_FEATURE_TYPES`), ``"start"``, and ``"end"`` (1-based,
        inclusive residue positions). Features with a missing or malformed
        location are dropped.
    """
    payload = _fetch_uniprot_features_json(uniprot_id)
    features = []
    for feature in payload.get("features", []):
        feature_type = feature.get("type")
        if feature_type not in CATALYTIC_FEATURE_TYPES:
            continue
        location = feature.get("location", {})
        start = location.get("start", {}).get("value")
        end = location.get("end", {}).get("value")
        if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
            continue
        features.append({"type": feature_type, "start": start, "end": end})
    return features


def fetch_uniprot_cleavage_offset(uniprot_id: str) -> int:
    """Calculate the exact N-terminal sequence offset due to signal/transit peptides.

    Parses 'Signal', 'Transit', 'Propeptide', and 'Initiator methionine' features
    starting from residue 1 to determine the mature sequence start index.

    Parameters
    ----------
    uniprot_id : str
        UniProt accession to compute the cleavage offset for.

    Returns
    -------
    int
        Number of N-terminal residues cleaved to obtain the mature
        sequence; 0 if no N-terminal cleavage is annotated.
    """
    payload = _fetch_uniprot_features_json(uniprot_id)
    cleavages = []
    for f in payload.get("features", []):
        if f.get("type") in ("Signal", "Transit", "Propeptide", "Initiator methionine"):
            start = f.get("location", {}).get("start", {}).get("value")
            end = f.get("location", {}).get("end", {}).get("value")
            if isinstance(start, int) and isinstance(end, int):
                cleavages.append((start, end))

    offset = 0
    current_start = 1
    while True:
        found = False
        for s, e in cleavages:
            if s == current_start:
                offset = max(offset, e)
                current_start = e + 1
                found = True
        if not found:
            break
    return offset


# ═══════════════════════════════════════════════════════════════════════════
#  Structure Fetching (AlphaFold / ESMFold)
# ═══════════════════════════════════════════════════════════════════════════

def check_alphafold_structure_exists(uniprot_id: str) -> bool:
    """Check (rate-limited) whether AlphaFold DB has a model for a UniProt ID.

    A plain HEAD request against the static file host, called from
    `dataset_validator.py`'s structure-availability check.

    Parameters
    ----------
    uniprot_id : str
        The UniProt accession.

    Returns
    -------
    bool
        True if AlphaFold DB has a model (HTTP 200), False otherwise
        (404, timeout, or any other error).
    """
    url = AFDB_URL_TEMPLATE.format(uniprot_id=uniprot_id)
    try:
        with alphafold_limiter:
            response = requests.head(url, timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def fetch_esmfold_pdb(uniprot_id: str, sequence: str, dest_dir: Path) -> Optional[Path]:
    """Predict and cache a structure via the ESM Atlas single-sequence folding API.

    Used as a fallback source for `uniprot_id`s that have no AlphaFold DB
    model (e.g. very recent, obscure, or non-Swiss-Prot entries), matching
    CatPred's own AlphaFold/ESMFold approach. Synchronous by design so it
    can be called identically from `dataset_validator.py`'s thread-pool-
    based validation and `geometry_processor.py`'s `asyncio.to_thread`-
    wrapped download step.

    Parameters
    ----------
    uniprot_id : str
        The UniProt accession, used only for the destination filename so
        downstream code treats it identically to an AlphaFold-sourced PDB.
    sequence : str
        The full-length (precursor) amino-acid sequence to fold.
    dest_dir : Path
        Destination directory (`PDB_CACHE_DIR`).

    Returns
    -------
    Path or None
        The path to the saved PDB file, or None if the sequence exceeds
        `ESMFOLD_MAX_SEQ_LEN` or the API call failed.
    """
    dest_path = dest_dir / f"{uniprot_id}.pdb"
    if dest_path.exists():
        return dest_path

    if len(sequence) > ESMFOLD_MAX_SEQ_LEN:
        logger.info(
            "Sequence for %s (%d residues) exceeds ESMFold's %d-residue limit.",
            uniprot_id, len(sequence), ESMFOLD_MAX_SEQ_LEN,
        )
        return None

    try:
        with esmfold_limiter:
            response = requests.post(ESMFOLD_API_URL, data=sequence, timeout=120)
        if response.status_code != 200 or not response.text.strip():
            logger.warning("ESMFold API call failed for %s (HTTP %s)", uniprot_id, response.status_code)
            return None
        dest_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_dir / f".tmp-{uuid.uuid4().hex}.pdb"
        tmp_path.write_text(response.text, encoding="utf-8")
        os.replace(tmp_path, dest_path)
        return dest_path
    except Exception as e:
        logger.warning("Error calling ESMFold API for %s: %s", uniprot_id, e)
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  Cache Persistence
# ═══════════════════════════════════════════════════════════════════════════

def _atomic_save(tensor: torch.Tensor, cache_path: Path, cache_dir: Path) -> None:
    """Write a tensor to `cache_path` atomically via a temp-file rename.

    geometry_processor.py processes many database entries concurrently on a
    worker thread pool; multiple threads can race to embed the same
    UniProt ID or SMILES. A write-then-rename avoids a torn/corrupt cache
    file if two threads' writes interleave on the same path.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to persist.
    cache_path : Path
        Final destination path.
    cache_dir : Path
        Parent directory, created if missing.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_dir / f".tmp-{uuid.uuid4().hex}.pt"
    torch.save(tensor, tmp_path)
    os.replace(tmp_path, cache_path)


# ═══════════════════════════════════════════════════════════════════════════
#  Embedding Extraction
# ═══════════════════════════════════════════════════════════════════════════

def get_protein_embedding(uniprot_id: str, sequence: str, cache_key: Optional[str] = None) -> torch.Tensor:
    """Compute (or load cached) per-residue ESM2 embeddings for a sequence.

    Parameters
    ----------
    uniprot_id : str
        The UniProt accession.
    sequence : str
        The full-length amino-acid sequence to embed. Must match the
        sequence the caller will index into (e.g. residue-order-derived
        from the same AlphaFold model), since positions are returned in
        input order.
    cache_key : str, optional
        Disk-cache key. Defaults to `uniprot_id`. Callers embedding a
        *mutated* sequence for the same UniProt ID (see
        geometry_processor.py's conservative point-mutation handling) must
        pass a distinct key (e.g. ``f"{uniprot_id}_{mutation_code}"``) -
        otherwise a mutant embedding could be cached under the wild-type's
        plain `uniprot_id` key (or vice versa) and silently reused for the
        other, since embeddings are never re-derived from `sequence` once a
        cache file exists.

    Returns
    -------
    torch.Tensor
        Per-residue embeddings of shape (len(sequence), ESM2_EMBED_DIM).

    Raises
    ------
    ValueError
        If `sequence` exceeds ESM2_MAX_SEQ_LEN residues.
    """
    safe_cache_key = (cache_key or uniprot_id).replace("/", "-")
    cache_path = PROTEIN_EMBEDDING_CACHE_DIR / f"{safe_cache_key}.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    if len(sequence) > ESM2_MAX_SEQ_LEN:
        raise ValueError(
            f"Sequence for {uniprot_id} has {len(sequence)} residues, "
            f"exceeding ESM2's {ESM2_MAX_SEQ_LEN}-residue context window. "
            f"dataset_validator.py should have dropped this entry."
        )

    tokenizer, model = _get_esm2()
    inputs = tokenizer(sequence, return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        # pyrefly: ignore [not-callable]
        outputs = model(**inputs, output_hidden_states=True)

    embedding = outputs.hidden_states[-1][0, 1:-1, :].to("cpu")
    if embedding.size(0) != len(sequence):
        raise RuntimeError(
            f"ESM2 output length ({embedding.size(0)}) does not match "
            f"input sequence length ({len(sequence)}) for {uniprot_id}."
        )

    _atomic_save(embedding, cache_path, PROTEIN_EMBEDDING_CACHE_DIR)
    return embedding


def _get_masked_position_log_probs(uniprot_id: str, full_sequence: str, seq_idx: int) -> torch.Tensor:
    """Compute (or load cached) ESM2 masked-LM log-probabilities at one position.

    Uses the masked-marginal method (Meier et al., ESM-1v): the residue at
    `seq_idx` is replaced with the mask token before the forward pass, so
    the model must infer that position purely from its surrounding
    context rather than from its own identity leaking through the input.
    This is the more accurate of ESM's two standard zero-shot scoring
    conventions for a single point substitution, at the cost of one
    forward pass per distinct position (versus one per protein for the
    wildtype-marginal alternative).

    Cached per (uniprot_id, seq_idx) rather than per mutation, since every
    substitution proposed at the same position shares the same masked
    forward pass.

    Parameters
    ----------
    uniprot_id : str
        The UniProt accession, used only for the cache key and error
        messages.
    full_sequence : str
        The wild-type full-length (precursor) sequence, before any
        mutation is applied and before the signal peptide is chopped -
        `seq_idx` indexes into this sequence directly.
    seq_idx : int
        0-based residue index (into `full_sequence`) to mask and score.

    Returns
    -------
    torch.Tensor
        Shape (vocab_size,) log-probabilities at the masked position.

    Raises
    ------
    ValueError
        If `full_sequence` exceeds ESM2_MAX_SEQ_LEN residues.
    """
    cache_path = MUTATION_LOG_ODDS_CACHE_DIR / f"{uniprot_id}_pos{seq_idx}.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    if len(full_sequence) > ESM2_MAX_SEQ_LEN:
        raise ValueError(
            f"Sequence for {uniprot_id} has {len(full_sequence)} residues, "
            f"exceeding ESM2's {ESM2_MAX_SEQ_LEN}-residue context window. "
            f"dataset_validator.py should have dropped this entry."
        )

    tokenizer, model = _get_esm2()
    inputs = tokenizer(full_sequence, return_tensors="pt").to(_DEVICE)
    # +1 to skip the leading <cls> token, matching get_protein_embedding's alignment.
    inputs["input_ids"][0, seq_idx + 1] = cast(int, tokenizer.mask_token_id)
    with torch.no_grad():
        # pyrefly: ignore [not-callable]
        outputs = model(**inputs)
    log_probs = torch.log_softmax(outputs.logits[0, seq_idx + 1, :], dim=-1).to("cpu")

    _atomic_save(log_probs, cache_path, MUTATION_LOG_ODDS_CACHE_DIR)
    return log_probs


def get_mutation_log_odds(uniprot_id: str, full_sequence: str, seq_idx: int, wt_res: str, mut_res: str) -> float:
    """Masked-marginal log-odds score for a point mutation.

    Computes ``log P(mut_res | masked context) - log P(wt_res | masked
    context)`` from ESM2's own masked-LM head - a standard zero-shot
    variant-effect proxy (Meier et al., ESM-1v). Large negative values
    flag substitutions the model considers structurally or functionally
    disruptive at that position; values near zero indicate the model
    considers the two residues roughly interchangeable in this context.

    Parameters
    ----------
    uniprot_id : str
        The UniProt accession.
    full_sequence : str
        The wild-type full-length (precursor) sequence.
    seq_idx : int
        0-based residue index (into `full_sequence`) of the mutation.
    wt_res, mut_res : str
        Single-letter wild-type and mutant amino-acid codes.

    Returns
    -------
    float
        The log-odds score.
    """
    log_probs = _get_masked_position_log_probs(uniprot_id, full_sequence, seq_idx)
    tokenizer, _ = _get_esm2()
    wt_id = tokenizer.convert_tokens_to_ids(wt_res)
    mut_id = tokenizer.convert_tokens_to_ids(mut_res)
    return (log_probs[mut_id] - log_probs[wt_id]).item()


def get_catalytic_site_mask(uniprot_id: str, seq_len: int, offset: int = 0) -> torch.Tensor:
    """Compute (or load cached) a per-residue catalytic-site mask aligned to the mature sequence.

    Parameters
    ----------
    uniprot_id : str
        The UniProt accession, used as the disk-cache key.
    seq_len : int
        Length of the sequence this mask must align with (e.g. len of mature sequence).
    offset : int
        N-terminal sequence offset (due to cleaved signal peptides) that the
        biological feature indices must be shifted by to align with the sequence.

    Returns
    -------
    torch.Tensor
        A float tensor of shape (seq_len, NUM_CATALYTIC_FEATURE_TYPES).
    """
    cache_path = CATALYTIC_SITE_CACHE_DIR / f"{uniprot_id}_offset{offset}.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    mask = torch.zeros((seq_len, NUM_CATALYTIC_FEATURE_TYPES), dtype=torch.float)
    for feature in _fetch_uniprot_features(uniprot_id):
        # Shift 1-based biological indices to 0-based array indices, applying the offset
        start = feature["start"] - 1 - offset
        end = feature["end"] - offset
        
        start = max(0, start)
        end = min(seq_len, end)
        if start < end:
            col = CATALYTIC_FEATURE_TYPES[feature["type"]]
            mask[start:end, col] = 1.0

    _atomic_save(mask, cache_path, CATALYTIC_SITE_CACHE_DIR)
    return mask


def get_ligand_embedding(smiles: str) -> torch.Tensor:
    """Compute (or load cached) a pooled whole-molecule ChemBERTa embedding.

    Parameters
    ----------
    smiles : str
        The canonical SMILES string of the molecule (or a dot-separated
        multi-component SMILES for combined co-substrates).

    Returns
    -------
    torch.Tensor
        A single pooled embedding of shape (CHEMBERTA_EMBED_DIM,), to be
        broadcast onto every atom of the molecule.
    """
    cache_key = hashlib.sha256(smiles.encode("utf-8")).hexdigest()
    cache_path = LIGAND_EMBEDDING_CACHE_DIR / f"{cache_key}.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    tokenizer, model = _get_chemberta()
    inputs = tokenizer(
        smiles, return_tensors="pt", truncation=True, max_length=CHEMBERTA_MAX_SEQ_LEN
    ).to(_DEVICE)
    with torch.no_grad():
        # pyrefly: ignore [not-callable]
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state[0]
    mask = inputs["attention_mask"][0].unsqueeze(-1).to(hidden.dtype)
    embedding = (hidden * mask).sum(dim=0) / mask.sum().clamp(min=1.0)
    embedding = embedding.to("cpu")

    _atomic_save(embedding, cache_path, LIGAND_EMBEDDING_CACHE_DIR)
    return embedding
