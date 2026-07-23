"""
===========================================================================
Geometry Processor
Description: 3D Structural Feature Extraction (AlphaFold/ESMFold + P2Rank)
===========================================================================

Workflow:
1. Download an AlphaFold v6 PDB file into data/cache/pdbs, falling back to
   an ESM Atlas-predicted structure (fetch_esmfold_pdb) for UniProt IDs
   AlphaFold has no model for.
2. Run P2Rank (-c alphafold config) to predict the highest-ranked binding
   pocket and crop protein atoms within a 10 A radius of its center.
3. Featurize pocket atoms (element, residue identity, backbone flag) and
   gather the mature-sequence-aligned ESM2 embedding/catalytic-site mask
   already computed by generate_tensors.py's 2D pipeline onto them by
   residue index - no second ESM2 forward pass.
4. Generate 3D conformers (RDKit ETKDG + MMFF) for the same ligand/
   co-substrate RDKit Mol objects the 2D D-MPNN already featurizes.
5. Build bipartite interacts_with edges (6 A cutoff, KD-tree) between
   protein_pocket_atoms, ligand_atoms, and co_substrate_atoms.

Known Caveats:
- Requires a P2Rank installation at tools/p2rank/prank (not distributed
  with this repository); predict_pocket logs an error and returns None
  if missing.
- AlphaFold models number residues 1..L matching the full UniProt
  precursor sequence with no gaps; residue_index_map is still built from
  parsed structure order rather than assumed from resseq arithmetic.
- Mutant entries approximate the true (unavailable) mutant structure with
  the wild-type AlphaFold/ESMFold coordinates - only the identity/ESM2/
  catalytic-mask features at the mutated position reflect the mutation.

Author: ThermoKP Team
License: MIT
"""

import asyncio
import logging
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import torch
from Bio.PDB import PDBParser
from scipy.spatial import KDTree
from torch_geometric.data import HeteroData

import rdkit.Chem as Chem
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

from src.data.processors.pretrained_embeddings import (
    AFDB_URL_TEMPLATE,
    PDB_CACHE_DIR,
    alphafold_limiter,
    fetch_esmfold_pdb,
)

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
P2RANK_EXECUTABLE = PROJECT_ROOT / "tools" / "p2rank" / "prank"
CROP_RADIUS_ANGSTROM = 10.0
INTERACTION_CUTOFF_ANGSTROM = 6.0

PROTEIN_ELEMENTS = ["C", "N", "O", "S"]
AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}

PROTEIN_POCKET_ONEHOT_DIM = len(PROTEIN_ELEMENTS) + len(AMINO_ACIDS) + 1  # + backbone flag
NUM_CATALYTIC_FEATURE_TYPES = 2  # duplicated from pretrained_embeddings.py

_P2RANK_SEMAPHORE = threading.Semaphore(4)


# ═══════════════════════════════════════════════════════════════════════════
#  PDB Fetching
# ═══════════════════════════════════════════════════════════════════════════
async def download_pdb(uniprot_id: str, mature_sequence: str) -> Optional[Path]:
    """Download (or load cached) an AlphaFold PDB, falling back to ESMFold.

    Parameters
    ----------
    uniprot_id : str
        The UniProt accession.
    mature_sequence : str
        Mature-sequence fallback input for `fetch_esmfold_pdb` if AlphaFold
        has no model for `uniprot_id`.

    Returns
    -------
    Path or None
        Path to the cached PDB file, or None if both sources failed.
    """
    dest_path = PDB_CACHE_DIR / f"{uniprot_id}.pdb"
    if dest_path.exists():
        return dest_path

    url = AFDB_URL_TEMPLATE.format(uniprot_id=uniprot_id)

    def fetch() -> requests.Response:
        with alphafold_limiter:
            return requests.get(url, timeout=30)

    try:
        response = await asyncio.to_thread(fetch)
        if response.status_code == 200:
            PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(response.content)
            return dest_path
    except Exception as e:
        logger.error(f"Error downloading {uniprot_id} from AFDB: {e}")

    return await asyncio.to_thread(fetch_esmfold_pdb, uniprot_id, mature_sequence, PDB_CACHE_DIR)


# ═══════════════════════════════════════════════════════════════════════════
#  Pocket Prediction & Cropping
# ═══════════════════════════════════════════════════════════════════════════
def predict_pocket(pdb_path: Path) -> Optional[Tuple[float, float, float]]:
    """Predict the highest-ranked binding pocket center using P2Rank.

    Parameters
    ----------
    pdb_path : Path
        Path to the protein structure PDB file to run P2Rank against.

    Returns
    -------
    tuple of (float, float, float) or None
        (center_x, center_y, center_z) of the top pocket, or None on
        failure or if no ligandable pocket was found.
    """
    if not P2RANK_EXECUTABLE.exists():
        logger.error(f"P2Rank executable not found at {P2RANK_EXECUTABLE}")
        return None

    pdb_filename = pdb_path.name
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / f"predict_{pdb_filename}"
        cmd = [
            str(P2RANK_EXECUTABLE.absolute()),
            "predict",
            "-c", "alphafold",
            "-f", str(pdb_path.absolute()),
            "-o", str(output_dir.absolute()),
            "-threads", "1",
        ]

        with _P2RANK_SEMAPHORE:
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, check=True, timeout=300, cwd=tmpdir
                )

                predictions_csv = output_dir / f"{pdb_filename}_predictions.csv"
                if not predictions_csv.exists():
                    logger.warning(
                        "P2Rank produced no predictions file for %s; last output:\n%s",
                        pdb_filename, result.stdout[-2000:],
                    )
                    return None

                df = pd.read_csv(predictions_csv)
                df.columns = df.columns.str.strip()
                if df.empty:
                    logger.info("P2Rank found no ligandable pocket for %s.", pdb_filename)
                    return None

                top_pocket = df.iloc[0]
                return float(top_pocket["center_x"]), float(top_pocket["center_y"]), float(top_pocket["center_z"])

            except subprocess.CalledProcessError as e:
                logger.error(
                    "P2Rank execution failed for %s (exit %s):\n%s",
                    pdb_filename, e.returncode, (e.stderr or "")[-2000:],
                )
                return None
            except Exception as e:
                logger.error(f"Pocket prediction failed for {pdb_filename}: {e}")
                return None


def extract_full_sequence(pdb_path: Path) -> Tuple[str, Dict[Tuple[str, int, str], int]]:
    """Extract the full-chain sequence and a residue-key-to-structure-index map.

    Parameters
    ----------
    pdb_path : Path
        Path to the protein structure PDB file to parse.

    Returns
    -------
    tuple of (str, dict)
        The one-letter sequence (non-standard residues mapped to 'X'), and
        a dict mapping each residue's (chain_id, resseq, icode) key to its
        zero-based structure-order index.
    """
    parser = PDBParser(QUIET=True)
    with open(pdb_path, "r", encoding="utf-8", errors="replace") as f:
        structure = parser.get_structure("protein", f)

    sequence_chars: List[str] = []
    residue_index_map: Dict[Tuple[str, int, str], int] = {}

    for residue in structure.get_residues():
        het_flag, resseq, icode = residue.id
        if het_flag.strip():
            continue
        chain_id = residue.get_parent().id
        one_letter = THREE_TO_ONE.get(residue.resname.upper(), "X")
        residue_index_map[(chain_id, resseq, icode)] = len(sequence_chars)
        sequence_chars.append(one_letter)

    return "".join(sequence_chars), residue_index_map


def crop_atoms(pdb_path: Path, center: Tuple[float, float, float], radius: float) -> List[Any]:
    """Filter PDB atoms to retain only those within `radius` of `center`.

    Parameters
    ----------
    pdb_path : Path
        Path to the protein structure PDB file to parse.
    center : tuple of (float, float, float)
        Cartesian coordinates of the pocket center (e.g. from `predict_pocket`).
    radius : float
        Cropping radius in Angstroms; atoms further than this from `center`
        are discarded.

    Returns
    -------
    list
        Bio.PDB `Atom` objects within `radius` of `center`, in structure order.
    """
    parser = PDBParser(QUIET=True)
    with open(pdb_path, "r", encoding="utf-8", errors="replace") as f:
        structure = parser.get_structure("protein", f)

    atoms = list(structure.get_atoms())
    if not atoms:
        return []

    coords = [atom.coord for atom in atoms]
    kdtree = KDTree(coords)
    indices = kdtree.query_ball_point(center, r=radius)
    return [atoms[i] for i in indices]


# ═══════════════════════════════════════════════════════════════════════════
#  Featurization
# ═══════════════════════════════════════════════════════════════════════════
def _one_hot(val: Any, choices: List[Any]) -> List[float]:
    encoding = [0.0] * len(choices)
    if val in choices:
        encoding[choices.index(val)] = 1.0
    return encoding


def _featurize_protein_pocket(
    atoms: List[Any],
    residue_index_map: Dict[Tuple[str, int, str], int],
    structure_offset: int,
    mature_sequence: str,
    catalytic_site_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Featurize cropped pocket atoms, gathering the mature-sequence catalytic-site mask.

    The per-residue ESM2 embedding is deliberately NOT gathered/duplicated here.
    Since ESM2 is 1280-dim and a residue typically has 5-15 atoms, storing it
    per-atom would multiply the on-disk/in-memory tensor size several-fold for
    no benefit - every atom of a residue would carry an identical copy of the
    same vector. Instead, each atom's mature-sequence residue index is returned
    so `generate_tensors.py` can wire it as a `residue_of` edge to
    `protein_sequence`, letting `StructuralEGNNEncoder` gather the embedding
    directly from the (already-batched) `protein_sequence.esm2_embedding` at
    forward time - PyG's own edge_index batching correctly offsets this
    cross-node-type reference per graph, so no custom collation logic is needed.

    Parameters
    ----------
    atoms : list of Bio.PDB.Atom.Atom
        Retained (post-crop) protein atoms.
    residue_index_map : dict
        (chain_id, resseq, icode) -> zero-based structure-order index, as
        returned by `extract_full_sequence`.
    structure_offset : int
        Signal-peptide offset (`fetch_uniprot_cleavage_offset`) mapping a
        structure-order index to the mature-sequence index used by
        `mature_sequence`/`catalytic_site_mask`.
    mature_sequence : str
        The (possibly mutated) mature sequence generate_tensors.py already
        built for the 2D pipeline - residue identity is read from here, not
        from the atom's own (wild-type) residue name.
    catalytic_site_mask : torch.Tensor
        Per-residue catalytic-site flags, shape (len(mature_sequence), 2).

    Returns
    -------
    tuple of torch.Tensor
        (x_tensor, pos_tensor, residue_index_tensor): per-atom one-hot +
        catalytic-flag features, 3D coordinates, and each atom's
        mature-sequence residue index (for the `residue_of` edge). Atoms
        whose residue falls outside the mature sequence (cleaved signal
        peptide) are dropped.
    """
    x_features = []
    positions = []
    embed_indices = []

    for atom in atoms:
        residue = atom.get_parent()
        chain_id = residue.get_parent().id
        _het_flag, resseq, icode = residue.id
        struct_idx = residue_index_map[(chain_id, resseq, icode)]
        mature_idx = struct_idx - structure_offset
        if not (0 <= mature_idx < len(mature_sequence)):
            continue

        elem_feat = _one_hot(atom.element.upper(), PROTEIN_ELEMENTS)
        res_three = ONE_TO_THREE.get(mature_sequence[mature_idx], "")
        res_feat = _one_hot(res_three, AMINO_ACIDS)
        name = atom.get_name().strip().upper()
        is_backbone = 1.0 if name in BACKBONE_ATOMS else 0.0

        x_features.append(elem_feat + res_feat + [is_backbone])
        positions.append(atom.coord.tolist())
        embed_indices.append(mature_idx)

    if not x_features:
        return (
            torch.empty((0, PROTEIN_POCKET_ONEHOT_DIM + NUM_CATALYTIC_FEATURE_TYPES), dtype=torch.float),
            torch.empty((0, 3), dtype=torch.float),
            torch.empty((0,), dtype=torch.long),
        )

    embed_index_tensor = torch.tensor(embed_indices, dtype=torch.long)
    catalytic_flags = catalytic_site_mask[embed_index_tensor]

    x_onehot = torch.tensor(x_features, dtype=torch.float)
    x_tensor = torch.cat([x_onehot, catalytic_flags.to(torch.float)], dim=-1)
    pos_tensor = torch.tensor(positions, dtype=torch.float)
    return x_tensor, pos_tensor, embed_index_tensor


def generate_conformer(mol: Optional[Chem.Mol]) -> Optional[torch.Tensor]:
    """Generate a 3D conformer for an RDKit molecule via ETKDG + MMFF.

    Parameters
    ----------
    mol : Chem.Mol, optional
        Molecule to embed a conformer for. None or an empty molecule
        yields an empty coordinate tensor rather than an error.

    Returns
    -------
    torch.Tensor or None
        Atomic coordinates (N_atoms, 3) in Angstroms, empty (0, 3) if `mol`
        is None/empty, or None if embedding failed.
    """
    if mol is None or mol.GetNumAtoms() == 0:
        return torch.empty((0, 3), dtype=torch.float)

    mol_h = Chem.AddHs(mol)
    params = ETKDGv3()
    params.randomSeed = 42
    if EmbedMolecule(mol_h, params) == -1:
        return None

    try:
        MMFFOptimizeMolecule(mol_h, maxIters=500)
    except Exception:
        logger.warning("MMFF optimization raised an exception; using raw ETKDG geometry.")

    conf = mol_h.GetConformer()
    coords = conf.GetPositions()[: mol.GetNumAtoms()]
    return torch.tensor(coords, dtype=torch.float)


def _build_bipartite_edges(pos_src: torch.Tensor, pos_dst: torch.Tensor, cutoff: float = INTERACTION_CUTOFF_ANGSTROM) -> torch.Tensor:
    """Build bipartite edge indices between two atom sets within `cutoff` Angstroms.

    Parameters
    ----------
    pos_src : torch.Tensor
        Shape (N_src, 3) Cartesian coordinates of the source atom set.
    pos_dst : torch.Tensor
        Shape (N_dst, 3) Cartesian coordinates of the destination atom set.
    cutoff : float, optional
        Maximum distance in Angstroms for an edge to be created between a
        source and destination atom. Defaults to `INTERACTION_CUTOFF_ANGSTROM`.

    Returns
    -------
    torch.Tensor
        Shape (2, num_edges) edge index (source indices, destination
        indices), empty if either atom set is empty or no pair is within
        `cutoff`.
    """
    if pos_src.size(0) == 0 or pos_dst.size(0) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    tree_dst = KDTree(pos_dst.numpy())
    neighbour_lists = tree_dst.query_ball_point(pos_src.numpy(), r=cutoff)

    src_indices: List[int] = []
    dst_indices: List[int] = []
    for src_idx, neighbours in enumerate(neighbour_lists):
        for dst_idx in neighbours:
            src_indices.append(src_idx)
            dst_indices.append(dst_idx)

    if not src_indices:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([src_indices, dst_indices], dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point (consumed by generate_tensors.py)
# ═══════════════════════════════════════════════════════════════════════════
async def augment_hetero_graph_with_3d(
    data: HeteroData,
    uniprot_id: str,
    mature_sequence: str,
    structure_offset: int,
    catalytic_site_mask: torch.Tensor,
    primary_mol: Chem.Mol,
    co_sub_mol: Optional[Chem.Mol],
) -> HeteroData:
    """Add 3D structural node types/edges to an existing 2D HeteroData object.

    Adds a `protein_pocket_atoms` node type (cropped AlphaFold/ESMFold
    pocket, 10 A radius), 3D conformer `pos` onto the existing `ligand_atoms`/
    `co_substrate_atoms` node types, `interacts_with` bipartite edges (6 A
    cutoff) between all three, and a `residue_of` edge from each pocket atom
    to its residue in `protein_sequence` - this lets the encoder gather the
    ESM2 embedding at forward time instead of duplicating it per atom on
    disk (see `_featurize_protein_pocket`). Reuses the existing
    `covalent_bond` edges/features already present on `ligand_atoms`/
    `co_substrate_atoms` for the EGNN's intra-molecular message passing - no
    duplicate topology is stored.

    Parameters
    ----------
    data : HeteroData
        The 2D-pipeline `HeteroData` object (from `generate_tensors.py`'s
        `build_hetero_graph`) to augment in place with 3D node types/edges.
    uniprot_id : str
        UniProt accession, used to download/predict the protein structure.
    mature_sequence : str
        The (possibly mutated) mature sequence already used by the 2D
        pipeline, kept aligned with `catalytic_site_mask`.
    structure_offset : int
        Signal-peptide/propeptide cleavage offset mapping a structure-order
        residue index to the mature-sequence index.
    catalytic_site_mask : torch.Tensor
        Per-residue catalytic-site flags aligned to `mature_sequence`,
        shape (len(mature_sequence), 2).
    primary_mol : Chem.Mol
        Primary substrate molecule to generate a 3D conformer for.
    co_sub_mol : Chem.Mol, optional
        Co-substrate molecule to generate a 3D conformer for, or None if
        the record has no co-substrate.

    Returns
    -------
    HeteroData
        The same `data` object, augmented in place with `protein_pocket_atoms`
        (`x`, `pos`), `pos` on `ligand_atoms`/`co_substrate_atoms`,
        `residue_of` edges from pocket atoms to `protein_sequence`, and
        bipartite `interacts_with` edges between ligand, co-substrate, and
        pocket atoms.

    Raises
    ------
    ValueError
        If PDB download, pocket prediction, cropping, or ligand conformer
        generation fails - callers should treat this the same as any other
        tensor-generation failure (drop the entry).
    """
    pdb_path = await download_pdb(uniprot_id, mature_sequence)
    if pdb_path is None:
        raise ValueError(f"PDB download failed for {uniprot_id}.")

    pocket_center = await asyncio.to_thread(predict_pocket, pdb_path)
    if pocket_center is None:
        raise ValueError(f"Pocket prediction failed for {uniprot_id}.")

    def build_pocket() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        retained_atoms = crop_atoms(pdb_path, pocket_center, radius=CROP_RADIUS_ANGSTROM)
        if not retained_atoms:
            raise ValueError(f"No atoms found within {CROP_RADIUS_ANGSTROM}A radius for {uniprot_id}.")
        _full_sequence, residue_index_map = extract_full_sequence(pdb_path)
        return _featurize_protein_pocket(
            retained_atoms, residue_index_map, structure_offset, mature_sequence,
            catalytic_site_mask,
        )

    pocket_x, pocket_pos, pocket_residue_idx = await asyncio.to_thread(build_pocket)

    def build_conformers() -> Tuple[torch.Tensor, torch.Tensor]:
        l_pos = generate_conformer(primary_mol)
        if l_pos is None:
            raise ValueError(f"Ligand conformer generation failed for {uniprot_id}.")
        c_pos = generate_conformer(co_sub_mol)
        if c_pos is None:
            raise ValueError(f"Co-substrate conformer generation failed for {uniprot_id}.")
        return l_pos, c_pos

    ligand_pos, co_sub_pos = await asyncio.to_thread(build_conformers)

    data["protein_pocket_atoms"].x = pocket_x
    data["protein_pocket_atoms"].pos = pocket_pos
    data["ligand_atoms"].pos = ligand_pos
    data["co_substrate_atoms"].pos = co_sub_pos

    data["protein_pocket_atoms", "residue_of", "protein_sequence"].edge_index = torch.stack([
        torch.arange(pocket_x.size(0), dtype=torch.long), pocket_residue_idx,
    ])

    data["ligand_atoms", "interacts_with", "protein_pocket_atoms"].edge_index = _build_bipartite_edges(ligand_pos, pocket_pos)
    data["protein_pocket_atoms", "interacts_with", "ligand_atoms"].edge_index = _build_bipartite_edges(pocket_pos, ligand_pos)
    data["ligand_atoms", "interacts_with", "co_substrate_atoms"].edge_index = _build_bipartite_edges(ligand_pos, co_sub_pos)
    data["co_substrate_atoms", "interacts_with", "ligand_atoms"].edge_index = _build_bipartite_edges(co_sub_pos, ligand_pos)

    return data
