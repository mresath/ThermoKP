"""
===========================================================================
Interactive Structure Viewer
Description: py3Dmol rendering of the predicted enzyme structure, its
P2Rank-predicted pocket, mutated residues, and an illustrative substrate
overlay, for the Streamlit inference dashboard.
===========================================================================

Workflow:
1. Fetch (or reuse the cached) AlphaFold/ESMFold PDB for a UniProt ID.
2. Run P2Rank to find the highest-ranked binding pocket and the residues
   within its crop radius (reusing geometry_processor.py's own functions,
   not a second featurization pass).
3. Resolve any queried mutation code(s) to raw precursor-sequence indices
   (generate_tensors.resolve_mutation_positions) and map them onto PDB
   residue numbers via geometry_processor.extract_full_sequence.
4. Render a py3Dmol cartoon, highlighting the pocket and mutated residues,
   optionally overlaying a substrate's freshly generated 3D conformer
   recentered on the pocket for illustration.

Known Caveats:
- The ligand overlay is positional only (recentered on the pocket
  centroid) - it is not a docked pose, since the model performs no
  docking.
- P2Rank pocket prediction is not cached; each call reruns it once
  (a few seconds), unlike the PDB fetch itself.

Author: ThermoKP Team
License: MIT
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import rdkit.Chem as Chem
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.processors.generate_tensors import resolve_mutation_positions  # noqa: E402
from src.data.processors.geometry_processor import (  # noqa: E402
    CROP_RADIUS_ANGSTROM,
    crop_atoms,
    download_pdb,
    extract_full_sequence,
    generate_conformer,
    predict_pocket,
)
from src.data.processors.pretrained_embeddings import (  # noqa: E402
    fetch_uniprot_cleavage_offset,
    fetch_uniprot_sequence,
)

import py3Dmol  # noqa: E402

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Mirrors images/generators/_style.py's navy/magenta/teal/amber palette (not
# imported directly - that module switches matplotlib's backend and creates
# an output directory as import side effects, which the dashboard shouldn't
# trigger just to reuse four hex strings). dashboard/app.py imports these
# same constants to keep its color legend in sync with the viewer.
PROTEIN_COLOR = "#c9d1d9"    # neutral cartoon, so highlighted residues stand out
POCKET_COLOR = "#1f8f7f"     # TEAL: 3D/geometric feature (P2Rank pocket)
MUTATION_COLOR = "#b5396f"   # MAGENTA: mutant/contrast category
LIGAND_COLOR = "#c17f16"     # AMBER: substrate overlay
VIEWER_BACKGROUND = "0xffffff"


# ═══════════════════════════════════════════════════════════════════════════
#  Residue Resolution
# ═══════════════════════════════════════════════════════════════════════════
def _pocket_residues(pdb_path: Path, center: Tuple[float, float, float]) -> List[Tuple[str, int]]:
    """Return the (chain_id, resseq) pairs of residues within the pocket crop radius."""
    atoms = crop_atoms(pdb_path, center, radius=CROP_RADIUS_ANGSTROM)
    residues = set()
    for atom in atoms:
        residue = atom.get_parent()
        chain_id = residue.get_parent().id
        _het_flag, resseq, _icode = residue.id
        residues.add((chain_id, resseq))
    return sorted(residues)


def _mutation_residues(
    pdb_path: Path, full_sequence: str, offset: int, mutation_code: Optional[str]
) -> List[Tuple[str, int]]:
    """Return the (chain_id, resseq) pairs of any queried mutation sites.

    Best-effort: the mutation code was already validated once by
    thermokp.build_enzyme_substrate_graph before this ever runs, so a
    resolution failure here only skips highlighting rather than aborting
    the structure view.
    """
    if not mutation_code:
        return []

    _full_seq_struct, residue_index_map = extract_full_sequence(pdb_path)
    struct_idx_to_key = {idx: key for key, idx in residue_index_map.items()}

    try:
        mutations = resolve_mutation_positions(full_sequence, offset, mutation_code)
    except ValueError:
        logger.warning(f"Could not resolve mutation {mutation_code!r} for structure display.")
        return []

    residues = []
    for _wt_res, mut_idx, _mut_res in mutations:
        key = struct_idx_to_key.get(mut_idx)
        if key is not None:
            chain_id, resseq, _icode = key
            residues.append((chain_id, resseq))
    return residues


# ═══════════════════════════════════════════════════════════════════════════
#  Ligand Overlay
# ═══════════════════════════════════════════════════════════════════════════
def _recentered_ligand_molblock(mol: Chem.Mol, pocket_center: Tuple[float, float, float]) -> Optional[str]:
    """Generate a 3D conformer for `mol` and translate it onto the pocket center.

    Returns
    -------
    str or None
        A V2000 molblock with the recentered conformer, or None if
        conformer generation failed or `mol` has no atoms.
    """
    positions = generate_conformer(mol)
    if positions is None or positions.size(0) == 0:
        return None

    centroid = positions.mean(dim=0)
    shift = torch.tensor(pocket_center, dtype=positions.dtype) - centroid
    recentered = positions + shift

    mol_with_pos = Chem.Mol(mol)
    conformer = Chem.Conformer(mol_with_pos.GetNumAtoms())
    for atom_idx, (x, y, z) in enumerate(recentered.tolist()):
        conformer.SetAtomPosition(atom_idx, (x, y, z))
    mol_with_pos.RemoveAllConformers()
    mol_with_pos.AddConformer(conformer, assignId=True)

    return Chem.MolToMolBlock(mol_with_pos)


# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════
def build_structure_view(
    uniprot_id: str,
    mutation: Optional[str] = None,
    primary_mol: Optional[Chem.Mol] = None,
    width: int = 700,
    height: int = 500,
) -> Optional[str]:
    """Build an interactive py3Dmol structure view as standalone HTML.

    Parameters
    ----------
    uniprot_id : str
        UniProt accession whose predicted structure to render.
    mutation : str, optional
        ``/``-separated point-mutation code(s) to highlight, using the
        same codes accepted by thermokp.predict_kinetics.
    primary_mol : rdkit.Chem.Mol, optional
        The primary substrate's RDKit molecule. If given, a freshly
        generated 3D conformer is overlaid near the predicted pocket
        center for illustration - not a docked pose.
    width, height : int, optional
        Viewer dimensions in pixels.

    Returns
    -------
    str or None
        Self-contained viewer HTML, or None if the structure or pocket
        could not be resolved (no AlphaFold/ESMFold model available, or no
        local P2Rank install / no ligandable pocket found).
    """
    full_sequence = fetch_uniprot_sequence(uniprot_id)
    if not full_sequence:
        logger.warning(f"No UniProt sequence available for {uniprot_id}.")
        return None

    offset = fetch_uniprot_cleavage_offset(uniprot_id)
    mature_sequence = full_sequence[offset:]

    pdb_path = asyncio.run(download_pdb(uniprot_id, mature_sequence))
    if pdb_path is None:
        logger.warning(f"No AlphaFold/ESMFold structure available for {uniprot_id}.")
        return None

    pocket_center = predict_pocket(pdb_path)
    if pocket_center is None:
        logger.warning(f"P2Rank could not resolve a pocket for {uniprot_id}.")
        return None

    pocket_residues = _pocket_residues(pdb_path, pocket_center)
    mutation_residues = _mutation_residues(pdb_path, full_sequence, offset, mutation)

    view = py3Dmol.view(width=width, height=height)
    view.setBackgroundColor(VIEWER_BACKGROUND)
    view.addModel(pdb_path.read_text(encoding="utf-8", errors="replace"), "pdb")
    view.setStyle({}, {"cartoon": {"color": PROTEIN_COLOR}})

    for chain_id, resseq in pocket_residues:
        view.addStyle(
            {"chain": chain_id, "resi": resseq},
            {"cartoon": {"color": POCKET_COLOR}, "stick": {"color": POCKET_COLOR, "radius": 0.2}},
        )

    for chain_id, resseq in mutation_residues:
        view.addStyle({"chain": chain_id, "resi": resseq}, {"stick": {"color": MUTATION_COLOR, "radius": 0.35}})

    if primary_mol is not None:
        molblock = _recentered_ligand_molblock(primary_mol, pocket_center)
        if molblock is not None:
            view.addModel(molblock, "mol")
            view.setStyle({"model": -1}, {"stick": {"color": LIGAND_COLOR, "radius": 0.25}})
        else:
            logger.warning("Ligand conformer generation failed; skipping structure overlay.")

    view.zoomTo()
    return view._make_html()
