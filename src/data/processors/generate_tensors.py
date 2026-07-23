"""
===========================================================================
Sequence Pipeline
Description: Sequence, 2D Graph, and 3D Structural Extraction Pipeline
===========================================================================

Workflow:
1. Fetch UniProt sequence for proteins.
2. Run ESM2-650M to get per-residue embeddings, fetch catalytic-site mask.
3. Fetch SMILES for primary substrate and co-substrates.
4. Featurize substrates into 2D molecular graphs using RDKit and ChemBERTa-2.
5. Construct PyTorch Geometric HeteroData with sequence and graph representations.
6. Augment the HeteroData with 3D structural features via
   src/data/processors/geometry_processor.py: AlphaFold/ESMFold pocket
   cropping, ligand/co-substrate 3D conformers, and interacts_with edges.
7. Save the resulting PyG tensors to disk.

Author: ThermoKP Team
License: MIT
"""

import os
import argparse
import asyncio
import logging
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

import pandas as pd
import torch
from torch_geometric.data import HeteroData
import math
from accelerate import Accelerator

import rdkit.Chem as Chem
from rdkit.Chem import rdPartialCharges
from Bio.Align import substitution_matrices

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Import SMILES fetching logic from validator
from src.data.processors.dataset_validator import get_smiles

# Import 3D structural feature extraction (AlphaFold/ESMFold + P2Rank)
from src.data.processors.geometry_processor import augment_hetero_graph_with_3d

# Import pretrained embedding extraction
from src.data.processors.pretrained_embeddings import (
    CHEMBERTA_EMBED_DIM,
    fetch_uniprot_sequence,
    fetch_uniprot_cleavage_offset,
    get_catalytic_site_mask,
    get_ligand_embedding,
    get_mutation_log_odds,
    get_protein_embedding,
    set_device,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Logging Configuration
# ═══════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
from accelerate.logging import get_logger
logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Constants & Configuration
# ═══════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_ROOT / "data" / "thermokp_database.db"
TENSOR_DIR = PROJECT_ROOT / "data" / "processed" / "tensors"
FAILED_TENSORS_FILE = PROJECT_ROOT / "data" / "failed_tensors.txt"

# Feature vocabularies
AMINO_ACIDS = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"
]
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
# Default index 20 for 'X' or unknown amino acids


MUTATION_CODE_PATTERN = re.compile(r"^([A-Z])(\d+)([A-Z])$")

LIGAND_ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
# Values outside these closed vocabularies (e.g. Fe in metal-ion
# cofactors) get a dedicated trailing "unknown" one-hot slot via
# _one_hot_encoding(..., include_unknown=True) rather than an ambiguous
# all-zero encoding.

# ═══════════════════════════════════════════════════════════════════════════
#  Mutation Descriptor
# ═══════════════════════════════════════════════════════════════════════════
# A point mutation perturbs a single residue among hundreds; that per-residue
# change is averaged away by the protein-side pooling before it reaches the
# kinetics head (ARCHITECTURE.md Section 1a). Every record therefore carries a
# fixed-width mutation descriptor as a top-level per-graph tensor
# (data.mutation_features), which - like pH/temperature - bypasses pooling
# entirely. A wild-type record's descriptor is the exact zero vector, keeping
# wild-type and mutant tensors structurally symmetric rather than giving
# mutants an information channel wild-types lack.

# Kyte-Doolittle hydropathy index (dimensionless).
KD_HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
# Residue side-chain volumes (A^3, Zamyatnin 1972).
RESIDUE_VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8,
    "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6,
    "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8,
    "Y": 193.6, "V": 140.0,
}
# Net side-chain charge at pH 7. Histidine (pKa ~6) is partially protonated.
RESIDUE_CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}

# BLOSUM62 substitution scores: a context-free evolutionary substitutability
# signal, complementary to the context-aware ESM2 masked-marginal log-odds.
# Typed Any because biopython's Array supports 2-key (row, col) indexing at
# runtime, which its stub does not model.
_BLOSUM62: Any = substitution_matrices.load("BLOSUM62")

# Ordered descriptor components (see _build_mutation_features).
MUTATION_FEATURE_DIM = 7


def _residue_property_delta(wt_res: str, mut_res: str) -> Tuple[float, float, float, float]:
    """Physicochemical and evolutionary change for a single substitution.

    Returns
    -------
    Tuple[float, float, float, float]
        ``(delta_hydropathy, delta_volume, delta_charge, blosum62_score)``,
        each ``mut - wt`` except the BLOSUM62 score, which is the pairwise
        substitution score. Unknown residues contribute 0.
    """
    d_hydro = KD_HYDROPATHY.get(mut_res, 0.0) - KD_HYDROPATHY.get(wt_res, 0.0)
    d_vol = RESIDUE_VOLUME.get(mut_res, 0.0) - RESIDUE_VOLUME.get(wt_res, 0.0)
    d_charge = RESIDUE_CHARGE.get(mut_res, 0.0) - RESIDUE_CHARGE.get(wt_res, 0.0)
    try:
        blosum = float(_BLOSUM62[wt_res, mut_res])
    except (KeyError, IndexError):
        blosum = 0.0
    return d_hydro, d_vol, d_charge, blosum


def _build_mutation_features(
    mutation_count: int,
    mutation_log_odds: float,
    sum_d_hydropathy: float,
    sum_d_volume: float,
    sum_d_charge: float,
    sum_blosum62: float,
    catalytic_hits: int,
) -> torch.Tensor:
    """Assemble the fixed-width mutation descriptor.

    All arguments are summed over the sites of a multi-mutant; a wild-type
    record passes zeros, yielding the exact neutral (no-perturbation) vector.

    Parameters
    ----------
    mutation_count : int
        Number of point substitutions in the record.
    mutation_log_odds : float
        Summed ESM2 masked-marginal log-odds (ESM-1v; Meier et al.).
    sum_d_hydropathy, sum_d_volume, sum_d_charge : float
        Summed wild-type->mutant deltas in Kyte-Doolittle hydropathy,
        side-chain volume (A^3), and net charge at pH 7.
    sum_blosum62 : float
        Summed BLOSUM62 substitution score.
    catalytic_hits : int
        Number of mutated positions on a UniProt-annotated catalytic/binding
        residue.

    Returns
    -------
    torch.Tensor
        Shape (MUTATION_FEATURE_DIM,), a top-level per-graph tensor bypassing
        protein pooling (see build_hetero_graph / ARCHITECTURE.md Section 1a).
    """
    return torch.tensor(
        [
            float(mutation_count),
            mutation_log_odds,
            sum_d_hydropathy,
            sum_d_volume,
            sum_d_charge,
            sum_blosum62,
            float(catalytic_hits),
        ],
        dtype=torch.float,
    )


def resolve_mutation_positions(
    full_sequence: str, offset: int, mutation_code: str
) -> List[Tuple[str, int, str]]:
    """Parse a mutation code into resolved (wt_res, mut_idx, mut_res) triples.

    `mut_idx` is a 0-based index into the raw precursor sequence
    (`full_sequence`), matching AlphaFold's 1..L residue numbering (see
    geometry_processor.py) - not the mature (signal-peptide-chopped)
    sequence. Shared by `apply_mutations` (which additionally mutates the
    sequence and accumulates the descriptor) and dashboard/structure_view.py
    (which only needs the resolved positions, to highlight mutated residues
    on the predicted structure).

    Parameters
    ----------
    full_sequence : str
        The full-length, unmutated precursor sequence.
    offset : int
        Signal-peptide/propeptide cleavage offset (see
        `fetch_uniprot_cleavage_offset`).
    mutation_code : str
        ``/``-separated point-mutation codes (e.g. ``"N41D/N281D"``).

    Returns
    -------
    list of (str, int, str)
        One `(wt_res, mut_idx, mut_res)` triple per mutation site, in the
        order given in `mutation_code`.

    Raises
    ------
    ValueError
        If a mutation code is malformed, or its wild-type residue matches
        neither the raw nor the offset-shifted sequence position.
    """
    resolved = []
    for m_code in mutation_code.split("/"):
        match = MUTATION_CODE_PATTERN.match(m_code)
        if not match:
            raise ValueError(f"Malformed mutation code: {m_code!r}")
        wt_res, pos_str, mut_res = match.groups()
        seq_idx = int(pos_str) - 1

        # Stage 1: Try offset match (literature typically uses mature sequence numbering)
        if 0 <= seq_idx + offset < len(full_sequence) and full_sequence[seq_idx + offset] == wt_res:
            mut_idx = seq_idx + offset
        # Stage 2: Try raw match as fallback
        elif 0 <= seq_idx < len(full_sequence) and full_sequence[seq_idx] == wt_res:
            mut_idx = seq_idx
        else:
            found = full_sequence[seq_idx] if 0 <= seq_idx < len(full_sequence) else None
            found_off = full_sequence[seq_idx + offset] if 0 <= seq_idx + offset < len(full_sequence) else None
            raise ValueError(f"Mutation {m_code} mismatch (raw: {found!r}, offset: {found_off!r}).")

        resolved.append((wt_res, mut_idx, mut_res))
    return resolved


def apply_mutations(
    uniprot_id: str,
    full_sequence: str,
    mutation_code: Optional[str],
    offset: int,
    catalytic_site_mask: torch.Tensor,
) -> Tuple[str, str, torch.Tensor]:
    """Inject point mutation(s) into the precursor sequence and assemble the mutation descriptor.

    Mutation positions are matched against the raw precursor sequence
    before the signal peptide is chopped (mutate-then-chop, so mutation
    indices are resolved against the full precursor sequence before any
    residues are stripped), trying the position first as given, then
    shifted by `offset` for
    callers numbering relative to the mature sequence. Shared by the
    tensor-generation pipeline (this module) and thermokp.py's inference
    path, so both featurize a mutation identically.

    Parameters
    ----------
    uniprot_id : str
        UniProt accession, used only to key the ESM2 masked-marginal cache.
    full_sequence : str
        The full-length, unmutated precursor sequence.
    mutation_code : str, optional
        ``/``-separated point-mutation codes (e.g. ``"N41D/N281D"``), or
        None for a wild-type record.
    offset : int
        Signal-peptide/propeptide cleavage offset (see
        `fetch_uniprot_cleavage_offset`), the number of N-terminal residues
        stripped to obtain the mature sequence.
    catalytic_site_mask : torch.Tensor
        Per-residue catalytic/binding-site mask aligned to the mature
        sequence (see `get_catalytic_site_mask`), used to count mutated
        positions falling on an annotated catalytic residue.

    Returns
    -------
    Tuple[str, str, torch.Tensor]
        ``(mature_sequence, embed_cache_key, mutation_features)``.
        `embed_cache_key` disambiguates a mutant's ESM2 embedding cache
        entry from the wild-type's; `mutation_features` has shape
        (MUTATION_FEATURE_DIM,) and is the exact zero vector for a
        wild-type record.

    Raises
    ------
    ValueError
        If a mutation code is malformed, or its wild-type residue matches
        neither the raw nor the offset-shifted sequence position.
    """
    embed_cache_key = uniprot_id
    mutated_full_sequence = full_sequence

    # Accumulators for the mutation descriptor, summed over the sites of a
    # multi-mutant. All remain zero for a wild-type record, which is the
    # exact neutral value (no perturbation), not a placeholder.
    mutation_count = 0
    mutation_log_odds = 0.0
    sum_d_hydro = sum_d_vol = sum_d_charge = sum_blosum = 0.0
    mutated_mature_positions = []

    if mutation_code:
        embed_cache_key = f"{uniprot_id}_{mutation_code}"
        # Support multi-mutants (e.g. "N41D/N281D")
        for wt_res, mut_idx, mut_res in resolve_mutation_positions(full_sequence, offset, mutation_code):
            # Apply mutation to raw sequence (order independent since length is conserved)
            mutated_full_sequence = mutated_full_sequence[:mut_idx] + mut_res + mutated_full_sequence[mut_idx + 1:]

            # Accumulate descriptor components, summed across the sites of a
            # multi-mutant. The log-odds is scored on the original
            # wild-type sequence (additive approximation per Meier et al.
            # ESM-1v); the physicochemical/BLOSUM deltas depend only on
            # the residue identities.
            mutation_count += 1
            mutation_log_odds += get_mutation_log_odds(uniprot_id, full_sequence, mut_idx, wt_res, mut_res)
            d_hydro, d_vol, d_charge, blosum = _residue_property_delta(wt_res, mut_res)
            sum_d_hydro += d_hydro
            sum_d_vol += d_vol
            sum_d_charge += d_charge
            sum_blosum += blosum
            # Mature-sequence coordinate, matching the frame the catalytic
            # mask is built in (mut_idx is in precursor space; the signal
            # peptide of length `offset` is chopped).
            mutated_mature_positions.append(mut_idx - offset)

    # Chop signal peptide
    mature_sequence = mutated_full_sequence[offset:]

    # Count mutated sites falling on a residue UniProt annotates as
    # catalytic/binding - a strong mechanistic signal, since active-site
    # substitutions dominate turnover effects.
    catalytic_hits = sum(
        1 for p in mutated_mature_positions
        if 0 <= p < catalytic_site_mask.size(0) and catalytic_site_mask[p].sum() > 0
    )

    mutation_features = _build_mutation_features(
        mutation_count, mutation_log_odds, sum_d_hydro,
        sum_d_vol, sum_d_charge, sum_blosum, catalytic_hits
    )

    return mature_sequence, embed_cache_key, mutation_features


# ═══════════════════════════════════════════════════════════════════════════
#  Featurization Logic
# ═══════════════════════════════════════════════════════════════════════════
def _one_hot_encoding(val, choices, include_unknown=False):
    """One-hot encode `val` against `choices`.

    Parameters
    ----------
    val
        The value to encode.
    choices : list
        The closed vocabulary to encode against.
    include_unknown : bool, optional
        If True, append one extra trailing slot set to 1.0 whenever `val`
        is not in `choices`, instead of returning an all-zero vector for
        that case - so "value outside the vocabulary" is a distinct,
        explicit signal rather than indistinguishable from a zeroed-out
        feature elsewhere. By default False.

    Returns
    -------
    list of float
        One-hot vector of length ``len(choices)`` (or ``len(choices) + 1``
        when `include_unknown` is True).
    """
    encoding = [0.0] * (len(choices) + (1 if include_unknown else 0))
    if val in choices:
        encoding[choices.index(val)] = 1.0
    elif include_unknown:
        encoding[-1] = 1.0
    return encoding

def sequence_to_indices(seq: str) -> torch.Tensor:
    """Encode an amino-acid sequence as integer indices into `AMINO_ACIDS`.

    Parameters
    ----------
    seq : str
        Amino-acid sequence (single-letter codes).

    Returns
    -------
    torch.Tensor
        Shape ``(len(seq),)`` tensor of indices, with unknown/non-standard
        residues mapped to the trailing "unknown" index (20).
    """
    indices = [AA_TO_INDEX.get(aa, 20) for aa in seq]
    return torch.tensor(indices, dtype=torch.long)

NUM_LIGAND_SCALAR_FEATURES = 8
LIGAND_ATOM_FEATURE_DIM = len(LIGAND_ELEMENTS) + 1 + len(HYBRIDIZATIONS) + 1 + NUM_LIGAND_SCALAR_FEATURES
LIGAND_BOND_FEATURE_DIM = len(BOND_TYPES) + 1

def featurize_ligand(mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Featurize a molecule into atom/bond-level structural tensors.

    Atom features: element, hybridization, formal charge, aromaticity,
    Gasteiger charge, ring membership, degree, hydrogen count, and a
    continuous electrophilic/nucleophilic charge z-score pair. Does not
    include the whole-molecule ChemBERTa embedding - see build_hetero_graph.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule to featurize, already fully resolved (e.g. by
        `_build_co_substrates_mol` or `combine_ligand_mols`).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(x, edge_index, edge_attr)``: per-atom feature matrix of shape
        ``(num_atoms, LIGAND_ATOM_FEATURE_DIM)``, a ``(2, num_edges)`` edge
        index with both directions of every bond, and a per-edge feature
        matrix of shape ``(num_edges, LIGAND_BOND_FEATURE_DIM)``. All three
        are empty tensors if `mol` is None or has no atoms.
    """
    if mol is None or mol.GetNumAtoms() == 0:
        return (
            torch.empty((0, LIGAND_ATOM_FEATURE_DIM), dtype=torch.float),
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, LIGAND_BOND_FEATURE_DIM), dtype=torch.float),
        )

    rdPartialCharges.ComputeGasteigerCharges(mol)

    x_features = []
    g_charges = []
    for atom in mol.GetAtoms():
        elem = atom.GetSymbol()
        elem_feat = _one_hot_encoding(elem, LIGAND_ELEMENTS, include_unknown=True)

        hyb = atom.GetHybridization()
        hyb_feat = _one_hot_encoding(hyb, HYBRIDIZATIONS, include_unknown=True)

        formal_charge = float(atom.GetFormalCharge())
        is_aromatic = 1.0 if atom.GetIsAromatic() else 0.0

        try:
            g_charge = float(atom.GetProp("_GasteigerCharge"))
            if not math.isfinite(g_charge):
                g_charge = 0.0
        except KeyError:
            g_charge = 0.0

        is_in_ring = 1.0 if atom.IsInRing() else 0.0
        degree_norm = float(atom.GetDegree()) / 4.0
        num_h_norm = float(atom.GetTotalNumHs()) / 4.0

        x_features.append(
            elem_feat + hyb_feat
            + [formal_charge, is_aromatic, g_charge, is_in_ring, degree_norm, num_h_norm]
        )
        g_charges.append(g_charge)

    charges_t = torch.tensor(g_charges, dtype=torch.float)
    charge_std, charge_mean = torch.std_mean(charges_t, unbiased=False)
    z_scores = (charges_t - charge_mean) / (charge_std + 1e-6)
    electrophilic_scores = torch.clamp(z_scores, min=0.0)
    nucleophilic_scores = torch.clamp(-z_scores, min=0.0)
    for i, row in enumerate(x_features):
        row.append(electrophilic_scores[i].item())
        row.append(nucleophilic_scores[i].item())

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        b_feat = _one_hot_encoding(bond.GetBondType(), BOND_TYPES, include_unknown=True)
        edge_indices.extend([[i, j], [j, i]])
        edge_attrs.extend([b_feat, b_feat])

    x_tensor = torch.tensor(x_features, dtype=torch.float)

    if not edge_indices:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.empty((0, LIGAND_BOND_FEATURE_DIM), dtype=torch.float)
    else:
        edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.float)

    return x_tensor, edge_index_tensor, edge_attr_tensor


def combine_ligand_mols(smiles_list: List[str]) -> Optional[Chem.Mol]:
    """Combine already-resolved SMILES strings into one disconnected Mol.

    Pure Mol-merging, with no name-resolution ambiguity: every entry in
    `smiles_list` must already be a SMILES string. Parses each with RDKit
    and merges them via `Chem.CombineMols`, matching how `featurize_ligand`
    treats `co_substrate_atoms` as one combined node set. Ring perception
    (`Chem.GetSSSR`) is recomputed once on the combined mol, since
    `CombineMols` does not refresh it automatically.

    Parameters
    ----------
    smiles_list : list of str
        SMILES strings to combine, already resolved by the caller
        (`_build_co_substrates_mol` for database names, thermokp.py's
        `resolve_smiles` for name-or-SMILES inference input).

    Returns
    -------
    Chem.Mol or None
        The combined molecule, or None if `smiles_list` is empty.

    Raises
    ------
    ValueError
        If any entry fails to parse as SMILES.
    """
    combined_mol = None
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Failed to parse co-substrate SMILES: {smiles}")
        combined_mol = mol if combined_mol is None else Chem.CombineMols(combined_mol, mol)

    if combined_mol is not None:
        Chem.GetSSSR(combined_mol)

    return combined_mol


def _build_co_substrates_mol(co_substrates_str: str) -> Optional[Chem.Mol]:
    """Resolve and combine a "; "-joined co-substrate NAME list into one Mol.

    Every entry is a database-style chemical name, resolved via
    `get_smiles` - this is the training pipeline's own `co_substrates`
    field, which is always names, never raw SMILES (unlike thermokp.py's
    inference input, which accepts either - see `resolve_smiles` and
    `combine_ligand_mols`).

    Parameters
    ----------
    co_substrates_str : str
        ``"; "``-joined list of co-substrate chemical names, or NaN/empty
        if the record has none.

    Returns
    -------
    Chem.Mol or None
        The combined molecule, or None if `co_substrates_str` is NaN or empty.

    Raises
    ------
    ValueError
        If any co-substrate name fails to resolve to a SMILES string.
    """
    if pd.isna(co_substrates_str) or not co_substrates_str.strip():
        return None

    names = [s.strip() for s in co_substrates_str.split(";") if s.strip()]
    smiles_list = []
    for name in names:
        smiles = get_smiles(name)
        if not smiles:
            raise ValueError(f"Failed to resolve co-substrate name: {name}")
        smiles_list.append(smiles)

    return combine_ligand_mols(smiles_list)


# ═══════════════════════════════════════════════════════════════════════════
#  Tensor Generation
# ═══════════════════════════════════════════════════════════════════════════
def build_hetero_graph(
    uniprot_id: str,
    mutation: str,
    sequence: str,
    primary_mol: Chem.Mol,
    co_sub_mol: Optional[Chem.Mol],
    protein_embedding: torch.Tensor,
    catalytic_site_mask: torch.Tensor,
    ph: float,
    temperature_celsius: float,
    mutation_features: Optional[torch.Tensor] = None,
    kcat: Optional[float] = None,
    km: Optional[float] = None,
) -> HeteroData:
    """Assemble the 2D HeteroData graph shared by training and inference.

    `kcat`/`km` are the empirical training targets (raw DB units: s^-1,
    mM) and are omitted entirely - not zeroed - when unknown, e.g. when
    called from thermokp.py to predict rather than fit them.

    Parameters
    ----------
    uniprot_id : str
        UniProt accession.
    mutation : str
        Raw mutation-code string (e.g. "N41D/N281D"), or "" for wild-type.
    sequence : str
        The mature (signal-peptide-chopped, possibly mutated) sequence.
    primary_mol, co_sub_mol : Chem.Mol, optional
        RDKit molecules for the primary substrate and combined co-substrates
        (`co_sub_mol` is None when no co-substrate is present).
    protein_embedding : torch.Tensor
        Per-residue ESM2 embedding aligned to `sequence`.
    catalytic_site_mask : torch.Tensor
        Per-residue catalytic/binding-site mask aligned to `sequence`.
    ph : float
        Assay pH.
    temperature_celsius : float
        Assay temperature in Celsius (converted to Kelvin internally).
    mutation_features : torch.Tensor, optional
        Shape (MUTATION_FEATURE_DIM,) descriptor from `apply_mutations`;
        defaults to the exact zero vector (wild-type) if omitted.
    kcat : float, optional
        Empirical k_cat target (s^-1), if known.
    km : float, optional
        Empirical K_m target (mM, converted to M internally), if known.

    Returns
    -------
    HeteroData
        The assembled 2D graph (protein_sequence/ligand_atoms/
        co_substrate_atoms node types). Callers needing the 3D structural
        branch augment this further via
        geometry_processor.augment_hetero_graph_with_3d.
    """
    data = HeteroData()
    data.uniprot_id = uniprot_id
    data.mutation = mutation
    if mutation_features is None:
        mutation_features = torch.zeros(MUTATION_FEATURE_DIM, dtype=torch.float)
    # Shape (1, MUTATION_FEATURE_DIM): a top-level per-graph tensor batched
    # along dim 0 like data.kcat/data.pH, so it reaches the head without
    # passing through protein pooling. Exact zero vector for a wild-type.
    data.mutation_features = mutation_features.to(torch.float).view(1, -1)

    # ── Protein Nodes ────────────────────────────────────────────────────────
    data["protein_sequence"].num_nodes = len(sequence)
    data["protein_sequence"].aa_indices = sequence_to_indices(sequence)
    data["protein_sequence"].catalytic_mask = catalytic_site_mask
    data["protein_sequence"].esm2_embedding = protein_embedding

    # ── Primary Ligand Nodes ─────────────────────────────────────────────────
    # Whole-molecule ChemBERTa embedding as a top-level per-graph tensor
    # (shape (1, CHEMBERTA_EMBED_DIM)), not broadcast onto every atom - PyG
    # batches top-level per-graph tensors along dim 0 like data.kcat/data.pH.
    # multimodal_encoder.py fuses this into the pooled D-MPNN output.
    primary_embedding = get_ligand_embedding(Chem.MolToSmiles(primary_mol))
    l_x, l_edge_index, l_edge_attr = featurize_ligand(primary_mol)
    data["ligand_atoms"].x = l_x
    data["ligand_atoms", "covalent_bond", "ligand_atoms"].edge_index = l_edge_index
    data["ligand_atoms", "covalent_bond", "ligand_atoms"].edge_attr = l_edge_attr
    data.ligand_embedding = primary_embedding.to(torch.float).unsqueeze(0)

    # ── Co-Substrate Nodes ───────────────────────────────────────────────────
    # Zero vector for an absent co-substrate: MultimodalEncoder.forward
    # treats an all-zero embedding as "not present".
    if co_sub_mol is not None:
        co_sub_embedding = get_ligand_embedding(Chem.MolToSmiles(co_sub_mol))
    else:
        co_sub_embedding = torch.zeros(CHEMBERTA_EMBED_DIM, dtype=torch.float)
    c_x, c_edge_index, c_edge_attr = featurize_ligand(co_sub_mol)
    data["co_substrate_atoms"].x = c_x
    data["co_substrate_atoms", "covalent_bond", "co_substrate_atoms"].edge_index = c_edge_index
    data["co_substrate_atoms", "covalent_bond", "co_substrate_atoms"].edge_attr = c_edge_attr
    data.co_substrate_embedding = co_sub_embedding.to(torch.float).unsqueeze(0)

    # ── Thermodynamic Metadata ────────────────────────────────────────────────
    data.temperature = torch.tensor([temperature_celsius + 273.15], dtype=torch.float)
    data.pH = torch.tensor([ph], dtype=torch.float)
    if kcat is not None and km is not None:
        data.kcat = torch.tensor([kcat], dtype=torch.float)
        data.km = torch.tensor([km * 1e-3], dtype=torch.float)
        data.kcat_km = torch.tensor([kcat / (km * 1e-3)], dtype=torch.float)

    return data


# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline Execution
# ═══════════════════════════════════════════════════════════════════════════
async def _process_database(limit: Optional[int] = None, start_entry: Optional[int] = None, accelerator: Optional[Accelerator] = None):
    """Drive the full tensor-generation pipeline over `clean_parameters`.

    For each qualifying row: resolves substrate/co-substrate SMILES, fetches
    the UniProt sequence, applies any point mutation, computes ESM2/ChemBERTa
    embeddings, builds the 2D `HeteroData` graph, augments it with 3D
    structural features via `augment_hetero_graph_with_3d`, and writes the
    resulting tensor to `TENSOR_DIR`. Runs entries concurrently under an
    `asyncio.Semaphore` and, when `accelerator` is provided, shards the
    query results across processes and gathers success/total counts back
    to the main process for the summary log. Failed entries are logged to
    `FAILED_TENSORS_FILE` and removed from `clean_parameters` so a re-run
    does not repeatedly attempt known-bad rows.

    Parameters
    ----------
    limit : int, optional
        Maximum number of database rows to process, applied after
        `start_entry` filtering. None processes every qualifying row.
    start_entry : int, optional
        If given, only rows with ``entry_id >= start_entry`` are processed,
        allowing a run to resume partway through the table.
    accelerator : Accelerator, optional
        `accelerate.Accelerator` used to shard work across multiple
        processes/devices. None runs single-process on CPU/default device.

    Returns
    -------
    None
    """
    if accelerator is None or accelerator.is_main_process:
        logger.info("=========================================================")
        logger.info("  ThermoKP Sequence Pipeline Initialization")
        logger.info("=========================================================")
    
    if accelerator is not None and accelerator.is_main_process:
        TENSOR_DIR.mkdir(parents=True, exist_ok=True)
    elif accelerator is None:
        TENSOR_DIR.mkdir(parents=True, exist_ok=True)
        
    if accelerator is not None:
        accelerator.wait_for_everyone()
    
    if not DB_PATH.exists():
        logger.error(f"Database {DB_PATH} not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    
    query = """
    SELECT * FROM clean_parameters
    WHERE uniprot_id IS NOT NULL
      AND kcat IS NOT NULL
      AND km IS NOT NULL
      AND kcat > 0
      AND km > 0
      AND temperature IS NOT NULL
      AND ph IS NOT NULL
      AND measured_substrate IS NOT NULL
    """
    if start_entry is not None:
        query += f" AND entry_id >= {start_entry}"
        
    query += " ORDER BY entry_id ASC"
    if limit is not None:
        query += f" LIMIT {limit}"
        
    df_params = pd.read_sql_query(query, conn)
    conn.close()

    global_total = len(df_params)
    start_idx = 0

    if accelerator is not None:
        num_processes = accelerator.num_processes
        process_index = accelerator.process_index
        chunk_size = math.ceil(len(df_params) / num_processes)
        start_idx = process_index * chunk_size
        end_idx = min((process_index + 1) * chunk_size, len(df_params))
        df_params = df_params.iloc[start_idx:end_idx]
    
    total = len(df_params)
    proc_msg = f"[Process {accelerator.process_index}] " if accelerator else ""
    logger.info(f"{proc_msg}Assigned {total} targets to this process (Global: {global_total}).")
    
    successful = 0
    failed_entries = []
    failed_entry_ids = []
    
    semaphore = asyncio.Semaphore(min(16, (os.cpu_count() or 4) * 2))
    
    async def process_entry(i, row):
        nonlocal successful
        uniprot_id = row["uniprot_id"]
        entry_id = row["entry_id"]
        substrate_name = row["measured_substrate"]
        
        async with semaphore:
            logger.info(f"[{i + 1}/{total}] Processing {uniprot_id} (Entry {entry_id})...")
            
            out_file = TENSOR_DIR / f"entry{entry_id}_{uniprot_id}.pt"
            if out_file.exists():
                logger.info(f"[{i + 1}/{total}] Skipping {uniprot_id} (Entry {entry_id}): Tensor already exists.")
                successful += 1
                return
            
            try:
                def prepare_ligands():
                    primary_smiles = get_smiles(substrate_name)
                    if not primary_smiles:
                        raise ValueError(f"Failed to fetch SMILES for primary substrate: {substrate_name}")

                    primary_mol = Chem.MolFromSmiles(primary_smiles)
                    if not primary_mol:
                        raise ValueError(f"Failed to parse SMILES with RDKit: {primary_smiles}")

                    co_sub_mol = _build_co_substrates_mol(row.get("co_substrates", ""))
                    return primary_mol, co_sub_mol

                primary_mol, co_sub_mol = await asyncio.to_thread(prepare_ligands)

                full_sequence = await asyncio.to_thread(fetch_uniprot_sequence, uniprot_id)
                if not full_sequence:
                    raise ValueError("Sequence fetch failed.")

                def process_features():
                    offset = fetch_uniprot_cleavage_offset(uniprot_id)
                    raw_mutation = row.get("mutation")
                    mutation_code = raw_mutation if isinstance(raw_mutation, str) and raw_mutation.strip() else None

                    mature_len = len(full_sequence) - offset
                    catalytic_site_mask = get_catalytic_site_mask(uniprot_id, mature_len, offset=offset)

                    mature_sequence, embed_cache_key, mutation_features = apply_mutations(
                        uniprot_id, full_sequence, mutation_code, offset, catalytic_site_mask
                    )

                    protein_embedding = get_protein_embedding(uniprot_id, mature_sequence, cache_key=embed_cache_key)

                    data = build_hetero_graph(
                        uniprot_id=row["uniprot_id"],
                        mutation=mutation_code or "",
                        sequence=mature_sequence,
                        primary_mol=primary_mol,
                        co_sub_mol=co_sub_mol,
                        protein_embedding=protein_embedding,
                        catalytic_site_mask=catalytic_site_mask,
                        ph=float(row["ph"]),
                        temperature_celsius=float(row["temperature"]),
                        mutation_features=mutation_features,
                        kcat=float(row["kcat"]),
                        km=float(row["km"]),
                    )
                    return data, mature_sequence, offset, protein_embedding, catalytic_site_mask

                data, mature_sequence, offset, _protein_embedding, catalytic_site_mask = (
                    await asyncio.to_thread(process_features)
                )

                data = await augment_hetero_graph_with_3d(
                    data, uniprot_id, mature_sequence, offset,
                    catalytic_site_mask, primary_mol, co_sub_mol,
                )

                await asyncio.to_thread(torch.save, data, out_file)
                successful += 1
                
            except Exception as e:
                logger.error(f"{proc_msg}[{start_idx + i + 1}/{global_total}] Skipping {uniprot_id}: {e}")
                failed_entries.append(f"Entry {entry_id} ({uniprot_id}): {e}")
                failed_entry_ids.append(entry_id)

    tasks = [process_entry(i, row) for i, (_, row) in enumerate(df_params.iterrows())]
    if tasks:
        await asyncio.gather(*tasks)

    if failed_entries:
        with open(FAILED_TENSORS_FILE, "w") as f:
            for fail_msg in failed_entries:
                f.write(f"{fail_msg}\n")
        logger.warning(f"Logged {len(failed_entries)} failures to {FAILED_TENSORS_FILE}")
        
    if failed_entry_ids:
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in failed_entry_ids)
            cursor.execute(f"DELETE FROM clean_parameters WHERE entry_id IN ({placeholders})", failed_entry_ids)
            conn.commit()
            conn.close()
            logger.warning(f"Removed {len(failed_entry_ids)} failed entries from clean_parameters.")
        except Exception as e:
            logger.error(f"Failed to remove bad entries from database: {e}")

    # Gather total/successful stats so the summary is accurate across all processes
    successful_tensor = torch.tensor([successful], device=accelerator.device if accelerator else "cpu")
    total_tensor = torch.tensor([total], device=accelerator.device if accelerator else "cpu")

    if accelerator is not None:
        accelerator.wait_for_everyone()
        successful_gathered = int(cast(torch.Tensor, accelerator.gather(successful_tensor)).sum().item())
        total_gathered = int(cast(torch.Tensor, accelerator.gather(total_tensor)).sum().item())
    else:
        successful_gathered = successful
        total_gathered = total

    if accelerator is None or accelerator.is_main_process:
        logger.info("==========================================================================")
        logger.info("==                        Sequence Pipeline Summary                     ==")
        logger.info("==========================================================================")
        logger.info(f"Tensors generated         : {successful_gathered} / {total_gathered}")
        logger.info(f"Output directory          : {TENSOR_DIR.absolute()}")
        logger.info("==========================================================================")


def main():
    """Parse CLI arguments and run the tensor-generation pipeline.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="ThermoKP Sequence Pipeline")
    parser.add_argument("-m", "--max_entries", type=int, default=None, help="Maximum number of entries to process")
    parser.add_argument("-e", "--entry", type=int, default=None, help="Start processing from this entry_id onwards")
    args = parser.parse_args()
    
    accelerator = Accelerator()
    set_device(accelerator.device)
    
    asyncio.run(_process_database(limit=args.max_entries, start_entry=args.entry, accelerator=accelerator))


if __name__ == "__main__":
    main()