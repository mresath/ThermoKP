"""Generates the tensor-shape reference diagram: the layout, shapes and dtypes
of the PyG HeteroData object produced per record by generate_tensors.py and
augmented with 3D structural fields by geometry_processor.py.

Run: python -m images.generators.tensor_shape
Output: images/generated/tensor_shape.svg
"""
from matplotlib.patches import FancyBboxPatch

from ._style import (
    GOOD, INK, INK_MUTED, INK_SECONDARY, MAGENTA, MONO, NAVY, TEAL,
    field_table, footnote, leader, new_figure, save,
)

FIG_W, FIG_H = 13.333, 7.20
fig, ax = new_figure(FIG_W, FIG_H)

ax.add_patch(FancyBboxPatch(
    (0.35, 0.47), 8.35, 6.55, boxstyle="round,pad=0,rounding_size=0.10",
    linewidth=1.1, edgecolor=INK_MUTED, facecolor="none", zorder=1))

PW, PH = 3.78, 2.0
LX, RX = 0.6, 4.6
TY, BY = 4.67, 2.37

p_prot = field_table(ax, LX, TY, PW, PH, NAVY, "Protein Sequence", "node · L residues", [
    ("aa_indices", "(L,)", "int64"),
    ("catalytic_mask", "(L, 2)", "float32"),
    ("esm2_embedding", "(L, 1280)", "float32"),
])
p_lig = field_table(ax, RX, TY, PW, PH, INK_SECONDARY, "Ligand Atoms", "node · N atoms", [
    ("x", "(N, 22)", "float32"),
    ("edge_index", "(2, E)", "int64"),
    ("edge_attr", "(E, 5)", "float32"),
    ("pos", "(N, 3)", "float32"),
], row_fs=7.6)
p_cos = field_table(ax, LX, BY, PW, PH, INK_SECONDARY, "Co-Substrate Atoms", "node · M atoms · 0 if absent", [
    ("x", "(M, 22)", "float32"),
    ("edge_index", "(2, F)", "int64"),
    ("edge_attr", "(F, 5)", "float32"),
    ("pos", "(M, 3)", "float32"),
], row_fs=7.6)
p_graph = field_table(ax, RX, BY, PW, PH, GOOD, "Graph-Level Features", "per-graph", [
    ("ligand_embedding", "(1, 384)", "float32"),
    ("co_substrate_emb.", "(1, 384)", "float32"),
    ("mutation_features", "(1, 7)", "float32"),
    ("kcat / km / pH / T", "(1,)", "float32"),
], row_fs=7.6)

p_pocket = field_table(ax, LX, 0.62, PW, 1.65, TEAL, "Protein Pocket Atoms", "node · P atoms · 0 if absent", [
    ("x", "(P, 27)", "float32"),
    ("residue_index", "(P,)", "int64"),
    ("pos", "(P, 3)", "float32"),
], row_fs=7.8)
ax.text(RX, 2.12, "Structural Interaction Edges", ha="left", va="top", fontsize=10.5,
       fontweight="bold", color=INK, zorder=4)
ax.text(RX, 1.80, "interacts_with: (2, K) int64 · 6A cutoff, no edge_attr\n"
                 "  pocket_atoms <-> ligand_atoms <-> co_substrate_atoms\n"
                 "residue_of: (2, P) int64 · pocket atom -> its residue\n"
                 "  in protein_sequence (ESM2 gathered at forward time)",
       ha="left", va="top", fontsize=7.2, color=INK_SECONDARY, zorder=4, **MONO)
ax.text(RX, 0.90, "10A crop around P2Rank pocket center (AlphaFold/ESMFold structure)",
       ha="left", va="top", fontsize=6.8, style="italic", color=TEAL, zorder=4)

BX, BW2 = 9.05, 3.9
p_atom = field_table(ax, BX, TY, BW2, PH, INK_SECONDARY, "Atom Features", "22 dims", [
    ("element + unk", "one-hot", "10"),
    ("hybrid. + unk", "one-hot", "4"),
    ("charge/arom/gast", "scalar", "3"),
    ("ring/degree/#H", "scalar", "3"),
    ("electro/nucleo", "z-score", "2"),
], row_fs=8.0, dashed=True)

p_mut = field_table(ax, BX, BY, BW2, PH, MAGENTA, "Mutation Features", "7 dims · 0 if WT", [
    ("count", "", "1"),
    ("ESM2 masked-LM LO", "", "2"),
    ("d-hydro/vol/charge", "", "3-5"),
    ("BLOSUM62", "", "6"),
    ("catalytic-site hits", "", "7"),
], row_fs=8.0, dashed=True)

leader(ax, p_lig, p_atom)
leader(ax, p_graph, p_mut)

footnote(ax, FIG_W,
        "L = residues · N, M = atoms · P = pocket atoms (10A crop) · K = interacts_with edges · "
        "E, F = 2x bond count (edges are bidirectional) · edge_attr = 4 bond types + unknown", y=0.14)

save(fig, "tensor_shape")
