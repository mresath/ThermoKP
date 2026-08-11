"""Generates the neural-network architecture diagram: MultimodalEncoder's fused
2D sequence/graph branch and 3D structural EGNN branch, representation fusion,
and the physics-constrained kinetics/Eyring heads with the Briggs-Haldane K_m
derivation.

Run: python -m images.generators.nn_architecture
Output: images/generated/nn_architecture.svg
"""
from ._style import (
    AMBER, GOOD, INK, INK_MUTED, INK_SECONDARY, NAVY, TEAL,
    box, footnote, legend, new_figure, route, save,
)

FIG_W, FIG_H = 13.333, 7.85
fig, ax = new_figure(FIG_W, FIG_H)

legend(ax, 0.35, 7.50, [
    (NAVY, "Protein"), (INK_SECONDARY, "Ligand & Co-Substrate"),
    (TEAL, "3D Structural"), (AMBER, "Physics Head"), (GOOD, "Output"),
], gap=2.60)

Y_PROT, Y_LIG, Y_COS, Y_3D = 6.55, 5.15, 3.70, 2.10
RH = 1.15
RH_3D = 1.35

# ── Column A: inputs ──
AX, AW = 0.35, 2.00
i_prot = box(ax, AX, Y_PROT - RH / 2, AW, RH, NAVY, "Protein Input",
            "aa_indices, ESM2 (L,1280),\ncatalytic_mask", title_size=9.5, sub_size=7.5,
            linespacing=1.7)
i_lig = box(ax, AX, Y_LIG - RH / 2, AW, RH, INK_SECONDARY, "Ligand Input",
           "atoms (N,22) + bonds,\nChemBERTa (384)", title_size=9.5, sub_size=7.5,
           linespacing=1.7)
i_cos = box(ax, AX, Y_COS - RH / 2, AW, RH, INK_SECONDARY, "Co-Substrate Input",
           "atoms (M,22) + bonds,\nChemBERTa (384)", title_size=9.5, sub_size=7.5,
           linespacing=1.7)

# ── Column B: encoders ──
BX, BW = 2.70, 2.30
e_prot = box(ax, BX, Y_PROT - RH / 2, BW, RH, NAVY, "Protein Adapter",
            "ESM2 proj + AA embed\n-> per-residue (L,64)", title_size=9.5, sub_size=7.5,
            linespacing=1.7)
e_lig = box(ax, BX, Y_LIG - RH / 2, BW, RH, INK_SECONDARY, "Ligand D-MPNN",
           "message-pass -> pool\n+ ChemBERTa -> (64)", title_size=9.5, sub_size=7.5,
           linespacing=1.7)
e_cos = box(ax, BX, Y_COS - RH / 2, BW, RH, INK_SECONDARY, "Co-Substrate D-MPNN",
           "pool + ChemBERTa\n-> (64), 0 if absent", title_size=9.5, sub_size=7.5,
           linespacing=1.7)

# ── Column C: protein pooling ──
CX, CW = 5.35, 1.95
p_pool = box(ax, CX, Y_PROT - RH / 2 - 0.20, CW, RH + 0.40, NAVY, "Protein Pooling",
            "catalytic-site + substrate-\nconditioned attention\n-> p_rep (64)",
            title_size=9.5, sub_size=7.3, linespacing=1.7)

# ── 3D structural branch (columns A-C merged into one summary box) ──
struct3d = box(ax, AX, Y_3D - RH_3D / 2, (CX + CW) - AX, RH_3D, TEAL,
               "3D Structural EGNN",
               "AlphaFold/ESMFold pocket (10A crop) + ligand/co-substrate 3D conformers, "
               "interacts_with (6A) + covalent_bond edges -> heterogeneous EGNN -> "
               "gated-attn pool -> structural_rep (96), 0 if absent",
               title_size=9.5, sub_size=7.6, wrap=70, linespacing=1.65)

# ── Column D: fused representation (tall bar) ──
DX, DW = 7.65, 1.10
d_top, d_bot = 7.15, 0.55
fused = box(ax, DX, d_bot, DW, d_top - d_bot, INK_SECONDARY, "", "", emphasis=True)
dcx, dc = DX + DW / 2, (d_top + d_bot) / 2
ax.text(dcx + 0.16, dc, "Fused Representation", ha="center", va="center", fontsize=11,
       fontweight="bold", color=INK, rotation=90, zorder=4)
ax.text(dcx - 0.28, dc, "p | l | c_rep | struct_rep | pH | T | mut(7)  =  297",
       ha="center", va="center", fontsize=6.8, color=INK_SECONDARY, rotation=90,
       family="monospace", zorder=4)

# ── metadata bar (feeds fusion) ──
m_bar = box(ax, AX, 0.30, CX + CW - AX, 0.95, INK_SECONDARY, "Metadata",
           "pH, temperature, mutation_features (7) - scaled to ~unit range",
           title_size=9.5, sub_size=7.6, wrap=52, linespacing=1.6)

# ── Column E: heads ──
EX, EW = 9.15, 1.95
h_eyr = box(ax, EX, 3.75, EW, 2.85, AMBER, "Thermo Layer",
           r"$\kappa=\sigma(\cdot)$" + "\n" +
           r"$\Delta G^{\ddagger}=\mathrm{softplus}(\cdot)$" + "\n" +
           r"$k_{cat}=\kappa\frac{k_BT}{h}e^{-\Delta G^{\ddagger}/RT}$",
           title_size=9.5, sub_size=7.5, linespacing=2.5)
h_kin = box(ax, EX, 1.55, EW, 1.95, AMBER, "Kinetics Head",
           r"$k_1 = 10^{10}\,\sigma(\cdot)$" + "\n" +
           r"$k_{-1} = \frac{k_BT}{h}\,\sigma(\cdot)$", title_size=9.5, sub_size=9.0,
           linespacing=2.5)

# ── Column F: outputs ──
FX, FW = 11.35, 1.60
o_kcat = box(ax, FX, 5.25, FW, 1.0, GOOD, r"$k_{cat}$", emphasis=True, title_size=12)
o_km = box(ax, FX, 3.25, FW, 1.75, GOOD, r"$K_m$",
          r"$\frac{k_{-1}+k_{cat}}{k_1}$" + "\nBriggs-Haldane", title_size=12, sub_size=8.4,
          emphasis=True, linespacing=1.75)
o_loss = box(ax, FX, 1.05, FW, 1.55, GOOD, "Loss",
            r"$w\cdot\mathrm{RMSE}(\log k_{cat})$" + "\n" + r"$+\ \mathrm{RMSE}(\log K_m)$",
            title_size=10, sub_size=8.2, linespacing=1.65)

# --- connections ---
route(ax, [(i_prot["r"], Y_PROT), (e_prot["l"], Y_PROT)])
route(ax, [(i_lig["r"], Y_LIG), (e_lig["l"], Y_LIG)])
route(ax, [(i_cos["r"], Y_COS), (e_cos["l"], Y_COS)])
route(ax, [(e_prot["r"], Y_PROT), (p_pool["l"], Y_PROT)])

route(ax, [(e_lig["r"], Y_LIG), (5.28, Y_LIG), (5.28, Y_PROT + 0.55), (p_pool["l"], Y_PROT + 0.55)],
     dashed=True, color=INK_MUTED, lw=1.1)
route(ax, [(e_cos["r"], Y_COS), (5.12, Y_COS), (5.12, Y_PROT + 0.30), (p_pool["l"], Y_PROT + 0.30)],
     dashed=True, color=INK_MUTED, lw=1.1)

route(ax, [(p_pool["r"], Y_PROT), (DX, Y_PROT)])
route(ax, [(e_lig["r"], Y_LIG), (DX, Y_LIG)])
route(ax, [(e_cos["r"], Y_COS), (DX, Y_COS)])
route(ax, [(struct3d["r"], Y_3D), (DX, Y_3D)])
route(ax, [(m_bar["r"], m_bar["cy"]), (DX, m_bar["cy"])])

route(ax, [(DX + DW, h_eyr["cy"]), (h_eyr["l"], h_eyr["cy"])])
route(ax, [(DX + DW, h_kin["cy"]), (h_kin["l"], h_kin["cy"])])

route(ax, [(h_eyr["r"], h_eyr["cy"]), (11.22, h_eyr["cy"]),
          (11.22, o_kcat["cy"]), (o_kcat["l"], o_kcat["cy"])])
route(ax, [(h_kin["r"], h_kin["cy"]), (11.22, h_kin["cy"]),
          (11.22, o_km["cy"]), (o_km["l"], o_km["cy"])])
route(ax, [(12.13, o_kcat["b"]), (12.13, o_km["t"])], dashed=True, color=INK_MUTED, lw=1.1)

route(ax, [(11.90, o_km["b"]), (11.90, o_loss["t"])], color=GOOD, lw=1.3)
route(ax, [(o_kcat["r"], 5.55), (13.06, 5.55), (13.06, 1.75), (o_loss["r"], 1.75)], color=GOOD, lw=1.3)

ax.text(EX + EW / 2, 1.45, "diffusion & TST ceilings", ha="center", va="top",
       fontsize=7.0, style="italic", color=AMBER, zorder=4)

footnote(ax, FIG_W,
        "All three rate constants are structurally bounded to their physical limits; "
        "K_m follows from Briggs-Haldane, never a free output.")

save(fig, "nn_architecture")
