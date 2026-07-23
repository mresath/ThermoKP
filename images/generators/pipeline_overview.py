"""Generates the end-to-end data pipeline diagram: raw database ingestion
(BRENDA/SABIO-RK) through cleaning into clean_parameters (panel a), and
featurization of clean_parameters into trained model checkpoints (panel b).
A single figure sharing one visual system (images/generators/_style.py)
covering both pipeline stages.

Run: python -m images.generators.pipeline_overview
Output: images/generated/pipeline_overview.svg
"""
from ._style import (
    GOOD, INK_SECONDARY, MAGENTA, NAVY,
    arrow, box, container, footnote, legend, new_figure, panel_label, route, save,
)

FIG_W, FIG_H = 13.333, 11.2
fig, ax = new_figure(FIG_W, FIG_H)


# ═══════════════════════════════════════════════════════════════════════════
#  Panel a — ingestion: BRENDA / SABIO-RK -> clean_parameters
# ═══════════════════════════════════════════════════════════════════════════
panel_label(ax, 0.30, FIG_H - 0.15, "a")

MARGIN_X, BW, GAP = 0.35, 1.55, 0.24
ROW_H_A = 1.55
ROW_Y_A = 8.30
xs = [MARGIN_X + i * (BW + GAP) for i in range(7)]

SRC_H, SRC_GAP = 0.80, 0.22
src_cy = ROW_Y_A + ROW_H_A / 2
b_brenda = box(ax, xs[0], src_cy + SRC_GAP / 2, BW, SRC_H, NAVY, "BRENDA",
              "Literature kcat / Km", wrap=26, title_size=10, sub_size=7.8)
b_sabio = box(ax, xs[0], src_cy - SRC_GAP / 2 - SRC_H, BW, SRC_H, NAVY, "SABIO-RK",
             "Kinetics + SBML laws", wrap=26, title_size=10, sub_size=7.8)

b_parse = box(ax, xs[1], ROW_Y_A, BW, ROW_H_A, INK_SECONDARY, "Parse & Extract",
             "Km, kcat, pH, T; wild-type vs. mutant", wrap=20, sub_size=7.8)
b_canon = box(ax, xs[2], ROW_Y_A, BW, ROW_H_A, INK_SECONDARY, "Canonicalize",
             "Standardize ligand names; resolve UniProt", wrap=20, sub_size=7.8)
b_merge = box(ax, xs[3], ROW_Y_A, BW, ROW_H_A, INK_SECONDARY, "Merge",
             "Unified raw parameter table", caption="raw_parameters", wrap=20, sub_size=7.8)
b_clean = box(ax, xs[4], ROW_Y_A, BW, ROW_H_A, INK_SECONDARY, "Clean & Aggregate",
             "De-duplicate; reject >5x outlier spread; median", wrap=20, sub_size=7.6)
b_validate = box(ax, xs[5], ROW_Y_A, BW, ROW_H_A, INK_SECONDARY, "Validate",
                 "Resolve SMILES; sequence & structure availability", wrap=20, sub_size=7.6)
b_output_a = box(ax, xs[6], ROW_Y_A, BW, ROW_H_A, GOOD, "Clean Parameters\nTable",
                 "Model-ready kinetic records", caption="clean_parameters",
                 title_size=9.5, wrap=16, sub_size=7.8, emphasis=True)

b_reserved = box(ax, xs[4], ROW_Y_A - 2.05, BW, 1.10, MAGENTA, "Reserved",
                "Held-out enzymes, zero-shot test", caption="benchmark_parameters",
                dashed=True, wrap=18, sub_size=7.4)

ch_x = (b_brenda["r"] + b_parse["l"]) / 2
route(ax, [(b_brenda["r"], b_brenda["cy"]), (ch_x, b_brenda["cy"]),
          (ch_x, b_parse["cy"] + 0.16), (b_parse["l"], b_parse["cy"] + 0.16)])
route(ax, [(b_sabio["r"], b_sabio["cy"]), (ch_x, b_sabio["cy"]),
          (ch_x, b_parse["cy"] - 0.16), (b_parse["l"], b_parse["cy"] - 0.16)])
for a_, b_ in [(b_parse, b_canon), (b_canon, b_merge), (b_merge, b_clean), (b_clean, b_validate), (b_validate, b_output_a)]:
    arrow(ax, a_, b_)
arrow(ax, b_clean, b_reserved, from_side="bottom", to_side="top", color=MAGENTA, dashed=True)

legend(ax, MARGIN_X, ROW_Y_A - 2.45, [
    (NAVY, "Source Database"), (INK_SECONDARY, "Processing Step"),
    (GOOD, "Final Validated Dataset"), (MAGENTA, "Reserved / Not Yet Used"),
], gap=2.9, dashed_flags=[False, False, False, True])

# ═══════════════════════════════════════════════════════════════════════════
#  Panel b — featurization: clean_parameters -> tensors -> trained checkpoint
# ═══════════════════════════════════════════════════════════════════════════
panel_label(ax, 0.30, 5.60, "b")

GT_X, GT_W = 1.95, 3.55
GT_Y, GT_H = 0.85, 4.30
c_gt = container(ax, GT_X, GT_Y, GT_W, GT_H, INK_SECONDARY, "Tensor Generation",
                caption="per-record featurization")
inner_x, inner_w = GT_X + 0.20, GT_W - 0.40
sub_h, sub_gap = 0.80, 0.10
sub_top = GT_Y + GT_H - 0.45
sub_labels = [
    ("Protein", "UniProt seq -> mutate-then-chop -> ESM2-650M"),
    ("Ligand & Co-Substrate", "SMILES -> RDKit 2D graph + ChemBERTa-2"),
    ("Mutation Descriptor", "7 features: ESM2 masked-LM log-odds + lookups"),
    ("3D Structural", "AlphaFold/ESMFold + P2Rank pocket -> EGNN input"),
]
for i, (title, sub) in enumerate(sub_labels):
    top = sub_top - i * (sub_h + sub_gap)
    box(ax, inner_x, top - sub_h, inner_w, sub_h, INK_SECONDARY, title, sub,
       wrap=34, title_size=9, sub_size=7.2)

CY_B = GT_Y + GT_H / 2
H_B = 1.30
Y_B = CY_B - H_B / 2

b_clean_b = box(ax, 0.35, Y_B, 1.35, H_B, GOOD, "Clean Parameters\nTable", "", wrap=14,
               title_size=9.0, emphasis=True)
b_tensor = box(ax, 6.00, Y_B, 1.35, H_B, GOOD, "Graph Tensor", ".pt per record",
              caption="processed/tensors", wrap=14, title_size=9.5, sub_size=7.4, emphasis=True)
b_data = box(ax, 7.75, Y_B, 1.60, H_B, INK_SECONDARY, "Dataset & Loader",
            "grouped 90/10 split; weighted sampler", wrap=17, title_size=9.5, sub_size=7.4)
b_model = box(ax, 9.75, Y_B, 1.60, H_B, INK_SECONDARY, "Model + Loss",
             "Eyring + Briggs-Haldane; log-space", wrap=17, title_size=9.5, sub_size=7.4)
b_train = box(ax, 11.75, Y_B, 1.25, H_B, GOOD, "Train Model", "AdamW, auto-stop",
             caption="models/*.pth", wrap=12, title_size=9.5, sub_size=7.0, emphasis=True)

for a_, b_ in [(b_clean_b, c_gt), (c_gt, b_tensor), (b_tensor, b_data), (b_data, b_model), (b_model, b_train)]:
    arrow(ax, a_, b_)

legend(ax, 0.35, 0.45, [
    (INK_SECONDARY, "Processing Step"), (GOOD, "Artifact / Checkpoint"),
], gap=2.9)

footnote(ax, FIG_W,
        "Frozen ESM2-650M / ChemBERTa-2 embeddings, catalytic masks, mutation log-odds and SMILES "
        "are disk-cached and reused across records.", y=0.15)

save(fig, "pipeline_overview")
