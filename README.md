# ThermoKP: Zero-Shot Enzyme Kinetics from Sequence, Structure, and Molecular Graphs

`ThermoKP` is a thermodynamically constrained, physics-informed neural network (PINN) that predicts enzyme kinetic constants ($k_{cat}$, $K_m$) directly from a protein's sequence, its predicted 3D structure, and the molecular graphs of its substrates. A hard-parameter-sharing multimodal encoder extracts a shared representation which branches into thermodynamically constrained heads, so that macroscopic kinetics are always derived from physically bounded microscopic rate constants rather than fit as free scalars.

By combining Eyring/Arrhenius transition-state theory with hard physical bounds on the underlying rate constants (the Smoluchowski diffusion limit and the transition-state-theory speed limit), ThermoKP acts as an *amortized predictor*: it produces physically consistent, zero-shot $k_{cat}$ & $K_m$ estimates for uncharacterized enzymes without requiring any experimental data at inference time. Sequence and structural signal is further augmented with frozen pretrained representations — per-residue ESM2 protein embeddings and whole-molecule ChemBERTa-2 ligand embeddings — supplying evolutionary and chemical context beyond what is available from structure alone.

---

## Architectural Principles

- **Sequence-Based Enzyme Modeling:** A lightweight residual adapter refines frozen ESM2 embeddings, which already encode per-residue evolutionary and positional context, and substrate-conditioned attention pooling lets each bound substrate's identity determine which residues matter for that specific reaction.
- **3D Structural Modeling:** An AlphaFold/ESMFold structure is cropped to its P2Rank-predicted binding pocket, and a heterogeneous, E(n)-equivariant graph neural network (EGNN) passes messages between the pocket and the ligand/co-substrate 3D conformers over proximity edges, giving the model direct geometric access to the binding site alongside the sequence-level signal.
- **Hard Thermodynamic Constraints:** The network predicts foundational micro-rate parameters ($k_1$, $k_{-1}$, $\Delta G^{\ddagger}$), each passed through a bounded parameterization (softplus, sigmoid) that structurally enforces the diffusion limit, the transition-state-theory speed limit, and $\Delta G^{\ddagger} \geq 0$. $k_{cat}$ and $K_m$ are then derived deterministically from these rates via the Eyring and Briggs-Haldane relations, rather than predicted as unconstrained scalars.
- **Unified Kinetics Database:** A curated database pairs $k_{cat}$ and $K_m$ measurements from matching assay conditions (protein, substrate, pH, temperature, mutation state), so the loss function always operates on a physically coherent reaction state.

---

## Core Research Question

> How can physics-informed neural networks be architected to predict accurate, zero-shot enzyme kinetic parameters from protein sequence, 3D structure, and molecular graphs?

---

## Repository Layout

```text
├── data/                     # Curation scripts and datasets
│   ├── raw/                  # SABIO-RK and BRENDA downloads (gitignored)
│   ├── processed/            # Cleaned outputs and ML-ready tensors (gitignored)
│   │   └── tensors/          # PyG HeteroData .pt files
│   ├── cache/                # Cached ESM2/ChemBERTa-2 embeddings and PDBs (gitignored)
│   │   └── pdbs/             # AlphaFold/ESMFold structures fetched by geometry_processor.py
│   ├── results/               # Ingestion/cleaning/evaluation summaries, eval.json
│   ├── enzyme_targets.json   # List of enzymes to process
│   └── failed_*.txt          # Pipeline attrition logs (chemicals, structures, sequences, tensors)
├── models/                   # Final model checkpoints & weights (tracked via Git LFS)
├── tools/
│   └── p2rank/                # P2Rank binary (external install, not distributed - see below)
├── src/                       # Core source code
│   ├── data/                  # Data generation and curation scripts
│   │   ├── parsers/           # Raw data ingestion (BRENDA, SABIO-RK)
│   │   ├── processors/        # Data cleaning, validation, PyG tensor generation, 3D structural extraction
│   │   ├── scripts/           # Shell scripts to run the pipelines
│   │   └── utils/             # Shared utilities (caching, EC numbers, SMILES resolution)
│   ├── encoders/               # Multimodal encoder: sequence/2D-graph adapter fused with a 3D structural EGNN
│   ├── evaluation/             # evaluate_dataset.py: benchmark/train/val regression metrics
│   ├── physics/                 # Arrhenius/Eyring thermodynamics & multi-task loss
│   └── training/                # AdamW training loop with slope-based auto-stop, plus the baseline NN's trainer
├── train.py                   # Main training script
├── train_baseline_nn.py       # Non-physics-informed baseline NN (ablation control, see ARCHITECTURE.md)
├── thermokp.py                 # Zero-shot inference: UniProt ID + mutation + substrates -> k_cat/K_m
├── dashboard/                   # Streamlit dashboard for interactive inference demos (see below)
│   ├── app.py                   # Entry point: `streamlit run dashboard/app.py`
│   ├── inference_helpers.py     # thermokp.py wrappers deriving k_a/k1/k-1/deltaG/kappa, CSV batch runner
│   └── structure_view.py        # Interactive py3Dmol structure/pocket/mutation viewer
├── images/                       # Figure generation for publication-quality visualizations
├── ARCHITECTURE.md              # Detailed mathematical & physics formulations
└── README.md                     # Project overview
```

### External Prerequisite: P2Rank

The 3D structural pipeline (`src/data/processors/geometry_processor.py`) predicts ligand-binding pockets using [P2Rank](https://github.com/rdk/p2rank). Install it under `tools/p2rank/` (so `tools/p2rank/prank` is executable) before running `dataset_validator.py` or `generate_tensors.py` — it is not bundled with this repository.

## Execution Workflow

1. **Local Development:** Data acquisition, curation, and tensor generation run locally via the scripts under `src/data/scripts/`.
2. **Cloud Training:** Pull this repository onto a GPU-equipped machine and run `train.py` (or `train_baseline_nn.py` for the ablation control) to train the full network.
3. **Zero-Shot Inference:** Run `thermokp.py` with a UniProt ID (optionally with a point mutation) and a substrate list (database-style chemical names or SMILES strings) to receive instant, biophysically bounded $k_{cat}$ & $K_m$ predictions. See `src/evaluation/evaluate_dataset.py` for evaluating this pipeline against the withheld benchmark holdout.
4. **Interactive Dashboard:** Run `streamlit run dashboard/app.py` for a local, interactive demo of the same zero-shot inference path — single-query and CSV batch predictions, a PINN/baseline model toggle, derived $k_{a} = k_{cat} / K_m$ and (PINN-only) $k_1$, $k_{-1}$, $\Delta G^{\ddagger}$, $\kappa$ quantities, and an interactive py3Dmol view of the predicted structure, binding pocket, and any queried mutation site.

## Data Sources

- **BRENDA:** The primary source for $k_{cat}$ and $K_m$ parameters, temperature, pH, and mutant annotations.
- **SABIO-RK:** A secondary source for kinetic parameters with highly structured reaction metadata.
- **UniProt:** For sequence retrieval, curated active-site/binding-site annotations, and EC-number/organism resolution.
- **RDKit:** For generating 2D molecular graphs and 3D ligand/co-substrate conformers.
- **ESM2 (`facebook/esm2_t33_650M_UR50D`) / ChemBERTa-2 (`DeepChem/ChemBERTa-77M-MLM`):** Frozen pretrained protein and ligand embeddings, via HuggingFace `transformers`.
- **AlphaFold DB / ESM Metagenomic Atlas:** Predicted protein 3D structures, with ESMFold as a fallback when AlphaFold has no model for a given UniProt ID.
- **P2Rank:** Ligand-binding pocket prediction used to crop each protein structure to its catalytic site.

## License

This project is open-source under the MIT License.
