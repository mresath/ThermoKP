# ThermoKP: Physics-Informed Zero-Shot Enzyme Kinetics

`ThermoKP` is a thermodynamically constrained, physics-informed neural network (PINN) that predicts enzyme kinetic constants ($k_{cat}$, $K_m$) directly from a protein's sequence, its predicted 3D structure, and the molecular graphs of its substrates. A hard-parameter-sharing multimodal encoder extracts a shared representation which branches into thermodynamically constrained heads, so that macroscopic kinetics are always derived from physically bounded microscopic rate constants rather than fit as free scalars.

By combining Eyring/Arrhenius transition-state theory with hard physical bounds on the underlying rate constants (the Smoluchowski diffusion limit and the transition-state-theory speed limit), ThermoKP acts as an *amortized predictor*: it produces physically consistent, zero-shot $k_{cat}$ & $K_m$ estimates for uncharacterized enzymes without requiring any experimental data at inference time. Sequence and structural signal is further augmented with frozen pretrained representations — per-residue ESM2 protein embeddings and whole-molecule ChemBERTa-2 ligand embeddings — supplying evolutionary and chemical context beyond what is available from structure alone.

---

## Architectural Principles

- **Sequence-Based Enzyme Modeling:** A lightweight residual adapter refines frozen ESM2 embeddings, which already encode per-residue evolutionary and positional context, and substrate-conditioned attention pooling lets each bound substrate's identity determine which residues matter for that specific reaction.
- **3D Structural Modeling:** An AlphaFold/ESMFold structure is cropped to its P2Rank-predicted binding pocket, and a heterogeneous, E(n)-equivariant graph neural network (EGNN) passes messages between the pocket and the ligand/co-substrate 3D conformers over proximity edges, giving the model direct geometric access to the binding site alongside the sequence-level signal.
- **Hard Thermodynamic Constraints:** The network predicts foundational micro-rate parameters ($k_1$, $k_{-1}$, $\Delta G^{\ddagger}$), each passed through a bounded parameterization (softplus, sigmoid) that structurally enforces the diffusion limit, the transition-state-theory speed limit, and $\Delta G^{\ddagger} \geq 0$. $k_{cat}$ and $K_m$ are then derived deterministically from these rates via the Eyring and Briggs-Haldane relations, rather than predicted as unconstrained scalars.
- **Unified Kinetics Database:** A curated database pairs $k_{cat}$ and $K_m$ measurements from matching assay conditions (protein, substrate, pH, temperature, mutation state), so the loss function always operates on a physically coherent reaction state.

## Repository Layout

```text
├── data/                     # Curation scripts and datasets (most cache-type files are gitignored, gitkeep used to maintain skeleton)
│   ├── cache/                # Cached ESM2/ChemBERTa-2 embeddings, PDBs, and SMILES
│   ├── cache_eval/           # Cached embeddings/PDBs specifically for the withheld benchmark set
│   ├── external_predictions/ # CSV outputs from running external baseline models
│   ├── processed/            # Cleaned outputs
│   │   └── tensors/          # PyG HeteroData .pt ML-ready tensors
│   ├── raw/                  # Raw SABIO-RK and BRENDA downloads
│   ├── results/              # Pipeline outputs (*_summary.txt)
│   ├── enzyme_targets.json   # Curated list of target enzymes
│   ├── thermokp_database.db  # Unified SQLite thermodynamics database (Git LFS)
│   └── failed_*.txt          # Pipeline attrition logs (chemicals, sequences, structures)
├── external/                 # External baseline model repositories for benchmarking
│   ├── catpred/              # CatPred codebase and environment
│   ├── dlkcat/               # DLKcat codebase and environment
│   └── unikp/                # UniKP codebase and environment
├── models/                   # Model checkpoints and weights (Git LFS)
│   ├── {best,final}_model.pth             # Physics-informed ThermoKP weights
│   ├── {best,final}_no_mutation_model.pth # Mutation feature ablation weights
│   └── {best,final}_baseline_model.pth    # Non-physics baseline ablation weights
├── tools/
│   └── p2rank/               # P2Rank binary (external dependency)
├── src/                      # Core source code
│   ├── data/                 # Dataset parsers, processors, utilities, and run scripts
│   ├── encoders/             # Neural network model components
│   ├── evaluation/           # Benchmarking and metrics
│   ├── physics/              # Thermodynamic constraints and multi-task loss
│   └── training/             # Optimizer configurations and training loops
├── train.py                  # Main training script
├── train_no_mutation_nn.py   # Mutation feature ablation training script (see ARCHITECTURE.md)
├── train_baseline_nn.py      # Physics feature baseline training script (see ARCHITECTURE.md)
├── thermokp.py               # Inference script
├── dashboard/                # Streamlit dashboard for interactive inference
├── images/                   # Figure generation for publication
├── ARCHITECTURE.md           # Model architecture
├── TASKS.md                  # Project task list and tracking
└── README.md                 # Project overview
```

### External Prerequisite: P2Rank

The 3D structural pipeline (`src/data/processors/geometry_processor.py`) predicts ligand-binding pockets using [P2Rank](https://github.com/rdk/p2rank). You can easily install it by running the `tools/install_tools.sh` script, which will download and configure P2Rank v2.5.1 under `tools/p2rank/` for you.

### Chemical Name Resolution

Before doing any data processing or zero-shot inference, you should run `uv run python3 -m src.data.utils.sync_brenda_ligand_names` and `uv run python3 -m src.data.utils.sync_sabio_ligand_names` once. These scripts pull SMILES strings for the chemical entries from those databases and cache them under `data/cache/smiles_cache.json`, making substrate name to SMILES conversion more reliable.

## Execution Workflow

1. **Local Development:** Data acquisition, curation, and tensor generation run locally via the scripts under `src/data/scripts/`.
2. **Cloud Training:** Pull this repository onto a GPU-equipped machine and run `uv run python3 -m train` (and optionally the ablation controls `uv run python3 -m train_no_mutation_nn` and `uv run python3 -m train_baseline_nn`) to train the full network.
3. **Zero-Shot Inference:** Run `uv run python3 -m thermokp` with a UniProt ID (or raw sequence and structure files) and a substrate list (database-style chemical names or SMILES strings) to receive instant, biophysically bounded $k_{cat}$ & $K_m$ predictions. It also supports bulk prediction through YAML, CSV, or JSON input and output files. Run `uv run python3 -m src.evaluation.evaluate_dataset` for evaluating this pipeline against the withheld benchmark holdout.
4. **Interactive Dashboard:** Run `uv run -m streamlit run dashboard/app.py` for a local, interactive version of the same zero-shot inference path — single-query and CSV batch predictions, a PINN/baseline model toggle, derived $k_{a} = k_{cat} / K_m$ and (PINN-only) $k_1$, $k_{-1}$, $\Delta G^{\ddagger}$, $\kappa$ quantities, and an interactive py3Dmol view of the predicted structure, binding pocket, and any queried mutation site.

## Data Sources

### Kinetic Parameters
- **BRENDA:** Primary source for $k_{cat}$ and $K_m$ values, along with temperature, pH, and mutant annotations.
- **SABIO-RK:** Secondary source for kinetic parameters featuring highly structured reaction metadata.

### Chemical Identifiers (SMILES)
- **BRENDA & SABIO-RK:** Direct structural annotations provided within the kinetic databases.
- **ChEBI:** Primary resolver for exact SMILES representations via database cross-references.
- **KEGG:** Secondary resolver for exact SMILES representations via database cross-references.
- **PubChem:** Primary text-to-structure resolver for compound names lacking database cross-references.
- **OPSIN:** Fast IUPAC text-to-structure resolver for systematic chemical names.
- **CACTUS (NCI/CADD):** Fallback text-to-structure resolver for edge-case compound names.

### Protein Sequence & Structure
- **UniProt:** Sequence retrieval, curated binding-site annotations, and EC-number resolution.
- **AlphaFold DB / ESM Metagenomic Atlas:** Predicted protein 3D structures (with ESMFold as a fallback).
- **P2Rank:** Ligand-binding pocket prediction to crop protein structures to their catalytic sites.

### Pretrained Models & Tooling
- **RDKit:** Generation of 2D molecular graphs and 3D ligand conformers.
- **ESM2 (`facebook/esm2_t33_650M_UR50D`) / ChemBERTa-2 (`DeepChem/ChemBERTa-77M-MLM`):** Frozen pretrained protein and ligand embeddings via HuggingFace `transformers`.

## License

This project is open-source under the MIT License.
