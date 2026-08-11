#!/bin/bash
set -e

# ==============================================================================
# ThermoKP - External Model Inference Orchestrator
# ==============================================================================
# Iterates through DLKcat, UniKP, and CatPred to generate standard CSV
# outputs in data/external_predictions/. Gracefully skips models if dependencies
# or virtual environments are not properly initialized.
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="$PROJECT_ROOT/external"
PREDICTIONS_DIR="$PROJECT_ROOT/data/external_predictions"

mkdir -p "$PREDICTIONS_DIR"
cd "$PROJECT_ROOT"

echo "==========================================================================="
echo "Starting ThermoKP Benchmark Cross-Evaluation"
echo "==========================================================================="

echo "[0/4] Exporting common benchmark inputs (sequences, SMILES, PDB paths)..."
uv run python "$PROJECT_ROOT/src/evaluation/export_benchmark_csv.py" || { echo "Input export failed."; exit 1; }

# 1. DLKcat
echo "[1/4] Running DLKcat inference..."
if [ -d "$EXTERNAL_DIR/dlkcat/.venv" ] && [ -f "$EXTERNAL_DIR/dlkcat/run_inference.py" ]; then
    "$EXTERNAL_DIR/dlkcat/.venv/bin/python" "$EXTERNAL_DIR/dlkcat/run_inference.py" || echo "DLKcat inference failed."
else
    echo "DLKcat isolated venv (.venv) not found. Skipping."
fi

# 2. UniKP
echo "[2/4] Running UniKP inference..."
if [ -d "$EXTERNAL_DIR/unikp/.venv" ] && [ -f "$EXTERNAL_DIR/unikp/run_inference.py" ]; then
    "$EXTERNAL_DIR/unikp/.venv/bin/python" "$EXTERNAL_DIR/unikp/run_inference.py" || echo "UniKP inference failed."
else
    echo "UniKP isolated venv (.venv) not found. Skipping."
fi

# 3. CatPred
echo "[3/4] Running CatPred inference..."
if [ -d "$EXTERNAL_DIR/catpred/.venv" ] && [ -f "$EXTERNAL_DIR/catpred/run_inference.py" ]; then
    "$EXTERNAL_DIR/catpred/.venv/bin/python" "$EXTERNAL_DIR/catpred/run_inference.py" || echo "CatPred inference failed."
else
    echo "CatPred isolated venv (.venv) not found. Skipping."
fi

echo "==========================================================================="
echo "All model inferences completed. Check data/external_predictions/"
echo "==========================================================================="
