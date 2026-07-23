#!/bin/bash
set -e

# Change to project root
cd "$(dirname "$0")/../../.."

echo "============================================================"
echo "Running Dataset Validator..."
echo "============================================================"
uv run python -m src.data.processors.dataset_validator > data/results/validator_summary.txt 2>&1
echo "Validation completed successfully."

echo "============================================================"
echo "Running Generate Tensors..."
echo "============================================================"
uv run python -m src.data.processors.generate_tensors > data/results/tensors_summary.txt 2>&1
echo "Tensors generated successfully."

echo "Processors completed."
