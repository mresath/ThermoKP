#!/bin/bash
set -e

# Change to project root
cd "$(dirname "$0")/../../.."

echo "============================================================"
echo "Running BRENDA Parser..."
echo "============================================================"
uv run python -m src.data.parsers.brenda_parser --full > data/results/raw_brenda.txt 2>&1
echo "BRENDA Parser completed successfully."

echo "============================================================"
echo "Running SABIO-RK Parser..."
echo "============================================================"
uv run python -m src.data.parsers.sabio_rk_parser --full > data/results/raw_sabio.txt 2>&1
echo "SABIO-RK Parser completed successfully."

echo "============================================================"
echo "Generating Raw Ingestion Summary..."
echo "============================================================"
uv run python -m src.data.parsers.raw_summary > data/results/raw_summary.txt

echo "============================================================"
echo "Running Data Cleanup & Aggregation..."
echo "============================================================"
uv run python -m src.data.processors.clean_records > data/results/clean_summary.txt 2>&1
echo "Cleanup completed successfully."

echo "Parsers completed."
