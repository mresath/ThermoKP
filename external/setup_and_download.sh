#!/bin/bash
set -e

# ==============================================================================
# ThermoKP - External Model Setup & Download Script
# ==============================================================================
# This script clones the repositories for DLKcat, UniKP, and CatPred, and 
# downloads their associated datasets for sequence-identity tracking.
# ==============================================================================

EXTERNAL_DIR="$(dirname "$0")"
cd "$EXTERNAL_DIR"

echo "Setting up external models in $EXTERNAL_DIR..."

# ------------------------------------------------------------------------------
# 1. Initialize Submodules
# ------------------------------------------------------------------------------
echo "--- Initializing Git Submodules ---"
git submodule update --init --recursive

# ------------------------------------------------------------------------------
# 2. DLKcat Setup
# ------------------------------------------------------------------------------
echo "--- Unzipping DLKcat Dataset ---"
if [ ! -d "dlkcat/src/DeeplearningApproach/Data/input" ]; then
    cd dlkcat/src/DeeplearningApproach/Data && unzip input.zip && cd ../../../../
fi

echo "--- Creating DLKcat Isolated Virtual Environment ---"
if [ ! -d "dlkcat/.venv" ]; then
    uv venv dlkcat/.venv --python 3.9
fi
uv pip install --python dlkcat/.venv/bin/python torch torchvision pandas numpy rdkit scikit-learn requests

# ------------------------------------------------------------------------------
# 3. UniKP Setup
# ------------------------------------------------------------------------------
echo "--- Fetching UniKP Checkpoints ---"
mkdir -p unikp/UniKP
cd unikp/UniKP
if [ ! -f "UniKP for kcat.pkl" ]; then
    wget -q --show-progress "https://huggingface.co/HanselYu/UniKP/resolve/main/UniKP%20for%20kcat.pkl"
    wget -q --show-progress "https://huggingface.co/HanselYu/UniKP/resolve/main/UniKP%20for%20Km.pkl"
    wget -q --show-progress "https://huggingface.co/HanselYu/UniKP/resolve/main/UniKP%20for%20kcat_Km.pkl"
fi
cd ../../

echo "--- Creating UniKP Isolated Virtual Environment (x86_64) ---"
if [ ! -d "unikp/.venv" ]; then
    if [ "$(uname)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        arch -x86_64 /usr/bin/python3 -m venv unikp/.venv
    else
        uv venv unikp/.venv --python 3.9
    fi
fi

if [ "$(uname)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    arch -x86_64 unikp/.venv/bin/pip install torch torchvision transformers sentencepiece pandas "numpy<2" "scikit-learn==0.24.2" rdkit
else
    uv pip install --python unikp/.venv/bin/python torch torchvision transformers sentencepiece pandas "numpy<2" "scikit-learn==0.24.2" rdkit
fi

# ------------------------------------------------------------------------------
# 4. CatPred Setup
# ------------------------------------------------------------------------------
echo "--- Creating CatPred Isolated Virtual Environment ---"
# CatPred dependencies are tricky, we just use uv to install the basic ones 
# that are possible on this machine. (pyg, pytorch-scatter excluded for simple execution)
if [ ! -d "catpred/.venv" ]; then
    uv venv catpred/.venv --python 3.9
fi

echo "--- Setting up CatPred Dependencies ---"
if [ ! -d "catpred/src" ]; then
    git clone https://github.com/maranasgroup/CatPred.git catpred/src
fi

echo "--- Fetching CatPred-DB Pretrained Models ---"
if [ ! -d "catpred/data/pretrained/production/kcat" ]; then
    cd catpred
    if [ ! -f "capsule_data_update.tar.gz" ]; then
        echo "Downloading capsule_data_update.tar.gz (this may take a while)..."
        wget -q --show-progress -c --tries=5 --timeout=30 https://catpred.s3.us-east-1.amazonaws.com/capsule_data_update.tar.gz || \
        wget -q --show-progress -c --tries=5 --timeout=30 https://catpred.s3.amazonaws.com/capsule_data_update.tar.gz
    else
        echo "capsule_data_update.tar.gz already exists, skipping download."
    fi
    tar -xzf capsule_data_update.tar.gz
    rm capsule_data_update.tar.gz
    cd ..
fi

uv pip install --python catpred/.venv/bin/python torch torchvision transformers sentencepiece pandas numpy "scikit-learn>=0.22.2" rdkit fair-esm progres descriptastorus seaborn rotary_embedding_torch==0.6.5 typed-argument-parser>=1.6.1 faiss-cpu ipdb

echo "=============================================================================="
echo "External models successfully downloaded!"
echo "Next step: Run inference wrappers to extract sequence IDs and run predictions."
echo "=============================================================================="
