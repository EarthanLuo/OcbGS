#!/bin/bash
set -e

CONDA_ENV_NAME="${1:-ocbgs}"

echo "=== Creating conda environment: ${CONDA_ENV_NAME} ==="
conda env create -f environment.yml -n "${CONDA_ENV_NAME}"

echo "=== Activating environment ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

echo "=== Building CUDA submodules ==="
cd ocbgs/submodules/diff-gaussian-rasterization
pip install -e .
cd ../simple-knn
pip install -e .
cd ../../..

echo "=== Setup complete ==="
echo "Activate with: conda activate ${CONDA_ENV_NAME}"
echo "Run training: cd ocbgs && python train.py -s <dataset_path>"
