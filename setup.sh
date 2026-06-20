#!/bin/bash
set -e

# OcbGS setup.
#
# This script reuses the server image's PRE-INSTALLED PyTorch environment
# (target image: PyTorch 2.5.1 + CUDA 12.4, Python 3.12, RTX 4090 / sm_89).
# It does NOT create a conda env and does NOT reinstall torch — activate the
# torch environment yourself before running this. It then:
#   1. installs torch-scatter from the matching pyg wheel,
#   2. installs the pip dependencies,
#   3. initializes the GLM submodule,
#   4. compiles the diff-gaussian-rasterization and simple-knn CUDA extensions.
#
# Usage: bash setup.sh        (run from the repository root)
#
# Override the build target arch if you are not on an RTX 4090:
#   TORCH_CUDA_ARCH_LIST="8.6" bash setup.sh

echo "=== Verifying PyTorch / CUDA ==="
python - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit("ERROR: no 'torch' in the active environment. Activate the image's "
             "PyTorch env (or `conda activate <env>`) before running setup.sh.")
print(f"torch {torch.__version__} | CUDA build {torch.version.cuda} | "
      f"cuda.is_available()={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("WARNING: CUDA not visible — the CUDA extensions will compile but "
          "cannot run. Continuing anyway.", file=sys.stderr)
PY

echo "=== Installing torch-scatter (wheel matched to the active torch) ==="
TORCH_VER="$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
CUDA_TAG="$(python -c 'import torch; print("cu" + (torch.version.cuda or "").replace(".", ""))')"
echo "    target wheel index: torch-${TORCH_VER}+${CUDA_TAG}"
pip install torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html"

echo "=== Installing pip dependencies ==="
pip install einops wandb lpips laspy jaxtyping colorama opencv-python plyfile tqdm pytest

echo "=== Initializing GLM submodule (required to build the rasterizer) ==="
git submodule update --init ocbgs/submodules/diff-gaussian-rasterization/third_party/glm

echo "=== Building CUDA extensions (TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}) ==="
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
cd ocbgs/submodules/diff-gaussian-rasterization
pip install -e .
cd ../simple-knn
pip install -e .
cd ../../..

echo "=== Setup complete ==="
echo "Run tests:    cd ocbgs && python -m pytest tests/test_00_walking_skeleton.py -v"
echo "Run training: cd ocbgs && python train.py -s <dataset_path>"