#!/bin/bash
# Builds .venv-furniture (Python 3.8) for the Furniture-Bench experiments (exp_6). Run on a compute node:
#   srun -p test -A olveczky_lab -c 8 --mem=16G -t 1:00:00 bash scripts/furniture/install_venv_furniture.sh
# Sources (all under third_party/, untracked):
#   isaacgym/                 IsaacGym Preview 4 (iai-robust-rearrangement S3 copy)
#   furniture-bench/          github.com/ankile/furniture-bench (commit in exp_6 Setup)
#   stable-baselines3-py38/   copy of dsrl/stable-baselines3 patched for Python 3.8 (see PY38_PATCH.txt)
# dppo is not installed (train_adsrl.py puts dsrl/dppo on sys.path); only its dependencies are.
set -euo pipefail
ROOT=/n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
cd $ROOT
export UV_CACHE_DIR=/n/netscratch/olveczky_lab/Lab/charleszhang/uv_cache
uv python install 3.8.20
[ -d .venv-furniture ] || uv venv .venv-furniture --python 3.8.20
source .venv-furniture/bin/activate
uv pip install "torch==2.4.0" "torchvision==0.19.0" --index-url https://download.pytorch.org/whl/cu121
uv pip install "numpy==1.23.5"  # isaacgym torch_utils uses np.float (removed in numpy 1.24)
uv pip install -e third_party/isaacgym/python "numpy==1.23.5"
uv pip install -e third_party/furniture-bench "numpy==1.23.5"
# dppo deps (pyproject pins) + furniture extra, without installing dppo itself
uv pip install "numpy==1.23.5" "torch==2.4.0" einops==0.8.0 gdown==5.2.0 gym==0.22.0 hydra-core==1.3.2 imageio==2.35.1 \
  matplotlib==3.7.5 omegaconf==2.3.0 pretty_errors==1.2.25 tqdm==4.66.5 wandb==0.17.7 "cython<3" ipdb==0.13.13
uv pip install -e third_party/stable-baselines3-py38 "gymnasium==1.0.0" "numpy==1.23.5" "torch==2.4.0" pytest
uv pip freeze > $ROOT/requirements-lock-furniture.txt
PYLIB=$(python -c "import sys; print(sys.base_prefix)")/lib
echo "libpython dir: $PYLIB"; ls $PYLIB | grep libpython || true
