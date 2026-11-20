#!/bin/bash
# Build .venv-pi05 (Python 3.11) for adaptive-step steering of pi0.5 on LIBERO, with upstream openpi (which has
# pi0.5; the dsrl_pi0/openpi fork does not). .venv-pi0 is untouched.
#   sbatch -p shared -A olveczky_lab -c 8 --mem=32G -t 2:00:00 -o logs/exp11/install_%j.out scripts/pi05/install_venv_pi05.sh
# Upstream openpi is cloned into third_party/openpi_upstream at OPENPI_COMMIT (LFS skipped).
# jaxrl2 (dsrl_pi0, for the SAC agent) is installed without dsrl_pi0/requirements.txt: its pins (jax==0.5.0,
# orbax 0.11.1, ml_dtypes 0.5.1, tensorflow 2.19) conflict with upstream openpi (jax 0.5.3, orbax 0.11.13,
# ml-dtypes override 0.4.1). jaxrl2's runtime imports only need distrax, tensorflow-probability (jax substrate),
# gym, matplotlib, wandb, which are added here at versions compatible with openpi. lerobot and gym-aloha are dropped via
# overrides.txt (training-only / unused; lerobot needs rerun-sdk wheels for glibc>=2.31, cluster is 2.28).
set -euxo pipefail
ROOT=/n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
OPENPI_COMMIT=215abfb217dbac7d5f1273282331b9b1866c0479
cd "$ROOT"
export UV_CACHE_DIR=/n/holylfs06/LABS/olveczky_lab/Users/charleszhang/.cache/uv
export UV_LINK_MODE=copy
export GIT_LFS_SKIP_SMUDGE=1

[ -d third_party/openpi_upstream ] || git clone https://github.com/Physical-Intelligence/openpi.git third_party/openpi_upstream
git -C third_party/openpi_upstream checkout -q "$OPENPI_COMMIT"

[ -d .venv-pi05 ] || uv venv .venv-pi05 --python 3.11.11
source .venv-pi05/bin/activate

# openpi pins torch==2.7.1; the CPU wheel is enough (LIBERO torch.load of init states; we use the JAX model).
uv pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cpu
uv pip install -e third_party/openpi_upstream -e third_party/openpi_upstream/packages/openpi-client \
    -e dsrl_pi0 -e dsrl_pi0/LIBERO \
    "distrax==0.1.5" "tensorflow-probability==0.25.0" "gym==0.26.2" matplotlib \
    "robosuite==1.4.1" "mujoco==3.3.1" "bddl==3.5.0" easydict future h5py pytest \
    --override scripts/pi05/overrides.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

# LIBERO has no top-level libero/__init__.py; put the repo root on sys.path (same as .venv-pi0).
echo "$ROOT/dsrl_pi0/LIBERO" > .venv-pi05/lib/python3.11/site-packages/libero_root.pth

uv pip freeze > requirements-lock-pi05.txt
python -c "import jax, flax, openpi, libero, robosuite, mujoco, jaxrl2, distrax; from openpi.training import config; c = config.get_config('pi05_libero'); print('jax', jax.__version__, 'flax', flax.__version__, 'mujoco', mujoco.__version__, c.model)"
python -c "import sys; sys.path.insert(0, '$ROOT'); import adsrl_pi0.agent; print('adsrl_pi0.agent imports OK')"

# pi0.5 LIBERO checkpoint into scratch.
export OPENPI_DATA_HOME=/n/netscratch/olveczky_lab/Lab/charleszhang/openpi_cache
python -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero', token='anon'))"
du -sh $OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero
