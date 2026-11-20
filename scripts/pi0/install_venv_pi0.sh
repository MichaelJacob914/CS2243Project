#!/bin/bash
# Build .venv-pi0 (Python 3.11) for DSRL-pi0 / adaptive-step steering on LIBERO.
# Run inside a CPU job, e.g.:
#   sbatch -p shared -A olveczky_lab -c 8 --mem=32G -t 2:00:00 -o logs/exp7/install_%j.out scripts/pi0/install_venv_pi0.sh
# Nested submodules (dsrl_pi0/openpi, dsrl_pi0/LIBERO) must be checked out first:
#   cd dsrl_pi0 && git -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init
set -euxo pipefail
ROOT=/n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
cd "$ROOT"
export UV_CACHE_DIR=/n/holylfs06/LABS/olveczky_lab/Users/charleszhang/.cache/uv
export UV_LINK_MODE=copy
export GIT_LFS_SKIP_SMUDGE=1

[ -d .venv-pi0 ] || uv venv .venv-pi0 --python 3.11.11
source .venv-pi0/bin/activate

# CPU torch first (LIBERO needs torch for torch.load of init states; openpi needs torch>=2.5.1).
uv pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu
# dsrl_pi0 (jaxrl2) + its pinned requirements. The README installs these with sequential pip calls, where later
# pins win; uv resolves jointly, so scripts/pi0/overrides.txt applies the same outcome: pillow>=11 (openpi) over
# dsrl's pillow==10.4.0, and mujoco==3.3.1 (run_libero.sh) over gym-aloha's mujoco<3 (aloha is unused).
# dsrl_pi0 (jaxrl2) + its pinned requirements, openpi, openpi-client, LIBERO, jax with CUDA 12.
uv pip install -e dsrl_pi0 -r dsrl_pi0/requirements.txt "jax[cuda12]==0.5.0" \
    -e dsrl_pi0/openpi -e dsrl_pi0/openpi/packages/openpi-client -e dsrl_pi0/LIBERO \
    "mujoco==3.3.1" "robosuite==1.4.1" pytest --override scripts/pi0/overrides.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

# LIBERO has no top-level libero/__init__.py, so its editable install maps nothing; put the repo root on sys.path.
echo "$ROOT/dsrl_pi0/LIBERO" > .venv-pi0/lib/python3.11/site-packages/libero_root.pth

# LIBERO config without the interactive prompt.
mkdir -p "$ROOT/.libero"
LIB=$ROOT/dsrl_pi0/LIBERO/libero/libero
cat > "$ROOT/.libero/config.yaml" <<YAML
benchmark_root: $LIB
bddl_files: $LIB/bddl_files
init_states: $LIB/init_files
datasets: $LIB/../datasets
assets: $LIB/assets
YAML

uv pip freeze > requirements-lock-pi0.txt
python -c "import jax, flax, openpi, libero, robosuite, mujoco; print('jax', jax.__version__, 'flax', flax.__version__, 'mujoco', mujoco.__version__)"

# pi0 LIBERO checkpoint into scratch (s3://openpi-assets no longer exists; see scripts/pi0/download_ckpt.sh).
bash scripts/pi0/download_ckpt.sh
