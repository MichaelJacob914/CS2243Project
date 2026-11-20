#!/bin/bash
# pi0.5 steering tests (+ the pi0 agent tests, which only need jaxrl2) in .venv-pi05.
#   sbatch -p gpu -A olveczky_lab scripts/pi05/tests.sbatch
#SBATCH --job-name=p11_tests
#SBATCH --output=logs/exp11/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
source .venv-pi05/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/n/netscratch/olveczky_lab/Lab/charleszhang/jax_cache_pi05
nvidia-smi --query-gpu=name --format=csv,noheader
python -m pytest -q -s -p no:cacheprovider tests/test_pi05_sampler.py tests/test_pi0_agent.py "$@"
