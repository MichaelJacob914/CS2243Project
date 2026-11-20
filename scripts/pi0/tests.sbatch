#!/bin/bash
#   sbatch -p kempner_h100 -A kempner_pehlevan_lab scripts/pi0/tests.sbatch
#SBATCH --job-name=p7_tests
#SBATCH --output=logs/exp7/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
source .venv-pi0/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m pytest -x -q -p no:cacheprovider tests/test_pi0_agent.py tests/test_pi0_sampler.py "$@"
