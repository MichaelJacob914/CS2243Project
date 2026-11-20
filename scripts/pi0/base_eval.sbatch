#!/bin/bash
# Base pi0 success on all LIBERO-10 tasks, full and tiled noise.
#   sbatch -p kempner -A kempner_pehlevan_lab scripts/pi0/base_eval.sbatch [extra args]
#SBATCH --job-name=p7_base
#SBATCH --output=logs/exp7/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=128G
#SBATCH --time=06:00:00
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
source .venv-pi0/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
python scripts/pi0/base_eval.py --tasks 0 1 2 3 4 5 6 7 8 9 --episodes 50 --n_envs 12 --noise full tiled \
    --out results/exp7/base_eval_libero10.csv "$@"
