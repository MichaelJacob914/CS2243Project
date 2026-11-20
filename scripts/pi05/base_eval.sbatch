#!/bin/bash
# Plain pi0.5 (full Gaussian noise) success on LIBERO tasks. Pass base_eval.py args, e.g.
#   sbatch -p gpu -A olveczky_lab scripts/pi05/base_eval.sbatch --suite libero_90 --tasks $(seq 0 29) \
#       --episodes 20 --n_envs 20 --out results/exp11/base_eval_libero90_a.csv
#SBATCH --job-name=p11_base
#SBATCH --output=logs/exp11/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=06:00:00
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
source .venv-pi05/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/n/netscratch/olveczky_lab/Lab/charleszhang/jax_cache_pi05
echo "host $(hostname) job $SLURM_JOB_ID git $(git rev-parse HEAD)"; nvidia-smi --query-gpu=name --format=csv,noheader
python scripts/pi0/base_eval.py --model pi05 --noise full "$@"
