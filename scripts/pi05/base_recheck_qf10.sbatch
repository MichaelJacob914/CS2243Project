#!/bin/bash
# exp12 step 1: plain pi0.5 at query_freq 10 (full 10-action chunk) on the exp12 tasks, 50 eps each.
#   sbatch -p gpu_h200 -A olveczky_lab scripts/pi05/base_recheck_qf10.sbatch
#SBATCH --job-name=p12_base_qf10
#SBATCH --output=logs/exp12/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=01:30:00
set -e
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
source .venv-pi05/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/n/netscratch/olveczky_lab/Lab/charleszhang/jax_cache_pi05
echo "host $(hostname) job $SLURM_JOB_ID"; nvidia-smi --query-gpu=name --format=csv,noheader
python scripts/pi0/base_eval.py --model pi05 --noise full --query_freq 10 --suite libero_10 --tasks 8 \
    --episodes 50 --n_envs 25 --out results/exp12/base_eval_qf10_libero10.csv
python scripts/pi0/base_eval.py --model pi05 --noise full --query_freq 10 --suite libero_90 --tasks 14 28 77 \
    --episodes 50 --n_envs 25 --out results/exp12/base_eval_qf10_libero90.csv
