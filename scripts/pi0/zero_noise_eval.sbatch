#!/bin/bash
# Zero-noise-at-k baseline (eval only), tasks 8 4 6 in sequence. Rerun to continue (skips finished conditions).
#   sbatch scripts/pi0/zero_noise_eval.sbatch
#SBATCH --job-name=p7_zeronoise
#SBATCH --output=logs/exp7/%x_%j.out
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:30:00
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
source .venv-pi0/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
nvidia-smi --query-gpu=name --format=csv
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi blackwell; then
  export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
fi
python scripts/pi0/zero_noise_eval.py --tasks 8 4 6 --episodes 200 --n_envs 12
python scripts/pi0/plot_zero_noise.py
