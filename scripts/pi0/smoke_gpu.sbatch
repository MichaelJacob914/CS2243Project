#!/bin/bash
# Usage (one partition per submission):
#   sbatch -p kempner_h100 -A kempner_pehlevan_lab scripts/pi0/smoke_gpu.sbatch
#SBATCH --job-name=p7_smoke
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
nvidia-smi
python scripts/pi0/smoke_gpu.py --task_id 0 --out results/exp7/smoke_gpu.jsonl
