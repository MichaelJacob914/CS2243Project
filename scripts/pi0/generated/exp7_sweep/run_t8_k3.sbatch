#!/bin/bash
#SBATCH --job-name=p7_t8_k3
#SBATCH --output=logs/exp7/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=05:15:00
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
source .venv-pi0/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
nvidia-smi --query-gpu=name --format=csv
python train_adsrl_pi0.py --task_id 8 --steer.mode fixed --steer.k 3 --steer.xt_layout full --seed 0 \
    --total_env_steps 1000000 --n_envs 12 --eval_every 50000 --eval_det 1 \
    --max_wall_hours 4.9
