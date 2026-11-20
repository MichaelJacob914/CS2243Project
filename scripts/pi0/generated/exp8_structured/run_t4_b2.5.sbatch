#!/bin/bash
#SBATCH --job-name=p8_t4_b2.5
#SBATCH --output=logs/exp8/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
source .venv-pi0/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
nvidia-smi --query-gpu=name --format=csv
python train_adsrl_pi0.py --task_id 4 --steer.mode fixed --steer.k 0 --steer.xt_layout full \
    --steer.w_layout structured --action_magnitude 2.5 --seed 0 \
    --total_env_steps 1000000 --n_envs 12 --eval_every 50000 --eval_det 1 \
    --max_wall_hours 5.6 --logdir_root /n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs/exp8 --wandb_group exp8_pi0
