#!/bin/bash
#SBATCH --job-name=p9_t4_ad_lf3_abase_b2.5
#SBATCH --output=logs/exp9/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
source .venv-pi0/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
nvidia-smi --query-gpu=name --format=csv
python train_adsrl_pi0.py --task_id 4 --steer.mode adaptive --steer.rho 0.5 --steer.xt_layout full \
    --steer.w_layout lowfreq --steer.n_modes 3 --steer.obs_extra abase --action_magnitude 2.5 --seed 0 \
    --total_env_steps 1000000 --n_envs 12 --eval_every 50000 --eval_det 1 --eval_at_start 1 \
    --max_wall_hours 7.6 --logdir_root /n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs/exp9 --wandb_group exp9_pi0
