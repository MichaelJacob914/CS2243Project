#!/bin/bash
# Short end-to-end smoke of train_adsrl_pi0.py (adaptive, full x_T, eval at start, evals every 15k).
#   sbatch -p kempner_h100 -A kempner_pehlevan_lab scripts/pi0/smoke_train.sbatch   (resubmit to resume)
#SBATCH --job-name=p7_smoketrain
#SBATCH --output=logs/exp7/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
source .venv-pi0/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
python train_adsrl_pi0.py --task_id 4 --steer.mode adaptive --steer.rho 0.5 --steer.xt_layout full --n_envs 12 \
    --total_env_steps 60000 --eval_every 15000 --eval_det 1 --eval_at_start 1 --max_wall_hours 0.9 \
    --name smoke_adapt_xtfull
