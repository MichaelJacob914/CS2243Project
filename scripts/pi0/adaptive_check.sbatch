#!/bin/bash
# Adaptive-k pipeline + resume check on one task. Resubmitting the same command resumes from the checkpoint.
#   TASK=4 WALL_H=0.75 sbatch -p kempner_h100 -A kempner_pehlevan_lab scripts/pi0/adaptive_check.sbatch
#SBATCH --job-name=p7_adapt
#SBATCH --output=logs/exp7/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
source .venv-pi0/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
python train_adsrl_pi0.py --task_id ${TASK:?} --steer.mode adaptive --steer.rho 0.5 --n_envs 12 \
    --total_env_steps ${STEPS:-400000} --eval_every ${EVAL_EVERY:-25000} --max_wall_hours ${WALL_H:-7.6} \
    --name ${NAME:-adaptcheck_t${TASK}} "$@"
