#!/bin/bash
# DSRL k=0 learning check on one task (DSRL-pi0 hyperparameters).
#   TASK=3 sbatch -p kempner -A kempner_pehlevan_lab scripts/pi0/learn_check.sbatch [extra args]
#SBATCH --job-name=p7_learn
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
python train_adsrl_pi0.py --task_id ${TASK:?} --steer.mode fixed --steer.k 0 --n_envs ${N_ENVS:-12} \
    --total_env_steps ${STEPS:-400000} --eval_every ${EVAL_EVERY:-25000} --eval_at_start 1 --max_wall_hours 7.6 \
    --name ${NAME:-learncheck_t${TASK}_k0} "$@"
