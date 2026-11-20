#!/bin/bash
# GPU smoke of exp9 (k=3, lowfreq m=3, abase, b=2.5, task 8): train 20k env steps (evals + checkpoints), resume to
# 30k, the real-checkpoint plan-feature tests, and per-round timing for the throughput estimate.
#   sbatch -p kempner_h100 -A kempner_pehlevan_lab scripts/pi0/smoke_exp9.sbatch
#SBATCH --job-name=p9_smoke
#SBATCH --output=logs/exp9/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -e
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
source .venv-pi0/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
nvidia-smi --query-gpu=name --format=csv
ROOT_DIR=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs/exp9_smoke
NAME=smoke_t8_k3_lf3_abase_b2.5
rm -rf $ROOT_DIR/$NAME
ARGS="--task_id 8 --steer.mode fixed --steer.k 3 --steer.xt_layout full --steer.w_layout lowfreq --steer.n_modes 3 \
    --steer.obs_extra abase --action_magnitude 2.5 --n_envs 12 --eval_every 10000 --eval_episodes 12 --eval_det 1 \
    --logdir_root $ROOT_DIR --name $NAME --use_wandb 0"
python train_adsrl_pi0.py $ARGS --total_env_steps 20000
echo "=== resume to 30k"
python train_adsrl_pi0.py $ARGS --total_env_steps 30000
D=$ROOT_DIR/$NAME
ls $D $D/checkpoint; cat $D/eval_summary.csv; cat $D/train_log.csv
python -c "
import json; d = json.load(open('$D/run_info.json'))
print({k: d[k] for k in ['w_layout', 'n_modes', 'obs_extra', 'action_dim', 'obs_keys', 'use_xt_obs', 'action_bound']})"
echo "=== ckpt plan-feature tests"
python -m pytest -q -p no:cacheprovider -W ignore tests/test_pi0_sampler.py -k "ckpt" 2>&1 | tail -5
