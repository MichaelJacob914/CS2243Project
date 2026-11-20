#!/bin/bash
# GPU check of --hybrid.k_entropy_bias detach (adaptive, lowfreq m=3, abase, b=2.5, task 8): 20k env steps, evals
# with the per-k head log_std log.
#   sbatch -p kempner_h100 -A kempner_pehlevan_lab scripts/pi0/smoke_exp9_ad_detach.sbatch
#SBATCH --job-name=p9_smoke_addet
#SBATCH --output=logs/exp9/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:40:00
set -e
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
source .venv-pi0/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
ROOT_DIR=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs/exp9_smoke_ad
NAME=smoke_t8_ad_lf3_abase_b2.5_kdetach
rm -rf $ROOT_DIR/$NAME
python train_adsrl_pi0.py --task_id 8 --steer.mode adaptive --steer.rho 0.5 --steer.xt_layout full \
    --steer.w_layout lowfreq --steer.n_modes 3 --steer.obs_extra abase --action_magnitude 2.5 \
    --hybrid.k_entropy_bias detach --n_envs 12 --total_env_steps 20000 --eval_every 10000 --eval_episodes 12 \
    --eval_det 1 --logdir_root $ROOT_DIR --name $NAME --use_wandb 0
D=$ROOT_DIR/$NAME
cat $D/eval_summary.csv $D/head_log_std.csv
python -c "import json; d = json.load(open('$D/run_info.json')); print(d['k_entropy_bias'], d['args']['k_entropy_bias'])"
