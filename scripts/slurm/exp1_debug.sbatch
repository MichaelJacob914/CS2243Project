#!/bin/bash
#SBATCH --job-name=exp1_debug
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --exclude=holygpu8a12204
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
# Tiny end-to-end check of every exp_1 code path on the real robomimic env (no MPS; sequential-ish).
set -uo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export MUJOCO_GL=egl WANDB_MODE=offline
SCR=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl/exp1_debug_${SLURM_JOB_ID}; mkdir -p $SCR
echo "git: $(git rev-parse HEAD) host: $(hostname)"
TINY="train.total_timesteps=4000 train.init_rollout_steps=50 eval_interval=50 num_evals=10 env.n_eval_envs=5 train.buffer_size=5000 log_dir=$SCR wandb.mode=offline"
# 1) fixed k=4 short, then resume to a longer total
python train_adsrl.py --config-name=can steer.mode=fixed steer.k=4 name=dbg_k4 $TINY > $SCR/k4.log 2>&1; echo "k4 exit $?"
python train_adsrl.py --config-name=can steer.mode=fixed steer.k=4 name=dbg_k4 $TINY train.total_timesteps=5600 resume=true > $SCR/k4_resume.log 2>&1; echo "k4 resume exit $?"
# 2) adaptive
python train_adsrl.py --config-name=can steer.mode=adaptive name=dbg_adapt $TINY > $SCR/adapt.log 2>&1 &
# 3) upstream with eval csv
(cd dsrl && python $ROOT/scripts/pilot_train_dsrl.py --config-path=$ROOT/dsrl/cfg/robomimic --config-name=dsrl_can.yaml \
  +total_timesteps=800 +buffer_size=5000 use_wandb=False deterministic_eval=True num_evals=10 env.n_eval_envs=5 \
  eval_interval=50 train.init_rollout_steps=50 save_model_interval=100000000 log_dir=$SCR name=dbg_up +eval_csv=$SCR/dbg_up/eval.csv > $SCR/up.log 2>&1; echo "up exit $?") &
# 4) base eval
python scripts/base_eval.py --task can --n_episodes 20 --n_envs 10 --out_dir $SCR/base_eval > $SCR/base.log 2>&1; echo "base can exit $?"
python scripts/base_eval.py --task transport --ddim_steps 10 --n_episodes 10 --n_envs 10 --out_dir $SCR/base_eval > $SCR/base_tr.log 2>&1; echo "base transport exit $?"
wait
echo "adapt done"; tail -5 $SCR/*.log
