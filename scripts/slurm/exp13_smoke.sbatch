#!/bin/bash
#SBATCH --job-name=x13_smoke
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=01:30:00
# exp13 Phase 1 smoke: ~10k learning env steps on transport for each new option, 5 runs in one MPS pack.
# Warm start shrunk to 250 chunks/env (8k env steps); total 18k env steps; regular eval every 4k env steps.
set -uo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export MUJOCO_GL=egl WANDB_MODE=disabled ADSRL_RUNS_PER_GPU=5
LOGROOT=/n/netscratch/olveczky_lab/Lab/charleszhang/exp13_smoke/$SLURM_JOB_ID
SCR=$LOGROOT/mps; mkdir -p $SCR
echo "host: $(hostname)"; nvidia-smi
export CUDA_MPS_PIPE_DIRECTORY=$SCR/pipe CUDA_MPS_LOG_DIRECTORY=$SCR/log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d && echo "MPS daemon started"
COMMON="--config-name=transport model.ddim_steps=20 env.n_eval_envs=25 use_wandb=false log_dir=$LOGROOT
 train.init_rollout_steps=250 train.total_timesteps=18000 eval_interval=125 +log_train_stdout=true seed=1"
t0=$(date +%s)
run() { local n=$1; shift; python train_adsrl.py $COMMON name=$n "$@" >> $LOGROOT/$n.log 2>&1; local rc=$?; echo "[$(date +%T)] $n exit $rc after $(( $(date +%s) - t0 ))s"; return $rc; }
run k3_kick_zi steer.mode=fixed steer.k=3 steer.param=kick steer.actor_zero_init=true &
run k3_abs_zi steer.mode=fixed steer.k=3 steer.actor_zero_init=true &
run adaptive_kick_zi steer.mode=adaptive steer.param=kick steer.actor_zero_init=true num_evals=50 &
run k3_ent005 steer.mode=fixed steer.k=3 train.ent_coef=auto_0.05 num_evals=50 &
# early evals, then resume to 26k from checkpoint/latest.pt (early window continues to 22k)
( run k3_early steer.mode=fixed steer.k=3 num_evals=50 train.early_eval_every=2000 train.early_eval_until=22000 \
  && run k3_early steer.mode=fixed steer.k=3 num_evals=50 train.early_eval_every=2000 train.early_eval_until=22000 \
     resume=true train.total_timesteps=26000 ) &
wait
echo quit | nvidia-cuda-mps-control
echo "done in $(( $(date +%s) - t0 ))s"
