#!/bin/bash
#SBATCH --job-name=f6
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
# exp_6 (Furniture-Bench): one pack of train_adsrl.py runs on one GPU, each its own IsaacGym process,
# under CUDA MPS. Usage (from project root):
#   sbatch -J f6_<pack> scripts/slurm/exp6_pack.sbatch <task> "<name>::<hydra overrides>" ["<name>::<overrides>" ...]
# e.g. scripts/slurm/exp6_pack.sbatch one_leg_low "k0_n1024_u8::steer.mode=fixed steer.k=0 env.n_envs=1024 train.utd=8"
# MPS=0 in the environment disables MPS. ADSRL_EXTRA adds overrides to every run.
# Logdirs: $LOGROOT/adsrl/<task>/<name>; per-run stdout in logs/exp6/<job>_<task>_<id>/<name>.log,
# per-process GPU memory every 30 s in gpu_mem.csv there.
set -uo pipefail
TASK=${1:?task}; shift
[ $# -ge 1 ] || { echo "no runs given"; exit 1; }
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source scripts/furniture/env.sh  # venv, libpython, gcc 12 for gymtorch
export WANDB_MODE=offline ADSRL_RUNS_PER_GPU=$# PYTHONUNBUFFERED=1
LOGROOT=${LOGROOT:-/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs/exp6}
SCR=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl/exp6_${SLURM_JOB_ID}; mkdir -p $SCR
OUT=$ROOT/logs/exp6/${SLURM_JOB_NAME}_${TASK}_${SLURM_JOB_ID}; mkdir -p $OUT
echo "git: $(git rev-parse HEAD) host: $(hostname) task: $TASK runs: $#"; nvidia-smi
if [ "${MPS:-1}" = 1 ]; then
  export CUDA_MPS_PIPE_DIRECTORY=$SCR/mps_pipe CUDA_MPS_LOG_DIRECTORY=$SCR/mps_log
  mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  nvidia-cuda-mps-control -d && echo "MPS daemon started"
fi
nvidia-smi dmon -s um -d 30 -o T > $OUT/dmon.txt 2>&1 &
DMON=$!
( while true; do nvidia-smi --query-compute-apps=timestamp,pid,process_name,used_memory --format=csv,noheader >> $OUT/gpu_mem.csv 2>&1; sleep 30; done ) &
MEMLOG=$!
COMMON="--config-name=$TASK wandb.mode=offline log_dir=$LOGROOT resume=true ${ADSRL_EXTRA:-}"
t0=$(date +%s)
PIDS=()
for spec in "$@"; do
  name=${spec%%::*}; ov=${spec#*::}; [ "$ov" = "$spec" ] && ov=""
  echo "$name: $ov"
  if [ "${SEQ:-0}" = 1 ]; then  # throughput pilot: one run at a time, each cut at RUN_SEC seconds
    export ADSRL_RUNS_PER_GPU=1
    timeout ${RUN_SEC:-900} python train_adsrl.py $COMMON name=$name $ov >> $OUT/$name.log 2>&1
    echo "[$(date +%T)] $name exit $? after $(( $(date +%s) - t0 ))s"
    continue
  fi
  # retry loop: IsaacGym's PhysX occasionally dies mid-run (GJK/EPA kernel failure, seen on H100 after
  # hours); relaunch with resume=true from latest.pt until checkpoint/final.pt exists (RETRIES attempts)
  ( for attempt in $(seq 1 ${RETRIES:-5}); do
      python train_adsrl.py $COMMON name=$name $ov >> $OUT/$name.log 2>&1
      [ -f $LOGROOT/adsrl/$TASK/$name/checkpoint/final.pt ] && break
      echo "[$(date +%T)] $name: no final.pt after attempt $attempt; resuming" | tee -a $OUT/$name.log
    done ) &
  PIDS+=($!); echo "started $name pid $!"
done
for p in "${PIDS[@]}"; do wait $p; echo "[$(date +%T)] pid $p exit $? after $(( $(date +%s) - t0 ))s"; done
kill $DMON $MEMLOG 2>/dev/null
[ "${MPS:-1}" = 1 ] && echo quit | nvidia-cuda-mps-control
echo "done in $(( $(date +%s) - t0 ))s"
