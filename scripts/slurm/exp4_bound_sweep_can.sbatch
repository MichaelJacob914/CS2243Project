#!/bin/bash
#SBATCH --job-name=e4_bound
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --exclude=holygpu8a12204
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=16:00:00
# exp_4 bound-scale sweep on can, seed 1, 1M env steps, MPS pack of 6 (same settings as exp_2 Wave 0).
# Usage (from project root):
#   sbatch scripts/slurm/exp4_bound_sweep_can.sbatch A   # fixed k=0,4 x w_scale 1.5,2,3
#   sbatch scripts/slurm/exp4_bound_sweep_can.sbatch B   # fixed k=6,7 x w_scale 1.5,2,3
# All runs pass resume=true: re-submitting the same pack continues from checkpoint/latest.pt.
# Logdirs: logs/adsrl/can/fixed_k<k>_c<c>_s1 (logs/adsrl is a symlink to scratch).
set -uo pipefail
PACK=${1:?pack A or B}
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export MUJOCO_GL=egl WANDB_MODE=offline ADSRL_RUNS_PER_GPU=6
LOGROOT=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs
SCR=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl/exp4_${SLURM_JOB_ID}; mkdir -p $SCR
OUT=$ROOT/logs/exp4/bound_${PACK}_${SLURM_JOB_ID}; mkdir -p $OUT
echo "git: $(git rev-parse HEAD) host: $(hostname) pack: $PACK"; nvidia-smi
export CUDA_MPS_PIPE_DIRECTORY=$SCR/mps_pipe CUDA_MPS_LOG_DIRECTORY=$SCR/mps_log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d && echo "MPS daemon started"
nvidia-smi dmon -s um -d 60 -o T > $OUT/dmon.txt 2>&1 &
DMON=$!
COMMON="--config-name=can seed=1 env.n_eval_envs=10 wandb.mode=offline log_dir=$LOGROOT resume=true steer.mode=fixed"
if [ "$PACK" = A ]; then KS="0 4"; else KS="6 7"; fi
t0=$(date +%s)
PIDS=()
for k in $KS; do for c in 1.5 2 3; do
  n=fixed_k${k}_c${c}_s1
  python train_adsrl.py $COMMON steer.k=$k steer.w_scale=$c name=$n >> $OUT/$n.log 2>&1 &
  PIDS+=($!); echo "started $n pid $!"
done; done
# wait only on training pids; a bare `wait` also waits on dmon and never returns
for p in "${PIDS[@]}"; do wait $p; echo "[$(date +%T)] pid $p exit $?"; done
kill $DMON 2>/dev/null
echo quit | nvidia-cuda-mps-control
echo "done in $(( $(date +%s) - t0 ))s"
