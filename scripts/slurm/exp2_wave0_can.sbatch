#!/bin/bash
#SBATCH --job-name=w0_can
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
# exp_2 Wave 0 on can, seed 1, 1M env steps, MPS pack. Usage (from project root):
#   sbatch scripts/slurm/exp2_wave0_can.sbatch A   # fixed k=0..5
#   sbatch scripts/slurm/exp2_wave0_can.sbatch B   # fixed k=6,7 + adaptive rho .5/.25 + uniform-k
# All runs pass resume=true: re-submitting the same pack continues from checkpoint/latest.pt.
# Logdirs: /n/netscratch/.../adsrl_runs/adsrl/can/<name> (logs/adsrl is a symlink there).
set -uo pipefail
PACK=${1:?pack A or B}
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export MUJOCO_GL=egl WANDB_MODE=offline
LOGROOT=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs
SCR=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl/exp2_${SLURM_JOB_ID}; mkdir -p $SCR
OUT=$ROOT/logs/exp2/w0_${PACK}_${SLURM_JOB_ID}; mkdir -p $OUT
echo "git: $(git rev-parse HEAD) host: $(hostname) pack: $PACK"; nvidia-smi
export CUDA_MPS_PIPE_DIRECTORY=$SCR/mps_pipe CUDA_MPS_LOG_DIRECTORY=$SCR/mps_log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d && echo "MPS daemon started"
nvidia-smi dmon -s um -d 60 -o T > $OUT/dmon.txt 2>&1 &
DMON=$!
COMMON="--config-name=can seed=1 env.n_eval_envs=10 wandb.mode=offline log_dir=$LOGROOT resume=true"
declare -A RUNS
if [ "$PACK" = A ]; then
  export ADSRL_RUNS_PER_GPU=6
  for k in 0 1 2 3 4 5; do RUNS[fixed_k${k}_s1]="steer.mode=fixed steer.k=$k"; done
else
  export ADSRL_RUNS_PER_GPU=5
  for k in 6 7; do RUNS[fixed_k${k}_s1]="steer.mode=fixed steer.k=$k"; done
  RUNS[adaptive_rho0.5_s1]="steer.mode=adaptive steer.rho=0.5"
  RUNS[adaptive_rho0.25_s1]="steer.mode=adaptive steer.rho=0.25"
  RUNS[uniformk_s1]="steer.mode=adaptive steer.freeze_k_uniform=true"
fi
t0=$(date +%s)
PIDS=()
for n in "${!RUNS[@]}"; do
  python train_adsrl.py $COMMON ${RUNS[$n]} name=$n >> $OUT/$n.log 2>&1 &
  PIDS+=($!); echo "started $n pid $!"
done
# wait only on training pids; a bare `wait` also waits on dmon and never returns
for p in "${PIDS[@]}"; do wait $p; echo "[$(date +%T)] pid $p exit $?"; done
kill $DMON 2>/dev/null
echo quit | nvidia-cuda-mps-control
echo "done in $(( $(date +%s) - t0 ))s"
