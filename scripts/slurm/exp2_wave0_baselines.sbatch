#!/bin/bash
#SBATCH --job-name=w0_base
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --exclude=holygpu8a12204
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# exp_2 Wave 0 baselines on can, seed 1, one MPS pack on one GPU:
#   residual SAC beta=0.1 and beta=0.3 (1M env steps each, resume=true) + constant-noise CEM (k=0).
# Submit from project root: sbatch scripts/slurm/exp2_wave0_baselines.sbatch
# Re-submitting continues the residual runs from checkpoint/latest.pt; CEM is skipped once it has results.
# Logdirs: logs/adsrl/can/{residual_b0.1,residual_b0.3,cem}/seed1 (logs/adsrl -> scratch adsrl_runs/adsrl).
set -uo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export MUJOCO_GL=egl WANDB_MODE=offline ADSRL_RUNS_PER_GPU=3
LOGROOT=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs
SCR=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl/exp2_${SLURM_JOB_ID}; mkdir -p $SCR
OUT=$ROOT/logs/exp2/w0_baselines_${SLURM_JOB_ID}; mkdir -p $OUT
echo "git: $(git rev-parse HEAD) host: $(hostname)"; nvidia-smi
export CUDA_MPS_PIPE_DIRECTORY=$SCR/mps_pipe CUDA_MPS_LOG_DIRECTORY=$SCR/mps_log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d && echo "MPS daemon started"
nvidia-smi dmon -s um -d 60 -o T > $OUT/dmon.txt 2>&1 &
DMON=$!
t0=$(date +%s)
PIDS=()
for b in 0.1 0.3; do
  python train_adsrl.py --config-name=can seed=1 env.n_eval_envs=10 wandb.mode=offline log_dir=$LOGROOT resume=true \
    steer.mode=residual steer.residual_beta=$b name=residual_b$b/seed1 >> $OUT/residual_b$b.log 2>&1 &
  PIDS+=($!); echo "started residual_b$b pid $!"
done
CEMDIR=$LOGROOT/adsrl/can/cem/seed1
if [ ! -f $CEMDIR/eval_summary.csv ]; then
  rm -rf $CEMDIR  # partial search from a killed job: restart it
  python scripts/cem_noise.py --task can --seed 1 --pop 16 --episodes_per_cand 8 --iters 15 --n_envs 16 \
    --final_episodes 200 --final_envs 10 --logdir $CEMDIR > $OUT/cem.log 2>&1 &
  PIDS+=($!); echo "started cem pid $!"
fi
for p in "${PIDS[@]}"; do wait $p; echo "[$(date +%T)] pid $p exit $? after $(( $(date +%s) - t0 ))s"; done
kill $DMON 2>/dev/null
echo quit | nvidia-cuda-mps-control
echo "done in $(( $(date +%s) - t0 ))s"
