#!/bin/bash
#SBATCH --job-name=exp1_trK
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
#SBATCH --time=02:00:00
# exp_1 Task B follow-up: 400 more transport base-eval episodes (eval seed 1) for K in {100,50,25,20,10}
# (divisors of 100, so the DDIM schedule starts at t=T-T/K like K=100 does); MPS pack on one GPU.
set -uo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export MUJOCO_GL=egl
SCR=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl/exp1trK_${SLURM_JOB_ID}; mkdir -p $SCR
OUT=$ROOT/logs/exp1/trK_${SLURM_JOB_ID}; mkdir -p $OUT
echo "git: $(git rev-parse HEAD) host: $(hostname)"
export CUDA_MPS_PIPE_DIRECTORY=$SCR/mps_pipe CUDA_MPS_LOG_DIRECTORY=$SCR/mps_log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d && echo "MPS daemon started"
t0=$(date +%s)
for K in 100 50 25 20 10; do
  (python scripts/base_eval.py --task transport --ddim_steps $K --n_episodes 400 --n_envs 10 --seed 1 \
     --out_dir $ROOT/results/base_eval_seed1 > $OUT/tr_K$K.log 2>&1; echo "[$(date +%T)] K=$K exit $? at $(( $(date +%s) - t0 ))s") &
done
python scripts/authority.py --task transport --ddim_steps 25 --state_source dataset --n_states 500 --n_w 64 > $OUT/auth_tr_K25.log 2>&1 &
python scripts/authority.py --task transport --ddim_steps 50 --state_source dataset --n_states 500 --n_w 64 > $OUT/auth_tr_K50.log 2>&1 &
wait
echo quit | nvidia-cuda-mps-control
grep -h BASE_EVAL $OUT/*.log
