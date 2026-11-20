#!/bin/bash
#SBATCH --job-name=exp1_base
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
#SBATCH --time=03:00:00
# exp_1 Task B: base-policy eval (200 episodes; can, square, transport K in {100,20,15,10}) packed with MPS,
# then the authority diagnostic at AUTH_K_TRANSPORT (pass as $1 once K is chosen; empty = skip transport authority).
# Submit from project root: sbatch scripts/slurm/exp1_base_eval.sbatch [transport_K]
set -uo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export MUJOCO_GL=egl
AUTH_K=${1:-}
SCR=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl/exp1base_${SLURM_JOB_ID}; mkdir -p $SCR
OUT=$ROOT/logs/exp1/base_${SLURM_JOB_ID}; mkdir -p $OUT
echo "git: $(git rev-parse HEAD) host: $(hostname)"; nvidia-smi
export CUDA_MPS_PIPE_DIRECTORY=$SCR/mps_pipe CUDA_MPS_LOG_DIRECTORY=$SCR/mps_log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d && echo "MPS daemon started"
t0=$(date +%s)
run() { local tag=$1; shift; python "$@" > $OUT/$tag.log 2>&1; echo "[$(date +%T)] $tag exit $? at $(( $(date +%s) - t0 ))s"; }
# 6 evals x 10 env procs = 60 procs on 24 cores; transport evals are the long ones, start them first.
run tr_K100 scripts/base_eval.py --task transport --ddim_steps 100 &
run tr_K20  scripts/base_eval.py --task transport --ddim_steps 20 &
run tr_K15  scripts/base_eval.py --task transport --ddim_steps 15 &
run tr_K10  scripts/base_eval.py --task transport --ddim_steps 10 &
run can     scripts/base_eval.py --task can &
run square  scripts/base_eval.py --task square &
# authority for can/square (GPU only) in parallel; transport at the chosen K (or all candidates)
AUTH="--state_source dataset --n_states 500 --n_w 64"
run auth_can scripts/authority.py --task can $AUTH &
run auth_square scripts/authority.py --task square $AUTH &
for K in ${AUTH_K:-100 20 15 10}; do run auth_tr_K$K scripts/authority.py --task transport --ddim_steps $K $AUTH & done
wait
echo quit | nvidia-cuda-mps-control
grep -h BASE_EVAL $OUT/*.log
echo "done in $(( $(date +%s) - t0 ))s"
