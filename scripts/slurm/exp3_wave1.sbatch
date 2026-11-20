#!/bin/bash
#SBATCH --job-name=w1
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --exclude=holygpu8a12204
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
# exp_3 Wave 1: one MPS pack of training runs on one GPU. Usage (from project root):
#   sbatch -J w1_<pack> scripts/slurm/exp3_wave1.sbatch <task> <run name> [<run name> ...]
# Run names set the method and seed:
#   fixed_k<k>_s<seed> | adaptive_rho<rho>_s<seed> | adaptivedetach_rho<rho>_s<seed> (k_entropy_bias=detach) | uniformk_s<seed> | residual_b<beta>_s<seed>
# Partition/account/CPUs/mem/time can be overridden on the sbatch command line (one partition per job).
# All runs pass resume=true and save latest.pt every eval and every 120 min, so re-submitting the
# exact same command continues every unfinished run; finished runs just re-run the final eval check and exit.
# Logdirs: logs/adsrl/<task>/<name> (logs/adsrl -> scratch adsrl_runs/adsrl).
set -uo pipefail
TASK=${1:?task}; shift
[ $# -ge 1 ] || { echo "no runs given"; exit 1; }
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export MUJOCO_GL=egl WANDB_MODE=offline ADSRL_RUNS_PER_GPU=$#
LOGROOT=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs
SCR=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl/exp3_${SLURM_JOB_ID}; mkdir -p $SCR
OUT=$ROOT/logs/exp3/${SLURM_JOB_NAME}_${TASK}_${SLURM_JOB_ID}; mkdir -p $OUT
echo "git: $(git rev-parse HEAD) host: $(hostname) task: $TASK runs: $*"; nvidia-smi
export CUDA_MPS_PIPE_DIRECTORY=$SCR/mps_pipe CUDA_MPS_LOG_DIRECTORY=$SCR/mps_log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d && echo "MPS daemon started"
nvidia-smi dmon -s um -d 60 -o T > $OUT/dmon.txt 2>&1 &
DMON=$!
COMMON="--config-name=$TASK env.n_eval_envs=10 wandb.mode=offline log_dir=$LOGROOT resume=true checkpoint_every_min=120"
[ "$TASK" = transport ] && COMMON="$COMMON model.ddim_steps=20"  # exp_1 decision: K=20
COMMON="$COMMON ${ADSRL_EXTRA:-}"  # extra hydra overrides, e.g. ADSRL_EXTRA="train.total_timesteps=6000000" to extend finished runs
t0=$(date +%s)
PIDS=()
for n in "$@"; do
  seed=${n##*_s}
  case $n in
    fixed_k*) k=${n#fixed_k}; k=${k%%_*}; ARGS="steer.mode=fixed steer.k=$k" ;;
    adaptive_rho*) r=${n#adaptive_rho}; r=${r%_s*}; ARGS="steer.mode=adaptive steer.rho=$r" ;;
    adaptivedetach_rho*) r=${n#adaptivedetach_rho}; r=${r%_s*}; ARGS="steer.mode=adaptive steer.rho=$r steer.k_entropy_bias=detach" ;;
    uniformk_s*) ARGS="steer.mode=adaptive steer.freeze_k_uniform=true" ;;
    residual_b*) b=${n#residual_b}; b=${b%_s*}; ARGS="steer.mode=residual steer.residual_beta=$b" ;;
    *) echo "unknown run name $n"; continue ;;
  esac
  echo "$n: seed=$seed $ARGS"
  python train_adsrl.py $COMMON seed=$seed $ARGS name=$n >> $OUT/$n.log 2>&1 &
  PIDS+=($!); echo "started $n pid $!"
done
# wait only on training pids; a bare `wait` also waits on dmon and never returns
for p in "${PIDS[@]}"; do wait $p; echo "[$(date +%T)] pid $p exit $? after $(( $(date +%s) - t0 ))s"; done
kill $DMON 2>/dev/null
echo quit | nvidia-cuda-mps-control
echo "done in $(( $(date +%s) - t0 ))s"
