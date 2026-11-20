#!/bin/bash
#SBATCH --job-name=x13dip
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
# exp_13 dip ablation: copy of exp3_wave1.sbatch (same config as the exp10 fixed-k baseline) with the arm in the
# run name. Usage (from project root):
#   sbatch -J x13_<pack> scripts/slurm/exp13_dip.sbatch <task> <run name> [<run name> ...]
# Run names: fixed_k<k>_<arm>_s<seed> or adaptivedetach_rho<rho>_<arm>_s<seed> (as exp3_wave1: steer.mode=adaptive
# steer.rho=<rho> steer.k_entropy_bias=detach), arm in
#   tr     : absolute noise, noise trust region on (bound 0.3)
#   kickzi : steer.param=kick, actor_zero_init, actor_log_std_init=-1.4, ent_coef=auto_0.05
#   ent005 : absolute noise, ent_coef=auto_0.05
#   eq     : equal control strength, steer.w_scale_dims = per-dim vector for k (adaptive: the per-k (K, A) file)
# Every run adds early det evals every 64k env steps up to 1.6M (50 episodes).
# DRY_RUN=1 prints the python commands without starting MPS or training.
# Partition/account/CPUs/mem/time can be overridden on the sbatch command line (one partition per job).
# All runs pass resume=true and save latest.pt every eval and every 120 min, so re-submitting the
# exact same command continues every unfinished run; finished runs just re-run the final eval check and exit.
# Logdirs: logs/adsrl/<task>/<name> (logs/adsrl -> scratch adsrl_runs/adsrl).
set -uo pipefail
TASK=${1:?task}; shift
[ $# -ge 1 ] || { echo "no runs given"; exit 1; }
ROOT=${SLURM_SUBMIT_DIR:-$PWD}; cd $ROOT; source .venv/bin/activate
export MUJOCO_GL=egl WANDB_MODE=offline ADSRL_RUNS_PER_GPU=$#
LOGROOT=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs
if [ -z "${DRY_RUN:-}" ]; then
  SCR=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl/exp13_${SLURM_JOB_ID}; mkdir -p $SCR
  OUT=$ROOT/logs/exp13/${SLURM_JOB_NAME}_${TASK}_${SLURM_JOB_ID}; mkdir -p $OUT
  echo "git: $(git rev-parse HEAD) host: $(hostname) task: $TASK runs: $*"; nvidia-smi
  export CUDA_MPS_PIPE_DIRECTORY=$SCR/mps_pipe CUDA_MPS_LOG_DIRECTORY=$SCR/mps_log
  mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  nvidia-cuda-mps-control -d && echo "MPS daemon started"
  nvidia-smi dmon -s um -d 60 -o T > $OUT/dmon.txt 2>&1 &
  DMON=$!
fi
COMMON="--config-name=$TASK env.n_eval_envs=10 wandb.mode=offline log_dir=$LOGROOT resume=true checkpoint_every_min=120"
[ "$TASK" = transport ] && COMMON="$COMMON model.ddim_steps=20"  # exp_1 decision: K=20
COMMON="$COMMON ${ADSRL_EXTRA:-}"  # extra hydra overrides, e.g. ADSRL_EXTRA="train.total_timesteps=6000000" to extend finished runs
t0=$(date +%s)
PIDS=()
for n in "$@"; do
  seed=${n##*_s}
  case $n in
    fixed_k*_*_s*) k=${n#fixed_k}; k=${k%%_*}; arm=${n#fixed_k${k}_}; arm=${arm%_s*}; M="steer.mode=fixed steer.k=$k" ;;
    adaptivedetach_rho*_*_s*) r=${n#adaptivedetach_rho}; r=${r%%_*}; arm=${n#adaptivedetach_rho${r}_}; arm=${arm%_s*}; k=allk
      M="steer.mode=adaptive steer.rho=$r steer.k_entropy_bias=detach" ;;
    *) echo "unknown run name $n"; continue ;;
  esac
  case $arm in
    tr) A="train.trust_region.enabled=true train.trust_region.bound=0.3" ;;
    kickzi) A="steer.param=kick steer.actor_zero_init=true steer.actor_log_std_init=-1.4 train.ent_coef=auto_0.05" ;;
    ent005) A="train.ent_coef=auto_0.05" ;;
    # equal control strength: per-action-dim noise scale matched to k=0 (exp13 Phase 1; k=0 reference = exp10 fixed_k0_s1)
    eq) if [ "$k" = allk ]; then f=w_scale_dims_allk_transport_K20.npy; else f=w_scale_dims_k${k}_transport_K20.npy; fi
        A="steer.w_scale=1.0 steer.w_scale_dims=results/exp13/authority/$f" ;;
    *) echo "unknown arm $arm in $n"; continue ;;
  esac
  ARGS="$M $A train.early_eval_every=64000 train.early_eval_until=1600000 train.early_eval_episodes=50"
  echo "$n: seed=$seed $ARGS"
  if [ -n "${DRY_RUN:-}" ]; then echo "python train_adsrl.py $COMMON seed=$seed $ARGS name=$n"; continue; fi
  python train_adsrl.py $COMMON seed=$seed $ARGS name=$n >> $OUT/$n.log 2>&1 &
  PIDS+=($!); echo "started $n pid $!"
done
# wait only on training pids; a bare `wait` also waits on dmon and never returns
[ -n "${DRY_RUN:-}" ] && exit 0
for p in "${PIDS[@]}"; do wait $p; echo "[$(date +%T)] pid $p exit $? after $(( $(date +%s) - t0 ))s"; done
kill $DMON 2>/dev/null
echo quit | nvidia-cuda-mps-control
echo "done in $(( $(date +%s) - t0 ))s"
