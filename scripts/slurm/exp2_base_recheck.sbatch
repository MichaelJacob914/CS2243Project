#!/bin/bash
#SBATCH --job-name=exp2_base_recheck
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --exclude=holygpu8a12204
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=04:00:00
# Base success re-check on can (exp_2): 3 seeds x 500 eps for base / zero_k1 / tanh_k0 via AdsrlCallback.evaluate,
# plus upstream DSRL's own init stochastic eval (3 seeds x 500). All packed under MPS.
set -uo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export MUJOCO_GL=egl
SCR=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl/recheck_${SLURM_JOB_ID}; mkdir -p $SCR
OUT=$ROOT/logs/exp2/recheck_${SLURM_JOB_ID}; mkdir -p $OUT
RES=$ROOT/results/base_recheck
echo "git: $(git rev-parse HEAD) host: $(hostname)"; nvidia-smi
export CUDA_MPS_PIPE_DIRECTORY=$SCR/mps_pipe CUDA_MPS_LOG_DIRECTORY=$SCR/mps_log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d && echo "MPS daemon started"
t0=$(date +%s)
run() { local tag=$1; shift; "$@" > $OUT/$tag.log 2>&1; echo "[$(date +%T)] $tag exit $? at $(( $(date +%s) - t0 ))s"; }
for s in 0 1 2; do
  for m in base zero_k1 tanh_k0; do
    run ${m}_s$s python scripts/base_recheck.py --task can --mode $m --seed $s --n_episodes 500 --n_envs 10 --out_csv $RES/parts/${m}_s$s.csv &
  done
  (cd dsrl && run upstream_s$s python ../scripts/upstream_init_eval.py --seed $((s+1)) --num_evals 500 --deterministic 0 --out_csv $RES/parts/upstream_s$s.csv) &
done
wait
echo quit | nvidia-cuda-mps-control
grep -h "RECHECK\|UPSTREAM_EVAL" $OUT/*.log
echo "done in $(( $(date +%s) - t0 ))s"
