#!/bin/bash
#SBATCH --job-name=render_base
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --exclude=holygpu8a12204
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
# Report videos: base-policy clips for can, square, transport (K=20), 10 episodes each (seeds 0-9),
# as 3 parallel processes. Then a smoke test of the fixed_k / adaptive checkpoint loaders on the
# exp_1 smoke runs (2 episodes each, written to $SMOKE_OUT, not to report/assets).
# Submit from project root: sbatch scripts/slurm/render_base.sbatch [smoke_runs_dir] [smoke_out_dir]
set -uo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
OUT=$ROOT/logs/render/base_${SLURM_JOB_ID}; mkdir -p $OUT
echo "git: $(git rev-parse HEAD) host: $(hostname)"; nvidia-smi -L
t0=$(date +%s)
run() { local tag=$1; shift; python "$@" > $OUT/$tag.log 2>&1; echo "[$(date +%T)] $tag exit $? at $(( $(date +%s) - t0 ))s"; }

run transport scripts/render_rollouts.py --task transport --policy base --ddim_steps 20 &
run can scripts/render_rollouts.py --task can --policy base &
run square scripts/render_rollouts.py --task square --policy base &
wait

SMOKE_RUNS=${1:-}
SMOKE_OUT=${2:-}
if [ -n "$SMOKE_RUNS" ] && [ -n "$SMOKE_OUT" ]; then
  run smoke_k4 scripts/render_rollouts.py --task can --policy fixed_k --k 4 --ckpt $SMOKE_RUNS/smoke_k4/checkpoint/final.pt \
    --n_episodes 2 --max_success 1 --max_fail 1 --out_dir $SMOKE_OUT &
  run smoke_adaptive scripts/render_rollouts.py --task can --policy adaptive --ckpt $SMOKE_RUNS/smoke_adaptive/checkpoint/final.pt \
    --n_episodes 2 --max_success 1 --max_fail 1 --out_dir $SMOKE_OUT &
  run one_cam scripts/render_rollouts.py --task transport --policy base --ddim_steps 20 --cameras agentview \
    --n_episodes 1 --max_success 1 --max_fail 1 --out_dir $SMOKE_OUT/one_cam &
  wait
fi
echo "done in $(( $(date +%s) - t0 ))s"
