#!/bin/bash
#SBATCH --job-name=render_w0
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --exclude=holygpu8a12204
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:30:00
# Report videos for the Wave 0 can runs (final checkpoints, seed 1), 10 episodes each.
set -uo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
OUT=$ROOT/logs/render/w0_${SLURM_JOB_ID}; mkdir -p $OUT
t0=$(date +%s)
run() { local tag=$1; shift; python "$@" > $OUT/$tag.log 2>&1; echo "[$(date +%T)] $tag exit $? at $(( $(date +%s) - t0 ))s"; }
R=logs/adsrl/can
run dsrl scripts/render_rollouts.py --task can --policy fixed_k --k 0 --ckpt $R/fixed_k0_s1/checkpoint/final.pt --method dsrl &
run best_k scripts/render_rollouts.py --task can --policy fixed_k --k 1 --ckpt $R/fixed_k1_s1/checkpoint/final.pt --method best_k &
run adaptive scripts/render_rollouts.py --task can --policy adaptive --ckpt $R/adaptive_rho0.5_s1/checkpoint/final.pt --method adaptive &
run uniform_k scripts/render_rollouts.py --task can --policy adaptive --ckpt $R/uniformk_s1/checkpoint/final.pt --method uniform_k &
run residual scripts/render_rollouts.py --task can --policy residual --ckpt $R/residual_b0.3/seed1/checkpoint/final.pt --method residual &
wait
