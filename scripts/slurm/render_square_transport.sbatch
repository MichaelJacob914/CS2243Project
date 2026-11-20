#!/bin/bash
#SBATCH --job-name=render_st
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --exclude=holygpu8a12204
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
# Report videos for square (final checkpoints) and transport (latest.pt of the runs still training to 6M;
# saved atomically, so reading it mid-run is safe). Seed 1 of every method, 10 episodes each.
set -uo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
OUT=$ROOT/logs/render/st_${SLURM_JOB_ID}; mkdir -p $OUT
t0=$(date +%s)
run() { local tag=$1; shift; python "$@" > $OUT/$tag.log 2>&1; echo "[$(date +%T)] $tag exit $? at $(( $(date +%s) - t0 ))s"; }
S=logs/adsrl/square
run sq_dsrl scripts/render_rollouts.py --task square --policy fixed_k --k 0 --ckpt $S/fixed_k0_s1/checkpoint/final.pt --method dsrl &
run sq_best_k scripts/render_rollouts.py --task square --policy fixed_k --k 1 --ckpt $S/fixed_k1_s1/checkpoint/final.pt --method best_k &
run sq_adaptive scripts/render_rollouts.py --task square --policy adaptive --ckpt $S/adaptive_rho0.5_s1/checkpoint/final.pt --method adaptive &
run sq_uniform_k scripts/render_rollouts.py --task square --policy adaptive --ckpt $S/uniformk_s1/checkpoint/final.pt --method uniform_k &
run sq_residual scripts/render_rollouts.py --task square --policy residual --ckpt $S/residual_b0.3_s1/checkpoint/final.pt --method residual &
T=logs/adsrl/transport
run tr_dsrl scripts/render_rollouts.py --task transport --policy fixed_k --k 0 --ckpt $T/fixed_k0_s1/checkpoint/latest.pt --method dsrl &
run tr_best_k scripts/render_rollouts.py --task transport --policy fixed_k --k 5 --ckpt $T/fixed_k5_s1/checkpoint/latest.pt --method best_k &
run tr_adaptive scripts/render_rollouts.py --task transport --policy adaptive --ckpt $T/adaptive_rho0.5_s1/checkpoint/latest.pt --method adaptive &
run tr_uniform_k scripts/render_rollouts.py --task transport --policy adaptive --ckpt $T/uniformk_s1/checkpoint/latest.pt --method uniform_k &
run tr_residual scripts/render_rollouts.py --task transport --policy residual --ckpt $T/residual_b0.3_s1/checkpoint/final.pt --method residual &
wait
