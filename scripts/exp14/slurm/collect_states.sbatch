#!/bin/bash
#SBATCH --job-name=e14_states
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=sapphire
#SBATCH --account=olveczky_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=02:00:00
# exp14 state collection: 192 base-policy rollouts (48 workers), demo states, env checks, 330-state selection.
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "commit $(git rev-parse HEAD) host $(hostname)"
python scripts/exp14/collect_states.py --out_dir results/exp14/states --n_workers 48 --n_episodes 192
