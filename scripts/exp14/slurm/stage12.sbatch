#!/bin/bash
#SBATCH --job-name=e14_s12
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_h200
#SBATCH --account=olveczky_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
# exp14 Stages 1, 2a, 2b on transport (one shard). Args: SHARD (e.g. 0/2) [LIMIT]
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl
SHARD=${1:-0/1}; LIMIT=${2:-}
echo "commit $(git rev-parse HEAD) host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) shard $SHARD limit $LIMIT"
python scripts/exp14/stage12.py --states results/exp14/states/states.npz --out_dir results/exp14/stage12 \
  --act_dir /n/netscratch/olveczky_lab/Lab/charleszhang/exp14/actions --shard $SHARD ${LIMIT:+--limit $LIMIT}
