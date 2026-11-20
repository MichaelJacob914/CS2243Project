#!/bin/bash
#SBATCH --job-name=e14_s0
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
# exp14 Stage 0: toy positive control (train small dppo DiffusionModel + full measurement pipeline).
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl
echo "commit $(git rev-parse HEAD) host $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python scripts/exp14/toy.py --out_dir results/exp14/stage0 --device cuda:0 "$@"
