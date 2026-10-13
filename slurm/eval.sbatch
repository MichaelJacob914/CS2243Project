#!/bin/bash
# Run scripts/eval.py on one GPU. Submit from the repo root:
#   sbatch slurm/eval.sbatch <config.yaml> [key=value ...] [--tag name]
#   sbatch --array=0-1 slurm/eval.sbatch <config.yaml>      # split tasks/seeds over 2 jobs
#   sbatch -t 01:00:00 slurm/eval.sbatch ...                # shorter limit, starts sooner
# gpu_test allows 2 jobs at a time and 12 h each. Resubmitting the same command resumes.
#SBATCH -J cs2243-eval
#SBATCH -p gpu_test
#SBATCH -A hankyang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o /n/netscratch/hankyang_lab/Everyone/mjammot/cs2243/runs/%u/slurm/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[ $# -ge 1 ] || { echo "usage: sbatch slurm/eval.sbatch <config.yaml> [key=value ...]" >&2; exit 2; }

venv=libero
grep -qE '^env: *aloha' "$1" && venv=aloha
set +u; source slurm/common.sh "$venv"; set -u    # venv activate scripts touch unset variables
python scripts/eval.py "$@"
