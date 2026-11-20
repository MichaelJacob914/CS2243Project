#!/bin/bash
#SBATCH --job-name=e14_analyze_kd
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=test
#SBATCH --account=olveczky_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:00:00
# exp14 analysis: preregistered rules, summaries, figures.
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export PYTHONUNBUFFERED=1
python scripts/exp14/analyze.py --states results/exp14/states/states.npz --stage12 results/exp14/stage12 --sim results/exp14/sim \
  --act_dir /n/netscratch/olveczky_lab/Lab/charleszhang/exp14/actions --out_dir results/exp14/analysis_keep_diverged --keep_diverged --figs results/exp14/analysis_keep_diverged
