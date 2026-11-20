#!/bin/bash
#SBATCH --job-name=e14_viddata
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=test
#SBATCH --account=olveczky_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:40:00
# Data for the exp14 toy animation and explainer video. Uses the netscratch torch env (toyenv): the project .venv on
# holylfs06 took >2 min just to import numpy on compute nodes on 2026-11-12.
set -euo pipefail
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
PY=/n/netscratch/olveczky_lab/Lab/charleszhang/toyenv/bin/python
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
echo "commit $(git rev-parse HEAD) host $(hostname) job $SLURM_JOB_ID"
$PY scripts/exp14/video/toy_anim_data.py --out results/exp14/anim/toy_anim_data.npz &
$PY scripts/exp14/video/extract_video_data.py &
wait
