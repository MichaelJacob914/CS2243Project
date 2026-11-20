#!/bin/bash
#SBATCH --job-name=e14_s4gen
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=test
#SBATCH --account=olveczky_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:45:00
# exp14 Stage 4 action generation (design: results/exp14/stage4_design.json)
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
python scripts/exp14/stage12.py --mode stage4 --states results/exp14/states/states.npz --out_dir results/exp14/stage12 \
  --act_dir /n/netscratch/olveczky_lab/Lab/charleszhang/exp14/actions --stage4_states 3,6,8,22,60,74,76,85,123,126,135,144,185,190,195,204,256,258,261,264 --stage4_ks 4,9,12,15,18 --device cpu
