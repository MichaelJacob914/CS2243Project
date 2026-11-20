#!/bin/bash
#SBATCH --job-name=e14_s0gate3
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=test
#SBATCH --account=olveczky_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:45:00
# exp14 Stage 0 gate rerun at the re-preregistered P=64 x M=16 design, fresh measurement seed 8, null by the identical procedure incl. chain-drawn clean points; rule 1 width-only, same trained toy model.
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
echo "commit $(git rev-parse HEAD) host $(hostname)"
python scripts/exp14/toy.py --out_dir results/exp14/stage0_gate3_P64_g8 --model_from results/exp14/stage0/toy_model.pt --P 64 --M 16 --gen_seed 8 --device cpu
