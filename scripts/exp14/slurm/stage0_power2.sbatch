#!/bin/bash
#SBATCH --job-name=e14_s0pow2
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=test
#SBATCH --account=olveczky_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=00:45:00
# exp14 Stage 0 DIAGNOSTIC (not a gate): pass rate of the final procedure (Design changes 1-3) over measurement seeds 9-16.
set -uo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
M=results/exp14/stage0/toy_model.pt
for s in 9 10 11 12 13 14 15 16; do
  python scripts/exp14/toy.py --out_dir results/exp14/stage0_power2/P64_g${s} --model_from $M --P 64 --gen_seed $s --device cpu > /dev/null 2>&1 &
done
wait
