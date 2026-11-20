#!/bin/bash
#SBATCH --job-name=e14_s0pow
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=sapphire
#SBATCH --account=olveczky_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:30:00
# exp14 Stage 0 diagnostic: repeat the toy measurement (same trained model) over measurement seeds at P=16 and P=64.
set -uo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
M=results/exp14/stage0/toy_model.pt
for P in 16 64; do
  for s in 2 3 4 5; do
    python scripts/exp14/toy.py --out_dir results/exp14/stage0_power/P${P}_g${s} --model_from $M --P $P --gen_seed $s --device cpu > /dev/null 2>&1 &
  done
done
wait
grep -h CRITERIA results/exp14/stage0_power/*/log.txt
