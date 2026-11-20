#!/bin/bash
#SBATCH --job-name=e14_sim
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=sapphire
#SBATCH --account=olveczky_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=06:00:00
# exp14 simulator executor. Args: STAGE [extra sim_exec args...]
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
STAGE=$1; shift
echo "commit $(git rev-parse HEAD) host $(hostname) stage $STAGE args $@"
python scripts/exp14/sim_exec.py --stage $STAGE --states results/exp14/states/states.npz \
  --act_dir /n/netscratch/olveczky_lab/Lab/charleszhang/exp14/actions --out_dir results/exp14/sim --n_workers $SLURM_CPUS_PER_TASK "$@"
