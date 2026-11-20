#!/bin/bash
#SBATCH --job-name=x13_wsd_allk
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
# exp13: per-k w_scale_dims for all k = 1..19 (transport, K=20), same settings as the k=3,5,8 calibration
# (job 59624201: 1000 dataset states x 128 w, seed 0, verify seed 1, 10 damped iterations).
# CALIB_ARGS adds options to both runs, e.g. CALIB_ARGS="--damp 0.2 --iters 25" (damp 0.5 oscillates on the gripper dims at k=11-17).
# Run 2 repeats it with fresh states and w (seed 2, verify seed 3) into scratch, to measure calibration noise.
set -euo pipefail
cd $SLURM_SUBMIT_DIR; source .venv/bin/activate
echo "git: $(git rev-parse HEAD) host: $(hostname)"; nvidia-smi -L
python scripts/exp13_wscale_dims_calib.py --device cuda:0 --allk ${CALIB_ARGS:-}
REP=/n/netscratch/olveczky_lab/Lab/charleszhang/exp13_wsd_allk_rep_${SLURM_JOB_ID}; mkdir -p $REP
python scripts/exp13_wscale_dims_calib.py --device cuda:0 --allk ${CALIB_ARGS:-} --seed 2 --verify_seed 3 --out_dir $REP
