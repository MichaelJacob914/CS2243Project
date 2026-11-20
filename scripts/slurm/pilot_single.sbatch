#!/bin/bash
#SBATCH --job-name=pilot_single
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=01:30:00
# Single-run profile of DSRL-NA on can (~20k training env steps after the init rollout + initial eval).
# Submit from project root: sbatch scripts/slurm/pilot_single.sbatch
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR
source $ROOT/.venv/bin/activate
export WANDB_MODE=offline WANDB_DIR=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl/wandb
export MUJOCO_GL=egl
mkdir -p $WANDB_DIR
OUT=$ROOT/logs/pilot/single_${SLURM_JOB_ID}; mkdir -p $OUT
echo "git: $(git -C $ROOT rev-parse HEAD) host: $(hostname)"; nvidia-smi
nvidia-smi dmon -s um -d 5 -o T > $OUT/dmon.txt 2>&1 &
DMON=$!
cd $ROOT/dsrl
python $ROOT/scripts/pilot_train_dsrl.py --config-path=$ROOT/dsrl/cfg/robomimic --config-name=dsrl_can.yaml \
  +profile=true +total_timesteps=5000 log_dir=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl/logs \
  2>&1 | tee $OUT/train.log
kill $DMON || true
