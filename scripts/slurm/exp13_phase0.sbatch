#!/bin/bash
#SBATCH --job-name=e13_p0
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=23
#SBATCH --mem=128G
#SBATCH --time=04:00:00
# exp13 Phase 0 (eval only): A = authority calibration, B = headroom for learned k (uniformk_s1 @ 6M).
# Runs from a frozen code snapshot in scratch (adsrl/ is being edited in parallel); MODE=smoke for a tiny run.
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
SNAP=/n/netscratch/olveczky_lab/Lab/charleszhang/exp13_phase0/code_snapshot
CKPT=/n/netscratch/olveczky_lab/Lab/charleszhang/exp13_phase0/uniformk_s1/checkpoint/final.pt
cp scripts/exp13_*.py $SNAP/scripts/
export EXP13_CODE_ROOT=$SNAP MUJOCO_GL=egl PYTHONUNBUFFERED=1
echo "commit $(git rev-parse HEAD) host $(hostname) snapshot $(cat $SNAP/SNAPSHOT_TIME)"; nvidia-smi -L
MODE=${MODE:-full}
if [ "$MODE" = smoke ]; then
  OUT=/n/netscratch/olveczky_lab/Lab/charleszhang/exp13_phase0/smoke; rm -rf $OUT
  python $SNAP/scripts/exp13_authority_calib.py --n_states 16 --n_w 8 --out_dir $OUT/authority
  python $SNAP/scripts/exp13_headroom.py --ckpt $CKPT --out_dir $OUT/headroom --n_episodes 5 --n_envs 5 \
    --n_states 20 --n_w 4 --n_cand 32
  python $SNAP/scripts/exp13_headroom_analyze.py --dir $OUT/headroom
else
  OUT=results/exp13
  python $SNAP/scripts/exp13_authority_calib.py --out_dir $OUT/authority
  python $SNAP/scripts/exp13_headroom.py --ckpt $CKPT --out_dir $OUT/headroom --n_episodes 200 --n_envs 20 \
    --n_states 300 --n_w 32 --n_cand ${NCAND:-32}
  python $SNAP/scripts/exp13_headroom_analyze.py --dir $OUT/headroom
fi
