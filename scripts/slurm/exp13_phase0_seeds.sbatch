#!/bin/bash
#SBATCH --job-name=e13_p0seeds
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=23
#SBATCH --mem=128G
#SBATCH --time=03:30:00
# exp13 Phase 0B seed replicate: uniformk_s2 / s3 final.pt (6M), paired arms + 300-state headroom.
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
SNAP=/n/netscratch/olveczky_lab/Lab/charleszhang/exp13_phase0/code_snapshot
cp scripts/exp13_*.py $SNAP/scripts/
export EXP13_CODE_ROOT=$SNAP MUJOCO_GL=egl PYTHONUNBUFFERED=1
echo "commit $(git rev-parse HEAD) host $(hostname) snapshot $(cat $SNAP/SNAPSHOT_TIME)"
for s in 2 3; do
  CKPT=/n/netscratch/olveczky_lab/Lab/charleszhang/exp13_phase0/uniformk_s$s/checkpoint/final.pt
  python -c "import torch,sys; n=torch.load('$CKPT',map_location='cpu',weights_only=False)['extra']['callback']['env_steps']; print('s$s env_steps',n); sys.exit(n!=6000000)"
  OUT=results/exp13/headroom_paired_s$s
  python $SNAP/scripts/exp13_headroom.py --ckpt $CKPT --out_dir $OUT --n_episodes 200 --n_envs 20 \
    --n_states 300 --n_w 32 --n_cand 32 --arms iii,headroom,iv,i,ii --paired
  python $SNAP/scripts/exp13_headroom_analyze.py --dir $OUT > $OUT/analyze.log 2>&1
  python $SNAP/scripts/exp13_paired_tests.py $OUT
done
