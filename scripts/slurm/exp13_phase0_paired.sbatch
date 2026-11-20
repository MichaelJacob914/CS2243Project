#!/bin/bash
#SBATCH --job-name=e13_p0pair
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=23
#SBATCH --mem=128G
#SBATCH --time=02:00:00
# exp13 Phase 0B rerun of arms (iii, iv, i, ii) with paired initial states (--paired seeds every
# worker's np RNG before each reset). A 2-round pairing check runs first and aborts on mismatch.
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
SNAP=/n/netscratch/olveczky_lab/Lab/charleszhang/exp13_phase0/code_snapshot
CKPT=/n/netscratch/olveczky_lab/Lab/charleszhang/exp13_phase0/uniformk_s1/checkpoint/final.pt
cp scripts/exp13_*.py $SNAP/scripts/
export EXP13_CODE_ROOT=$SNAP MUJOCO_GL=egl PYTHONUNBUFFERED=1
echo "commit $(git rev-parse HEAD) host $(hostname) snapshot $(cat $SNAP/SNAPSHOT_TIME)"
CHK=/n/netscratch/olveczky_lab/Lab/charleszhang/exp13_phase0/pair_check; rm -rf $CHK
python $SNAP/scripts/exp13_headroom.py --ckpt $CKPT --out_dir $CHK --n_episodes 4 --n_envs 2 --arms iii,iv --paired
python - <<PY
import glob, numpy as np
o = {}
for a in ["iii_stoch", "iv_det"]:
    d = np.load(glob.glob("$CHK/arm_%s/eval_rollouts/*.npz" % a)[0], allow_pickle=True)
    m = d["chunk"] == 0
    o[a] = d["obs"][m][np.argsort(d["episode"][m])][:, :59]
diff = np.abs(o["iii_stoch"] - o["iv_det"]).max()
spread = np.abs(o["iii_stoch"][0] - o["iii_stoch"][1:]).max()
print("PAIR CHECK max|diff| across arms", diff, "| across episodes", spread)
assert diff < 1e-6 and spread > 1e-3, "pairing failed"
PY
python $SNAP/scripts/exp13_headroom.py --ckpt $CKPT --out_dir results/exp13/headroom_paired --n_episodes 200 \
  --n_envs 20 --n_cand 32 --arms iii,iv,i,ii --paired
python $SNAP/scripts/exp13_headroom_analyze.py --dir results/exp13/headroom_paired
