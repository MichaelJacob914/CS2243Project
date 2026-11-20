#!/bin/bash
# Short GPU smoke of the structured noise layout (exp8): train 20k env steps with evals (stoch + det) and
# checkpoints, restore the checkpoint for an --eval_only pass, and print the stats of decoded noise.
#   sbatch -p kempner_h100 -A kempner_pehlevan_lab scripts/pi0/smoke_exp8.sbatch
#SBATCH --job-name=p8_smoke
#SBATCH --output=logs/exp8/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
set -e
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
source .venv-pi0/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
ROOT_DIR=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs/exp8_smoke
rm -rf $ROOT_DIR/smoke_t8_wstruct_b2.5
ARGS="--task_id 8 --steer.mode fixed --steer.k 0 --steer.xt_layout full --steer.w_layout structured \
    --action_magnitude 2.5 --n_envs 12 --total_env_steps 20000 --eval_every 10000 --eval_episodes 12 --eval_det 1 \
    --logdir_root $ROOT_DIR --name smoke_t8_wstruct_b2.5 --use_wandb 0"
python train_adsrl_pi0.py $ARGS
python train_adsrl_pi0.py $ARGS --eval_only 1
D=$ROOT_DIR/smoke_t8_wstruct_b2.5
ls $D $D/checkpoint; cat $D/eval_summary.csv $D/eval_final.csv
python - <<PY
import json, numpy as np, pickle, sys
sys.path.insert(0, ".")
from adsrl_pi0.rollout import structured_noise
info = json.load(open("$D/run_info.json"))
print("run_info:", {k: info[k] for k in ["w_layout", "xt_layout", "action_bound", "use_xt_obs"]})
rng = np.random.default_rng(0)
x_T, z = rng.standard_normal((50, 32)), rng.standard_normal(32)
w = structured_noise(x_T, z)
print("one decoded w (z ~ N(0,I)): shape", w.shape, "mean %.4f std %.4f" % (w.mean(), w.std()))
print("max |time-mean - z/sqrt(50)| = %.2e" % np.abs(w.mean(0) - z / np.sqrt(50)).max())
W = structured_noise(rng.standard_normal((2000, 50, 32)), rng.standard_normal((2000, 32)))
print("2000 decoded w: per-entry mean in [%.3f, %.3f], per-entry std in [%.3f, %.3f]"
      % (W.mean(0).min(), W.mean(0).max(), W.std(0).min(), W.std(0).max()))
# transitions in the buffer: first ones are random N(0,I) z, the last ones the agent's z (bound 2.5)
d = pickle.load(open("$D/checkpoint/buffer.pkl", "rb"))["data"]
z, xt = d["actions"][:, 0], d["observations"]["x_T"].reshape(-1, 50, 32)
print("buffer: n=%d, x_T obs shape %s" % (len(z), d["observations"]["x_T"].shape[1:]))
for nm, sl in [("first 200 (random z)", slice(0, 200)), ("last 200 (agent z)", slice(-200, None))]:
    w = structured_noise(xt[sl], z[sl])
    print("%s: |z| max %.3f, z std %.3f; decoded w mean %.3f std %.3f; time-mean err %.1e"
          % (nm, np.abs(z[sl]).max(), z[sl].std(), w.mean(), w.std(),
             np.abs(w.mean(1) - z[sl] / np.sqrt(50)).max()))
PY
