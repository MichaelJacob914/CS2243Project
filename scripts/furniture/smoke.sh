#!/bin/bash
# exp_6 smoke test on one GPU: py3.8 test suite, base eval on a small batch, tiny train_adsrl runs
# (fixed k=0, adaptive with k_subset, residual) through the shared-env eval path.
set -uo pipefail
ROOT=/n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
cd $ROOT; source scripts/furniture/env.sh
export WANDB_MODE=offline PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
[ "${SKIP_PYTEST:-0}" = 1 ] || { echo "== pytest (py3.8)"; python -m pytest -q tests/ 2>&1 | tail -8; }
echo "== base eval smoke"; python scripts/furniture/base_eval_furniture.py --task ${TASK:-one_leg_low} --n_envs 64 --ks 20 --out $ROOT/logs/exp6/smoke_base_eval.json 2>&1 | grep -E "^K=|sim created|wrote|Error|error|Traceback" | tail -25
SM=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs/exp6_smoke; rm -rf $SM
TINY="--config-name=${TASK:-one_leg_low} log_dir=$SM use_wandb=false env.n_envs=64 eval_interval=176 train.init_rollout_steps=88 train.total_timesteps=$((64*8*88*5)) train.utd=2 model.ddim_steps=20 eval_dump_every=1"
for m in "fixed_k0::steer.mode=fixed steer.k=0" "adaptive::steer.mode=adaptive steer.k_subset=[0,3,6,9,12,15,17,19]" "residual::steer.mode=residual steer.residual_beta=0.3"; do
  name=${m%%::*}; ov=${m#*::}
  echo "== train smoke $name"; python train_adsrl.py $TINY name=$name $ov 2>&1 | grep -E "^\[|Error|error|Traceback|assert|saved checkpoint .*final" | tail -20
  cat $SM/adsrl/*/$name/eval_summary.csv 2>/dev/null
done
