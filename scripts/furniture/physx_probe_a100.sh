#!/bin/bash
# exp_6: does the lamp/round_table PhysX GJK/EPA kernel failure also happen on A100 (sm_80)? n=256, K=20, 1 episode.
set -uo pipefail
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project; source scripts/furniture/env.sh
export PYTHONUNBUFFERED=1
T=$1; O=logs/exp6/physx_probe_$T; mkdir -p $O
nvidia-smi --query-gpu=name --format=csv
python scripts/furniture/base_eval_furniture.py --task $T --ks 20 --n_envs 256 --out $O/a100_n256.json > $O/a100_n256.log 2>&1; echo "exit $?"
grep -E "^K=|internal error|CUDA error" $O/a100_n256.log | head -4
