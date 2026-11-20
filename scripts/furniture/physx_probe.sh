#!/bin/bash
# exp_6: lamp/round_table PhysX GPU-narrowphase crash at 1024 envs -- probe smaller N and larger PhysX buffers.
# Usage: bash scripts/furniture/physx_probe.sh <task>   (inside a 1-GPU allocation)
set -uo pipefail
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project; source scripts/furniture/env.sh
export PYTHONUNBUFFERED=1
T=$1; O=logs/exp6/physx_probe_$T; mkdir -p $O
run() { tag=$1; shift; echo "== $tag $*"; python scripts/furniture/base_eval_furniture.py --task $T --ks 20 --out $O/$tag.json "$@" > $O/$tag.log 2>&1; echo "exit $?"; grep -E "^K=|physx\.|internal error|CUDA error" $O/$tag.log | head -4; }
run n256 --n_envs 256
run n1024_buf --n_envs 1024 --set env.furniture.physx.max_gpu_contact_pairs=8388608 env.furniture.physx.default_buffer_size_multiplier=8.0
