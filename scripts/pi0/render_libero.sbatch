#!/bin/bash
# LIBERO-10 task 8 rollout videos for the report (scripts/pi0/render_libero.py): pi0.5 k=3, pi0.5 adaptive and
# plain pi0.5 in .venv-pi05, pi0 k=0 in .venv-pi0, in parallel on one GPU.
#   sbatch -p kempner_h100 -A kempner_pehlevan_lab scripts/pi0/render_libero.sbatch
#SBATCH --job-name=render_libero8
#SBATCH --output=logs/render/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:45:00
set -e
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "host $(hostname) job $SLURM_JOB_ID git $(git rev-parse HEAD)"; nvidia-smi --query-gpu=name --format=csv,noheader
R=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs
(
  source .venv-pi05/bin/activate
  JAX_COMPILATION_CACHE_DIR=/n/netscratch/olveczky_lab/Lab/charleszhang/jax_cache_pi05 \
  python scripts/pi0/render_libero.py \
    --run $R/exp12/pi05_libero_10_t8_qf10_k3_xtfull_lf3_abase_b2.5_s2 \
          $R/exp12/pi05_libero_10_t8_qf10_adaptive_rho0.5_xtfull_lf3_abase_kdetach_b2.5_s0 \
    --base pi05 --query_freq 10 \
    --method pi05_k3 pi05_adaptive pi05_base \
    --label "π0.5 steered, k=3 (seed 2)" "π0.5 steered, adaptive k (seed 0)" "π0.5 base (no steering)"
) > logs/render/libero8_pi05_${SLURM_JOB_ID}.log 2>&1 &
P1=$!
(
  source .venv-pi0/bin/activate
  python scripts/pi0/render_libero.py --run $R/exp9/pi0_t8_k0_xtfull_lf3_abase_b2.5_s0 \
    --method pi0_k0 --label "π0 steered, k=0 (seed 0)"
) > logs/render/libero8_pi0_${SLURM_JOB_ID}.log 2>&1 &
P2=$!
wait $P1; S1=$?
wait $P2; S2=$?
echo "pi05 exit $S1, pi0 exit $S2"
