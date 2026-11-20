#!/bin/bash
# Noise-pattern diagnostic (eval only), one LIBERO task per job: 16 conditions x 100 episodes. Rerun to continue.
#   sbatch scripts/pi0/noise_pattern_eval.sbatch <task>     (tasks 8, 4, 6)
#SBATCH --job-name=p7_noisepat
#SBATCH --output=logs/exp7/%x_%j.out
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:30:00
set -e
TASK=${1:?usage: sbatch noise_pattern_eval.sbatch <task>}
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
source .venv-pi0/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
nvidia-smi --query-gpu=name --format=csv
echo "git $(git rev-parse --short HEAD) job $SLURM_JOB_ID task $TASK"
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi blackwell; then
  export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
fi
python scripts/pi0/noise_pattern_eval.py --task "$TASK" --episodes 100 --n_envs 12
python scripts/pi0/plot_noise_pattern.py
