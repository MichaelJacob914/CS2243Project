#!/bin/bash
#SBATCH --job-name=p12_L90t14_k0_s1
#SBATCH --output=logs/exp12/%x_%j.out
#SBATCH --partition=gpu_h200
#SBATCH --account=olveczky_lab
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=09:00:00
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
source .venv-pi05/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/n/netscratch/olveczky_lab/Lab/charleszhang/jax_cache_pi05
echo "host $(hostname) job $SLURM_JOB_ID"; nvidia-smi --query-gpu=name --format=csv,noheader
python train_adsrl_pi0.py --model pi05 --suite libero_90 --task_id 14 --query_freq 10 --steer.mode fixed --steer.k 0 \
    --steer.xt_layout full --steer.w_layout lowfreq --steer.n_modes 3 --steer.obs_extra abase --action_magnitude 2.5 --seed 1 \
    --total_env_steps 1000000 --n_envs 12 --eval_every 50000 --eval_det 1 --eval_at_start 1 \
    --max_wall_hours 8.6 --logdir_root /n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs/exp12 --name pi05_libero_90_t14_qf10_k0_xtfull_lf3_abase_b2.5_s1 --wandb_group exp12_pi05
