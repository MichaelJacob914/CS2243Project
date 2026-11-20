#!/bin/bash
# Throughput probe: runs the given train_adsrl_pi0.py configs concurrently on one GPU for WALL_H hours each
# (no evals), then train_log.csv in each logdir has env steps/hour, pi0 s/call, update time.
#   CONFIGS="k0;adaptive" WALL_H=0.4 N_ENVS=16 sbatch -p kempner -A kempner_pehlevan_lab scripts/pi0/throughput.sbatch
#SBATCH --job-name=p7_tput
#SBATCH --output=logs/exp7/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=160G
#SBATCH --time=01:30:00
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
source .venv-pi0/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
TASK=${TASK:-0}; WALL_H=${WALL_H:-0.4}; N_ENVS=${N_ENVS:-16}
nvidia-smi --query-gpu=name --format=csv
(while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader >> logs/exp7/tput_${SLURM_JOB_ID}_gpu.csv; sleep 10; done) &
MON=$!
pids=()
IFS=';' read -ra CFGS <<< "${CONFIGS:-k0}"
i=0
for c in "${CFGS[@]}"; do
  if [[ $c == k* ]]; then args="--steer.mode fixed --steer.k ${c#k}"; else args="--steer.mode adaptive --steer.rho 0.5"; fi
  python train_adsrl_pi0.py --task_id $TASK $args --n_envs $N_ENVS --total_env_steps 100000000 --eval_every 100000000 \
      --max_wall_hours $WALL_H --use_wandb 0 --logdir_root /n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs/exp7/throughput \
      --name tput_${SLURM_JOB_ID}_${i}_${c}_n${N_ENVS} > logs/exp7/tput_${SLURM_JOB_ID}_${i}_${c}.log 2>&1 &
  pids+=($!)
  i=$((i+1))
done
wait "${pids[@]}"
kill $MON
