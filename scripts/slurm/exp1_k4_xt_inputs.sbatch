#!/bin/bash
#SBATCH --job-name=exp1_k4xt
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --exclude=holygpu8a12204
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=04:00:00
# exp_1: fixed k=4 on can, steer.xT_inputs=all vs actor_noise_critic (Q(s,a) without x_T), 2 seeds, 200k env steps; 4-run MPS pack.
set -uo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export MUJOCO_GL=egl WANDB_MODE=offline ADSRL_RUNS_PER_GPU=4
SCR=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl/exp1k4xt_${SLURM_JOB_ID}; mkdir -p $SCR
OUT=$ROOT/logs/exp1/k4xt_${SLURM_JOB_ID}; mkdir -p $OUT
echo "git: $(git rev-parse HEAD) host: $(hostname) scratch: $SCR"; nvidia-smi
export CUDA_MPS_PIPE_DIRECTORY=$SCR/mps_pipe CUDA_MPS_LOG_DIRECTORY=$SCR/mps_log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d && echo "MPS daemon started"
ADS="env.n_eval_envs=10 num_evals=200 deterministic_eval=True eval_interval=1500 wandb.mode=offline train.total_timesteps=200000 log_dir=$SCR steer.mode=fixed steer.k=4"
t0=$(date +%s)
for x in all actor_noise_critic; do for s in 1 2; do
  python train_adsrl.py --config-name=can $ADS steer.xT_inputs=$x seed=$s name=k4_${x}_s$s > $OUT/k4_${x}_s$s.log 2>&1 &
done; done
wait
echo quit | nvidia-cuda-mps-control
rsync -a --exclude checkpoint --exclude wandb $SCR/adsrl/ $ROOT/results/exp1_k4_xt_inputs/
echo "done in $(( $(date +%s) - t0 ))s"
