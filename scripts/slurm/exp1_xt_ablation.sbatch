#!/bin/bash
#SBATCH --job-name=exp1_xt
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
# exp_1 diagnostic: is x_T in the network inputs what makes train_adsrl k=0 lag upstream on can?
# 5-run MPS pack, 200k env steps each: k=0 with x_T dropped from all network inputs (seeds 1-3),
# plus a 3rd seed of upstream and of k=0 with x_T.
set -uo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export MUJOCO_GL=egl WANDB_MODE=offline ADSRL_RUNS_PER_GPU=5
SCR=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl/exp1xt_${SLURM_JOB_ID}; mkdir -p $SCR
OUT=$ROOT/logs/exp1/xt_${SLURM_JOB_ID}; mkdir -p $OUT
echo "git: $(git rev-parse HEAD) host: $(hostname) scratch: $SCR"; nvidia-smi
export CUDA_MPS_PIPE_DIRECTORY=$SCR/mps_pipe CUDA_MPS_LOG_DIRECTORY=$SCR/mps_log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d && echo "MPS daemon started"
COMMON="env.n_eval_envs=10 num_evals=200 deterministic_eval=True eval_interval=1500"
ADS="$COMMON wandb.mode=offline train.total_timesteps=200000 log_dir=$SCR steer.mode=fixed steer.k=0"
t0=$(date +%s)
for s in 1 2 3; do
  python train_adsrl.py --config-name=can $ADS steer.obs_xT=false seed=$s name=k0_noxt_s$s > $OUT/k0_noxt_s$s.log 2>&1 &
done
python train_adsrl.py --config-name=can $ADS seed=3 name=k0_s3 > $OUT/k0_s3.log 2>&1 &
(cd dsrl && exec python $ROOT/scripts/pilot_train_dsrl.py --config-path=$ROOT/dsrl/cfg/robomimic --config-name=dsrl_can.yaml \
  $COMMON +total_timesteps=44000 +buffer_size=300000 use_wandb=False seed=3 save_model_interval=100000000 \
  log_dir=$SCR/upstream name=upstream_s3 +eval_csv=$SCR/upstream/upstream_s3/eval.csv) > $OUT/upstream_s3.log 2>&1 &
wait
echo quit | nvidia-cuda-mps-control
rsync -a --exclude checkpoint --exclude wandb $SCR/adsrl/ $SCR/upstream $ROOT/results/exp1_xt_ablation/
echo "done in $(( $(date +%s) - t0 ))s"
