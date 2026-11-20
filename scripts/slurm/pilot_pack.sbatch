#!/bin/bash
#SBATCH --job-name=pilot_pack
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --time=02:00:00
# Packing throughput: N in {1,2,4,6,8} copies of DSRL-NA can on one GPU, MPS on or off ($1 = mps|nomps).
# Eval disabled and short init rollout so we measure steady-state training throughput.
# Submit from project root: sbatch scripts/slurm/pilot_pack.sbatch mps
# Used: sbatch -p kempner_h100 --gres=gpu:2 -c 46 scripts/slurm/pilot_pack.sbatch {mps|nomps}  (2nd GPU idle; buys CPUs)
set -uo pipefail
MODE=${1:-nomps}
DUR=${DUR:-420}     # seconds per configuration
ROOT=$SLURM_SUBMIT_DIR
source $ROOT/.venv/bin/activate
export WANDB_MODE=disabled MUJOCO_GL=egl
SCR=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl/pilot_${SLURM_JOB_ID}
OUT=$ROOT/logs/pilot/pack_${MODE}_${SLURM_JOB_ID}; mkdir -p $OUT $SCR
# If the job holds >1 GPU (to get more CPUs under kempner's per-GPU core cap), use only the first one.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
echo "git: $(git -C $ROOT rev-parse HEAD) host: $(hostname) mode: $MODE"; nvidia-smi
if [ "$MODE" = mps ]; then
  export CUDA_MPS_PIPE_DIRECTORY=$SCR/mps_pipe CUDA_MPS_LOG_DIRECTORY=$SCR/mps_log
  mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  nvidia-cuda-mps-control -d && echo "MPS daemon started"
fi
cd $ROOT/dsrl
for N in 1 2 4 6 8; do
  D=$OUT/n$N; mkdir -p $D
  nvidia-smi dmon -i $CUDA_VISIBLE_DEVICES -s um -d 5 -o T > $D/dmon.txt 2>&1 &
  DMON=$!
  PIDS=()
  for i in $(seq 1 $N); do
    python $ROOT/scripts/pilot_train_dsrl.py --config-path=$ROOT/dsrl/cfg/robomimic --config-name=dsrl_can.yaml \
      +profile=true +total_timesteps=10000000 +buffer_size=1000000 use_wandb=False seed=$i \
      num_evals=0 env.n_eval_envs=1 train.init_rollout_steps=300 save_model_interval=100000000 \
      log_dir=$SCR/logs name=pack_n${N}_r${i} > $D/run$i.log 2>&1 &
    PIDS+=($!)
  done
  sleep $DUR
  free -g > $D/free.txt
  for p in "${PIDS[@]}"; do pkill -TERM -P $p 2>/dev/null; kill $p 2>/dev/null; done
  sleep 5; for p in "${PIDS[@]}"; do pkill -KILL -P $p 2>/dev/null; kill -9 $p 2>/dev/null; done
  pkill -u $USER -f "pack_n${N}_r" 2>/dev/null
  kill $DMON 2>/dev/null; sleep 10
done
if [ "$MODE" = mps ]; then echo quit | nvidia-cuda-mps-control; fi
