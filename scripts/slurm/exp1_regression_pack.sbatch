#!/bin/bash
#SBATCH --job-name=exp1_regr
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
#SBATCH --time=05:00:00
# exp_1 Task A: k=0 regression (upstream DSRL-NA vs train_adsrl fixed k=0, 2 seeds each, 200k env steps)
# + smoke runs (adaptive, fixed k=4 with a kill/resume) as one 6-run MPS pack on one GPU.
# Submit from project root: sbatch scripts/slurm/exp1_regression_pack.sbatch
set -uo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export MUJOCO_GL=egl WANDB_MODE=offline ADSRL_RUNS_PER_GPU=6
SCR=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl/exp1_${SLURM_JOB_ID}; mkdir -p $SCR
OUT=$ROOT/logs/exp1/regr_${SLURM_JOB_ID}; mkdir -p $OUT
echo "git: $(git rev-parse HEAD) host: $(hostname) scratch: $SCR"; nvidia-smi
export CUDA_MPS_PIPE_DIRECTORY=$SCR/mps_pipe CUDA_MPS_LOG_DIRECTORY=$SCR/mps_log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d && echo "MPS daemon started"
nvidia-smi dmon -s um -d 30 -o T > $OUT/dmon.txt 2>&1 &
DMON=$!

COMMON="env.n_eval_envs=10 num_evals=200 deterministic_eval=True"
ADS="wandb.mode=offline"
t0=$(date +%s)
PIDS=()
for s in 1 2; do
  (cd dsrl && exec python $ROOT/scripts/pilot_train_dsrl.py --config-path=$ROOT/dsrl/cfg/robomimic --config-name=dsrl_can.yaml \
    $COMMON +total_timesteps=44000 +buffer_size=300000 use_wandb=False seed=$s eval_interval=1500 \
    save_model_interval=100000000 log_dir=$SCR/upstream name=upstream_s$s +eval_csv=$SCR/upstream/upstream_s$s/eval.csv) \
    > $OUT/upstream_s$s.log 2>&1 &
  PIDS+=($!)
  python train_adsrl.py --config-name=can steer.mode=fixed steer.k=0 seed=$s name=k0_s$s $COMMON $ADS \
    train.total_timesteps=200000 eval_interval=1500 log_dir=$SCR > $OUT/k0_s$s.log 2>&1 &
  PIDS+=($!)
done
SMOKE="$COMMON $ADS train.total_timesteps=50000 eval_interval=500 log_dir=$SCR"
python train_adsrl.py --config-name=can steer.mode=adaptive seed=1 name=smoke_adaptive $SMOKE > $OUT/smoke_adaptive.log 2>&1 &
PIDS+=($!)
# fixed k=4: kill after its 2nd checkpoint, then resume
python train_adsrl.py --config-name=can steer.mode=fixed steer.k=4 seed=1 name=smoke_k4 $SMOKE > $OUT/smoke_k4.log 2>&1 &
K4=$!
until [ "$(grep -c 'saved checkpoint' $OUT/smoke_k4.log 2>/dev/null)" -ge 2 ] || ! kill -0 $K4 2>/dev/null; do sleep 20; done
echo "[$(date +%T)] killing smoke_k4 after: $(grep 'saved checkpoint' $OUT/smoke_k4.log | tail -1)"
pkill -TERM -P $K4; kill -TERM $K4; sleep 15; pkill -KILL -P $K4 2>/dev/null; kill -KILL $K4 2>/dev/null
cp $SCR/adsrl/can/smoke_k4/eval_summary.csv $OUT/smoke_k4_eval_before_kill.csv
python train_adsrl.py --config-name=can steer.mode=fixed steer.k=4 seed=1 name=smoke_k4 $SMOKE resume=true > $OUT/smoke_k4_resume.log 2>&1 &
PIDS+=($!)

for p in "${PIDS[@]}"; do wait $p; echo "[$(date +%T)] pid $p exit $? after $(( $(date +%s) - t0 ))s"; done
kill $DMON 2>/dev/null
echo quit | nvidia-cuda-mps-control
# keep summaries in the lab dir (checkpoints/buffers stay on scratch)
rsync -a --exclude checkpoint --exclude 'wandb' $SCR/adsrl/ $SCR/upstream $ROOT/results/exp1_regression/
echo "done in $(( $(date +%s) - t0 ))s"
