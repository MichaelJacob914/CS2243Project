#!/bin/bash
# exp12 step 2: pi0.5 adaptive smoke (LIBERO-90 task 28, query_freq 10, exp9 recipe), 30k env steps, evals every 10k
# + at start. Resume check: the first run is SIGKILLed right after its first checkpoint, then the identical command
# is rerun and must print "resumed at". A sampler logs total RSS of the job's processes and GPU memory every 15 s.
#   sbatch -p gpu_h200 -A olveczky_lab scripts/pi05/smoke_exp12.sbatch
#SBATCH --job-name=p12_smoke
#SBATCH --output=logs/exp12/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
source .venv-pi05/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/n/netscratch/olveczky_lab/Lab/charleszhang/jax_cache_pi05
echo "host $(hostname) job $SLURM_JOB_ID"; nvidia-smi --query-gpu=name --format=csv,noheader
ROOT_DIR=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs/exp12_smoke
NAME=smoke_pi05_libero_90_t28_qf10_ad_lf3_abase_kdetach_b2.5_s0
rm -rf $ROOT_DIR/$NAME
mkdir -p $ROOT_DIR
MEMLOG=logs/exp12/smoke_mem_${SLURM_JOB_ID}.csv
echo "t,rss_gb_total,gpu_mem_mib" > $MEMLOG
( while true; do
    rss=$(ps -u $USER -o rss= --no-headers | awk '{s+=$1} END {printf "%.2f", s/1048576}')
    g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1)
    echo "$(date +%s),$rss,$g" >> $MEMLOG; sleep 15; done ) &
SAMPLER=$!
CMD=(python train_adsrl_pi0.py --model pi05 --suite libero_90 --task_id 28 --query_freq 10 \
    --steer.mode adaptive --steer.rho 0.5 --hybrid.k_entropy_bias detach --steer.xt_layout full \
    --steer.w_layout lowfreq --steer.n_modes 3 --steer.obs_extra abase --action_magnitude 2.5 --seed 0 \
    --total_env_steps 30000 --n_envs 12 --eval_every 10000 --eval_det 1 --eval_at_start 1 \
    --logdir_root $ROOT_DIR --name $NAME --use_wandb 0)
LOG1=logs/exp12/smoke_run1_${SLURM_JOB_ID}.log
setsid "${CMD[@]}" > $LOG1 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  if grep -q "^checkpoint at" $LOG1; then echo "first checkpoint seen; SIGKILL run 1"; kill -9 -- -$PID; break; fi
  sleep 5
done
sleep 10; pkill -9 -f "train_adsrl_pi0.py.*$NAME" || true; sleep 5
echo "===== run 1 log"; cat $LOG1
echo "===== run 2 (same command)"
"${CMD[@]}"
echo "run2 exit $?"
kill $SAMPLER
D=$ROOT_DIR/$NAME
echo "===== eval_summary"; cat $D/eval_summary.csv
echo "===== train_log"; cat $D/train_log.csv
ls $D
awk -F, 'NR>1 {if ($2>r) r=$2; if ($3>g) g=$3} END {print "peak total RSS GB", r, "peak GPU MiB", g}' $MEMLOG
