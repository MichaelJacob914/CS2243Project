#!/bin/bash
# exp7 sweep launcher. Writes one sbatch per run (one run per GPU) and prints the sbatch commands; with --submit
# it submits them, keeping at most MAX_CONCURRENT of these jobs running at once: job i >= MAX_CONCURRENT waits
# (afterany) for job i - MAX_CONCURRENT. Runs resume from their own checkpoint, so a timed-out job can be
# resubmitted with the same generated script.
#
#   bash scripts/pi0/sweep_exp7.sh            # dry run: writes scripts/pi0/generated/exp7_sweep/*.sbatch
#   bash scripts/pi0/sweep_exp7.sh --submit   # submit (prints job ids to scripts/pi0/generated/exp7_sweep/jobs.txt)
#
# Settings approved 2026-11-07: tasks 8, 4, 6; DSRL k=0, fixed k in {3, 6, 8}, adaptive rho=0.5; 1 seed;
# 1M env steps; w = DSRL-pi0's tiled 32-d row, x_T = full 50x32 noise in the obs (--steer.xt_layout full; no
# effect on k=0's inputs); stochastic eval (50 episodes, main metric) + deterministic eval (one episode per init
# state) every 50k env steps, both rows in eval_summary.csv (column `mode`).
set -euo pipefail
ROOT=/n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
cd "$ROOT"

TASKS=(${TASKS:-8 4 6})                     # weakest LIBERO-10 tasks (base 0.675 / 0.72 / 0.79 over 200 episodes)
FIXED_KS=(${FIXED_KS:-0 3 6 8})             # k=0 is DSRL-pi0
RHO=${RHO:-0.5}
SEED=${SEED:-0}
TOTAL_ENV_STEPS=${TOTAL_ENV_STEPS:-1000000}
XT_LAYOUT=${XT_LAYOUT:-full}
EVAL_EVERY=${EVAL_EVERY:-50000}
N_ENVS=${N_ENVS:-12}
CPUS=${CPUS:-16}                            # kempner_h100 allows < 24 cores per GPU
MEM=${MEM:-64G}                             # ~40 GB RSS measured at 12 envs
PARTITION=${PARTITION:-kempner_h100}
ACCOUNT=${ACCOUNT:-kempner_pehlevan_lab}
# Expected ~3.4 h per run on H100: 2.2 h training (450k env steps/h) + 20 eval points x ~3 min (stoch + det)
# + ~10 min startup/checkpoints. Walltime ~1.5x that; the run checkpoints and exits ~20 min before it.
WALL=${WALL:-05:15:00}
WALL_HOURS=${WALL_HOURS:-4.9}               # t0 is after ~5 min of imports, so ~20 min margin
MAX_CONCURRENT=${MAX_CONCURRENT:-13}        # 15-GPU kempner cap minus the 2 running x6_* transport jobs
OUTDIR=scripts/pi0/generated/exp7_sweep
mkdir -p "$OUTDIR" logs/exp7

runs=()
names=()
for t in "${TASKS[@]}"; do
  for k in "${FIXED_KS[@]}"; do
    runs+=("--task_id $t --steer.mode fixed --steer.k $k"); names+=("t${t}_k${k}")
  done
  runs+=("--task_id $t --steer.mode adaptive --steer.rho $RHO"); names+=("t${t}_ad")
done

n=${#runs[@]}
echo "$n runs, <= $MAX_CONCURRENT concurrent, $PARTITION, $TOTAL_ENV_STEPS env steps each"
files=()
for ((r = 0; r < n; r++)); do
  f=$OUTDIR/run_${names[$r]}.sbatch
  cat > "$f" <<EOF
#!/bin/bash
#SBATCH --job-name=p7_${names[$r]}
#SBATCH --output=logs/exp7/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$WALL
cd $ROOT
source .venv-pi0/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
nvidia-smi --query-gpu=name --format=csv
python train_adsrl_pi0.py ${runs[$r]} --steer.xt_layout $XT_LAYOUT --seed $SEED \\
    --total_env_steps $TOTAL_ENV_STEPS --n_envs $N_ENVS --eval_every $EVAL_EVERY --eval_det 1 \\
    --max_wall_hours $WALL_HOURS
EOF
  files+=("$f")
done

if [[ "${1:-}" != "--submit" ]]; then
  for ((r = 0; r < n; r++)); do
    dep=""; ((r >= MAX_CONCURRENT)) && dep=" --dependency=afterany:<job of ${names[$((r - MAX_CONCURRENT))]}>"
    echo "sbatch -p $PARTITION -A $ACCOUNT$dep ${files[$r]}"
  done
  exit 0
fi

ids=()
: > "$OUTDIR/jobs.txt"
for ((r = 0; r < n; r++)); do
  dep=()
  ((r >= MAX_CONCURRENT)) && dep=(--dependency=afterany:${ids[$((r - MAX_CONCURRENT))]})
  id=$(sbatch --parsable -p "$PARTITION" -A "$ACCOUNT" "${dep[@]}" "${files[$r]}")
  ids+=("$id")
  echo "${names[$r]} $id ${dep[*]:-}" | tee -a "$OUTDIR/jobs.txt"
done
