#!/bin/bash
# exp8 sweep launcher: structured noise layout (--steer.w_layout structured) for DSRL k=0 on pi0/LIBERO-10.
# Writes one sbatch per run (one run per H100) and prints the sbatch commands; --submit submits them.
# Runs resume from their own checkpoint, so a timed-out job can be resubmitted with the same generated script.
#
#   bash scripts/pi0/sweep_exp8.sh            # dry run: writes scripts/pi0/generated/exp8_structured/*.sbatch
#   bash scripts/pi0/sweep_exp8.sh --submit   # submit (job ids to scripts/pi0/generated/exp8_structured/jobs.txt)
#
# tasks 8, 4, 6 x action bound b in {1.0 (DSRL LIBERO, = exp7), 2.5 (DSRL real robot)}; fixed k=0, full x_T in
# the obs (now used: the noise is built from it). Everything else as exp7's pi0_t{4,6,8}_k0_xtfull_s0.
set -euo pipefail
ROOT=/n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
cd "$ROOT"

TASKS=(${TASKS:-8 4 6})
BOUNDS=(${BOUNDS:-1.0 2.5})
SEED=${SEED:-0}
TOTAL_ENV_STEPS=${TOTAL_ENV_STEPS:-1000000}
EVAL_EVERY=${EVAL_EVERY:-50000}
N_ENVS=${N_ENVS:-12}
CPUS=${CPUS:-16}                            # kempner_h100 allows < 24 cores per GPU
MEM=${MEM:-64G}
PARTITION=${PARTITION:-kempner_h100}
ACCOUNT=${ACCOUNT:-kempner_pehlevan_lab}
LOGDIR_ROOT=${LOGDIR_ROOT:-/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs/exp8}
# exp7 k=0 runs took 2.8-3.5 h (t8 slowest: 3.54 h); k=3 runs (which also carry the 1600-d x_T obs) up to 3.75 h.
WALL=${WALL:-06:00:00}
WALL_HOURS=${WALL_HOURS:-5.6}               # t0 is after ~5 min of imports: checkpoint+exit ~20 min before walltime
OUTDIR=scripts/pi0/generated/exp8_structured
mkdir -p "$OUTDIR" logs/exp8

files=()
names=()
for t in "${TASKS[@]}"; do
  for b in "${BOUNDS[@]}"; do
    name=t${t}_b${b}
    f=$OUTDIR/run_${name}.sbatch
    cat > "$f" <<EOS
#!/bin/bash
#SBATCH --job-name=p8_${name}
#SBATCH --output=logs/exp8/%x_%j.out
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
python train_adsrl_pi0.py --task_id $t --steer.mode fixed --steer.k 0 --steer.xt_layout full \\
    --steer.w_layout structured --action_magnitude $b --seed $SEED \\
    --total_env_steps $TOTAL_ENV_STEPS --n_envs $N_ENVS --eval_every $EVAL_EVERY --eval_det 1 \\
    --max_wall_hours $WALL_HOURS --logdir_root $LOGDIR_ROOT --wandb_group exp8_pi0
EOS
    files+=("$f"); names+=("$name")
  done
done

n=${#files[@]}
echo "$n runs, $PARTITION, $TOTAL_ENV_STEPS env steps each"
if [[ "${1:-}" != "--submit" ]]; then
  for f in "${files[@]}"; do echo "sbatch -p $PARTITION -A $ACCOUNT $f"; done
  exit 0
fi
: > "$OUTDIR/jobs.txt"
for ((r = 0; r < n; r++)); do
  id=$(sbatch --parsable -p "$PARTITION" -A "$ACCOUNT" "${files[$r]}")
  echo "${names[$r]} $id" | tee -a "$OUTDIR/jobs.txt"
done
