#!/bin/bash
# exp9 sweep launcher: low-frequency noise (--steer.w_layout lowfreq) and the a_base plan feature in the obs
# (--steer.obs_extra abase). Writes one sbatch per run (one run per H100) into scripts/pi0/generated/exp9/ and
# prints the sbatch commands; --submit submits them (job ids to scripts/pi0/generated/exp9/jobs.txt).
# Runs resume from their own checkpoint, so a timed-out job can be resubmitted with the same generated script.
#
#   main     : tasks 8,4,6 x k in {0,3} x lowfreq m=3 x abase x b=2.5          (6 runs)
#   ablation : tasks 8,4,6 x k=0 x structured (time-mean) x abase x b=1.0     (3 runs; vs exp8 structured b=1, x_T obs)
# Everything else as exp7's pi0_t{4,6,8}_k0_xtfull_s0 (1M env steps, 12 envs, seed 0, stoch + det eval every 50k).
set -euo pipefail
ROOT=/n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
cd "$ROOT"
TASKS=(${TASKS:-8 4 6})
PARTITION=${PARTITION:-kempner_h100}
ACCOUNT=${ACCOUNT:-kempner_pehlevan_lab}
WALL=${WALL:-08:00:00}
WALL_HOURS=${WALL_HOURS:-7.6}               # t0 is after ~5 min of imports: checkpoint+exit ~20 min before walltime
LOGDIR_ROOT=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs/exp9
OUTDIR=scripts/pi0/generated/exp9
mkdir -p "$OUTDIR" logs/exp9

names=(); args=()
for t in "${TASKS[@]}"; do
  for k in 0 3; do
    names+=("t${t}_k${k}_lf3_abase_b2.5")
    args+=("--task_id $t --steer.k $k --steer.w_layout lowfreq --steer.n_modes 3 --steer.obs_extra abase --action_magnitude 2.5")
  done
done
for t in "${TASKS[@]}"; do
  names+=("t${t}_k0_wstruct_abase_b1.0")
  args+=("--task_id $t --steer.k 0 --steer.w_layout structured --steer.obs_extra abase --action_magnitude 1.0")
done

files=()
for ((r = 0; r < ${#names[@]}; r++)); do
  f=$OUTDIR/run_${names[$r]}.sbatch
  cat > "$f" <<EOS
#!/bin/bash
#SBATCH --job-name=p9_${names[$r]}
#SBATCH --output=logs/exp9/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=$WALL
cd $ROOT
source .venv-pi0/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
nvidia-smi --query-gpu=name --format=csv
python train_adsrl_pi0.py ${args[$r]} --steer.mode fixed --steer.xt_layout full --seed 0 \\
    --total_env_steps 1000000 --n_envs 12 --eval_every 50000 --eval_det 1 --eval_at_start 1 \\
    --max_wall_hours $WALL_HOURS --logdir_root $LOGDIR_ROOT --wandb_group exp9_pi0
EOS
  files+=("$f")
done
echo "${#files[@]} runs, $PARTITION"
if [[ "${1:-}" != "--submit" ]]; then
  for f in "${files[@]}"; do echo "sbatch -p $PARTITION -A $ACCOUNT $f"; done
  exit 0
fi
: > "$OUTDIR/jobs.txt"
for ((r = 0; r < ${#files[@]}; r++)); do
  id=$(sbatch --parsable -p "$PARTITION" -A "$ACCOUNT" "${files[$r]}"); echo "${names[$r]} $id" | tee -a "$OUTDIR/jobs.txt"
done
