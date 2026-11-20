#!/bin/bash
# exp12: adaptive-step DSRL on pi0.5 (.venv-pi05), exp9 recipe, query_freq 10 (full chunk before replanning).
# Tasks: LIBERO-10 task 8; LIBERO-90 tasks 14, 28, 77. Methods: k0 (standard DSRL), k3 (fixed k=3), ad (adaptive,
# rho 0.5, k-entropy-bias detach). Seeds 0 1 2. 4 x 3 x 3 = 36 runs.
# Writes scripts/pi05/generated/exp12/run_<name>.sbatch (one per run) and jobs.txt (one `sbatch ...` line per run).
# Does not submit unless called with --submit (job ids appended to scripts/pi05/generated/exp12/submitted.txt).
# A run that hits --max_wall_hours checkpoints and exits; resubmitting the same file resumes it ("resumed at").
# Run names (= logdir) carry model, suite, task, query_freq, method and seed, so they are unique per run, and
# LIBERO-10 vs LIBERO-90 runs with the same task id cannot collide.
set -euo pipefail
ROOT=/n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
cd "$ROOT"
PARTITION=gpu_h200
ACCOUNT=olveczky_lab
CPUS=${CPUS:-16}
MEM=${MEM:-64G}
# Walltime ~1.5x the expected time for 1M env steps (smoke 59248273, L90 t28, H200: 228k env steps/h in training
# rounds + ~230 s per stoch+det eval pair -> ~5.8 h). LIBERO-10 t8 is extrapolated (2x sim cost per step, ~2x longer
# episodes): ~9.7 h. --max_wall_hours leaves ~25 min for the final checkpoint and exit.
declare -A TIME=([libero_90]=09:00:00 [libero_10]=14:30:00)
declare -A MAX_WALL=([libero_90]=8.6 [libero_10]=14.0)
SEEDS=(${SEEDS:-0 1 2})
LOGDIR_ROOT=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs/exp12
OUTDIR=scripts/pi05/generated/exp12
RECIPE="--steer.xt_layout full --steer.w_layout lowfreq --steer.n_modes 3 --steer.obs_extra abase --action_magnitude 2.5"
# suite:task pairs
RUNS_TASKS=("libero_10:8" "libero_90:14" "libero_90:28" "libero_90:77")
declare -A METHOD_ARGS=(
  [k0]="--steer.mode fixed --steer.k 0"
  [k3]="--steer.mode fixed --steer.k 3"
  [ad]="--steer.mode adaptive --steer.rho 0.5 --hybrid.k_entropy_bias detach"
)
declare -A METHOD_TAG=(
  [k0]="k0_xtfull_lf3_abase_b2.5"
  [k3]="k3_xtfull_lf3_abase_b2.5"
  [ad]="adaptive_rho0.5_xtfull_lf3_abase_kdetach_b2.5"
)
declare -A SUITE_SHORT=([libero_10]=L10 [libero_90]=L90)
mkdir -p "$OUTDIR" logs/exp12
: > "$OUTDIR/jobs.txt"
files=()
for st in "${RUNS_TASKS[@]}"; do
  suite=${st%%:*}; t=${st##*:}
  for m in k0 k3 ad; do
    for s in "${SEEDS[@]}"; do
      name=pi05_${suite}_t${t}_qf10_${METHOD_TAG[$m]}_s${s}
      f=$OUTDIR/run_${name}.sbatch
      cat > "$f" <<EOS
#!/bin/bash
#SBATCH --job-name=p12_${SUITE_SHORT[$suite]}t${t}_${m}_s${s}
#SBATCH --output=logs/exp12/%x_%j.out
#SBATCH --partition=$PARTITION
#SBATCH --account=$ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=${TIME[$suite]}
cd $ROOT
source .venv-pi05/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/n/netscratch/olveczky_lab/Lab/charleszhang/jax_cache_pi05
echo "host \$(hostname) job \$SLURM_JOB_ID"; nvidia-smi --query-gpu=name --format=csv,noheader
python train_adsrl_pi0.py --model pi05 --suite $suite --task_id $t --query_freq 10 ${METHOD_ARGS[$m]} \\
    $RECIPE --seed $s \\
    --total_env_steps 1000000 --n_envs 12 --eval_every 50000 --eval_det 1 --eval_at_start 1 \\
    --max_wall_hours ${MAX_WALL[$suite]} --logdir_root $LOGDIR_ROOT --name $name --wandb_group exp12_pi05
EOS
      files+=("$f")
      echo "sbatch $f" >> "$OUTDIR/jobs.txt"
    done
  done
done
n_unique=$(printf '%s\n' "${files[@]}" | sort -u | wc -l)
[[ $n_unique -eq ${#files[@]} ]] || { echo "duplicate run names" >&2; exit 1; }
echo "wrote ${#files[@]} sbatch files to $OUTDIR (list: $OUTDIR/jobs.txt)"
if [[ "${1:-}" == "--submit" ]]; then
  for f in "${files[@]}"; do
    id=$(sbatch --parsable "$f"); echo "$(basename "$f" .sbatch) $id" | tee -a "$OUTDIR/submitted.txt"
  done
fi
