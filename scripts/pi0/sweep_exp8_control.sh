#!/bin/bash
# exp8 control: exp7's DSRL k=0 (tiled w, b=1.0) plus x_T (1600-d) in the agent obs (--steer.force_xt_obs 1).
# Separates "structured noise" from "the agent sees the 1600-d x_T" in the exp8 comparison.
# Writes scripts/pi0/generated/exp8_structured/run_t{8,4,6}_tiled_xtobs.sbatch only (never the structured files);
# prints the sbatch commands, or submits them with --submit (job ids appended to .../jobs_control.txt).
set -euo pipefail
ROOT=/n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
cd "$ROOT"
TASKS=(${TASKS:-8 4 6})
PARTITION=${PARTITION:-kempner_h100}
ACCOUNT=${ACCOUNT:-kempner_pehlevan_lab}
OUTDIR=scripts/pi0/generated/exp8_structured
mkdir -p "$OUTDIR" logs/exp8
files=()
for t in "${TASKS[@]}"; do
  f=$OUTDIR/run_t${t}_tiled_xtobs.sbatch
  cat > "$f" <<EOS
#!/bin/bash
#SBATCH --job-name=p8_t${t}_tiled_xtobs
#SBATCH --output=logs/exp8/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
cd $ROOT
source .venv-pi0/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
nvidia-smi --query-gpu=name --format=csv
python train_adsrl_pi0.py --task_id $t --steer.mode fixed --steer.k 0 --steer.xt_layout full \\
    --steer.force_xt_obs 1 --action_magnitude 1.0 --seed 0 \\
    --total_env_steps 1000000 --n_envs 12 --eval_every 50000 --eval_det 1 \\
    --max_wall_hours 5.6 --logdir_root /n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs/exp8 --wandb_group exp8_pi0
EOS
  files+=("$f")
done
if [[ "${1:-}" != "--submit" ]]; then
  for f in "${files[@]}"; do echo "sbatch -p $PARTITION -A $ACCOUNT $f"; done
  exit 0
fi
for f in "${files[@]}"; do
  id=$(sbatch --parsable -p "$PARTITION" -A "$ACCOUNT" "$f"); echo "$(basename "$f" .sbatch) $id" | tee -a "$OUTDIR/jobs_control.txt"
done
