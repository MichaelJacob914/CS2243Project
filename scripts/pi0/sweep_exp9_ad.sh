#!/bin/bash
# exp9 adaptive: tasks 8,4,6 x adaptive k (all 10 steps, rho 0.5 as exp7) x lowfreq m=3 x abase (k=0 plan feature)
# x b=2.5. Writes only scripts/pi0/generated/exp9/run_t{8,4,6}_ad_lf3_abase_b2.5.sbatch (never the 9 fixed-k files);
# prints the sbatch commands, or submits with --submit (job ids appended to scripts/pi0/generated/exp9/jobs_ad.txt).
# KBIAS=detach writes run_t{8,4,6}_ad_lf3_abase_kdetach_b2.5.sbatch with --hybrid.k_entropy_bias detach instead.
set -euo pipefail
ROOT=/n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
cd "$ROOT"
TASKS=(${TASKS:-8 4 6})
PARTITION=${PARTITION:-kempner_h100}
ACCOUNT=${ACCOUNT:-kempner_pehlevan_lab}
KBIAS=${KBIAS:-keep}
OUTDIR=scripts/pi0/generated/exp9
TAG=""; EXTRA=""
[[ "$KBIAS" == detach ]] && TAG="_kdetach" && EXTRA=" --hybrid.k_entropy_bias detach"
mkdir -p "$OUTDIR" logs/exp9
files=()
for t in "${TASKS[@]}"; do
  f=$OUTDIR/run_t${t}_ad_lf3_abase${TAG}_b2.5.sbatch
  cat > "$f" <<EOS
#!/bin/bash
#SBATCH --job-name=p9_t${t}_ad_lf3_abase${TAG}_b2.5
#SBATCH --output=logs/exp9/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
cd $ROOT
source .venv-pi0/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
nvidia-smi --query-gpu=name --format=csv
python train_adsrl_pi0.py --task_id $t --steer.mode adaptive --steer.rho 0.5 --steer.xt_layout full \\
    --steer.w_layout lowfreq --steer.n_modes 3 --steer.obs_extra abase --action_magnitude 2.5${EXTRA} --seed 0 \\
    --total_env_steps 1000000 --n_envs 12 --eval_every 50000 --eval_det 1 --eval_at_start 1 \\
    --max_wall_hours 7.6 --logdir_root /n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs/exp9 --wandb_group exp9_pi0
EOS
  files+=("$f")
done
if [[ "${1:-}" != "--submit" ]]; then
  for f in "${files[@]}"; do echo "sbatch -p $PARTITION -A $ACCOUNT $f"; done
  exit 0
fi
for f in "${files[@]}"; do
  id=$(sbatch --parsable -p "$PARTITION" -A "$ACCOUNT" "$f"); echo "$(basename "$f" .sbatch) $id" | tee -a "$OUTDIR/jobs_ad.txt"
done
