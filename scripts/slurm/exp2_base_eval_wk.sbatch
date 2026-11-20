#!/bin/bash
#SBATCH --job-name=base_wk
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
#SBATCH --time=14:00:00
# exp_2 no-RL reference per k: w=0 and w~N(0,I) at step k, fresh x_T each chunk (scripts/base_eval_wk.py).
# can/square K=8 k=0..7 (200 eps); transport K=20 k in round(linspace(0,19,8)) (600 eps). 16 procs, 6 at a time, MPS.
set -uo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT; source .venv/bin/activate
export MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
OUT=$ROOT/results/base_eval_wk; PARTS=$OUT/parts; mkdir -p $PARTS; rm -f $PARTS/*.csv
SCR=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl/wk_${SLURM_JOB_ID}; mkdir -p $SCR
echo "git: $(git rev-parse HEAD) host: $(hostname)"; nvidia-smi
export CUDA_MPS_PIPE_DIRECTORY=$SCR/mps_pipe CUDA_MPS_LOG_DIRECTORY=$SCR/mps_log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d && echo "MPS daemon started"
t0=$(date +%s)
{
  for k in 0 3 5 8 11 14 16 19; do echo "transport --ddim_steps 20 --n_episodes 600 --ks $k"; done
  for t in square can; do for ks in "0 1" "2 3" "4 5" "6 7"; do echo "$t --ks $ks"; done; done
} | while read -r task rest; do echo "$task $rest"; done > $SCR/tasks.txt
cat $SCR/tasks.txt
i=0
while read -r task rest; do
  i=$((i+1))
  echo "python scripts/base_eval_wk.py --task $task $rest --out_csv $PARTS/${task}_$i.csv > $SCR/p$i.log 2>&1; echo \"[\$(date +%T)] part $i ($task $rest) exit \$?\""
done < $SCR/tasks.txt | xargs -P 6 -d '\n' -I{} bash -c {}
python - <<PY
import glob, pandas as pd
d = pd.concat([pd.read_csv(f) for f in glob.glob("$PARTS/*.csv")]).sort_values(["task", "k", "w_mode"])
for t, g in d.groupby("task"):
    g[["task", "k", "w_mode", "success", "se", "n"]].to_csv(f"$OUT/{t}.csv", index=False)
print(d.to_string(index=False))
PY
cp $SCR/p*.log $OUT/parts/ 2>/dev/null
echo quit | nvidia-cuda-mps-control
echo "done in $(( $(date +%s) - t0 ))s"
