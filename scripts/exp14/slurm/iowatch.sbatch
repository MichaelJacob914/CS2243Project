#!/bin/bash
#SBATCH --job-name=e14_iowatch
#SBATCH --output=logs/exp14/%x_%j.out
#SBATCH --partition=test
#SBATCH --account=olveczky_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=12:00:00
# Waits for the lab filesystem to recover, then submits exp14 Stage 4 and the final analysis.
# Every 20 min it submits a 1-core probe job (a fresh job, usually on another node, so its page cache
# is cold) that times `import numpy` and `import torch` from the project venv. When a probe finishes
# with numpy < 15 s and torch < 90 s, it submits Stage 4 (112 cores) and the analysis (afterok).
ROOT=$SLURM_SUBMIT_DIR; cd $ROOT
for v in $(env | grep -o '^SLURM_[A-Z_]*'); do unset $v; done  # don't leak this job's settings into sbatch
PROBE='source .venv/bin/activate; s=$(date +%s); timeout 600 python -c "import numpy" || exit 1; n=$(( $(date +%s)-s )); s=$(date +%s); timeout 900 python -c "import torch" || exit 1; t=$(( $(date +%s)-s )); echo "PROBE numpy=$n torch=$t host=$(hostname)"'
while true; do
  jid=$(sbatch --parsable -p test -A olveczky_lab -c 1 --mem=4G -t 00:30:00 -J e14_probe -o logs/exp14/probe_%j.out --wrap "$PROBE")
  echo "[$(date '+%F %T')] probe $jid submitted"
  while squeue -h -j $jid 2>/dev/null | grep -q .; do sleep 30; done
  line=$(grep PROBE logs/exp14/probe_$jid.out 2>/dev/null)
  echo "[$(date '+%F %T')] probe $jid: ${line:-timed out or failed}"
  if [ -n "$line" ]; then
    n=$(sed 's/.*numpy=\([0-9]*\).*/\1/' <<<"$line"); t=$(sed 's/.*torch=\([0-9]*\).*/\1/' <<<"$line")
    if [ "$n" -lt 15 ] && [ "$t" -lt 90 ]; then
      s4=$(sbatch --parsable --time=06:00:00 --cpus-per-task=112 --mem=300G --job-name=e14_sim4 scripts/exp14/slurm/sim.sbatch stage4)
      an=$(sbatch --parsable --dependency=afterok:$s4 scripts/exp14/slurm/analyze.sbatch)
      echo "[$(date '+%F %T')] FS recovered; submitted stage4 $s4, analysis $an"
      exit 0
    fi
  fi
  sleep 1200
done
