#!/bin/bash
#SBATCH --job-name=f6_base
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
# exp_6 base-policy eval (w = x_T ~ N(0, I)) at several DDIM K. Usage (from project root):
#   sbatch -J f6_base_<task> scripts/slurm/exp6_base_eval.sbatch <task> [n_envs] [rounds] [Ks...]
set -uo pipefail
TASK=${1:?task}; N=${2:-1024}; R=${3:-1}; shift 3 2>/dev/null || shift $#
KS=${@:-100 20 10}
cd $SLURM_SUBMIT_DIR; source scripts/furniture/env.sh
export PYTHONUNBUFFERED=1
echo "git: $(git rev-parse HEAD) host: $(hostname) task: $TASK n_envs=$N rounds=$R Ks=$KS"; nvidia-smi --query-gpu=name,memory.used --format=csv
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 30 > logs/exp6/base_eval_${TASK}_${SLURM_JOB_ID}_gpu.csv 2>&1 &
python scripts/furniture/base_eval_furniture.py --task $TASK --n_envs $N --rounds $R --ks $KS 2>&1 | grep --line-buffered -v "^\s*$"
kill %1 2>/dev/null
