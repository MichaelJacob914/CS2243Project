#!/bin/bash
# exp_6 main sweep launcher: one sbatch job (one GPU, one run) per (task, condition), seed 1.
#
#   bash scripts/furniture/launch_sweep.sh            # dry run: prints the sbatch commands
#   SUBMIT=1 bash scripts/furniture/launch_sweep.sh   # submits
#
# Edit or override the lists to change the sweep, e.g.
#   TASKS="one_leg_low" CONDITIONS="fixed_k0 adaptive uniformk" bash scripts/furniture/launch_sweep.sh
# Conditions: fixed_k<k> | adaptive | uniformk | residual | cem. Default (19 runs): fixed k {0,2,4,7},
# adaptive rho 0.5 and uniform-k on GRID, on all 3 tasks; residual on one_leg only; no CEM.
# Budgets, K=10, n_envs=1024, utd 64, noise critic 32, batch 256, buffer 4M, residual beta 0.05,
# partial reward (lamp/round_table) all come from cfg/adsrl/<task>.yaml.
#
# Resubmitting the same command is safe: train_adsrl runs use resume=true (latest.pt + replay buffer,
# saved every 5 evals and every 120 min) and finished runs only redo the final-eval check; a finished
# CEM run (result.json) exits immediately, an unfinished one restarts its search.
# Placement: one_leg on kempner_h100; lamp/round_table on kempner (A100) -- IsaacGym's PhysX crashes
# on H100 for those two. 1 run per GPU (MPS crashes IsaacGym). Kempner cap: 15 GPUs across all jobs.
set -euo pipefail
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
TASKS=${TASKS:-"one_leg_low lamp_low round_table_low"}
GRID=${GRID:-"0 2 3 4 5 6 7 8"}  # 8-point k grid under K=10 (see exp_6 Setup)
FIXED_K=${FIXED_K:-"0 2 4 7"}  # fixed-k runs (user decision 19:35); adaptive/uniform-k use GRID
CONDITIONS=${CONDITIONS:-"$(for k in $FIXED_K; do printf 'fixed_k%s ' $k; done)adaptive uniformk"}
RESIDUAL_TASKS=${RESIDUAL_TASKS:-"one_leg_low"}  # residual beta=0.05 only on these; add "cem" to CONDITIONS to run CEM
SEED=${SEED:-1}
TIME=${TIME:-1-04:00:00}  # 28 h; adaptive is budgeted for ~24 h
LOGROOT=${LOGROOT:-/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs/exp6}
SUBMIT=${SUBMIT:-0}
MAX_JOBS=${MAX_JOBS:-9}  # concurrency cap: job i waits (afterany) for job i-MAX_JOBS; transport uses 4-6 of the 15
KSUB="[$(echo $GRID | tr ' ' ',')]"

partition() { [ "$1" = one_leg_low ] && echo kempner_h100 || echo kempner; }
budget() { python3 -c "import re;print(re.search(r'total_timesteps: (\d+)', open('cfg/adsrl/$1.yaml').read()).group(1))"; }

IDS=()
dep() {  # dependency flag that keeps at most MAX_JOBS of this sweep running
  local n=${#IDS[@]}
  if [ $n -ge $MAX_JOBS ]; then echo "--dependency=afterany:${IDS[$((n - MAX_JOBS))]}"; fi
}
run() {
  echo "+ $*"
  if [ "$SUBMIT" = 1 ]; then id=$("$@" | awk '{print $NF}'); echo "  -> $id"; else id="<job${#IDS[@]}>"; fi
  IDS+=("$id")
}

for task in $TASKS; do
  P=$(partition $task)
  conds=$CONDITIONS
  case " $RESIDUAL_TASKS " in *" $task "*) conds="$conds residual" ;; esac
  for c in $conds; do
    case $c in
      fixed_k*) k=${c#fixed_k}; ov="steer.mode=fixed steer.k=$k" ;;
      adaptive) ov="steer.mode=adaptive steer.rho=0.5 steer.k_subset=$KSUB"; c=adaptive_rho0.5 ;;
      uniformk) ov="steer.mode=adaptive steer.freeze_k_uniform=true steer.k_subset=$KSUB" ;;
      residual) ov="steer.mode=residual"; c=residual_b0.05 ;;
      cem)
        B=$(budget $task)
        run sbatch $(dep) -J f6_${task}_cem -p $P -A kempner_pehlevan_lab --gres=gpu:1 -c 16 --mem=96G -t $TIME \
          -o logs/slurm/%x_%j.out -e logs/slurm/%x_%j.err \
          --wrap "source scripts/furniture/env.sh; export PYTHONUNBUFFERED=1 ADSRL_RUNS_PER_GPU=1; python scripts/cem_noise.py --task $task --seed $SEED --pop 64 --episodes_per_cand 16 --budget_env_steps $B --logdir $LOGROOT/adsrl/$task/cem/seed$SEED"
        continue ;;
      *) echo "unknown condition $c"; exit 1 ;;
    esac
    name=${c}_s${SEED}
    run env MPS=0 LOGROOT=$LOGROOT sbatch $(dep) -J f6_${task}_${name} -p $P --mem=96G -t $TIME \
      scripts/slurm/exp6_pack.sbatch $task "$name::seed=$SEED $ov"
  done
done
