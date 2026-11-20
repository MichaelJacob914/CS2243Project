#!/bin/bash
# exp_6 pre-sweep smoke on one GPU (A100 for lamp): tiny versions of the sweep conditions with the final
# configs (K=10, partial reward), a kill + resume of a fixed-k run, and a tiny furniture CEM.
set -uo pipefail
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project; source scripts/furniture/env.sh
export WANDB_MODE=offline PYTHONUNBUFFERED=1
T=${TASK:-lamp_low}; C=125  # chunks per episode for lamp/round_table (88 for one_leg)
[ $T = one_leg_low ] && C=88
SM=/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs/exp6_smoke2; rm -rf $SM; mkdir -p $SM
TINY="--config-name=$T log_dir=$SM use_wandb=false resume=true env.n_envs=64 eval_interval=$C train.init_rollout_steps=$C train.total_timesteps=$((64*8*C*4)) train.utd=2 train.noise_critic_grad_steps=1 checkpoint_every_evals=1 checkpoint_every_min=0 eval_dump_every=1"
G="[0,2,3,4,5,6,7,8]"
show() { grep -E "^\[|Traceback|Error|resumed from|saved checkpoint .*final" $1 | grep -v INFO | tail -${2:-12}; }
echo "== kill + resume (fixed k0)"
python train_adsrl.py $TINY name=k0 steer.mode=fixed steer.k=0 > $SM/k0_a.log 2>&1 &
PID=$!
until grep -q "saved checkpoint .*latest.pt at env_steps=[1-9]" $SM/k0_a.log 2>/dev/null || ! kill -0 $PID 2>/dev/null; do sleep 5; done
kill $PID; wait $PID 2>/dev/null; echo "killed after:"; show $SM/k0_a.log 6
python train_adsrl.py $TINY name=k0 steer.mode=fixed steer.k=0 > $SM/k0_b.log 2>&1; echo "resume exit $?"; show $SM/k0_b.log
cat $SM/adsrl/$T/k0/eval_summary.csv
python train_adsrl.py $TINY name=k0 steer.mode=fixed steer.k=0 > $SM/k0_c.log 2>&1; echo "rerun-finished exit $?"; show $SM/k0_c.log 4
for m in "adaptive::steer.mode=adaptive steer.rho=0.5 steer.k_subset=$G" "uniformk::steer.mode=adaptive steer.freeze_k_uniform=true steer.k_subset=$G" "residual::steer.mode=residual"; do
  n=${m%%::*}; ov=${m#*::}; echo "== $n"
  python train_adsrl.py $TINY name=$n $ov > $SM/$n.log 2>&1; echo "exit $?"; show $SM/$n.log 8
done
echo "== cem"
python scripts/cem_noise.py --task $T --seed 1 --pop 4 --episodes_per_cand 16 --budget_env_steps $((64*8*C*2)) --set env.n_envs=64 --logdir $SM/adsrl/$T/cem/seed1 > $SM/cem.log 2>&1; echo "exit $?"
grep -E "^\[cem|budget|CEM |Traceback|Error" $SM/cem.log | tail -6
python scripts/cem_noise.py --task $T --seed 1 --logdir $SM/adsrl/$T/cem/seed1 2>&1 | tail -1
