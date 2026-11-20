#!/bin/bash
# exp_6 k=0 stabilization pilot (one_leg_low, final sweep config: K=10, utd 64, NC 32, batch 256, buffer 4M),
# one GPU per variant, <=5 h, eval (1024 det episodes; 512 at n=256) every 5 episodes, no buffer checkpoints.
set -euo pipefail
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
C="steer.mode=fixed steer.k=0 use_wandb=false checkpoint_every_min=0 checkpoint_every_evals=100000 eval_interval=440 train.init_rollout_steps=88 train.total_timesteps=1000000000"
TR="train.trust_region.enabled=true train.trust_region.bound=0.3"
declare -A V=(
  [stab_n256]="env.n_envs=256 num_evals=512"
  [stab_tr_lr1e-4]="$TR train.actor_lr=1e-4"
  [stab_ws0.5]="steer.w_scale=0.5"
  [stab_ags8_buf8M]="train.actor_gradient_steps=8 train.buffer_size=8000000"
  [stab_n256_tr_lr1e-4_ags16]="env.n_envs=256 num_evals=512 $TR train.actor_lr=1e-4 train.actor_gradient_steps=16"
  [stab_ws0.5_ags8_buf8M]="steer.w_scale=0.5 train.actor_gradient_steps=8 train.buffer_size=8000000"
  [stab_ws0.25]="steer.w_scale=0.25"
  [stab_ags4_buf8M]="train.actor_gradient_steps=4 train.buffer_size=8000000"
  # larger minibatch + more updates per env step (user, 22:10); actor steps kept at utd/8; w_scale 0.5; 8M buffer
  [stab_ws0.5_b1024_u128]="steer.w_scale=0.5 train.buffer_size=8000000 train.batch_size=1024 train.utd=128 train.noise_critic_grad_steps=64 train.actor_gradient_steps=16"
  [stab_ws0.5_b2048_u256]="steer.w_scale=0.5 train.buffer_size=8000000 train.batch_size=2048 train.utd=256 train.noise_critic_grad_steps=128 train.actor_gradient_steps=32"
  [stab_ags8_buf8M_s2]="seed=2 train.actor_gradient_steps=8 train.buffer_size=8000000"
  [stab_b1024_u128_ags16_buf8M]="train.buffer_size=8000000 train.batch_size=1024 train.utd=128 train.noise_critic_grad_steps=64 train.actor_gradient_steps=16"
  [stab_ws0.5_b1024_u256]="steer.w_scale=0.5 train.buffer_size=8000000 train.batch_size=1024 train.utd=256 train.noise_critic_grad_steps=128 train.actor_gradient_steps=32"
)
for name in ${WHICH:-${!V[@]}}; do
  MPS=0 sbatch -J f6_$name -t 05:00:00 scripts/slurm/exp6_pack.sbatch one_leg_low "$name::$C ${V[$name]}"
done
