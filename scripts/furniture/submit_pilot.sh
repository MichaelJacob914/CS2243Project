#!/bin/bash
# exp_6 throughput pilot on one_leg_low (K=20). Usage: bash scripts/furniture/submit_pilot.sh <which>
#   seq<N>   : one GPU, runs one at a time (SEQ=1): fixed k=0 and adaptive (8-point k grid) x utd {4,16}
#   mps<P>   : P concurrent fixed-k0 runs (n=1024, utd 16) under MPS
#   big<N>   : fixed k0 and adaptive at utd 16 with n_envs=N (e.g. big2048)
#   mpsad<P> : P concurrent adaptive runs under MPS
#   learn    : fixed k=0 learning check, n=1024, utd 8, ~75 min of training, eval every 5 episodes
set -euo pipefail
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
GRID="[0,3,6,9,12,15,17,19]"
base() {  # $1 = n_envs, $2 = train episodes after the warm-start episode
  echo "env.n_envs=$1 model.ddim_steps=20 use_wandb=false checkpoint_every_min=0 eval_interval=88000 train.init_rollout_steps=88 train.total_timesteps=$(( (1 + $2) * 88 * $1 * 8 ))"
}
which=$1
case $which in
  seq*) N=${which#seq}; B=$(base $N 3)
    SEQ=1 RUN_SEC=2400 MPS=0 sbatch -J f6_p$which --time=05:00:00 scripts/slurm/exp6_pack.sbatch one_leg_low \
      "p_n${N}_k0_u4::$B steer.mode=fixed steer.k=0 train.utd=4 train.noise_critic_grad_steps=2" \
      "p_n${N}_k0_u16::$B steer.mode=fixed steer.k=0 train.utd=16 train.noise_critic_grad_steps=8" \
      "p_n${N}_ad_u4::$B steer.mode=adaptive steer.k_subset=$GRID train.utd=4 train.noise_critic_grad_steps=2" \
      "p_n${N}_ad_u16::$B steer.mode=adaptive steer.k_subset=$GRID train.utd=16 train.noise_critic_grad_steps=8" ;;
  mps*) P=${which#mps}; B=$(base 1024 3); specs=()
    for i in $(seq 1 $P); do specs+=("p_mps${P}_${i}::$B seed=$i steer.mode=fixed steer.k=0 train.utd=16 train.noise_critic_grad_steps=8"); done
    sbatch -J f6_p$which --time=03:00:00 scripts/slurm/exp6_pack.sbatch one_leg_low "${specs[@]}" ;;
  big*) N=${which#big}; B=$(base $N 3)  # larger batch, fixed k0 + adaptive at utd 16
    SEQ=1 RUN_SEC=2400 MPS=0 sbatch -J f6_p$which --time=03:00:00 scripts/slurm/exp6_pack.sbatch one_leg_low \
      "p_n${N}_k0_u16::$B steer.mode=fixed steer.k=0 train.utd=16 train.noise_critic_grad_steps=8" \
      "p_n${N}_ad_u16::$B steer.mode=adaptive steer.k_subset=$GRID train.utd=16 train.noise_critic_grad_steps=8" ;;
  share*) P=${which#share}; B=$(base 1024 3); specs=()  # P processes on one GPU WITHOUT MPS (time-slicing)
    for i in $(seq 1 $P); do specs+=("p_share${P}_${i}::$B seed=$i steer.mode=fixed steer.k=0 train.utd=16 train.noise_critic_grad_steps=8"); done
    MPS=0 sbatch -J f6_p$which --time=03:00:00 scripts/slurm/exp6_pack.sbatch one_leg_low "${specs[@]}" ;;
  mpsad*) P=${which#mpsad}; B=$(base 1024 3); specs=()
    for i in $(seq 1 $P); do specs+=("p_mpsad${P}_${i}::$B seed=$i steer.mode=adaptive steer.k_subset=$GRID train.utd=16 train.noise_critic_grad_steps=8"); done
    sbatch -J f6_p$which --time=03:00:00 scripts/slurm/exp6_pack.sbatch one_leg_low "${specs[@]}" ;;
  learn2) B="env.n_envs=1024 model.ddim_steps=20 use_wandb=false checkpoint_every_min=0 eval_interval=440 train.init_rollout_steps=88 train.total_timesteps=$(( 31 * 88 * 1024 * 8 ))"
    MPS=0 sbatch -J f6_plearn2 --time=04:00:00 scripts/slurm/exp6_pack.sbatch one_leg_low \
      "p_learn_k0_u32_b1024::$B steer.mode=fixed steer.k=0 train.utd=32 train.noise_critic_grad_steps=16 train.batch_size=1024" ;;
  learn3) B="env.n_envs=1024 model.ddim_steps=20 use_wandb=false checkpoint_every_min=0 eval_interval=440 train.init_rollout_steps=88 train.total_timesteps=$(( 41 * 88 * 1024 * 8 ))"
    MPS=0 sbatch -J f6_plearn3 --time=05:00:00 scripts/slurm/exp6_pack.sbatch one_leg_low \
      "p_learn_k0_u64::$B steer.mode=fixed steer.k=0 train.utd=64 train.noise_critic_grad_steps=32" ;;
  learn4) B="env.n_envs=1024 model.ddim_steps=20 use_wandb=false checkpoint_every_min=0 eval_interval=440 train.init_rollout_steps=88 train.total_timesteps=$(( 41 * 88 * 1024 * 8 ))"
    MPS=0 sbatch -J f6_plearn4 --time=05:00:00 scripts/slurm/exp6_pack.sbatch one_leg_low \
      "p_learn_k0_u64_tr03::$B steer.mode=fixed steer.k=0 train.utd=64 train.noise_critic_grad_steps=32 train.trust_region.enabled=true train.trust_region.bound=0.3" ;;
  learn) B="env.n_envs=1024 model.ddim_steps=20 use_wandb=false checkpoint_every_min=0 eval_interval=440 train.init_rollout_steps=88 train.total_timesteps=$(( 31 * 88 * 1024 * 8 ))"
    MPS=0 sbatch -J f6_plearn --time=03:00:00 scripts/slurm/exp6_pack.sbatch one_leg_low \
      "p_learn_k0_u8::$B steer.mode=fixed steer.k=0 train.utd=8 train.noise_critic_grad_steps=4" ;;
esac
