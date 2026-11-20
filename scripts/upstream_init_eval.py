"""Cross-check: upstream DSRL's own eval (dsrl/utils.py LoggingCallback.evaluate) of an untrained dsrl_na agent,
built exactly as dsrl/train_dsrl.py builds it (cfg dsrl_can.yaml), before any training.

    cd dsrl && python ../scripts/upstream_init_eval.py --seed 1 --num_evals 500 --deterministic 0 --out_csv X.csv

Appends (seed, deterministic, success, se, n) to --out_csv. Replay buffer is shrunk (eval only, unused).
"""

import argparse
import csv
import math
import os
import sys
import types

import numpy as np
import torch

DSRL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dsrl")
sys.path.insert(0, DSRL_DIR)
sys.path.append("./dppo")

import hydra  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from stable_baselines3 import DSRL  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: E402

import utils as dsrl_utils  # noqa: E402
from env_utils import ActionChunkWrapper, ObservationWrapperRobomimic, make_robomimic_env  # noqa: E402

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
OmegaConf.register_new_resolver("round_down", math.floor, replace=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num_evals", type=int, default=500)
    p.add_argument("--deterministic", type=int, default=0)
    p.add_argument("--out_csv", required=True)
    a = p.parse_args()

    with hydra.initialize_config_dir(config_dir=os.path.join(DSRL_DIR, "cfg/robomimic"), version_base=None):
        cfg = hydra.compose("dsrl_can.yaml", overrides=[f"seed={a.seed}", "use_wandb=False", f"num_evals={a.num_evals}"])
    cfg.name, cfg.logdir = "upstream_init_eval", "/dev/null"
    OmegaConf.resolve(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    MAX_STEPS = int(cfg.env.max_episode_steps / cfg.act_steps)
    num_env = cfg.env.n_envs

    def make_env():
        env = make_robomimic_env(env=cfg.env_name, normalization_path=cfg.normalization_path,
                                 low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys, dppo_path=cfg.dppo_path)
        env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
        return ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)

    base_policy = dsrl_utils.load_base_policy(cfg)
    env = make_vec_env(make_env, n_envs=num_env, vec_env_cls=SubprocVecEnv)
    env.seed(cfg.seed + 1)
    net_arch = [cfg.train.layer_size] * cfg.train.num_layers
    policy_kwargs = dict(net_arch=dict(pi=net_arch, qf=net_arch), activation_fn=torch.nn.Tanh, log_std_init=0.0,
                         post_linear_modules=[torch.nn.LayerNorm] if cfg.train.use_layer_norm else None,
                         n_critics=cfg.train.n_critics)
    model = DSRL("MlpPolicy", env, learning_rate=cfg.train.actor_lr, buffer_size=1000, learning_starts=1,
                 batch_size=cfg.train.batch_size, tau=cfg.train.tau, gamma=cfg.train.discount,
                 train_freq=cfg.train.train_freq, gradient_steps=cfg.train.utd, ent_coef="auto",
                 target_update_interval=1, target_entropy=cfg.train.target_ent, policy_kwargs=policy_kwargs,
                 diffusion_policy=base_policy, diffusion_act_dim=(cfg.act_steps, cfg.action_dim),
                 noise_critic_grad_steps=cfg.train.noise_critic_grad_steps,
                 critic_backup_combine_type=cfg.train.critic_backup_combine_type)
    n_eval_env = cfg.env.n_eval_envs
    eval_env = make_vec_env(make_env, n_envs=n_eval_env, vec_env_cls=SubprocVecEnv)
    eval_env.seed(cfg.seed + num_env + 1)
    cb = dsrl_utils.LoggingCallback(action_chunk=cfg.act_steps, eval_episodes=int(cfg.num_evals / n_eval_env),
                                    log_freq=MAX_STEPS, use_wandb=True, eval_env=eval_env, eval_freq=cfg.eval_interval,
                                    num_train_env=num_env, num_eval_env=n_eval_env, rew_offset=cfg.env.reward_offset,
                                    algorithm=cfg.algorithm, max_steps=MAX_STEPS, deterministic_eval=False)
    got = {}
    dsrl_utils.wandb = types.SimpleNamespace(log=lambda d, step=None: got.update(d))  # capture instead of logging
    cb.evaluate(model, deterministic=bool(a.deterministic))
    s = float(got["eval/success_rate_deterministic" if a.deterministic else "eval/success_rate"])
    n = int(cfg.num_evals / n_eval_env) * n_eval_env
    row = {"seed": a.seed, "deterministic": a.deterministic, "success": s, "se": math.sqrt(s * (1 - s) / n), "n": n}
    new = not os.path.exists(a.out_csv)
    with open(a.out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print("UPSTREAM_EVAL", row, flush=True)
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
