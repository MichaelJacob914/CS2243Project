"""Evaluate the unsteered base policy (deterministic DDIM from x_T ~ N(0, I)) with the exact eval
protocol of train_adsrl.py (AdsrlCallback.evaluate: n_eval_envs parallel envs, num_evals episodes,
success = reward > -reward_offset at any chunk within max_episode_steps).

    python scripts/base_eval.py --task transport --ddim_steps 20 --n_episodes 200 --n_envs 10

Outputs in <out_dir>/<task>_K<K>/: eval_summary.csv, eval_rollouts/*.npz (same keys as training
dumps), result.json (success, binomial SE, timing). Needs a GPU node (robosuite uses EGL).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train_adsrl  # noqa: E402  (registers resolvers, puts dsrl/ and dppo/ on sys.path)
from omegaconf import OmegaConf  # noqa: E402

from adsrl.base_policy import SteeredBasePolicy  # noqa: E402
from adsrl.env import XTAugmentWrapper  # noqa: E402


class BaseAgent:
    """Agent stand-in for AdsrlCallback.evaluate: always plays a_base(s, x_T)."""

    def __init__(self, steered):
        self.steered = steered
        self.last_k, self.last_w, self.last_probs = 0, None, None

    def predict_diffused(self, obs, deterministic=True):
        return self.steered.base_action(obs).reshape(len(obs), -1).cpu().numpy(), None


def load_cfg(task, ddim_steps, n_envs, n_episodes, logdir, seed):
    cfg = OmegaConf.load(os.path.join(ROOT, "cfg", "adsrl", f"{task}.yaml"))
    for key in ["base_policy_path", "offline_data_path", "dppo_path", "normalization_path"]:
        if cfg.get(key) is not None:
            cfg[key] = train_adsrl.abs_path(cfg[key])
    if ddim_steps is not None:
        cfg.model.ddim_steps = int(ddim_steps)
    cfg.env.n_eval_envs = n_envs
    cfg.num_evals = n_episodes
    cfg.seed = seed
    cfg.logdir = logdir
    cfg.log_dir = logdir
    cfg.name = os.path.basename(logdir)
    cfg.steer.mode = "base"
    cfg.steer.k = 0
    OmegaConf.resolve(cfg)
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["can", "square", "transport"])
    p.add_argument("--ddim_steps", type=int, default=None, help="override K (default: config value)")
    p.add_argument("--n_episodes", type=int, default=200)
    p.add_argument("--n_envs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_dir", default=os.path.join(ROOT, "results", "base_eval"))
    args = p.parse_args()
    assert args.n_episodes % args.n_envs == 0, "n_episodes must be a multiple of n_envs"

    t0 = time.time()
    torch.manual_seed(args.seed)
    tmp_cfg = OmegaConf.load(os.path.join(ROOT, "cfg", "adsrl", f"{args.task}.yaml"))
    K = args.ddim_steps or tmp_cfg.model.ddim_steps
    logdir = os.path.join(args.out_dir, f"{args.task}_K{K}")
    os.makedirs(logdir, exist_ok=True)
    cfg = load_cfg(args.task, args.ddim_steps, args.n_envs, args.n_episodes, logdir, args.seed)
    cfg.device = args.device
    cfg.model.device = args.device

    import hydra

    base_model = hydra.utils.instantiate(OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))).eval()
    steered = SteeredBasePolicy(base_model, cfg.obs_dim, cfg.act_steps, cfg.action_dim)
    assert steered.K == K, (steered.K, K)

    from env_utils import ActionChunkWrapper, ObservationWrapperRobomimic, make_robomimic_env
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    def make_env():
        env = make_robomimic_env(
            env=cfg.env_name, normalization_path=cfg.normalization_path,
            low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys, dppo_path=cfg.dppo_path,
        )  # fmt: skip
        env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
        return ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)

    eval_env = XTAugmentWrapper(make_vec_env(make_env, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv), cfg.act_steps, cfg.action_dim)
    eval_env.seed(args.seed + 1000)
    norm = np.load(cfg.normalization_path)
    cb = train_adsrl.AdsrlCallback(cfg, eval_env, steered, norm["obs_min"], norm["obs_max"], None)
    cb.model = BaseAgent(steered)
    t1 = time.time()
    res = cb.evaluate(deterministic=True, dump=True)
    eval_s = time.time() - t1
    eval_env.close()

    n = args.n_episodes
    out = {
        "task": args.task, "K": K, "n_episodes": n, "n_envs": args.n_envs, "seed": args.seed,
        "success": res["success"], "se": float(np.sqrt(res["success"] * (1 - res["success"]) / n)),
        "mean_ep_len_success": res["mean_ep_len_success"], "mean_a_minus_base": res["mean_a_minus_base"],
        "eval_s": eval_s, "total_s": time.time() - t0, "run_info": train_adsrl.run_info(cfg),
    }  # fmt: skip
    with open(os.path.join(logdir, "result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("BASE_EVAL " + json.dumps({k: v for k, v in out.items() if k != "run_info"}), flush=True)


if __name__ == "__main__":
    main()
