"""No-RL reference per steering step k: evaluate a(s) = steered_ddim(s, x_T, k, w) with x_T fresh each
chunk and w fixed by a rule instead of an actor, using the AdsrlCallback.evaluate protocol of base_eval.py.

    w_mode=zero  : w = 0 at step k (what an untrained actor mean ~0 does)
    w_mode=gauss : w ~ N(0, I) fresh each chunk (DDPM-like re-noising; should match base success)

    python scripts/base_eval_wk.py --task can --ks 1 2 --w_modes zero gauss --out_csv results/base_eval_wk/parts/can_k1.csv

Appends rows (task, k, w_mode, success, se, n, K, eval_s) to --out_csv. Needs a GPU node.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import base_eval  # noqa: E402  (sets sys.path, imports train_adsrl)
from base_eval import ROOT, load_cfg, train_adsrl  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from adsrl.base_policy import SteeredBasePolicy  # noqa: E402
from adsrl.env import XTAugmentWrapper  # noqa: E402


class FixedWAgent:
    def __init__(self, steered, k, w_mode, seed):
        self.steered, self.k, self.w_mode = steered, int(k), w_mode
        self.rng = np.random.default_rng(seed)
        self.last_k, self.last_w, self.last_probs = self.k, None, None

    def predict_diffused(self, obs, deterministic=True):
        B, D = len(obs), self.steered.horizon * self.steered.act_dim
        w = np.zeros((B, D), np.float32) if self.w_mode == "zero" else self.rng.standard_normal((B, D)).astype(np.float32)
        self.last_w = w
        return self.steered(obs, self.k, w).reshape(B, -1).cpu().numpy(), None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["can", "square", "transport"])
    p.add_argument("--ddim_steps", type=int, default=None)
    p.add_argument("--ks", type=int, nargs="+", required=True)
    p.add_argument("--w_modes", nargs="+", default=["zero", "gauss"], choices=["zero", "gauss"])
    p.add_argument("--n_episodes", type=int, default=200)
    p.add_argument("--n_envs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_csv", required=True)
    args = p.parse_args()
    assert args.n_episodes % args.n_envs == 0

    torch.manual_seed(args.seed)
    logdir = os.path.join(os.path.dirname(os.path.abspath(args.out_csv)), "logdirs", f"{args.task}_k{'-'.join(map(str, args.ks))}")
    os.makedirs(logdir, exist_ok=True)
    cfg = load_cfg(args.task, args.ddim_steps, args.n_envs, args.n_episodes, logdir, args.seed)
    cfg.device = cfg.model.device = args.device

    import hydra

    base_model = hydra.utils.instantiate(OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))).eval()
    steered = SteeredBasePolicy(base_model, cfg.obs_dim, cfg.act_steps, cfg.action_dim)
    K = steered.K
    assert all(0 <= k < K for k in args.ks), (args.ks, K)

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
    norm = np.load(cfg.normalization_path)
    cb = train_adsrl.AdsrlCallback(cfg, eval_env, steered, norm["obs_min"], norm["obs_max"], None)
    new = not os.path.exists(args.out_csv)
    for k in args.ks:
        for wm in args.w_modes:
            eval_env.seed(args.seed + 1000)  # same episode seeds as base_eval.py
            cb.model = FixedWAgent(steered, k, wm, seed=args.seed * 1000 + k)
            t1 = time.time()
            res = cb.evaluate(deterministic=True, dump=False)
            s, n = float(res["success"]), args.n_episodes
            row = {"task": args.task, "k": k, "w_mode": wm, "success": s, "se": float(np.sqrt(s * (1 - s) / n)),
                   "n": n, "K": K, "eval_s": round(time.time() - t1, 1)}  # fmt: skip
            with open(args.out_csv, "a", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    wr.writeheader()
                    new = False
                wr.writerow(row)
            print("WK_EVAL", row, flush=True)
    eval_env.close()


if __name__ == "__main__":
    main()
