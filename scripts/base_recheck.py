"""Base-success re-check on the standard eval path (AdsrlCallback.evaluate), one mode per process.

    base     : a_base(s, x_T), x_T ~ N(0, I) fresh each chunk (scripts/base_eval.py BaseAgent)
    zero_k1  : w = 0 at steering step k = 1 (scripts/base_eval_wk.py FixedWAgent, w_mode=zero)
    tanh_k0  : x_T = tanh(u), u ~ N(0, I) fresh each chunk = what an untrained upstream DSRL actor
               (log_std_init=0, mean ~0, tanh-squashed to the env action box [-1, 1]) feeds as noise

    python scripts/base_recheck.py --task can --mode base --seed 0 --n_episodes 500 --out_csv results/base_recheck/x.csv
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
from base_eval import BaseAgent, load_cfg, train_adsrl  # noqa: E402
from base_eval_wk import FixedWAgent  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from adsrl.base_policy import SteeredBasePolicy  # noqa: E402
from adsrl.env import XTAugmentWrapper  # noqa: E402


class TanhK0Agent:
    def __init__(self, steered, seed):
        self.steered, self.rng = steered, np.random.default_rng(seed)
        self.last_k, self.last_w, self.last_probs = 0, None, None

    def predict_diffused(self, obs, deterministic=True):
        B, D = len(obs), self.steered.horizon * self.steered.act_dim
        w = np.tanh(self.rng.standard_normal((B, D))).astype(np.float32)
        self.last_w = w
        return self.steered(obs, 0, w).reshape(B, -1).cpu().numpy(), None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="can")
    p.add_argument("--mode", required=True, choices=["base", "zero_k1", "tanh_k0"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_episodes", type=int, default=500)
    p.add_argument("--n_envs", type=int, default=25)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_csv", required=True)
    a = p.parse_args()
    assert a.n_episodes % a.n_envs == 0
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    logdir = os.path.join(os.path.dirname(os.path.abspath(a.out_csv)), "logdirs", f"{a.task}_{a.mode}_s{a.seed}")
    os.makedirs(logdir, exist_ok=True)
    cfg = load_cfg(a.task, None, a.n_envs, a.n_episodes, logdir, a.seed)
    cfg.device = cfg.model.device = a.device

    import hydra

    base_model = hydra.utils.instantiate(OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))).eval()
    steered = SteeredBasePolicy(base_model, cfg.obs_dim, cfg.act_steps, cfg.action_dim)

    from env_utils import ActionChunkWrapper, ObservationWrapperRobomimic, make_robomimic_env
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    def make_env():
        env = make_robomimic_env(env=cfg.env_name, normalization_path=cfg.normalization_path,
                                 low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys, dppo_path=cfg.dppo_path)
        env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
        return ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)

    eval_env = XTAugmentWrapper(make_vec_env(make_env, n_envs=a.n_envs, vec_env_cls=SubprocVecEnv), cfg.act_steps, cfg.action_dim)
    eval_env.seed(a.seed + 1000)
    norm = np.load(cfg.normalization_path)
    cb = train_adsrl.AdsrlCallback(cfg, eval_env, steered, norm["obs_min"], norm["obs_max"], None)
    cb.model = {"base": lambda: BaseAgent(steered), "zero_k1": lambda: FixedWAgent(steered, 1, "zero", a.seed),
                "tanh_k0": lambda: TanhK0Agent(steered, a.seed)}[a.mode]()
    t1 = time.time()
    res = cb.evaluate(deterministic=True, dump=False)
    eval_env.close()
    s, n = float(res["success"]), a.n_episodes
    row = {"task": a.task, "mode": a.mode, "seed": a.seed, "success": s, "se": float(np.sqrt(s * (1 - s) / n)), "n": n,
           "n_envs": a.n_envs, "K": steered.K, "eval_s": round(time.time() - t1, 1)}
    new = not os.path.exists(a.out_csv)
    with open(a.out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print("RECHECK", row, flush=True)


if __name__ == "__main__":
    main()
