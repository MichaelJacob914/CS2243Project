"""Best-of-N noise at eval for a fixed-k or adaptive train_adsrl.py checkpoint (eval only).

    python scripts/best_of_n_eval.py --ckpt logs/adsrl/can/<run>/checkpoint/final.pt --n 1 4 16

For every chunk, N candidate (k, w) are drawn from the agent's stochastic policy, decoded by the
steered sampler, scored with Q(s, a) (min over critics, as in the backup), and the argmax is executed
(adsrl/best_of_n.py). The episode protocol is AdsrlCallback.evaluate's (--n_episodes episodes on
--n_envs envs). Each N gets the same eval-env seed, so all N start from the same initial states.
N = 1 is the plain stochastic policy, i.e. the "stoch" rows of the run's own eval_summary.csv.

Outputs: <out_dir>/<task>/<run>/N<N>/eval_summary.csv (+ eval_rollouts/*.npz with --dump), all in
the training CSV format with mode=stoch, and one row per N appended to <out_dir>/best_of_n.csv.
Needs a GPU node (robosuite renders/steps with EGL).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import train_adsrl  # noqa: E402  (registers resolvers, puts dsrl/ and dppo/ on sys.path)
from render_rollouts import build_agent, build_steered, method_name, run_cfg  # noqa: E402

from adsrl.best_of_n import agent_fns, best_of_n  # noqa: E402
from adsrl.env import XTAugmentWrapper  # noqa: E402

SUMMARY_FIELDS = ["task", "method", "run", "seed", "ckpt", "ckpt_env_steps", "N", "n_episodes", "success", "se",
                  "mean_a_minus_base", "mean_ep_len_success", "eval_s"]  # fmt: skip


class BestOfNAgent:
    """Agent stand-in for AdsrlCallback.evaluate that plays the best of N sampled candidates."""

    def __init__(self, model, steered, n: int):
        self.model, self.n = model, int(n)
        self.sample_fn, self.q_fn = agent_fns(model, steered)
        self.num_k = steered.K
        self.last_k = self.last_w = self.last_probs = None

    @torch.no_grad()
    def predict_diffused(self, obs, deterministic=False):
        obs_t = torch.as_tensor(np.asarray(obs), device=self.model.device, dtype=torch.float32)
        out = best_of_n(obs_t, self.n, self.sample_fn, self.q_fn)
        self.last_k = out["k"].cpu().numpy()
        self.last_w = out["w"].cpu().numpy()
        if hasattr(self.model, "hybrid_actor"):  # pi(k|s) over all K, for the dumps
            probs = self.model.hybrid_actor.all_heads(obs_t)["probs"].cpu().numpy()
            self.last_probs = np.zeros((len(obs), self.num_k), dtype=np.float32)
            self.last_probs[:, self.model.hybrid_actor.allowed_k.cpu().numpy()] = probs
        return out["action"].cpu().numpy(), None


def make_eval_env(cfg, n_envs):
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

    return XTAugmentWrapper(make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv), cfg.act_steps, cfg.action_dim)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="checkpoint/final.pt (or latest.pt) of a fixed-k/adaptive run")
    p.add_argument("--n", type=int, nargs="+", required=True, help="one or more N values")
    p.add_argument("--n_episodes", type=int, default=200)
    p.add_argument("--n_envs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0, help="eval-env seed offset (same for every N)")
    p.add_argument("--dump", action="store_true", help="also write eval_rollouts/*.npz")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_dir", default=os.path.join(ROOT, "results", "best_of_n"))
    args = p.parse_args()
    assert args.n_episodes % args.n_envs == 0, "n_episodes must be a multiple of n_envs"

    cfg = run_cfg(args.ckpt, args.device)
    assert cfg.steer.mode in ("fixed", "adaptive"), f"best-of-N needs a noise-space agent, not {cfg.steer.mode}"
    steered = build_steered(cfg)
    model = build_agent(cfg, steered, args.ckpt)
    model.policy.set_training_mode(False)
    if hasattr(model, "hybrid_actor"):
        model.hybrid_actor.train(False)
    ckpt_steps = int(torch.load(args.ckpt, map_location="cpu", weights_only=False)["extra"]["callback"]["env_steps"])
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt)))
    task_dir = run_dir.split(os.sep + cfg.env_name + os.sep, 1)
    run = task_dir[1] if len(task_dir) == 2 else os.path.basename(run_dir)
    ckpt_abs = os.path.abspath(args.ckpt)
    ckpt_label = os.path.relpath(ckpt_abs, ROOT) if ckpt_abs.startswith(ROOT + os.sep) else ckpt_abs
    method = method_name(cfg, "adaptive" if cfg.steer.mode == "adaptive" else "fixed_k")

    cfg.env.n_eval_envs = args.n_envs
    cfg.num_evals = args.n_episodes
    cfg.eval_dump_every = 1
    env = make_eval_env(cfg, args.n_envs)
    norm = np.load(cfg.normalization_path)
    summary_path = os.path.join(args.out_dir, "best_of_n.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    for n in args.n:
        cfg.logdir = os.path.join(args.out_dir, cfg.env_name, run, f"N{n}")
        env.seed(cfg.seed + 10_000 + args.seed)
        cb = train_adsrl.AdsrlCallback(cfg, env, steered, norm["obs_min"], norm["obs_max"], None,
                                       state={"env_steps": ckpt_steps})  # fmt: skip
        cb.model = BestOfNAgent(model, steered, n)
        t0 = time.time()
        res = cb.evaluate(deterministic=False, dump=args.dump)
        eval_s = time.time() - t0
        se = float(np.sqrt(res["success"] * (1 - res["success"]) / args.n_episodes))
        row = {"task": cfg.env_name, "method": method, "run": run, "seed": int(cfg.seed),
               "ckpt": ckpt_label, "ckpt_env_steps": ckpt_steps, "N": n,
               "n_episodes": args.n_episodes, "success": res["success"], "se": se,
               "mean_a_minus_base": res["mean_a_minus_base"], "mean_ep_len_success": res["mean_ep_len_success"],
               "eval_s": round(eval_s, 1)}  # fmt: skip
        new = not os.path.exists(summary_path)
        with open(summary_path, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            if new:
                wr.writeheader()
            wr.writerow(row)
        print("BEST_OF_N " + json.dumps(row), flush=True)
    env.close()


if __name__ == "__main__":
    main()
