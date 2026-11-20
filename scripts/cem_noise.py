"""Constant-noise CEM baseline (Golden Ticket style, arXiv 2603.15757): one fixed noise w per task,
no RL. Every chunk is decoded with x_T = w (steering step k = 0), whatever the state.

    python scripts/cem_noise.py --task can --seed 1

Search: CEM over w in R^(H*A) with a diagonal Gaussian N(mu, sigma^2), initialised at the prior
N(0, I). Each iteration draws --pop candidates (clipped to +-randn_clip_value of the base model),
scores each by its success rate over --episodes_per_cand episodes (the train_adsrl.py eval protocol:
max_episode_steps per episode, success = any successful chunk), and refits mu, sigma to the top
--elite_frac candidates (sigma floored at --min_std). Candidate episodes run in parallel on --n_envs
envs. The final w = mu is then evaluated with the standard protocol (--final_episodes episodes on
--final_envs envs, AdsrlCallback.evaluate).

Outputs in --logdir (default logs/adsrl/<task>/cem/seed<seed>, where scripts/plots.py finds it):
  eval_summary.csv   one mode=det row for the final w at env_steps = env steps spent searching, in the
                     train_adsrl.py format (wall_s/gpu_s include the search; gpu_s = wall_s / ADSRL_RUNS_PER_GPU)
  cem_log.csv        one row per iteration: env steps, GPU-s, candidate success stats, sigma, ||mu||
  cem_w.npy          final w (H*A,), plus cem_state.npz (mu, sigma, all candidates and scores)
  eval_rollouts/     the final eval dump; run_info.json with steer.mode=cem (k=0)

Furniture tasks (one_leg_low, lamp_low, round_table_low; .venv-furniture): one IsaacGym sim with
cfg.env.n_envs lockstep envs is used for the search and the final eval (--n_envs/--final_envs default
to it), episodes are ceil(max_episode_steps / act_steps) chunks, and success is the env's
"all parts assembled" flag (info["chunk_success"]), not the reward. Size the search with
--budget_env_steps (iterations = budget / env steps per iteration). A finished run (result.json
present) exits immediately, so resubmitting is safe; an unfinished one restarts from scratch.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import train_adsrl  # noqa: E402  (registers resolvers, puts dsrl/ and dppo/ on sys.path)
import torch  # noqa: E402  (after train_adsrl, which imports isaacgym first when installed)
from omegaconf import OmegaConf  # noqa: E402

from adsrl.base_policy import SteeredBasePolicy  # noqa: E402
from adsrl.env import XTAugmentWrapper  # noqa: E402

LOG_FIELDS = ["iter", "env_steps", "wall_s", "gpu_s", "cand_success_mean", "cand_success_max", "elite_success_mean",
              "sigma_mean", "mu_norm_per_dim"]  # fmt: skip


def cem_update(cands: np.ndarray, scores: np.ndarray, n_elite: int, min_std: float, rng=None):
    """Refit (mu, sigma) to the n_elite best candidates. Ties are broken uniformly at random
    (success rates over a few episodes tie often)."""
    rng = rng or np.random.default_rng()
    order = np.lexsort((rng.random(len(scores)), -np.asarray(scores)))  # sort by -score, then random
    elite = cands[order[:n_elite]]
    return elite.mean(0), np.maximum(elite.std(0), min_std), order[:n_elite]


class ConstantNoiseAgent:
    """Agent stand-in for AdsrlCallback.evaluate: decodes every chunk with x_T = w (k = 0)."""

    def __init__(self, steered, w: np.ndarray):
        self.steered, self.w = steered, np.asarray(w, dtype=np.float32)
        self.last_k, self.last_probs = 0, None
        self.last_w = None

    def predict_diffused(self, obs, deterministic=True):
        W = np.broadcast_to(self.w, (len(obs), self.w.size))
        self.last_w = np.array(W)
        return self.steered(obs, 0, self.last_w).reshape(len(obs), -1).cpu().numpy(), None


@torch.no_grad()
def run_episodes(env, steered, W: np.ndarray, n_chunks: int, rew_offset: float, use_flag: bool = False) -> np.ndarray:
    """One episode per env with constant noise W[i] in env i (same protocol as AdsrlCallback.evaluate).
    Returns success per env."""
    obs = env.reset()
    success = np.zeros(len(W), dtype=bool)
    for _ in range(n_chunks):
        action = steered(obs, 0, W).reshape(len(W), -1).cpu().numpy()
        obs, reward, _, infos = env.step(action)
        success |= train_adsrl.chunk_success(reward, infos, rew_offset, use_flag)
    return success


def load_cfg(task, seed, logdir, device, overrides=()):
    cfg = OmegaConf.load(os.path.join(ROOT, "cfg", "adsrl", f"{task}.yaml"))
    cfg.merge_with_dotlist(list(overrides))
    for key in ["base_policy_path", "offline_data_path", "dppo_path", "normalization_path"]:
        if cfg.get(key) is not None:
            cfg[key] = train_adsrl.abs_path(cfg[key])
    cfg.seed, cfg.logdir, cfg.log_dir, cfg.name = seed, logdir, logdir, f"cem_s{seed}"
    cfg.steer.mode, cfg.steer.k = "cem", 0
    cfg.device = cfg.model.device = device
    cfg.use_wandb = False
    OmegaConf.resolve(cfg)
    return cfg


def make_vec(cfg, n, furniture_venv=None):
    if furniture_venv is not None:  # one IsaacGym sim per process: always the same env
        assert n == furniture_venv.num_envs, (n, furniture_venv.num_envs)
        return XTAugmentWrapper(furniture_venv, cfg.act_steps, cfg.action_dim)
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

    return XTAugmentWrapper(make_vec_env(make_env, n_envs=n, vec_env_cls=SubprocVecEnv), cfg.act_steps, cfg.action_dim)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=["can", "square", "transport", "one_leg_low", "lamp_low", "round_table_low"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--pop", type=int, default=16, help="candidates per iteration")
    p.add_argument("--episodes_per_cand", type=int, default=8)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--elite_frac", type=float, default=0.25)
    p.add_argument("--init_std", type=float, default=1.0)
    p.add_argument("--min_std", type=float, default=0.05)
    p.add_argument("--n_envs", type=int, default=None, help="parallel envs for the search (default 16; furniture: cfg.env.n_envs)")
    p.add_argument("--final_episodes", type=int, default=None, help="default 200; furniture: cfg.env.n_envs")
    p.add_argument("--final_envs", type=int, default=None, help="default 10; furniture: cfg.env.n_envs")
    p.add_argument("--budget_env_steps", type=float, default=None, help="if set, iters = budget // env steps per iteration")
    p.add_argument("--set", nargs="*", default=[], help="config overrides (dotlist), e.g. env.n_envs=1024")
    p.add_argument("--ddim_steps", type=int, default=None, help="override K (transport uses its config value)")
    p.add_argument("--logdir", default=None)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    t0 = time.time()
    runs_per_gpu = float(os.environ.get("ADSRL_RUNS_PER_GPU", 1))
    logdir = os.path.abspath(args.logdir or os.path.join(ROOT, "logs", "adsrl", args.task, "cem", f"seed{args.seed}"))
    os.makedirs(logdir, exist_ok=True)
    if os.path.exists(os.path.join(logdir, "result.json")):
        print(f"{logdir} already finished (result.json); nothing to do", flush=True)
        return
    assert not os.path.exists(os.path.join(logdir, "eval_summary.csv")), f"{logdir} already has results"
    cfg = load_cfg(args.task, args.seed, logdir, args.device, args.set)
    if args.ddim_steps is not None:
        cfg.model.ddim_steps = args.ddim_steps
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    furniture = train_adsrl.env_type(cfg) == "furniture"
    furniture_venv = None
    if furniture:  # create the sim before torch initializes CUDA; it serves the search and the final eval
        from adsrl.furniture_env import make_furniture_vec_env

        furniture_venv = make_furniture_vec_env(cfg)
        n = furniture_venv.num_envs
        args.n_envs, args.final_envs, args.final_episodes = n, n, args.final_episodes or n
    args.n_envs = args.n_envs or 16
    args.final_envs = args.final_envs or 10
    args.final_episodes = args.final_episodes or 200
    slots = args.pop * args.episodes_per_cand
    assert slots % args.n_envs == 0, "pop * episodes_per_cand must be a multiple of n_envs"
    assert args.final_episodes % args.final_envs == 0

    import hydra

    base_model = hydra.utils.instantiate(OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))).eval()
    steered = SteeredBasePolicy(base_model, cfg.obs_dim, cfg.act_steps, cfg.action_dim)
    D, clip = cfg.act_steps * cfg.action_dim, float(cfg.model.randn_clip_value)
    steps = cfg.env.max_episode_steps / cfg.act_steps
    n_chunks = int(np.ceil(steps)) if furniture else int(steps)
    steps_per_round = args.n_envs * n_chunks * cfg.act_steps
    if args.budget_env_steps is not None:
        args.iters = max(1, int(args.budget_env_steps // (steps_per_round * (slots // args.n_envs))))
        print(f"budget {args.budget_env_steps:.3g} env steps -> {args.iters} CEM iterations", flush=True)
    n_elite = max(1, int(round(args.elite_frac * args.pop)))
    info = train_adsrl.run_info(cfg)
    info["cem_args"] = vars(args)
    print(json.dumps(info, indent=1), flush=True)

    env = make_vec(cfg, args.n_envs, furniture_venv)
    env.seed(args.seed + 1)
    mu, sigma = np.zeros(D), np.full(D, args.init_std)
    env_steps, hist = 0, {"cands": [], "scores": []}
    log_path = os.path.join(logdir, "cem_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()
    for it in range(args.iters):
        cands = np.clip(mu + sigma * rng.standard_normal((args.pop, D)), -clip, clip).astype(np.float32)
        slot_cand = np.repeat(np.arange(args.pop), args.episodes_per_cand)
        succ = np.zeros(slots)
        for r in range(slots // args.n_envs):
            idx = slot_cand[r * args.n_envs : (r + 1) * args.n_envs]
            succ[r * args.n_envs : (r + 1) * args.n_envs] = run_episodes(
                env, steered, cands[idx], n_chunks, cfg.env.reward_offset, use_flag=furniture
            )
            env_steps += steps_per_round
        scores = np.bincount(slot_cand, weights=succ, minlength=args.pop) / args.episodes_per_cand
        mu, sigma, elite = cem_update(cands, scores, n_elite, args.min_std, rng)
        hist["cands"].append(cands)
        hist["scores"].append(scores)
        wall = time.time() - t0
        row = {"iter": it, "env_steps": env_steps, "wall_s": round(wall, 1), "gpu_s": round(wall / runs_per_gpu, 1),
               "cand_success_mean": scores.mean(), "cand_success_max": scores.max(),
               "elite_success_mean": scores[elite].mean(), "sigma_mean": sigma.mean(),
               "mu_norm_per_dim": float(np.sqrt((mu**2).mean()))}  # fmt: skip
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)
        print(f"[cem {it}] env_steps={env_steps} cand_mean={scores.mean():.3f} max={scores.max():.3f} "
              f"elite={scores[elite].mean():.3f} sigma={sigma.mean():.3f} ({wall:.0f}s)", flush=True)  # fmt: skip
    if not furniture:
        env.close()

    w = mu.astype(np.float32)
    np.save(os.path.join(logdir, "cem_w.npy"), w)
    np.savez_compressed(os.path.join(logdir, "cem_state.npz"), mu=mu, sigma=sigma, cands=np.stack(hist["cands"]),
                        scores=np.stack(hist["scores"]))  # fmt: skip
    cfg.train.total_timesteps = env_steps  # the search budget; scripts/plots.py treats the run as finished
    cfg.env.n_eval_envs, cfg.num_evals, cfg.deterministic_eval = args.final_envs, args.final_episodes, True
    with open(os.path.join(logdir, "run_info.json"), "w") as f:
        json.dump({"run": info, "cfg": OmegaConf.to_container(cfg, resolve=True)}, f, indent=1)

    eval_env = make_vec(cfg, args.final_envs, furniture_venv)
    eval_env.seed(args.seed + 10_000)
    if furniture:
        norm = {"obs_min": furniture_venv.obs_min, "obs_max": furniture_venv.obs_max}
    else:
        norm = np.load(cfg.normalization_path)
    wall = time.time() - t0
    cb = train_adsrl.AdsrlCallback(cfg, eval_env, steered, norm["obs_min"], norm["obs_max"], None,
                                   state={"env_steps": env_steps, "wall_s": wall, "gpu_s": wall / runs_per_gpu})  # fmt: skip
    cb.model = ConstantNoiseAgent(steered, w)
    res = cb.evaluate(deterministic=True, dump=True)
    eval_env.close()
    out = {"task": args.task, "seed": args.seed, "env_steps": env_steps, "final_success": res["success"],
           "se": float(np.sqrt(res["success"] * (1 - res["success"]) / args.final_episodes)),
           "wall_s": round(time.time() - t0, 1), "gpu_s": round((time.time() - t0) / runs_per_gpu, 1)}  # fmt: skip
    with open(os.path.join(logdir, "result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("CEM " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
