"""Steering-authority diagnostic: how much does w move the final action when injected at step k?

For N states, fix one x_T per state, draw M w ~ N(0, I), and for every k compute the per-dim std of
the final action over the M draws, averaged over states. Also reports the mean distance of the
steered action to the unsteered base action a_base(s, x_T).

Outputs (in --out_dir): authority_<task>.csv (one row per k, h, a) and authority_<task>.png.

State sources:
  dataset : random states from dsrl/dppo/log/robomimic/<task>/train.npz (normalized like the env)
  rollout : states visited by the base policy in the robomimic env (needs robosuite)
  random  : uniform in [-1, 1] (smoke test only)

Example:
  python scripts/authority.py --task can --n_states 500 --device cuda:0
  python scripts/authority.py --task transport --ddim_steps 10 --state_source rollout
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from adsrl.model_utils import DSRL_DIR, abs_dsrl_path, build_model, load_task_cfg  # noqa: E402
from adsrl.sampler import num_ddim_steps, steered_ddim_sample  # noqa: E402


def dataset_states(cfg, n, rng):
    data = np.load(os.path.join(DSRL_DIR, "dppo", "log", "robomimic", cfg.env_name, "train.npz"))
    states = data["states"].astype(np.float32)
    if np.abs(states).max() > 1.5:  # stored unnormalized: normalize like the robomimic_lowdim wrapper
        norm = np.load(abs_dsrl_path(cfg.normalization_path))
        lo, hi = norm["obs_min"], norm["obs_max"]
        states = 2 * ((states - lo) / (hi - lo + 1e-6) - 0.5)
    idx = rng.choice(len(states), size=min(n, len(states)), replace=False)
    return states[idx].astype(np.float32)


def rollout_states(cfg, model, n, rng, device, n_episodes=None):
    """Run the unsteered base policy (x_T ~ N(0, I)) and collect states at chunk boundaries."""
    sys.path.insert(0, DSRL_DIR)
    from env_utils import ActionChunkWrapper, ObservationWrapperRobomimic, make_robomimic_env

    env = make_robomimic_env(
        env=cfg.env_name,
        normalization_path=abs_dsrl_path(cfg.normalization_path),
        low_dim_keys=list(cfg.env.wrappers.robomimic_lowdim.low_dim_keys),
        dppo_path=abs_dsrl_path(cfg.dppo_path),
    )
    env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
    env = ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)
    H, A = cfg.act_steps, cfg.action_dim
    states = []
    ep = 0
    n_episodes = n_episodes or max(1, n // 20)
    while ep < n_episodes:
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        done = False
        while not done:
            states.append(np.asarray(obs, np.float32))
            s = torch.as_tensor(obs, dtype=torch.float32, device=device)[None]
            x_T = torch.randn(1, H, A, device=device)
            with torch.no_grad():
                a = steered_ddim_sample(model, {"state": s}, x_T, 0, x_T)
            obs, _, done, _, _ = env.step(a[0].cpu().numpy())
        ep += 1
        print(f"rollout episode {ep}/{n_episodes}, {len(states)} states", flush=True)
    states = np.stack(states)
    idx = rng.choice(len(states), size=min(n, len(states)), replace=False)
    return states[idx]


@torch.no_grad()
def authority(model, states, n_w, batch, seed, w_scale=1.0, clip_w=False):
    """Returns std[K, H, A] (mean over states of per-state std over w), dist[K] (mean ||a - a_base||)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    device = model.ddim_alphas.device
    K = num_ddim_steps(model)
    H, A = model.horizon_steps, model.action_dim
    N = len(states)
    s_all = torch.as_tensor(states, device=device)
    xT_all = torch.randn(N, H, A, generator=g).to(device)
    std_sum = torch.zeros(K, H, A, device=device)
    dist_sum = torch.zeros(K, device=device)
    per_chunk = max(1, batch // n_w)
    for i0 in range(0, N, per_chunk):
        s = s_all[i0 : i0 + per_chunk]
        n = len(s)
        x_T = xT_all[i0 : i0 + per_chunk]
        base = steered_ddim_sample(model, {"state": s}, x_T, torch.zeros(n, dtype=torch.long), x_T)
        s_rep = s.repeat_interleave(n_w, 0)
        xT_rep = x_T.repeat_interleave(n_w, 0)
        w = torch.randn(n * n_w, H, A, generator=g).to(device)
        if clip_w:  # the trained actor's w lives in [-1, 1]^D (tanh squash), so mimic that bound
            w = w.clamp(-1.0, 1.0)
        w = w * w_scale
        for k in range(K):
            kk = torch.full((n * n_w,), k, dtype=torch.long, device=device)
            a = steered_ddim_sample(model, {"state": s_rep}, xT_rep, kk, w).view(n, n_w, H, A)
            std_sum[k] += a.std(dim=1).sum(0)
            dist_sum[k] += (a - base[:, None]).flatten(2).norm(dim=-1).mean(1).sum()
    return (std_sum / N).cpu().numpy(), (dist_sum / N).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="can", choices=["lift", "can", "square", "transport"])
    p.add_argument("--n_states", type=int, default=500)
    p.add_argument("--n_w", type=int, default=64)
    p.add_argument("--ddim_steps", type=int, default=None, help="override K (e.g. for transport)")
    p.add_argument("--state_source", default="dataset", choices=["dataset", "rollout", "random"])
    p.add_argument("--rollout_episodes", type=int, default=None)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=8192, help="max samples per sampler call")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--w_scale", type=float, default=1.0, help="multiply w by this (bound sweep)")
    p.add_argument("--clip_w", action="store_true", help="clip w to [-1, 1] before scaling, like the actor's bound")
    p.add_argument("--out_dir", default=os.path.join(ROOT, "results", "authority"))
    args = p.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    cfg = load_task_cfg(args.task)
    model = build_model(cfg, device=args.device, ddim_steps=args.ddim_steps)
    K = num_ddim_steps(model)
    if args.state_source == "dataset":
        states = dataset_states(cfg, args.n_states, rng)
    elif args.state_source == "rollout":
        states = rollout_states(cfg, model, args.n_states, rng, args.device, args.rollout_episodes)
    else:
        states = rng.uniform(-1, 1, (args.n_states, model.obs_dim)).astype(np.float32)
    print(f"task={args.task} K={K} states={states.shape} source={args.state_source}", flush=True)

    std, dist = authority(model, states, args.n_w, args.batch, args.seed, args.w_scale, args.clip_w)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.task}_K{K}_{args.state_source}"
    if args.clip_w or args.w_scale != 1.0:
        tag += f"_clip{int(args.clip_w)}_c{args.w_scale:g}"
    csv_path = os.path.join(args.out_dir, f"authority_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["k", "h", "a", "std", "mean_std_k", "mean_dist_to_base_k"])
        for k in range(K):
            for h in range(std.shape[1]):
                for a in range(std.shape[2]):
                    wr.writerow([k, h, a, std[k, h, a], std[k].mean(), dist[k]])
    try:
        commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    with open(os.path.join(args.out_dir, f"authority_{tag}.json"), "w") as f:
        json.dump({**vars(args), "K": K, "commit": commit, "n_states_used": len(states),
                   "mean_std_per_k": std.mean((1, 2)).tolist(), "dist_to_base_per_k": dist.tolist(),
                   "seconds": time.time() - t0}, f, indent=2)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ks = np.arange(K)
    per_dim = std.mean(1)  # [K, A], averaged over chunk positions
    for a in range(per_dim.shape[1]):
        axes[0].plot(ks, per_dim[:, a], color="0.75", lw=1)
    axes[0].plot(ks, std.mean((1, 2)), color="C0", lw=2.5, marker="o", label="mean over dims")
    axes[0].set(xlabel="steering step k", ylabel="std of final action over w",
                title="per-dim authority (grey: action dims)")
    axes[0].legend(frameon=False)
    axes[1].plot(ks, dist, color="C1", lw=2.5, marker="o")
    axes[1].set(xlabel="steering step k", ylabel="E ||a(k, w) - a_base||", title="distance to base action")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{args.task}: K={K}, N={len(states)} states, {args.n_w} w draws")
    fig.tight_layout()
    png_path = os.path.join(args.out_dir, f"authority_{tag}.png")
    fig.savefig(png_path, dpi=150)
    print(f"wrote {csv_path} and {png_path} in {time.time() - t0:.1f}s")
    for k in range(K):
        print(f"k={k}: mean std={std[k].mean():.4f}  dist_to_base={dist[k]:.4f}")


if __name__ == "__main__":
    main()
