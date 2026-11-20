"""Redraws for the animated toy figure: real toy model, real sampler (windows.ddpm_regenerate).

For each state (start, past obstacle) and for the policy and its Gaussian control: pick 4 clean chunks,
noise each once to every DDIM step k (one noise draw reused over k, as in measure.stage2a_policy), and redraw
M times. Saves positions (state + cumulative velocity) for plotting. Run with the project .venv:

    python scripts/exp14/video/toy_anim_data.py --out results/exp14/anim/toy_anim_data.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import toy  # noqa: E402
import windows as W  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def pick(clean_np, n_each=2):
    """Two chunks that go up and two that go down (by total y displacement), the most typical of each side."""
    dy = clean_np[:, :, 1].sum(1)
    up = np.where(dy > 0)[0]
    dn = np.where(dy <= 0)[0]
    med_up, med_dn = np.median(dy[up]), np.median(dy[dn])
    up = up[np.argsort(np.abs(dy[up] - med_up))][:n_each]
    dn = dn[np.argsort(np.abs(dy[dn] - med_dn))][:n_each]
    return np.concatenate([up, dn])


def pick_spread(clean_np, n=4):
    """Four chunks spread over the main direction of variation (quantiles 0.1, 0.37, 0.63, 0.9)."""
    X = clean_np.reshape(len(clean_np), -1)
    Xc = X - X.mean(0)
    v = np.linalg.svd(Xc, full_matrices=False)[2][0]
    s = Xc @ v
    order = np.argsort(s)
    return order[(np.array([0.1, 0.37, 0.63, 0.9]) * (len(s) - 1)).astype(int)]


@torch.no_grad()
def redraws(model, obs, clean, levels, M, gen):
    """clean (P, H, A) -> (L, P, M, H, A) normalized actions, the stage2a_policy procedure."""
    P = clean.shape[0]
    eps = torch.randn(clean.shape, generator=gen)
    cond = {"state": obs[None].expand(P * M, -1)}
    out = []
    for t in levels:
        x_t = W.q_sample(model, clean, int(t), eps)
        a = W.ddpm_regenerate(model, cond, x_t.repeat_interleave(M, 0), int(t), gen)
        out.append(a.reshape(P, M, *a.shape[1:]).numpy())
    return np.stack(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--M", type=int, default=32)
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    obs_d, act = toy.make_dataset(6000, rng)  # same call as toy.py, so amin / amax match training
    amin, amax = act.reshape(-1, toy.A).min(0), act.reshape(-1, toy.A).max(0)
    model = toy.build_model("cpu", amin, amax)
    model.network.load_state_dict(torch.load(os.path.join(ROOT, "results/exp14/stage0/toy_model.pt"), map_location="cpu"))
    model.eval()
    levels = W.ddim_levels(model)
    gen = torch.Generator().manual_seed(args.seed)
    nrng = np.random.default_rng(args.seed)

    def unnorm(a):
        return (a + 1) / 2 * (amax - amin) + amin

    out = {"levels": levels, "logsnr": W.logsnr(model, levels), "amin": amin, "amax": amax}
    for name, pos in {"start": toy.START, "past": np.array([0.31, 0.55])}.items():
        o = torch.as_tensor(pos, dtype=torch.float32)
        samp = W.ddpm_sample(model, {"state": o[None]}, 256, gen)[0].numpy()
        null = W.GaussNull(samp)
        gm = W.GaussEpsModel(model, null)
        pol_pool = W.ddpm_sample(model, {"state": o[None]}, 64, gen)[0]
        nul_pool = W.ddpm_sample(gm, {"state": torch.zeros(1, 1)}, 64, gen)[0]
        if name == "start":
            ip = pick(unnorm(pol_pool.numpy()))
            inn = pick(unnorm(nul_pool.numpy()))
        else:
            ip, inn = pick_spread(pol_pool.numpy()), pick_spread(nul_pool.numpy())
        for arm, model_, pool, idx in (("policy", model, pol_pool, ip), ("control", gm, nul_pool, inn)):
            clean = pool[idx]
            acts = redraws(model_, o, clean, levels, args.M, gen)  # (L, P, M, H, A)
            vel = unnorm(acts)
            traj = pos + np.cumsum(vel, axis=-2)
            traj = np.concatenate([np.broadcast_to(pos, traj[..., :1, :].shape), traj], axis=-2)
            cv = unnorm(clean.numpy())
            ctraj = np.concatenate([np.broadcast_to(pos, (len(cv), 1, 2)), pos + np.cumsum(cv, 1)], 1)
            out[f"{name}__{arm}__traj"] = traj.astype(np.float32)
            out[f"{name}__{arm}__clean"] = ctraj.astype(np.float32)
            # also 200 fresh samples, for the "what this state's actions look like" backdrop
        fresh_p = unnorm(samp[:200])
        fresh_n = unnorm(null.sample(200, nrng))
        for arm, fv in (("policy", fresh_p), ("control", fresh_n)):
            out[f"{name}__{arm}__fresh"] = np.concatenate([np.broadcast_to(pos, (200, 1, 2)), pos + np.cumsum(fv, 1)], 1).astype(np.float32)
        print(name, "done", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
