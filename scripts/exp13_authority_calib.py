"""exp13 Phase 0A: authority calibration of steer.w_scale per steering step k (transport, DDIM K=20).

For each k in --ks and each c in --cs, decode M w per state (w ~ N(0, I) clipped to [-1, 1] like the
actor's tanh bound, then scaled by c; the same w draws for every (k, c)) and measure
  - per-dim std of the final action over w (mean over states)            -> std[k, c, H, A]
  - mean |a - a_base| (per action dim, averaged)  and  mean ||a - a_base||_2 over the chunk
where a_base is the unsteered DDIM action from the state's x_T.

c_k is the c at which the mean per-dim std at step k equals the k=0, c=1 reference (linear
interpolation on the c grid). A second estimate fits all per-dim stds at once (least squares on
log std). Also reports, at c_k, the interpolated mean |a - a_base| and the range of the scaled
noise c * clip(w) against N(0, 1).

States come from scripts/authority.py's dataset loader (same normalization).

    python scripts/exp13_authority_calib.py --out_dir results/exp13/authority
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

ROOT = os.environ.get("EXP13_CODE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from authority import dataset_states  # noqa: E402

from adsrl.model_utils import build_model, load_task_cfg  # noqa: E402
from adsrl.sampler import num_ddim_steps, steered_ddim_sample  # noqa: E402


@torch.no_grad()
def measure(model, states, ks, cs, n_w, batch, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    device = model.ddim_alphas.device
    H, A = model.horizon_steps, model.action_dim
    N = len(states)
    s_all = torch.as_tensor(states, device=device)
    xT_all = torch.randn(N, H, A, generator=g).to(device)
    nk, nc = len(ks), len(cs)
    std_sum = torch.zeros(nk, nc, H, A, device=device, dtype=torch.float64)
    abs_sum = torch.zeros(nk, nc, device=device, dtype=torch.float64)
    l2_sum = torch.zeros(nk, nc, device=device, dtype=torch.float64)
    per_chunk = max(1, batch // n_w)
    for i0 in range(0, N, per_chunk):
        s, x_T = s_all[i0 : i0 + per_chunk], xT_all[i0 : i0 + per_chunk]
        n = len(s)
        base = steered_ddim_sample(model, {"state": s}, x_T, torch.zeros(n, dtype=torch.long), x_T)
        s_rep, xT_rep = s.repeat_interleave(n_w, 0), x_T.repeat_interleave(n_w, 0)
        w = torch.randn(n * n_w, H, A, generator=g).to(device).clamp(-1.0, 1.0)
        for ik, k in enumerate(ks):
            kk = torch.full((n * n_w,), k, dtype=torch.long, device=device)
            for ic, c in enumerate(cs):
                a = steered_ddim_sample(model, {"state": s_rep}, xT_rep, kk, w * c).view(n, n_w, H, A)
                d = a - base[:, None]
                std_sum[ik, ic] += a.std(dim=1).sum(0).double()
                abs_sum[ik, ic] += d.abs().mean(dim=(1, 2, 3)).sum().double()
                l2_sum[ik, ic] += d.flatten(2).norm(dim=-1).mean(1).sum().double()
    return (std_sum / N).cpu().numpy(), (abs_sum / N).cpu().numpy(), (l2_sum / N).cpu().numpy()


def interp_c(cs, y, target):
    """Smallest c where the piecewise-linear y(c) crosses target (y increasing in c); None if never."""
    cs, y = np.asarray(cs, float), np.asarray(y, float)
    if y[0] >= target:
        return float(cs[0]) if np.isclose(y[0], target, rtol=1e-6) else None
    for i in range(len(cs) - 1):
        if y[i] <= target <= y[i + 1]:
            return float(cs[i] + (target - y[i]) / (y[i + 1] - y[i]) * (cs[i + 1] - cs[i]))
    return None


def noise_stats(c, n=2_000_000, seed=0):
    """Range of c * clip(z, -1, 1), z ~ N(0,1), vs N(0,1) itself."""
    z = np.random.default_rng(seed).standard_normal(n).clip(-1, 1) * c
    return {"max_abs": float(c), "rms": float(np.sqrt((z**2).mean())),
            "frac_abs_gt2": float((np.abs(z) > 2).mean()), "frac_abs_gt3": float((np.abs(z) > 3).mean()),
            "frac_at_bound": float((np.abs(z) >= c - 1e-9).mean())}  # fmt: skip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="transport")
    p.add_argument("--ddim_steps", type=int, default=20)
    p.add_argument("--ks", type=int, nargs="+", default=[0, 3, 5, 8])
    p.add_argument("--cs", type=float, nargs="+", default=[1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5])
    p.add_argument("--n_states", type=int, default=500)
    p.add_argument("--n_w", type=int, default=64)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_dir", default="results/exp13/authority")
    args = p.parse_args()
    t0 = time.time()
    assert args.ks[0] == 0 and args.cs[0] == 1.0, "reference is k=0, c=1 (first entries)"

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    cfg = load_task_cfg(args.task)
    model = build_model(cfg, device=args.device, ddim_steps=args.ddim_steps)
    K = num_ddim_steps(model)
    states = dataset_states(cfg, args.n_states, rng)
    print(f"task={args.task} K={K} states={states.shape}", flush=True)
    std, mabs, l2 = measure(model, states, args.ks, args.cs, args.n_w, args.batch, args.seed)

    ref = std[0, 0]  # [H, A], k=0 c=1
    ref_mean = float(ref.mean())
    mean_std = std.mean((2, 3))  # [nk, nc]
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.task}_K{K}"
    with open(os.path.join(args.out_dir, f"authority_grid_{tag}.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["k", "c", "mean_std", "std_ratio_to_ref", "per_dim_ratio_min", "per_dim_ratio_max",
                     "mean_abs_a_minus_base", "mean_l2_a_minus_base"])  # fmt: skip
        for ik, k in enumerate(args.ks):
            for ic, c in enumerate(args.cs):
                r = std[ik, ic].mean(0) / ref.mean(0)  # per action dim, averaged over chunk positions
                wr.writerow([k, c, mean_std[ik, ic], mean_std[ik, ic] / ref_mean, r.min(), r.max(),
                             mabs[ik, ic], l2[ik, ic]])  # fmt: skip

    cs = np.asarray(args.cs)
    rows = []
    for ik, k in enumerate(args.ks):
        ck = interp_c(cs, mean_std[ik], ref_mean)
        # least-squares fit over per-dim stds: c minimizing sum_d (log std_d(c) - log ref_d)^2 on a fine grid
        fine = np.linspace(cs[0], cs[-1], 801)
        per_dim = std[ik].reshape(len(cs), -1)
        interp_dims = np.stack([np.interp(fine, cs, per_dim[:, j]) for j in range(per_dim.shape[1])], 1)
        sse = ((np.log(interp_dims) - np.log(ref.reshape(1, -1))) ** 2).sum(1)
        ck_ls = float(fine[sse.argmin()])
        row = {"k": k, "c_k_mean_std": ck, "c_k_lsq_per_dim": ck_ls,
               "std_ratio_at_c1": float(mean_std[ik, 0] / ref_mean)}  # fmt: skip
        if ck is not None:
            per_dim_at = np.array([np.interp(ck, cs, std[ik, :, h, a]) for h in range(ref.shape[0])
                                   for a in range(ref.shape[1])]).reshape(ref.shape)  # fmt: skip
            r = per_dim_at.mean(0) / ref.mean(0)
            row.update({"per_action_dim_std_ratio_at_c_k": r.tolist(),
                        "per_dim_ratio_min": float(r.min()), "per_dim_ratio_max": float(r.max()),
                        "mean_abs_a_minus_base_at_c_k": float(np.interp(ck, cs, mabs[ik])),
                        "mean_l2_a_minus_base_at_c_k": float(np.interp(ck, cs, l2[ik])),
                        "noise_at_c_k": noise_stats(ck)})  # fmt: skip
        rows.append(row)
        print(json.dumps({kk: v for kk, v in row.items() if kk != "per_action_dim_std_ratio_at_c_k"}), flush=True)
    try:
        commit = subprocess.check_output(["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                          "rev-parse", "HEAD"], text=True).strip()  # fmt: skip
    except Exception:
        commit = None
    out = {**vars(args), "K": K, "commit": commit, "code_root": ROOT, "w_dist": "clip(N(0,1), -1, 1) * c",
           "ref_mean_std_k0_c1": ref_mean, "ref_mean_abs_a_minus_base_k0_c1": float(mabs[0, 0]),
           "ref_mean_l2_a_minus_base_k0_c1": float(l2[0, 0]), "noise_N01": {"rms": 1.0, "frac_abs_gt2": 0.0455,
           "frac_abs_gt3": 0.0027}, "calibration": rows,
           "grid": {"mean_std": mean_std.tolist(), "mean_abs": mabs.tolist(), "mean_l2": l2.tolist()},
           "seconds": time.time() - t0}  # fmt: skip
    with open(os.path.join(args.out_dir, f"authority_calib_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez(os.path.join(args.out_dir, f"authority_grid_{tag}.npz"), std=std, mean_abs=mabs, mean_l2=l2,
             ks=np.asarray(args.ks), cs=cs)  # fmt: skip
    print(f"done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
