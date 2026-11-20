"""exp13: per-action-dim noise scale (steer.w_scale_dims) for equal control strength (transport, DDIM K=20).

For each k in --ks, find v (length A, tiled over the horizon) such that the per-action-dim std of the final
action over w (w = clip(N(0,1), -1, 1) * v, the actor's bound times the scale) matches k=0 at v = 1. The std
is averaged over states and chunk positions, as in scripts/exp13_authority_calib.py, whose `measure` is reused.

Start from the per-dim c_k of results/exp13/authority/authority_per_dim_ck_transport_K20.json and iterate the
damped fixed point v <- v * (ref / std(v))^damp (the noise -> action map is not diagonal, so one interpolated c
per dim is only a first guess; the undamped update oscillates on the gripper dims at k=8). Keep the iterate with
the smallest max |ratio - 1|. Then verify on fresh w draws (another seed). Saves
results/exp13/authority/w_scale_dims_k<k>_transport_K20.npy (length A, for steer.w_scale_dims) and
w_scale_dims_calib_transport_K20.json.

--allk calibrates every k = 1..K-1 and saves one (K, A) array instead (row 0 = ones, row k = v_k; for
steer.w_scale_dims with the adaptive agent, each sample uses its own k's row):
results/exp13/authority/w_scale_dims_allk_transport_K20.npy and w_scale_dims_allk_calib_transport_K20.json.
A k missing from the init file starts from the previous k's result (k=1 from ones).

    python scripts/exp13_wscale_dims_calib.py --device cuda:0
    python scripts/exp13_wscale_dims_calib.py --device cuda:0 --allk
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
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from authority import dataset_states  # noqa: E402
from exp13_authority_calib import measure  # noqa: E402

from adsrl.model_utils import build_model, load_task_cfg  # noqa: E402


def per_dim_std(model, states, k, v, n_w, batch, seed):
    """std [H, A] at step k with w = clip(N(0,1)) * v (v: [A] tiled over H)."""
    H, A = model.horizon_steps, model.action_dim
    scale = torch.as_tensor(np.tile(v, (H, 1)), dtype=torch.float32, device=model.ddim_alphas.device)
    std, _, _ = measure(model, states, [k], [scale], n_w, batch, seed)
    return std[0, 0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="transport")
    p.add_argument("--ddim_steps", type=int, default=20)
    p.add_argument("--ks", type=int, nargs="+", default=[3, 5, 8])
    p.add_argument("--n_states", type=int, default=1000)
    p.add_argument("--n_w", type=int, default=128)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--damp", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verify_seed", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_dir", default=os.path.join(ROOT, "results", "exp13", "authority"))
    p.add_argument("--init_dir", default=os.path.join(ROOT, "results", "exp13", "authority"))
    p.add_argument("--allk", action="store_true", help="calibrate k = 1..K-1 and save one (K, A) array")
    args = p.parse_args()
    if args.allk:
        args.ks = list(range(1, args.ddim_steps))
    t0 = time.time()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    cfg = load_task_cfg(args.task)
    model = build_model(cfg, device=args.device, ddim_steps=args.ddim_steps)
    A = model.action_dim
    states = dataset_states(cfg, args.n_states, rng)
    tag = f"{args.task}_K{args.ddim_steps}"
    with open(os.path.join(args.init_dir, f"authority_per_dim_ck_{tag}.json")) as f:
        init = json.load(f)

    ref = per_dim_std(model, states, 0, np.ones(A), args.n_w, args.batch, args.seed).mean(0)  # [A]
    ref_v = per_dim_std(model, states, 0, np.ones(A), args.n_w, args.batch, args.verify_seed)
    out = {**vars(args), "w_dist": "clip(N(0,1), -1, 1) * v", "ref_k0_per_action_dim_std": ref.tolist(), "per_k": {}}
    prev = np.ones(A)
    for k in args.ks:
        v = np.asarray(init[str(k)]["c_per_action_dim"] if str(k) in init else prev, dtype=np.float64)
        hist = []
        for it in range(args.iters + 1):
            s = per_dim_std(model, states, k, v, args.n_w, args.batch, args.seed).mean(0)
            r = s / ref
            hist.append({"iter": it, "v": v.tolist(), "ratio": r.tolist(), "max_abs_dev": float(np.abs(r - 1).max())})
            print(f"k={k} iter={it} ratio min/max {r.min():.4f}/{r.max():.4f}", flush=True)
            if it < args.iters:
                v = v * r ** (-args.damp)
        best = min(hist, key=lambda h: h["max_abs_dev"])
        v = np.asarray(best["v"])
        prev = v
        # verification on fresh w draws: per action dim, and per (h, a) chunk position
        sv = per_dim_std(model, states, k, v, args.n_w, args.batch, args.verify_seed)
        r_a = sv.mean(0) / ref_v.mean(0)
        r_ha = sv / ref_v
        r_1 = per_dim_std(model, states, k, np.ones(A), args.n_w, args.batch, args.verify_seed).mean(0) / ref_v.mean(0)
        if not args.allk:
            np.save(os.path.join(args.out_dir, f"w_scale_dims_k{k}_{tag}.npy"), v.astype(np.float32))
        out["per_k"][str(k)] = {
            "w_scale_dims": v.tolist(), "v_min": float(v.min()), "v_max": float(v.max()),
            "verify_ratio_per_action_dim": r_a.tolist(),
            "verify_ratio_min": float(r_a.min()), "verify_ratio_max": float(r_a.max()),
            "verify_ratio_per_h_a_min": float(r_ha.min()), "verify_ratio_per_h_a_max": float(r_ha.max()),
            "unscaled_ratio_per_action_dim": r_1.tolist(),
            "unscaled_ratio_min": float(r_1.min()), "unscaled_ratio_max": float(r_1.max()),
            "best_iter": best["iter"], "iterations": hist,
        }  # fmt: skip
        print(f"k={k} v min/max {v.min():.4f}/{v.max():.4f}  verify ratio {r_a.min():.4f}/{r_a.max():.4f} "
              f"(per h,a {r_ha.min():.4f}/{r_ha.max():.4f}); unscaled {r_1.min():.4f}/{r_1.max():.4f}", flush=True)  # fmt: skip
    name = f"w_scale_dims_calib_{tag}.json"
    if args.allk:
        allk = np.ones((args.ddim_steps, A), dtype=np.float32)
        for k in args.ks:
            allk[k] = np.asarray(out["per_k"][str(k)]["w_scale_dims"], dtype=np.float32)
        np.save(os.path.join(args.out_dir, f"w_scale_dims_allk_{tag}.npy"), allk)
        out["allk_shape"] = list(allk.shape)
        name = f"w_scale_dims_allk_calib_{tag}.json"
    out["seconds"] = time.time() - t0
    with open(os.path.join(args.out_dir, name), "w") as f:
        json.dump(out, f, indent=2)
    print(f"done in {out['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
