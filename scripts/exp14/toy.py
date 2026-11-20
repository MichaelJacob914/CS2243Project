"""exp14 Stage 0: positive control on a 2D point mass with two routes around an obstacle.

Trains a small dppo DiffusionModel (same class, 100 DDPM steps, cosine schedule, DDIM 20) and runs the
same measurement pipeline (measure.py / windows.py) as transport. Pass criteria are in the exp14 doc.

    python scripts/exp14/toy.py --out_dir results/exp14/stage0 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402,F401  (paths: snapshot, dppo, pylib)
import measure  # noqa: E402
import windows as W  # noqa: E402

H, A = 8, 2
START = np.array([-1.0, 0.0])
GOAL = np.array([1.0, 0.0])
WAY_Y = 0.8
SPEED_MU, SPEED_SD = 0.05, 0.006
OBS_JIT, ACT_NOISE = 0.02, 0.0015


def path_points(route, speed, n_max=80):
    """Positions along start -> (0, route*0.8) -> goal at constant speed; stays at goal once reached."""
    W1 = np.array([0.0, WAY_Y * route])
    legs = [(START, W1), (W1, GOAL)]
    L1 = np.linalg.norm(W1 - START)
    L2 = np.linalg.norm(GOAL - W1)
    s = np.arange(n_max) * speed
    pts = np.empty((n_max, 2))
    for i, si in enumerate(s):
        if si <= L1:
            pts[i] = START + (W1 - START) * si / L1
        elif si <= L1 + L2:
            pts[i] = W1 + (GOAL - W1) * (si - L1) / L2
        else:
            pts[i] = GOAL
    del legs
    return pts


def make_dataset(n_demos, rng):
    obs, act = [], []
    for _ in range(n_demos):
        route = rng.choice([-1, 1])
        speed = float(np.clip(rng.normal(SPEED_MU, SPEED_SD), 0.03, 0.07))
        pts = path_points(route, speed)
        vel = np.diff(pts, axis=0)  # (n-1, 2)
        n = len(vel) - H
        for t in range(0, n):
            chunk = vel[t : t + H] + rng.normal(0, ACT_NOISE, (H, A))
            o = pts[t] + rng.normal(0, OBS_JIT, 2)
            obs.append(o)
            act.append(chunk)
    return np.array(obs, np.float32), np.array(act, np.float32)


def build_model(device, amin, amax):
    import adsrl  # noqa: F401
    from model.diffusion.diffusion import DiffusionModel
    from model.diffusion.mlp_diffusion import DiffusionMLP

    net = DiffusionMLP(action_dim=A, horizon_steps=H, cond_dim=2, time_dim=16, mlp_dims=[512, 512, 512], residual_style=True)
    m = DiffusionModel(
        network=net, horizon_steps=H, obs_dim=2, action_dim=A, device=device, denoised_clip_value=1.0,
        randn_clip_value=3, denoising_steps=100, predict_epsilon=True, use_ddim=True, ddim_steps=20,
    )  # fmt: skip
    m.amin, m.amax = amin, amax
    return m


def train(model, obs_n, act_n, steps, device, bs=2048, lr=1e-3, log=None):
    opt = torch.optim.AdamW(model.network.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-5)
    O = torch.as_tensor(obs_n, device=device)
    X = torch.as_tensor(act_n, device=device)
    g = torch.Generator(device=device).manual_seed(0)
    model.train()
    t0 = time.time()
    for it in range(steps):
        idx = torch.randint(0, len(O), (bs,), device=device, generator=g)
        loss = model.loss(X[idx], {"state": O[idx]})
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if log is not None and (it % 2000 == 0 or it == steps - 1):
            log(f"train it={it} loss={loss.item():.5f} t={time.time() - t0:.0f}s")
    model.eval()


def toy_features(stage1):
    """Features at one toy state: cluster label (if a labeler is given), PC1, PC2, chunk mean."""
    pcs, mu, labeler = stage1["pcs"], stage1["mu"], stage1.get("labeler_used")

    def f(a):
        X = a.reshape(len(a), -1).astype(np.float64)
        out = {"pc1": ((X - mu) @ pcs[0])[:, None], "pc2": ((X - mu) @ pcs[1])[:, None], "chunk_mean": a.mean(1)}
        if labeler is not None:
            out["cluster_label"] = W.cluster_label_onehot(labeler, a)
        return out

    return f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--train_steps", type=int, default=30000)
    p.add_argument("--n_demos", type=int, default=6000)
    p.add_argument("--P", type=int, default=64)  # re-preregistered 2026-11-10 (16 failed the toy power check)
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gen_seed", type=int, default=1, help="seed of the measurement RNGs (not the dataset / training)")
    p.add_argument("--model_from", default=None, help="load a trained toy_model.pt instead of training")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    logf = open(os.path.join(args.out_dir, "log.txt"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log(f"=== toy start {time.ctime()} args={vars(args)} {common.env_info()}")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    obs, act = make_dataset(args.n_demos, rng)
    amin, amax = act.reshape(-1, A).min(0), act.reshape(-1, A).max(0)
    act_n = 2 * (act - amin) / (amax - amin) - 1
    log(f"dataset {obs.shape} {act.shape} amin={amin} amax={amax}")
    model = build_model(args.device, amin, amax)
    ck = args.model_from or os.path.join(args.out_dir, "toy_model.pt")
    if os.path.exists(ck):
        model.network.load_state_dict(torch.load(ck, map_location=args.device))
        model.eval()
        log("loaded trained toy model")
    else:
        train(model, obs, act_n, args.train_steps, args.device, log=log)
        torch.save(model.network.state_dict(), ck)

    dev = args.device
    levels = W.ddim_levels(model)  # (20,) DDPM index per DDIM step k, noisiest first
    x_ls = W.logsnr(model, levels)
    ab_levels = W.abar(model)[levels]
    states = {
        "start": START.copy(),
        "past_obstacle": np.array([0.31, 0.55]),  # on the upper route's second leg
        "first_leg_up": np.array([-0.5, 0.4]),
    }
    gen = torch.Generator(device=dev).manual_seed(args.gen_seed)
    rng = np.random.default_rng(1000 + args.gen_seed)
    results = {"levels": levels.tolist(), "logsnr": x_ls.tolist()}
    curves_all = {}
    for name, pos in states.items():
        o = torch.as_tensor(pos, dtype=torch.float32, device=dev)
        samp = W.ddpm_sample(model, {"state": o[None]}, 256, gen)[0].cpu().numpy()  # (256, H, A)
        ct = W.cluster_tests(samp, rng_seed=0)
        null = W.GaussNull(samp)
        lam_max = float(null.lam.max())
        ct_null = W.cluster_tests(null.sample(256, rng), rng_seed=0)
        st1 = {"pcs": ct["pcs"], "mu": samp.reshape(256, -1).mean(0), "labeler_used": ct["labeler"] if ct["clustered"] else None}
        feat = toy_features(st1)
        # top-of-chain check: regenerate from t = 99 and compare with fresh samples
        # policy clean points: fresh stochastic draws, separate from the 256 Stage-1 samples that fit the
        # labeler / PCs / null (review item 4)
        clean = W.ddpm_sample(model, {"state": o[None]}, args.P, gen)[0]
        eps = torch.randn(clean.shape, device=dev, generator=gen)
        lv = list(levels) + [model.denoising_steps - 1]
        summ, acts = measure.stage2a_policy(model, o, clean, eps, lv, args.M, gen, feat, keep_actions=True)
        top_acts = acts[-1].reshape(-1, H, A)
        e_stat, e_p = measure.energy_test(top_acts[:256], samp, rng=rng)
        cur = measure.curves_from_summary(summ, args.M, args.n_boot, rng)
        top_R = {n: float(c["R"][-1]) for n, c in cur.items()}
        # null and an independent null replicate (false-positive calibration)
        # null: identical sampler (ddpm_regenerate) with the null's closed-form epsilon; clean points are exact
        # null draws (review item 1)
        gm = W.GaussEpsModel(model, null)
        e_n1 = torch.randn(clean.shape, device=dev, generator=gen)
        e_n2 = torch.randn(clean.shape, device=dev, generator=gen)
        nsum, _ = measure.stage2a_null(gm, args.P, e_n1, levels, args.M, gen, feat)
        nsum2, _ = measure.stage2a_null(gm, args.P, e_n2, levels, args.M, gen, feat)
        ncur = measure.curves_from_summary(nsum, args.M, args.n_boot, rng)
        ncur2 = measure.curves_from_summary(nsum2, args.M, args.n_boot, rng)
        st = {"lambda_max": lam_max, "speciation_logsnr": -np.log(lam_max), "cluster": {k: v for k, v in ct.items() if k not in ("labeler", "pcs", "pc_ev")},
              "cluster_null": {k: ct_null[k] for k in ("dbic", "dip_p", "clustered")}, "top_check": {"energy_stat": e_stat, "energy_p": e_p, "R_t99": top_R},
              "frac_outside_unit_box": float((np.abs(samp) > 1).mean()), "features": {}}  # fmt: skip
        curves_all[name] = {}
        for fn in cur:
            Rp = cur[fn]["R"][:-1]
            Bp = cur[fn]["boot"][:, :-1]
            sp, sn, sn2 = W.curve_stats(x_ls, Rp), W.curve_stats(x_ls, ncur[fn]["R"]), W.curve_stats(x_ls, ncur2[fn]["R"])
            bp, bn, bn2 = W.curve_stats(x_ls, Bp)["width"], W.curve_stats(x_ls, ncur[fn]["boot"])["width"], W.curve_stats(x_ls, ncur2[fn]["boot"])["width"]
            bmid = W.curve_stats(x_ls, Bp)["mid"]
            wt, wt_nn = W.width_test(bp, bn), W.width_test(bn2, bn)
            tr, tr_null, tr_null2 = W.two_rise_test(x_ls, Rp), W.two_rise_test(x_ls, ncur[fn]["R"]), W.two_rise_test(x_ls, ncur2[fn]["R"])
            st["features"][fn] = {
                "mid": float(sp["mid"]), "width": float(sp["width"]), "null_mid": float(sn["mid"]), "null_width": float(sn["width"]),
                "null2_width": float(sn2["width"]), "mid_ci": np.nanpercentile(bmid, [2.5, 97.5]).tolist(),
                "width_test": wt, "width_test_null_vs_null": wt_nn, "two_rise": tr, "two_rise_null": tr_null, "two_rise_null2": tr_null2,
                "R": Rp.tolist(), "R_null": ncur[fn]["R"].tolist(),
            }  # fmt: skip
            curves_all[name][fn] = {"R": Rp, "lo": np.nanpercentile(Bp, 2.5, 0), "hi": np.nanpercentile(Bp, 97.5, 0),
                                    "Rn": ncur[fn]["R"], "lon": np.nanpercentile(ncur[fn]["boot"], 2.5, 0), "hin": np.nanpercentile(ncur[fn]["boot"], 97.5, 0)}  # fmt: skip
        # Stage 2b on the toy (deterministic), plan form + steering form
        Bb, Cc = 8, 32
        xT = torch.randn((Bb, H, A), device=dev, generator=gen)
        e2 = torch.randn((Bb, Cc, H, A), device=dev, generator=gen)
        w2 = torch.randn((Bb, Cc, H, A), device=dev, generator=gen)
        ks = list(range(len(levels)))
        r2b, _, _ = measure.stage2b_policy(model, o, xT, e2, ks, feat, steer_w=w2)
        fresh = W.ddim_sample(model, {"state": o[None].expand(256, -1)}, torch.randn((256, H, A), device=dev, generator=gen)).cpu().numpy()
        ff = feat(fresh)
        st["open"] = {form: {fn: measure.open_curves(v, W.total_var(ff[fn])).tolist() for fn, v in d.items()} for form, d in r2b.items()}
        results[name] = st
        log(f"[{name}] lam_max={lam_max:.3f} -ln(lam)={-np.log(lam_max):.2f} clustered={ct['clustered']} dbic={ct['dbic']:.1f} dip_p={ct['dip_p']:.3g} "
            f"top-check energy p={e_p:.3f} R_t99={top_R}")  # fmt: skip
        for fn, d in st["features"].items():
            log(f"   {fn:14s} mid={d['mid']:.2f} width={d['width']:.2f} | null mid={d['null_mid']:.2f} width={d['null_width']:.2f} | "
                f"CI pol={np.round(d['width_test']['pol_ci'], 2)} null={np.round(d['width_test']['null_ci'], 2)} narrower={d['width_test']['narrower']} "
                f"| null-vs-null narrower={d['width_test_null_vs_null']['narrower']} | two-rise dBIC={d['two_rise']['dbic']:.1f} pref={d['two_rise']['preferred']}")  # fmt: skip

    # pass criteria
    s, po = results["start"], results["past_obstacle"]
    crit = {}
    lab = s["features"].get("cluster_label")
    crit["start_clustered"] = bool(s["cluster"]["clustered"])
    crit["a_label_narrower_than_null"] = bool(lab is not None and lab["width_test"]["narrower"])
    crit["b_mid_near_speciation"] = bool(lab is not None and abs(lab["mid"] - s["speciation_logsnr"]) <= 1.0)
    pc1 = po["features"]["pc1"]
    ci_overlap = not (pc1["width_test"]["pol_ci"][1] < pc1["width_test"]["null_ci"][0] or pc1["width_test"]["null_ci"][1] < pc1["width_test"]["pol_ci"][0])
    # Design change 3: rule 1 = width test only; two-rise is descriptive (reported with its null-vs-null rate)
    crit["c_past_obstacle_matches_null"] = bool(not pc1["width_test"]["censored"] and ci_overlap)
    crit["two_rise_descriptive"] = {f"{sn}/{fn}": {"policy": d["two_rise"]["preferred"], "null": d["two_rise_null"]["preferred"], "null2": d["two_rise_null2"]["preferred"]}
                                    for sn in states for fn, d in results[sn]["features"].items()}
    nn = {f"{sn}/{fn}": bool(d["width_test_null_vs_null"]["narrower"]) for sn in states for fn, d in results[sn]["features"].items()}
    crit["null_vs_null_narrower_any"] = bool(any(nn.values()))
    crit["null_vs_null_gate_features_ok"] = bool(not nn.get("start/cluster_label", True) and not nn.get("past_obstacle/pc1", True))
    crit["censored"] = {f"{sn}/{fn}": d["width_test"]["censored"] for sn in states for fn, d in results[sn]["features"].items()}
    crit["top_check"] = bool(all(abs(r) < 0.05 for st_ in (s, po) for r in st_["top_check"]["R_t99"].values()) and s["top_check"]["energy_p"] > 0.01 and po["top_check"]["energy_p"] > 0.01)
    # gate = the plan's three criteria; the top-of-chain check is a reported sanity check, not part of the gate
    crit["PASS"] = bool(crit["a_label_narrower_than_null"] and crit["b_mid_near_speciation"] and crit["c_past_obstacle_matches_null"]
                        and crit["null_vs_null_gate_features_ok"])
    results["criteria"] = crit
    log("CRITERIA " + json.dumps(crit))
    with open(os.path.join(args.out_dir, "stage0_results.json"), "w") as f:
        json.dump(results, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    np.savez(os.path.join(args.out_dir, "stage0_curves.npz"), **{f"{sn}__{fn}__{k}": v for sn, d in curves_all.items() for fn, dd in d.items() for k, v in dd.items()}, logsnr=x_ls)

    # figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    for r in (-1, 1):
        pts = path_points(r, SPEED_MU)
        ax.plot(pts[:, 0], pts[:, 1], color="0.6", lw=1)
    ax.add_patch(plt.Circle((0, 0), 0.5, color="0.85"))
    for n, pos in states.items():
        ax.plot(*pos, "o", label=n)
    ax.set_aspect("equal")
    ax.set_title("toy: two routes around an obstacle")
    ax.legend(frameon=False, fontsize=7)
    for ax, (sn, fn, title) in zip(axes[1:], [("start", "cluster_label", "start state: route label"), ("past_obstacle", "pc1", "past obstacle: PC1")]):
        if fn not in curves_all[sn]:
            continue
        c = curves_all[sn][fn]
        ax.fill_between(x_ls, c["lo"], c["hi"], color="C0", alpha=0.25)
        ax.plot(x_ls, c["R"], "o-", color="C0", label="policy")
        ax.fill_between(x_ls, c["lon"], c["hin"], color="0.5", alpha=0.25)
        ax.plot(x_ls, c["Rn"], "s--", color="0.4", label="Gaussian null")
        if sn == "start":
            ax.axvline(results["start"]["speciation_logsnr"], color="C3", ls=":", label=r"$-\ln\lambda_{max}$")
        ax.set(xlabel="log-SNR of the re-noised level", ylabel="resolved fraction R", title=title, ylim=(-0.1, 1.1))
        ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Stage 0 positive control: PASS={crit['PASS']}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "stage0.png"), dpi=150)
    log(f"=== toy done {time.ctime()}")


if __name__ == "__main__":
    main()
