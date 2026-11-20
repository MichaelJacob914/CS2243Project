"""exp14 analysis (CPU): preregistered decision rules, Stage 1 / 3 / 4 summaries, RL-gain validation, figures.

Rules (exp14 doc Setup, after Design changes 1-4):
  1  window at a state: for a primary feature (cluster label [clustered states], grip0, grip1, value), the policy
     width's 95% bootstrap interval lies below the null's (windows.width_test, >= 95% complete replicates per
     arm). Population: policy share - null-vs-null share >= 0.10 and one-sided Fisher p < 0.01.
  2  phase differences: per primary feature, midpoints in DDIM step index (300 phase states); statistic = max over
     phase pairs |median_i - median_j|; 10,000 label permutations; holds if > 1.0 step and p < 0.01.
  3  usefulness: Spearman rho between the median (over all non-excluded states) value open fraction (plan form) at
     k in {0,3,5,8,11,14,16,19} and the RL gain g_k; holds if rho >= 0.7 and exact one-sided p < 0.05.

    python scripts/exp14/analyze.py --states results/exp14/states/states.npz --stage12 results/exp14/stage12 \
        --sim results/exp14/sim --act_dir /n/netscratch/.../exp14/actions --out_dir results/exp14/analysis
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402
import windows as W  # noqa: E402

PHASES = ["before_contact", "open_grasp", "trash", "handover", "place", "ambiguous"]
PRIMARY = ["cluster_label", "grip0", "grip1", "value"]
EXPLORATORY = ["pc1", "pc2", "pc3", "pc4", "pc5", "arm0", "arm1", "slow", "fast", "grip0_mean", "grip1_mean"]
SWEPT = [0, 3, 5, 8, 11, 14, 16, 19]
GAIN_STEPS = (5_560_000, 5_816_000, 6_000_000)


def logsnr_levels():
    """log-SNR of the 20 DDIM step inputs (t = 95 - 5k), cosine schedule, T = 100 (as the policy)."""
    sys.path.insert(0, os.path.join(common.DSRL, "dppo"))
    from model.diffusion.sampling import cosine_beta_schedule

    b = cosine_beta_schedule(100).double().numpy()
    ac = np.cumprod(1 - b)
    t = 95 - 5 * np.arange(20)
    return np.log(ac[t] / (1 - ac[t]))


# ----------------------------------------------------------------------------------------
def rule1(res, feats):
    """Per state x feature: width tests (policy vs null; null2 vs null), point stats. Returns list of rows."""
    rows = []
    for r in res:
        for f in feats:
            if f not in r["pol"]:
                continue
            wt = W.width_test(r["pol"][f]["boot_width"], r["null"][f]["boot_width"])
            nn = W.width_test(r["null2"][f]["boot_width"], r["null"][f]["boot_width"])
            rows.append({"i": r["i"], "phase": r["phase"], "source": r["source"], "feature": f, "narrower": wt["narrower"],
                         "censored": wt["censored"], "p": wt["p"],
                         # Design change 5: separate "null degenerate" (the null's feature is constant, so its curve
                         # is undefined) from "policy censored" (policy bootstrap < 95% complete)
                         "null_degenerate": bool(np.all(np.isnan(r["null"][f]["R"][:20]))),
                         "policy_censored": bool(wt["complete"][0] < 0.95), "null_censored": bool(wt["complete"][1] < 0.95), "nn_narrower": nn["narrower"], "nn_censored": nn["censored"],
                         "nn_p": nn["p"], "pol_ci": wt["pol_ci"], "null_ci": wt["null_ci"]})  # fmt: skip
    return rows


def population_test(rows, f):
    from scipy.stats import fisher_exact

    rr = [x for x in rows if x["feature"] == f]
    n = len(rr)
    if n == 0:
        return {"n": 0}
    a = sum(x["narrower"] for x in rr)
    b = sum(x["nn_narrower"] for x in rr)
    _, p = fisher_exact([[a, n - a], [b, n - b]], alternative="greater")
    return {"n": n, "policy_narrower": a, "null_vs_null_narrower": b, "share_policy": a / n, "share_nn": b / n,
            "n_censored_policy_test": sum(x["censored"] for x in rr), "n_null_degenerate": sum(x["null_degenerate"] for x in rr),
            "n_policy_censored": sum(x["policy_censored"] for x in rr), "n_null_censored_not_degenerate": sum(x["null_censored"] and not x["null_degenerate"] for x in rr),
            "fisher_p": float(p),
            "holds": bool(a / n - b / n >= 0.10 and p < 0.01)}  # fmt: skip


def bh(p, q=0.05):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    out = np.zeros(len(p), bool)
    if ok.sum() == 0:
        return out
    ps = p[ok]
    order = np.argsort(ps)
    m = len(ps)
    thr = q * (np.arange(1, m + 1) / m)
    passed = ps[order] <= thr
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    sig = np.zeros(m, bool)
    sig[order[:k]] = True
    out[np.where(ok)[0]] = sig
    return out


def midpoints_k(res, f):
    x_k = np.arange(20, dtype=float)
    out = []
    for r in res:
        if f in r["pol"] and r["phase"] < 5:
            m = W.curve_stats(x_k, r["pol"][f]["R"][:20])["mid"]
            out.append((r["i"], r["phase"], float(m)))
    return out


def rule2(res, f, n_perm=10_000, seed=0):
    mk = [(i, ph, m) for i, ph, m in midpoints_k(res, f) if np.isfinite(m)]
    if len(mk) < 10:
        return {"n": len(mk)}
    ph = np.array([x[1] for x in mk])
    m = np.array([x[2] for x in mk])

    def stat(lab):
        meds = [np.median(m[lab == p]) for p in range(5) if (lab == p).sum() >= 3]
        return max(abs(a - b) for a, b in itertools.combinations(meds, 2)) if len(meds) >= 2 else np.nan

    s0 = stat(ph)
    if not np.isfinite(s0):  # fewer than two phases with >= 3 uncensored midpoints
        return {"n": len(mk), "stat_max_pair_diff_steps": np.nan, "perm_p": np.nan, "holds": False}
    rng = np.random.default_rng(seed)
    perm = np.array([stat(rng.permutation(ph)) for _ in range(n_perm)])
    p = float((np.sum(perm >= s0) + 1) / (n_perm + 1))
    return {"n": len(mk), "medians": {PHASES[p_]: float(np.median(m[ph == p_])) for p_ in range(5) if (ph == p_).any()},
            "n_per_phase": {PHASES[p_]: int((ph == p_).sum()) for p_ in range(5)}, "stat_max_pair_diff_steps": float(s0),
            "perm_p": p, "holds": bool(s0 > 1.0 and p < 0.01)}  # fmt: skip


# ----------------------------------------------------------------------------------------
def rl_gains():
    base = {}
    with open(os.path.join(common.PROJECT, "results", "base_eval_wk", "transport.csv")) as f:
        for row in csv.DictReader(f):
            if row["w_mode"] == "gauss":
                base[int(row["k"])] = float(row["success"])
    out = {}
    for k in SWEPT:
        rows = list(csv.DictReader(open(os.path.join(common.PROJECT, "logs", "adsrl", "transport", f"fixed_k{k}_s1", "eval_summary.csv"))))
        det = {int(r["env_steps"]): float(r["success"]) for r in rows if r["mode"] == "det"}
        sto = {int(r["env_steps"]): float(r["success"]) for r in rows if r["mode"] == "stoch"}
        missing = [s for s in GAIN_STEPS if s not in det]
        assert not missing, (k, missing)
        out[k] = {"no_learning": base[k], "det_final3": float(np.mean([det[s] for s in GAIN_STEPS])),
                  "gain": float(np.mean([det[s] for s in GAIN_STEPS]) - base[k]),
                  "gain_4.79M": det.get(4_792_000, np.nan) - base[k], "gain_6.0M": det[6_000_000] - base[k],
                  "gain_stoch_final3": float(np.mean([sto[s] for s in GAIN_STEPS if s in sto]) - base[k])}  # fmt: skip
    return out


def spearman_exact(x, y):
    from scipy.stats import rankdata

    rx, ry = rankdata(x), rankdata(y)

    def rho(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    r0 = rho(rx, ry)
    perms = [rho(rx, ry[list(pm)]) for pm in itertools.permutations(range(len(ry)))]
    return r0, float(np.mean(np.array(perms) >= r0 - 1e-12))


def rule3(res, gains):
    out = {}
    g = np.array([gains[k]["gain"] for k in SWEPT])
    for form in ("plan", "steer"):
        O = np.array([r["open"][form]["value"]["O"] for r in res if "value" in r["open"][form]])  # (S, 20)
        med = np.nanmedian(O, 0)
        rho, p = spearman_exact(med[SWEPT], g)
        out[form] = {"open_median": med.tolist(), "open_q25": np.nanpercentile(O, 25, 0).tolist(), "open_q75": np.nanpercentile(O, 75, 0).tolist(),
                     "rho": rho, "p_exact_one_sided": p, "holds": bool(rho >= 0.7 and p < 0.05)}  # fmt: skip
        for alt in ("gain_4.79M", "gain_6.0M", "gain_stoch_final3"):
            ga = np.array([gains[k][alt] for k in SWEPT])
            out[form][f"rho_{alt}"], out[form][f"p_{alt}"] = spearman_exact(med[SWEPT], ga)
    out["primary_holds"] = out["plan"]["holds"]
    return out


# ----------------------------------------------------------------------------------------
def sim_feature_curves(arr_dict, rng, x_ls, nboot=1000):
    """arr_dict from a stage3 a2a npz: disp (L,P,M,9), eef_disp (L,P,M,6), toward0/1 (L,P,M). -> name -> stats."""
    feats = {"disp": arr_dict["disp"].astype(np.float64), "eef_disp": arr_dict["eef_disp"].astype(np.float64),
             "toward0": np.eye(3)[arr_dict["toward0"].astype(int)], "toward1": np.eye(3)[arr_dict["toward1"].astype(int)]}  # fmt: skip
    out = {}
    for n, f in feats.items():
        fbar, wvar = W.point_summary(f)
        M = f.shape[2]
        R = W.resolved_from_summary(fbar, wvar, M)
        Rb = W.bootstrap_resolved_gram(fbar, wvar, M, nboot, rng)
        out[n] = {"R": R, "boot_width": W.curve_stats(x_ls, Rb)["width"]}
    return out


def stage3(res_by_i, sim_dir, rng, x_ls):
    """Returns (rows, curves, skipped): states not in res_by_i (excluded, e.g. diverged) are skipped and listed."""
    rows, curves, skipped = [], {}, []
    for f in sorted(glob.glob(os.path.join(sim_dir, "stage3", "s3_*_a2a.npz"))):
        i = int(os.path.basename(f)[3:6])
        if i not in res_by_i:
            skipped.append(i)
            continue
        fn = f.replace("_a2a.npz", "_a2a_null.npz")
        if not os.path.exists(fn):
            continue
        cp = sim_feature_curves(dict(np.load(f)), rng, x_ls)
        cn = sim_feature_curves(dict(np.load(fn)), rng, x_ls)
        curves[i] = {"pol": {k: v["R"].tolist() for k, v in cp.items()}, "null": {k: v["R"].tolist() for k, v in cn.items()}}
        for k in cp:
            wt = W.width_test(cp[k]["boot_width"], cn[k]["boot_width"])
            st = W.curve_stats(np.arange(20.0), cp[k]["R"])
            rows.append({"i": i, "phase": res_by_i[i]["phase"], "feature": "sim_" + k, "narrower": wt["narrower"], "censored": wt["censored"],
                         "p": wt["p"], "mid_k": float(st["mid"])})  # fmt: skip
        # open fraction of sim features (plan / steer forms), total variance from the Stage 1 executions
        s1 = os.path.join(sim_dir, "stage1", f"s1_{i:03d}.npz")
        if os.path.exists(s1):
            d1 = dict(np.load(s1))
            tv = {"disp": W.total_var(d1["disp"].astype(np.float64)), "eef_disp": W.total_var(d1["eef_disp"].astype(np.float64)),
                  "toward0": W.total_var(np.eye(3)[d1["toward0"].astype(int)]), "toward1": W.total_var(np.eye(3)[d1["toward1"].astype(int)])}  # fmt: skip
            for form in ("a2b_plan", "a2b_steer"):
                fp = f.replace("_a2a.npz", f"_{form}.npz")
                if os.path.exists(fp):
                    d = dict(np.load(fp))
                    feats = {"disp": d["disp"].astype(np.float64), "eef_disp": d["eef_disp"].astype(np.float64),
                             "toward0": np.eye(3)[d["toward0"].astype(int)], "toward1": np.eye(3)[d["toward1"].astype(int)]}  # fmt: skip
                    curves[i][form] = {k: W.open_fraction(v, tv[k]).tolist() for k, v in feats.items()}
    return rows, curves, skipped


def stage1_sim(res, sim_dir):
    out = {p: {"n": 0, "split0": 0, "split1": 0} for p in range(6)}
    for r in res:
        f = os.path.join(sim_dir, "stage1", f"s1_{r['i']:03d}.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f)
        o = out[r["phase"]]
        o["n"] += 1
        for g in (0, 1):
            share = np.bincount(d[f"toward{g}"].astype(int), minlength=3) / len(d[f"toward{g}"])
            o[f"split{g}"] += int(np.sort(share)[-2] >= 0.2)  # second most common target >= 20%
    return {PHASES[p]: v for p, v in out.items()}


def stage4(sim_dir, act_dir, res_by_i, x_ks):
    from sklearn.metrics import roc_auc_score

    files = sorted(glob.glob(os.path.join(act_dir, "s4_*.npz")))
    if not files:
        return None
    out = {"states": {}, "pooled": {}, "skipped_states": []}
    allq, alls = [], []
    for f in files:
        i = int(os.path.basename(f)[3:6])
        if i not in res_by_i:  # excluded (e.g. diverged) state: skipped like everywhere else, and listed
            out["skipped_states"].append(i)
            continue
        d = np.load(f)
        q, ks = d["q"], d["ks"]  # q (L, P4, M4)
        L, P4, M4 = q.shape
        succ = np.full((L, P4, M4), np.nan)
        for lv in range(L):
            for pt in range(P4):
                g = os.path.join(sim_dir, "stage4", f"s4_{i:03d}_{lv}_{pt:02d}.npz")
                if os.path.exists(g):
                    succ[lv, pt] = np.load(g)["success"]
        if np.isnan(succ).any():
            continue
        Rs = W.resolved_fraction(succ[..., None])
        Rq = W.resolved_fraction(q[..., None].astype(np.float64))
        within_auc = roc_auc_score(succ.ravel(), q.ravel()) if 0 < succ.mean() < 1 else np.nan
        Rv = [float(res_by_i[i]["pol"]["value"]["R"][k]) for k in ks]
        out["states"][i] = {"phase": res_by_i[i]["phase"], "ks": ks.tolist(), "success_rate": float(succ.mean()),
                            "R_success": Rs.tolist(), "R_value_stage4": Rq.tolist(), "R_value_stage2a": Rv, "auc_within": within_auc}  # fmt: skip
        allq.append(q.ravel())
        alls.append(succ.ravel())
    if alls:
        q, s = np.concatenate(allq), np.concatenate(alls)
        out["pooled"] = {"n_episodes": int(len(s)), "success_rate": float(s.mean()), "auc_pooled": float(roc_auc_score(s, q)) if 0 < s.mean() < 1 else np.nan,
                         "auc_within_mean": float(np.nanmean([v["auc_within"] for v in out["states"].values()])),
                         "mean_R_success": np.nanmean([v["R_success"] for v in out["states"].values()], 0).tolist(),
                         "mean_R_value": np.nanmean([v["R_value_stage4"] for v in out["states"].values()], 0).tolist()}  # fmt: skip
    return out


def memorization(act_dir, rng, ks=(0, 10, 17, 18, 19), n_per=128, max_states=40, include=None):
    """Distance from regenerated 2a actions to the nearest training chunk (dppo train.npz, 8-step chunks within
    demos, normalized like the policy's actions)."""
    import torch

    tr = np.load(os.path.join(common.DSRL, "dppo", "log", "robomimic", "transport", "train.npz"))
    norm = np.load(common.NORM)
    acts = tr["actions"].astype(np.float64)
    if np.abs(acts).max() > 1.0 + 1e-6:
        acts = 2 * (acts - norm["action_min"]) / (norm["action_max"] - norm["action_min"]) - 1
    chunks, s = [], 0
    for L in tr["traj_lengths"]:
        for t in range(s, s + L - 7, 2):
            chunks.append(acts[t : t + 8].ravel())
        s += L
    T = torch.as_tensor(np.array(chunks), dtype=torch.float32)
    out = {k: [] for k in ks}
    files = sorted(glob.glob(os.path.join(act_dir, "s3_*.npz")))
    if include is not None:  # skip excluded states consistently
        files = [f for f in files if int(os.path.basename(f)[3:6]) in include]
    files = files[:max_states]
    for f in files:
        a = np.load(f)["a2a"]  # (20, P, M, H, A)
        for k in ks:
            x = a[k].reshape(-1, 112)
            x = x[rng.choice(len(x), size=min(n_per, len(x)), replace=False)]
            d = torch.cdist(torch.as_tensor(x, dtype=torch.float32), T).min(1).values.numpy()
            out[k].append(float(np.median(d)))
    return {f"k{k}": {"median_over_states": float(np.nanmedian(v)), "n_states": int(np.isfinite(v).sum())} for k, v in out.items() if v}


def two_rise_rates(res, feats):
    from multiprocessing import Pool

    x_ls = logsnr_levels()
    jobs = [(r["i"], f, arm, r[arm][f]["R"][:20]) for r in res for f in feats if f in r["pol"] for arm in ("pol", "null2")]
    with Pool(min(32, os.cpu_count() or 4)) as pool:
        outs = pool.starmap(_tr, [(x_ls, j[3]) for j in jobs])
    rates = {}
    for (i, f, arm, _), o in zip(jobs, outs):
        d = rates.setdefault(f, {"pol": [0, 0], "null2": [0, 0]})
        d[arm][0] += int(o)
        d[arm][1] += 1
    return {f: {arm: {"preferred": v[0], "n": v[1], "rate": v[0] / max(v[1], 1)} for arm, v in d.items()} for f, d in rates.items()}


def _tr(x, R):
    if np.all(np.isnan(R)):
        return False
    return W.two_rise_test(x, R)["preferred"]


# ----------------------------------------------------------------------------------------
# Display names for the report figures (the code names stay in the JSON / CSV outputs).
FEAT_LABEL = {"cluster_label": "action cluster", "grip0": "gripper, arm 0", "grip1": "gripper, arm 1", "value": "critic value Q"}
PHASE_LABEL = {"before_contact": "before contact", "open_grasp": "open & grasp", "trash": "trash", "handover": "handover",
               "place": "place", "ambiguous": "ambiguous"}  # fmt: skip


def figures(res, r1rows, r2, r3, gains, s4, out_figs, x_ls):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = plt.cm.viridis(np.linspace(0, 0.9, 6))
    # 1. width vs null width per primary feature
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, f in zip(axes, PRIMARY):
        for r in res:
            if f not in r["pol"]:
                continue
            wp, wn = np.nanmedian(r["pol"][f]["boot_width"]), np.nanmedian(r["null"][f]["boot_width"])
            ax.scatter(wn, wp, s=10, color=cols[r["phase"]], alpha=0.7)
        lim = [0, 10]
        ax.plot(lim, lim, color="0.5", lw=1)
        rr = [x for x in r1rows if x["feature"] == f]
        ax.set(title=f"{FEAT_LABEL[f]}: {sum(x['narrower'] for x in rr)} of {len(rr)} states narrower\n"
               f"(control vs control: {sum(x['nn_narrower'] for x in rr)})",
               xlabel="Gaussian control width (log-SNR)", ylabel="policy width (log-SNR)", xlim=lim, ylim=lim)  # fmt: skip
        ax.spines[["top", "right"]].set_visible(False)
    for p in range(6):
        axes[0].scatter([], [], color=cols[p], label=PHASE_LABEL[PHASES[p]])
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_figs, "exp14_width_vs_null.png"), dpi=150)
    plt.close(fig)
    # 2. midpoints by phase
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, f in zip(axes, PRIMARY):
        mk = midpoints_k(res, f)
        for p in range(5):
            m = [x[2] for x in mk if x[1] == p and np.isfinite(x[2])]
            ax.scatter(np.full(len(m), p) + np.random.default_rng(p).uniform(-0.2, 0.2, len(m)), m, s=8, color=cols[p], alpha=0.6)
            if m:
                ax.plot([p - 0.3, p + 0.3], [np.median(m)] * 2, color="k", lw=2)
        info = r2.get(f, {})
        ax.set(title=f"{FEAT_LABEL[f]}\nlargest phase gap {info.get('stat_max_pair_diff_steps', np.nan):.2f} steps, "
               f"p = {info.get('perm_p', np.nan):.2g}", xticks=range(5), xlabel="task phase")  # fmt: skip
        ax.set_xticklabels([PHASE_LABEL[p] for p in PHASES[:5]], rotation=30, fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("DDIM step where half is resolved")
    fig.tight_layout()
    fig.savefig(os.path.join(out_figs, "exp14_midpoints_by_phase.png"), dpi=150)
    plt.close(fig)
    # 3. open fraction of the value vs RL gain
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ks = np.arange(20)
    for form, c in (("plan", "C0"), ("steer", "C1")):
        lab = {"plan": "re-noise the clean action", "steer": "inject new noise (as in steering)"}[form]
        axes[0].plot(ks, r3[form]["open_median"], color=c, lw=2, label=f"{lab} (median)")
        axes[0].fill_between(ks, r3[form]["open_q25"], r3[form]["open_q75"], color=c, alpha=0.2)
    axes[0].set(xlabel="DDIM step k", ylabel="share of critic value Q still undecided", title=f"Undecided share of the critic value, {len(res)} states")
    axes[0].legend(frameon=False)
    g = [gains[k]["gain"] for k in SWEPT]
    o = [r3["plan"]["open_median"][k] for k in SWEPT]
    axes[1].scatter(o, g, color="C0")
    for k, oo, gg in zip(SWEPT, o, g):
        axes[1].annotate(f"k={k}", (oo, gg), fontsize=8, xytext=(3, 3), textcoords="offset points")
    axes[1].set(xlabel="median undecided share of critic value Q", ylabel="RL gain in success over no RL\n(det., 5.56-6.0M)",
                title=f"Steering step k: Spearman ρ = {r3['plan']['rho']:.2f}, p = {r3['plan']['p_exact_one_sided']:.1e}")  # fmt: skip
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_figs, "exp14_open_vs_rlgain.png"), dpi=150)
    plt.close(fig)
    # 4. example curves (first state of each phase with a clustered label if possible)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, p in zip(axes.ravel(), range(6)):
        cand = [r for r in res if r["phase"] == p]
        if not cand:
            continue
        r = next((c for c in cand if "cluster_label" in c["pol"]), cand[0])
        for f, c in zip(["grip0", "grip1", "value", "cluster_label"], ["C0", "C1", "C2", "C3"]):
            if f not in r["pol"]:
                continue
            ax.plot(x_ls, r["pol"][f]["R"][:20], "o-", color=c, ms=3, label=FEAT_LABEL[f])
            ax.fill_between(x_ls, r["pol"][f]["R_lo"], r["pol"][f]["R_hi"], color=c, alpha=0.15)
            ax.plot(x_ls, r["null"][f]["R"][:20], "--", color=c, lw=1)
        ax.set(title=f"{PHASE_LABEL[PHASES[p]]} (state {r['i']}; dashed = Gaussian control)", xlabel="log-SNR",
               ylabel="resolved fraction R", ylim=(-0.1, 1.1))  # fmt: skip
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_figs, "exp14_example_curves.png"), dpi=150)
    plt.close(fig)
    # 5. stage 4
    if s4 and s4.get("pooled"):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        st = list(s4["states"].values())
        ks = st[0]["ks"]
        ax.plot(ks, s4["pooled"]["mean_R_success"], "o-", label="episode success (full rollouts)")
        ax.plot(ks, s4["pooled"]["mean_R_value"], "s--", label="critic value Q (same chunks)")
        ax.set(xlabel="DDIM step k where the first chunk is regenerated", ylabel="share decided by the chunk\n(mean over 20 states)",
               title=f"Critic ranks states (AUC {s4['pooled']['auc_pooled']:.2f}) but not chunks within a state "
               f"({s4['pooled']['auc_within_mean']:.2f})")  # fmt: skip
        ax.title.set_fontsize(9)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_figs, "exp14_stage4_success_vs_value.png"), dpi=150)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--states", required=True)
    p.add_argument("--stage12", required=True)
    p.add_argument("--sim", required=True)
    p.add_argument("--act_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--figs", default=os.path.join(common.PROJECT, "report", "assets", "figs"))
    p.add_argument("--keep_diverged", action="store_true", help="sensitivity: keep states where the unclamped chain diverged")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(14)
    x_ls = logsnr_levels()
    res = []
    for f in sorted(glob.glob(os.path.join(args.stage12, "state_*.pkl"))):
        with open(f, "rb") as fh:
            res.append(pickle.load(fh))
    # Diverged states: with x0 clipping off (plan), the stochastic DDPM chain blows up at a few states
    # (Stage 1 samples ~1e30, NaN regenerations at the noisiest levels). Their null fit and curves are invalid;
    # they are excluded from every analysis and listed (--keep_diverged for the sensitivity run).
    diverged = []
    for r in res:
        s1 = np.load(os.path.join(args.act_dir, f"s1_{r['i']:03d}.npz"))["samp"]
        bad = (not np.isfinite(s1).all()) or np.abs(s1).max() > 10
        bad = bad or any(np.isnan(np.asarray(r[arm]["pc1"]["R"][:20])).any() for arm in ("pol", "null", "null2"))
        if bad:
            diverged.append({"i": r["i"], "phase": r["phase"], "source": r["source"], "max_abs_stage1": float(np.nanmax(np.abs(s1)))})
    if not args.keep_diverged:
        dset = {d["i"] for d in diverged}
        res = [r for r in res if r["i"] not in dset]
    res_by_i = {r["i"]: r for r in res}
    out = {"n_states": len(res), "diverged_states": diverged, "keep_diverged": args.keep_diverged, "env": common.env_info()}
    # Stage 1
    st1 = {}
    for p_ in range(6):
        rr = [r for r in res if r["phase"] == p_]
        if rr:
            st1[PHASES[p_]] = {"n": len(rr), "share_clustered": float(np.mean([r["cluster"]["clustered"] for r in rr])),
                               "share_clustered_null": float(np.mean([r["cluster_null"]["clustered"] for r in rr])),
                               "median_lam1": float(np.median([r["lam"][0] for r in rr])),
                               "median_frac_outside_box": float(np.median([r["frac_outside_box"] for r in rr]))}  # fmt: skip
    out["stage1"] = st1
    out["stage1_sim_target_split"] = stage1_sim(res, args.sim)
    out["top_check"] = {"share_energy_p_lt_0.01": float(np.mean([r["top_check"]["energy_p"] < 0.01 for r in res])),
                        "max_abs_R_t99": float(np.nanmax([abs(v) for r in res for v in r["top_check"]["R_t99"].values()])),
                        "share_R_t99_abs_gt_0.05": float(np.nanmean([abs(v) > 0.05 for r in res for v in r["top_check"]["R_t99"].values() if np.isfinite(v)]))}  # fmt: skip
    # Rule 1
    r1rows = rule1(res, PRIMARY + EXPLORATORY)
    out["rule1"] = {f: population_test(r1rows, f) for f in PRIMARY}
    out["rule1"]["holds"] = bool(any(out["rule1"][f].get("holds", False) for f in PRIMARY))
    out["rule1_by_phase"] = {f: {PHASES[p_]: {"n": sum(1 for x in r1rows if x["feature"] == f and x["phase"] == p_),
                                              "narrower": sum(x["narrower"] for x in r1rows if x["feature"] == f and x["phase"] == p_)} for p_ in range(6)} for f in PRIMARY}  # fmt: skip
    # exploratory (policy features) + Stage 3 sim features, BH across states x features
    s3rows, s3curves, s3skipped = stage3(res_by_i, args.sim, rng, x_ls)
    out["stage3_skipped_states"] = s3skipped
    ex = [x for x in r1rows if x["feature"] in EXPLORATORY] + s3rows
    sig = bh([x["p"] for x in ex])
    out["exploratory"] = {}
    for f in sorted(set(x["feature"] for x in ex)):
        idx = [j for j, x in enumerate(ex) if x["feature"] == f]
        out["exploratory"][f] = {"n": len(idx), "narrower_ci": int(sum(ex[j]["narrower"] for j in idx)), "bh_significant": int(sig[idx].sum()),
                                 "null_vs_null_narrower": int(sum(ex[j].get("nn_narrower", False) for j in idx)),
                                 "censored": int(sum(ex[j]["censored"] for j in idx))}  # fmt: skip
    # Rule 2
    out["rule2"] = {f: rule2(res, f) for f in PRIMARY}
    out["rule2"]["holds"] = bool(any(out["rule2"][f].get("holds", False) for f in PRIMARY))
    out["rule2_exploratory_medians"] = {f: rule2(res, f, n_perm=2000) for f in ["pc1", "arm0", "arm1", "slow"]}
    # Rule 3
    gains = rl_gains()
    out["rl_gains"] = gains
    out["rule3"] = rule3(res, gains)
    # Stage 4, memorization, two-rise descriptive
    out["stage4"] = stage4(args.sim, args.act_dir, res_by_i, None)
    out["memorization_nn_dist"] = memorization(args.act_dir, rng, include=set(res_by_i))
    out["two_rise_descriptive"] = two_rise_rates(res, PRIMARY)
    # summary row
    r1, r2_ = out["rule1"]["holds"], out["rule2"]["holds"]
    out["summary_row"] = "Windows exist and move across phases" if (r1 and r2_) else ("Windows exist at one place" if r1 else "Only a smooth drift")
    out["seconds"] = time.time() - t0
    with open(os.path.join(args.out_dir, "analysis.json"), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else (float(o) if isinstance(o, (np.floating,)) else str(o)))
    with open(os.path.join(args.out_dir, "rule1_rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(r1rows[0]))
        w.writeheader()
        w.writerows(r1rows)
    with open(os.path.join(args.out_dir, "stage3_curves.json"), "w") as f:
        json.dump(s3curves, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    figures(res, r1rows, out["rule2"], out["rule3"], gains, out["stage4"], args.figs, x_ls)
    print(json.dumps({k: out[k] for k in ("summary_row", "rule1", "rule2", "rule3")}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
