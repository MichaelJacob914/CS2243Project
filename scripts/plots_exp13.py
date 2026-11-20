"""exp13 report figures (robomimic transport, K=20 DDIM): early-dip ablation, equal control strength, headroom.

    python scripts/plots_exp13.py
    python scripts/plots_exp13.py --only exp13_equal_strength

Inputs (<runs>/adsrl/transport/<name>/eval_summary.csv, 200 det episodes per eval, 50 for early evals):
  dip ablation:   fixed_k{0,3}_{tr,kickzi,ent005}_s{1,2,3} vs the exp10 baselines fixed_k{0,3}_s{1,2,3}
  equal strength: fixed_k{3,5,8}_eq_s1 vs fixed_k{0,3,5,8}_s1; adaptivedetach_rho0.5_eq_s1 vs adaptivedetach_rho0.5_s1
  headroom:       results/exp13/pooled_seeds.json
Outputs: report/assets/figs/<name>.{svg,png}, report/assets/data/<name>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plots_seeds as S  # noqa: E402  (shared loaders, colors, rcParams)

P, plt, Line2D = S.P, S.plt, S.Line2D
TR = os.path.join(S.RUNS, "adsrl", "transport")


def runs_of(name, seeds, max_steps=None):
    out = []
    for s in seeds:
        d = S.load_det(os.path.join(TR, f"{name}_s{s}", "eval_summary.csv"), max_steps=max_steps)
        if d is not None:  # early (50-ep) and regular (200-ep) evals can share a step: keep the later-written row
            d = d.assign(_r=(d["env_steps"] / 1000).round()).drop_duplicates("_r", keep="last")  # e.g. 896000 and 896032
            out.append((s, d.drop(columns="_r").reset_index(drop=True)))
    return out


def mean_line(ax, runs, color, lw=2.0, ls="-", label=None, grid=1000, z=3):
    for _, d in runs:
        ax.plot(d["env_steps"], d["success"], color=color, lw=0.6, alpha=0.3, ls=ls, zorder=z - 1)
    x, y, n = S.seed_mean(runs, grid)
    full = n == max(n) if len(n) else n
    ax.plot(x[full], y[full], color=color, lw=lw, ls=ls, marker="o", ms=2.2, zorder=z, label=label)
    return {"env_steps": x.tolist(), "mean": y.tolist(), "n_seeds": n.tolist()}


# ---------------------------------------------------------------------------------------------- dip ablation
DIP_ARMS = {"base": ("exp10 baseline", P.INK, "-"), "tr": ("noise trust region (bound 0.3)", "#4a7fb5", "-"),
            "ent005": ("lower starting entropy (α 0.05)", "#c9893a", "-"),
            "kickzi": ("kick + zero-init actor + α 0.05", "#b04a6a", (0, (4, 2)))}  # fmt: skip
DIP_MAX = 1_700_000


def exp13_dip(ctx):
    name = "exp13_dip"
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True, gridspec_kw={"wspace": 0.08})
    data = {"description": "Transport, det success vs env steps up to 1.7M, 3 seeds per arm (thin lines) and their "
            "mean (thick). Early evals every 64k steps (50 episodes) from the end of the 640k-step warm start.",
            "panels": {}}  # fmt: skip
    base = S.transport_base()
    for ax, k in zip(axs, (0, 3)):
        data["panels"][k] = {}
        for arm, (lab, c, ls) in DIP_ARMS.items():
            nm = f"fixed_k{k}_s" if arm == "base" else f"fixed_k{k}_{arm}_s"
            runs = runs_of(nm[:-2], S.SEEDS_T, DIP_MAX)
            if runs:
                data["panels"][k][arm] = mean_line(ax, runs, c, ls=ls, label=lab, grid=16000)
        if base:
            S.base_line(ax, base[0])
        ax.set_xlim(0, DIP_MAX)
        ax.set_ylim(-0.02, 0.5)
        ax.set_title(f"k = {k}" + (" (DSRL)" if k == 0 else ""), fontsize=8)
        ax.set_xlabel("Env steps")
        P.kfmt_steps(ax)
        P._style_ax(ax)
    axs[0].set_ylabel("Det. success")
    P.panel_label(axs[0], "a")
    P.panel_label(axs[1], "b")
    hs = [Line2D([], [], color=c, lw=2, ls=ls, label=lab) for lab, c, ls in DIP_ARMS.values()]
    hs.append(Line2D([], [], color=S.C["base"], lw=1.3, ls=S.BASE_LS, label="base policy"))
    fig.legend(handles=hs, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.17), fontsize=7)
    data["data_as_of"] = S.as_of()
    return P.save(ctx, fig, name, data, check_alignment=False)


# ---------------------------------------------------------------------------------------------- equal strength
EQ_C = {0: S.C["k0"], 3: S.C["k3"], 5: "#6aa9a9", 8: "#b5d4d4"}


def exp13_equal_strength(ctx):
    name = "exp13_equal_strength"
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.5), sharey=True, gridspec_kw={"wspace": 0.08})
    data = {"description": "Transport seed 1, det success vs env steps. Solid: steering at step k with the "
            "standard noise scale (exp10/exp3 runs). Dashed: same k with the noise scaled per action dimension so the "
            "action spread matches k=0 (equal control strength). k=0 (black) is the same reference in every panel. Panel d: adaptive k (k-entropy fix) with the standard scale vs with the per-step scales for all 20 steps. Dotted vertical line: latest eval of the equal-strength run in that panel.",
            "panels": {}}  # fmt: skip
    base = S.transport_base()
    k0 = runs_of("fixed_k0", [1])
    xmax = {}
    for ax, k in zip(axs, (3, 5, 8, "adaptive")):
        p = data["panels"][k] = {}
        if k0:
            p["k0"] = mean_line(ax, k0, EQ_C[0], lw=1.4, label="k=0 (DSRL)")
        stem = "adaptivedetach_rho0.5" if k == "adaptive" else f"fixed_k{k}"
        kl = "adaptive k" if k == "adaptive" else f"k={k}"
        std = runs_of(stem, [1])
        eq = runs_of(f"{stem}_eq", [1])
        if std:
            p["standard"] = mean_line(ax, std, S.C["k3"], lw=1.4, label=f"{kl}, standard noise")
        if eq:
            p["equal"] = mean_line(ax, eq, "#c45c2a", lw=1.8, ls=(0, (4, 2)), label=f"{kl}, equal strength")
            xmax[ax] = int(eq[0][1]["env_steps"].iloc[-1])
        if base:
            S.base_line(ax, base[0])
        ax.set_title(kl.replace("=", " = "), fontsize=8)
        ax.set_xlabel("Env steps")
        P.kfmt_steps(ax)
        P._style_ax(ax)
    for ax in axs:
        ax.set_xlim(0, 9.1e6)
        ax.set_xticks(np.arange(0, 9.01e6, 3e6))
        if ax in xmax:
            ax.axvline(xmax[ax], color=P.INK2, lw=0.6, ls=":", zorder=0)
    axs[0].set_ylim(-0.02, 1.0)
    axs[0].set_ylabel("Det. success (seed 1)")
    for a, l in zip(axs, "abcd"):
        P.panel_label(a, l)
    hs = [Line2D([], [], color=EQ_C[0], lw=1.4, label="k=0 (DSRL)"),
          Line2D([], [], color=S.C["k3"], lw=1.4, label="standard noise"),
          Line2D([], [], color="#c45c2a", lw=1.8, ls=(0, (4, 2)), label="equal control strength"),
          Line2D([], [], color=S.C["base"], lw=1.3, ls=S.BASE_LS, label="base policy")]  # fmt: skip
    fig.legend(handles=hs, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12), fontsize=7)
    data["equal_strength_latest_step"] = {a.get_title(): v for a, v in xmax.items()}
    data["data_as_of"] = S.as_of()
    return P.save(ctx, fig, name, data, check_alignment=False)


# ---------------------------------------------------------------------------------------------- headroom
HR_ARMS = [("iv_det", "actor,\nmean noise"), ("iii_stoch", "actor,\nsampled noise"),
           ("ii_k0_N640", "critic: best of\n640 at k=0"), ("i_allk_N32", "critic: best k\n× 32 noises")]  # fmt: skip


def exp13_headroom(ctx):
    name = "exp13_headroom"
    js = json.load(open(os.path.join(P.ROOT, "results", "exp13", "pooled_seeds.json")))
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    cols = [S.C["uniform"], S.C["uniform"], S.C["k0"], S.C["adaptive"]]
    data = {"description": "Transport, the uniform-random-k agent trained to 6M (seed 1), no further training. Arms: its actor with the mean "
            "noise, its actor with sampled noise, and two critic-guided choices per chunk. "
            "Success over 200 episodes per seed (3 seeds, 600 pooled); bars pooled, dots per seed, whiskers ±1 "
            "binomial SE of the pooled rate. Best-of-640 at k=0 and the critic's choice over all 20 steps × 32 "
            "noises use the same 640 critic evaluations per chunk.", "arms": {}}  # fmt: skip
    for i, ((a, lab), c) in enumerate(zip(HR_ARMS, cols)):
        r = js["arms"][a]
        ax.bar(i, r["pooled"], width=0.62, color=c, alpha=0.8, lw=0, zorder=2)
        ax.errorbar(i, r["pooled"], yerr=r["se_binom_600"], color=P.INK, lw=0.9, capsize=2, zorder=4)
        ps = list(r["per_seed"].values())
        ax.plot(i + np.linspace(-0.15, 0.15, len(ps)), ps, "o", ms=3.2, mfc="white", mec=P.INK, mew=0.8, zorder=5)
        ax.text(i, 0.985, f"{r['pooled']:.3f}", ha="center", va="top", fontsize=6.8, color=P.INK)
        data["arms"][a] = r
    ax.set_xticks(range(len(HR_ARMS)))
    ax.set_xticklabels([l for _, l in HR_ARMS], fontsize=6.4)
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("Success (200 ep / seed)")
    P._style_ax(ax)
    data["paired"] = js.get("paired")
    return P.save(ctx, fig, name, data, check_alignment=False)


FIGURES = {"exp13_dip": exp13_dip, "exp13_equal_strength": exp13_equal_strength, "exp13_headroom": exp13_headroom}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(P.ROOT, "report", "assets", "figs"))
    p.add_argument("--data_out", default=os.path.join(P.ROOT, "report", "assets", "data"))
    p.add_argument("--only", default=None)
    a = p.parse_args()
    ctx = S.Ctx(a.out, a.data_out)
    st = {n: ("written" if FIGURES[n](ctx) else "skipped") for n in (a.only.split(",") if a.only else FIGURES)}
    print(json.dumps(st, indent=1))
    if S.MISSING:
        print("missing:\n  " + "\n  ".join(sorted(set(S.MISSING))))


if __name__ == "__main__":
    main()
