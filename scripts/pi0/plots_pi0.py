"""Report figures for the pi0 / LIBERO-10 steering runs (exp7, exp8, exp9), from the run CSVs on scratch.

    python scripts/pi0/plots_pi0.py
    python scripts/pi0/plots_pi0.py --only pi0_curves,pi0_final

Inputs (<runs>/ = /n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs):
  <runs>/exp{7,8,9}/pi0_t{4,6,8}_<run>/eval_summary.csv   (train_adsrl_pi0.py; 50 episodes per eval)
  <runs>/exp7/zero_noise_baseline/task{4,6,8}.csv          (scripts/pi0/zero_noise_eval.py; includes plain pi0)
  <runs>/exp7/noise_pattern/task{4,6,8}.csv                (scripts/pi0/noise_pattern_eval.py)

Outputs: report/assets/figs/<name>.{svg,png} (PNG at 192 dpi) and report/assets/data/<name>.json with the
numbers drawn. Style, save() and helpers come from scripts/plots.py. Missing runs are skipped and printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plots as P  # noqa: E402  (sets the shared rcParams)

plt, matplotlib = P.plt, P.matplotlib
RUNS = "/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs"
TASKS = [4, 6, 8]
N_EVAL = 50  # episodes per eval
N_LAST = 3   # final = mean of the last 3 det evals

# ---- methods: (round, key) -> run-dir pattern, label, color, linestyle ---------------------------------------
# Colors follow scripts/plots.py: k=0 (DSRL) neutral dark, adaptive the orange accent, k=3 the muted teal
# ("fixed k"), structured noise the muted plum. The same hue is reused for a method across rounds; older
# rounds are drawn dashed and faded in the curve figure.
C = {"k0": P.METHOD_COLORS["dsrl"], "k3": P.METHOD_COLORS["best_fixed"], "k6": "#7fb2b2", "k8": "#b6d3d3",
     "adaptive": P.METHOD_COLORS["adaptive"], "struct": P.METHOD_COLORS["residual_sac"], "struct25": "#b98aa6",
     "xtobs": "#a8906a", "base": P.METHOD_COLORS["base"]}  # fmt: skip
METHODS = [
    # round, key, run name (with {t}), label, color key
    ("exp7", "k0", "pi0_t{t}_k0_xtfull_s0", "k=0 (DSRL)", "k0"),
    ("exp7", "k3", "pi0_t{t}_k3_xtfull_s0", "k=3", "k3"),
    ("exp7", "k6", "pi0_t{t}_k6_xtfull_s0", "k=6", "k6"),
    ("exp7", "k8", "pi0_t{t}_k8_xtfull_s0", "k=8", "k8"),
    ("exp7", "adaptive", "pi0_t{t}_adaptive_rho0.5_xtfull_s0", "adaptive k (before fix)", "adaptive"),
    ("exp8", "struct", "pi0_t{t}_k0_xtfull_wstruct_s0", "k=0 structured, b=1", "struct"),
    ("exp8", "struct25", "pi0_t{t}_k0_xtfull_wstruct_b2.5_s0", "k=0 structured, b=2.5", "struct25"),
    ("exp8", "xtobs", "pi0_t{t}_k0_xtfull_xtobs_s0", "k=0 repeated + raw noise in obs", "xtobs"),
    ("exp9", "k0", "pi0_t{t}_k0_xtfull_lf3_abase_b2.5_s0", "k=0 low-freq", "k0"),
    ("exp9", "k3", "pi0_t{t}_k3_xtfull_lf3_abase_b2.5_s0", "k=3 low-freq", "k3"),
    ("exp9", "adaptive", "pi0_t{t}_adaptive_rho0.5_xtfull_lf3_abase_kdetach_b2.5_s0", "adaptive k low-freq (k-entropy fix)", "adaptive"),
    ("exp9", "struct", "pi0_t{t}_k0_xtfull_wstruct_abase_s0", "k=0 structured + plan obs, b=1", "struct"),
]  # fmt: skip
ROUND_LABEL = {"exp7": "exp7: repeated noise", "exp8": "exp8: structured noise", "exp9": "exp9: low-freq + π0 plan obs"}
MISSING: list[str] = []


class Ctx:
    def __init__(self, out, data_out):
        self.out, self.data_out, self.qa_dir = out, data_out, None


def se(p, n):
    p = np.asarray(p, float)
    return np.sqrt(np.clip(p * (1 - p), 0, None) / n)


def load(rnd, run, t):
    f = os.path.join(RUNS, rnd, run.format(t=t), "eval_summary.csv")
    if not os.path.exists(f):
        MISSING.append(f)
        return None
    return pd.read_csv(f)


def det(df):
    return df[df["mode"] == "det"].sort_values("env_steps").reset_index(drop=True)


def plain_pi0():
    """task -> (success, se, n) from the zero-noise baseline CSV's 'base' row (200 episodes)."""
    out = {}
    for t in TASKS:
        f = os.path.join(RUNS, "exp7", "zero_noise_baseline", f"task{t}.csv")
        if os.path.exists(f):
            b = pd.read_csv(f, dtype={"k": str})
            b = b[b["k"] == "base"].iloc[0]
            out[t] = (float(b["success"]), float(b["se"]), int(b["n"]))
    return out


def base_ref(ax, base, t, horizontal=True, label=True):
    if t not in base:
        return
    b, s, n = base[t]
    if horizontal:
        ax.axhspan(b - s, b + s, color="0.92", lw=0, zorder=0)
        ax.axhline(b, color=C["base"], lw=1.3, ls=(0, (1, 1.4)), zorder=1, label=f"plain π0 (±1 SE, n={n})" if label else None)
    else:
        ax.axvspan(b - s, b + s, color="0.92", lw=0, zorder=0)
        ax.axvline(b, color=C["base"], lw=1.2, ls=(0, (1, 1.4)), zorder=1)


# =====================================================================================
def pi0_curves(ctx):
    name = "pi0_curves"
    base = plain_pi0()
    show = [("exp7", "k0"), ("exp7", "k3"), ("exp7", "adaptive"),
            ("exp9", "struct"), ("exp9", "k3"), ("exp9", "adaptive"), ("exp9", "k0")]  # drawn last = on top
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True, gridspec_kw={"wspace": 0.12})
    data = {"description": "Deterministic eval success (50 episodes per eval) vs env steps; band = ±1 binomial SE "
            "(n=50) for exp9 runs only. Plain pi0 from the zero-noise baseline base row (200 episodes).", "runs": []}
    handles = {}
    for ax, t in zip(axes, TASKS):
        base_ref(ax, base, t)
        for rnd, key in show:
            m = next(x for x in METHODS if x[0] == rnd and x[1] == key)
            df = load(rnd, m[2], t)
            if df is None:
                continue
            d = det(df)
            x, y = d["env_steps"].to_numpy(), d["success"].to_numpy()
            col = C[m[4]]
            if rnd == "exp9":
                (h,) = ax.plot(x, y, color=col, lw=1.5, zorder=4 + (key == "k0"))
                ax.fill_between(x, y - se(y, N_EVAL), y + se(y, N_EVAL), color=col, alpha=0.08, lw=0, zorder=2)
            else:
                (h,) = ax.plot(x, y, color=col, lw=1.1, ls=(0, (3, 1.6)), alpha=0.45, zorder=3)
            handles[f"{ROUND_LABEL[rnd].split(':')[0]} {m[3]}"] = h
            data["runs"].append({"task": t, "round": rnd, "method": key, "run": m[2].format(t=t),
                                 "env_steps": x.tolist(), "success": y.tolist()})
        ax.set_title(f"LIBERO-10 task {t}")
        ax.set_xlabel("Env steps")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, 1.02e6)
        P.kfmt_steps(ax)
        P._style_ax(ax)
    axes[0].set_ylabel("Det. success (50 episodes / eval)")
    order = ["exp9 k=0 low-freq", "exp9 k=3 low-freq", "exp9 adaptive k low-freq (k-entropy fix)", "exp9 k=0 structured + plan obs, b=1",
             "exp7 k=0 (DSRL)", "exp7 k=3", "exp7 adaptive k (before fix)"]
    hs = [handles[k] for k in order if k in handles]
    labs = [k for k in order if k in handles]
    bh = axes[0].get_legend_handles_labels()
    fig.legend(hs + bh[0][:1], labs + ["plain π0 (±1 SE, n=200)"], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.17), handlelength=2.2, columnspacing=1.4)
    return P.save(ctx, fig, name, data, check_alignment=False)


def pi0_final(ctx):
    name = "pi0_final"
    base = plain_pi0()
    rows = [m for m in METHODS]
    # y positions: rounds top to bottom; each round gets a bold header row above its methods
    ys, heads, y, prev = [], {}, 0.0, None
    for m in rows:
        if m[0] != prev:
            y += 0.4 if prev is not None else 0.0
            heads[m[0]] = y
            y += 1
        ys.append(y)
        y += 1
        prev = m[0]
    ys = np.array(ys)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.9), sharey=True, gridspec_kw={"wspace": 0.08})
    data = {"description": f"Final det success = mean of the last {N_LAST} det evals ({N_LAST * N_EVAL} episodes), "
            "±1 binomial SE; start = det eval at env step 0 (before training, z=0), exp9 only.", "rows": []}
    for ax, t in zip(axes, TASKS):
        base_ref(ax, base, t, horizontal=False)
        for yy, m in zip(ys, rows):
            df = load(m[0], m[2], t)
            if df is None:
                continue
            d = det(df)
            last = d["success"].iloc[-N_LAST:]
            p = float(last.mean())
            e = float(se(p, N_LAST * N_EVAL))
            col = C[m[4]]
            ax.errorbar(p, yy, xerr=e, fmt="o", ms=4.5, color=col, mec=col, capsize=0, elinewidth=1.2, zorder=4)
            s0 = d[d["env_steps"] == 0]["success"]
            start = float(s0.iloc[0]) if len(s0) else None
            if start is not None:
                ax.plot(start, yy, "o", ms=4.5, mfc="white", mec=col, mew=1.0, zorder=3)
                ax.plot([start, p], [yy, yy], color=col, lw=0.6, alpha=0.35, zorder=2)
            data["rows"].append({"task": t, "round": m[0], "method": m[1], "label": m[3], "run": m[2].format(t=t),
                                 "final": p, "se": e, "n_episodes": N_LAST * N_EVAL,
                                 "last_evals": last.tolist(), "last_env_steps": d["env_steps"].iloc[-N_LAST:].tolist(),
                                 "start_success": start})
        ax.set_xlim(-0.02, 1.04)
        ax.set_title(f"Task {t}" + (f"  (plain π0 {base[t][0]:.2f})" if t in base else ""))
        ax.grid(True, axis="x", color=P.GRID, lw=0.6)
        ax.grid(False, axis="y")
        ax.set_axisbelow(True)
    hy = [heads[r] for r in ROUND_LABEL if r in heads]
    axes[0].set_yticks(list(ys) + hy)
    axes[0].set_yticklabels([m[3] for m in rows] + [ROUND_LABEL[r] for r in ROUND_LABEL if r in heads])
    axes[0].invert_yaxis()
    labs = axes[0].get_yticklabels()
    for lab, m in zip(labs[: len(rows)], rows):
        lab.set_color(C[m[4]] if m[4] not in ("k8", "k6", "struct25") else P.INK2)
    for lab in labs[len(rows):]:
        lab.set_fontweight("bold")
        lab.set_color(P.INK)
    for ax in axes:
        ax.tick_params(axis="y", length=0)
        for yy in hy[1:]:
            ax.axhline(yy - 0.7, color=P.GRID, lw=0.8)
    fig.supxlabel("Deterministic success, mean of last 3 evals (150 episodes)", fontsize=8, y=0.02)
    from matplotlib.lines import Line2D

    hs = [Line2D([], [], marker="o", ls="", color=P.INK, ms=4.5, label="final (±1 SE)"),
          Line2D([], [], marker="o", ls="", mfc="white", mec=P.INK, ms=4.5, label="step 0, before training (exp9 only)"),
          Line2D([], [], color=C["base"], ls=(0, (1, 1.4)), lw=1.2, label="plain π0 (band ±1 SE, n=200)")]
    fig.legend(handles=hs, loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.07))
    return P.save(ctx, fig, name, data, check_alignment=False)


def _hist(s):
    return np.array([float(v) for v in str(s).split()])


def pi0_adaptive_k(ctx):
    name = "pi0_adaptive_k"
    cols = [("exp7", "pi0_t{t}_adaptive_rho0.5_xtfull_s0", "exp7 adaptive (before k-entropy fix)"),
            ("exp9", "pi0_t{t}_adaptive_rho0.5_xtfull_lf3_abase_kdetach_b2.5_s0", "exp9 adaptive (with fix)")]
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 5.6), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.15, "wspace": 0.06})  # fmt: skip
    data = {"description": "Fraction of training chunks (sampled k, train_k_hist: counts over the ~50k env steps "
            "before each eval) choosing each denoising step k, per eval point.", "panels": []}
    im = None
    for i, t in enumerate(TASKS):
        for j, (rnd, run, lab) in enumerate(cols):
            ax = axes[i, j]
            ax.grid(False)
            df = load(rnd, run, t)
            if df is None:
                ax.set_title(f"{lab}: task {t} missing")
                continue
            d = det(df)
            H = np.stack([_hist(s) for s in d["train_k_hist"]])
            keep = H.sum(1) > 0
            src = "train_k_hist"
            if not keep.any():
                H = np.stack([_hist(s) for s in d["k_hist"]])
                keep, src = H.sum(1) > 0, "k_hist (det)"
            x = d["env_steps"].to_numpy()[keep]
            F = H[keep] / H[keep].sum(1, keepdims=True)
            edges = np.concatenate([[x[0] - (x[1] - x[0]) / 2], (x[1:] + x[:-1]) / 2, [x[-1] + (x[-1] - x[-2]) / 2]])
            im = ax.pcolormesh(edges, np.arange(11) - 0.5, F.T, cmap="Blues", vmin=0, vmax=0.5, shading="flat")
            if i == 0:
                ax.set_title(lab, fontsize=8.5, fontweight="bold")
            ax.set_yticks(range(10))
            ax.tick_params(axis="y", labelsize=6.5)
            P.kfmt_steps(ax)
            data["panels"].append({"task": t, "round": rnd, "source": src, "env_steps": x.tolist(),
                                   "frac": F.tolist(), "chunks": H[keep].sum(1).tolist()})
    for i in range(3):
        axes[i, 0].set_ylabel(f"Task {TASKS[i]}\nchosen k")
    for j in range(2):
        axes[2, j].set_xlabel("Env steps")
    if im is not None:
        cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02, extend="max")
        cb.set_label("Fraction of training chunks (uniform = 0.1)")
        cb.ax.axhline(0.1, color=P.INK, lw=0.8)
    return P.save(ctx, fig, name, data, check_alignment=False)


def pi0_zero_noise(ctx):
    name = "pi0_zero_noise"
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True, gridspec_kw={"wspace": 0.12})
    data = {"description": "No training. Success with w = 0 injected at step k (200 episodes per point, ±1 SE) "
            "and plain pi0 (200 episodes).", "tasks": []}
    for ax, t in zip(axes, TASKS):
        f = os.path.join(RUNS, "exp7", "zero_noise_baseline", f"task{t}.csv")
        if not os.path.exists(f):
            MISSING.append(f)
            continue
        z = pd.read_csv(f, dtype={"k": str})
        b = z[z["k"] == "base"].iloc[0]
        r = z[z["k"] != "base"].assign(k=lambda d: d["k"].astype(int)).sort_values("k")
        base_ref(ax, {t: (float(b["success"]), float(b["se"]), int(b["n"]))}, t)
        ax.errorbar(r["k"], r["success"], yerr=r["se"], fmt="o-", color=C["k0"], ms=3.5, lw=1.3, capsize=0,
                    label="zero noise at step k")
        ax.set_title(f"LIBERO-10 task {t}")
        ax.set_xticks(range(10))
        ax.set_xlabel("Injection step k")
        ax.set_ylim(0, 1)
        P._style_ax(ax)
        data["tasks"].append({"task": t, "plain": [float(b["success"]), float(b["se"]), int(b["n"])],
                              "k": r["k"].tolist(), "success": r["success"].tolist(), "se": r["se"].tolist(),
                              "n": r["n"].tolist()})
    axes[0].set_ylabel("Success (±1 SE)")
    axes[0].legend(loc="lower left")
    return P.save(ctx, fig, name, data, check_alignment=False)


def pi0_noise_pattern(ctx):
    name = "pi0_noise_pattern"
    style = {"iid": ("#2a78d6", "o", "fresh N(0,I) per step"),
             "tiled": ("#eda100", "s", "one N(0,I) row, repeated"),
             "agent": ("#c2410c", "D", "untrained agent's noise, repeated"),
             "agent_iid": ("#1baf7a", "^", "untrained agent's noise, fresh per step")}  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True, gridspec_kw={"wspace": 0.12})
    base = plain_pi0()
    data = {"description": "No training. Success when pi0 is steered at step k with noise drawn from each pattern "
            "(100 episodes per point, ±1 SE); plain pi0 from the zero-noise baseline (200 episodes).", "tasks": []}
    for ax, t in zip(axes, TASKS):
        f = os.path.join(RUNS, "exp7", "noise_pattern", f"task{t}.csv")
        if not os.path.exists(f):
            MISSING.append(f)
            continue
        r = pd.read_csv(f)
        base_ref(ax, base, t, label=(t == TASKS[0]))
        ent = {"task": t}
        for i, (pat, (col, mk, lab)) in enumerate(style.items()):
            q = r[r["pattern"] == pat].sort_values("k")
            ax.errorbar(q["k"] + 0.12 * (i - 1.5), q["success"], yerr=q["se"], fmt=mk + "-", color=col, ms=3.5,
                        lw=1.2, capsize=0, label=lab)
            ent[pat] = {"k": q["k"].tolist(), "success": q["success"].tolist(), "se": q["se"].tolist(),
                        "n": q["n"].tolist()}
        ax.set_title(f"LIBERO-10 task {t}")
        ax.set_xticks([0, 3, 6, 8])
        ax.set_xlabel("Injection step k")
        ax.set_ylim(0, 1)
        P._style_ax(ax)
        data["tasks"].append(ent)
    axes[0].set_ylabel("Success (±1 SE)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.2), columnspacing=1.2)
    return P.save(ctx, fig, name, data, check_alignment=False)


def pi0_dist(ctx):
    name = "pi0_dist"
    show = [("exp7", "k0"), ("exp9", "struct"), ("exp9", "k3"), ("exp9", "adaptive"), ("exp9", "k0")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True, gridspec_kw={"wspace": 0.12})
    data = {"description": "mean |a - a_base| (mean absolute difference between the steered action and pi0's "
            "action for the same starting noise, normalized action space) in deterministic evals vs env steps.",
            "runs": []}
    hs = {}
    for ax, t in zip(axes, TASKS):
        for rnd, key in show:
            m = next(x for x in METHODS if x[0] == rnd and x[1] == key)
            df = load(rnd, m[2], t)
            if df is None:
                continue
            d = det(df)
            x, y = d["env_steps"].to_numpy(), d["mean_a_minus_base"].to_numpy()
            if rnd == "exp9":
                (h,) = ax.plot(x, y, color=C[m[4]], lw=1.5)
            else:
                (h,) = ax.plot(x, y, color=C[m[4]], lw=1.1, ls=(0, (3, 1.6)), alpha=0.45)
            hs[f"{rnd} {m[3]}"] = h
            data["runs"].append({"task": t, "round": rnd, "method": key, "env_steps": x.tolist(), "mean_a_minus_base": y.tolist()})
        ax.set_title(f"LIBERO-10 task {t}")
        ax.set_xlabel("Env steps")
        ax.set_ylim(bottom=0)
        P.kfmt_steps(ax)
        P._style_ax(ax)
    axes[0].set_ylabel("Mean |a − a_π0| (det. eval)")
    fig.legend(list(hs.values()), [k.replace("(fixed)", "(k-entropy fix)") for k in hs], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.2))
    return P.save(ctx, fig, name, data, check_alignment=False)


FIGURES = {"pi0_curves": pi0_curves, "pi0_final": pi0_final, "pi0_adaptive_k": pi0_adaptive_k,
           "pi0_zero_noise": pi0_zero_noise, "pi0_noise_pattern": pi0_noise_pattern, "pi0_dist": pi0_dist}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(P.ROOT, "report", "assets", "figs"))
    p.add_argument("--data_out", default=os.path.join(P.ROOT, "report", "assets", "data"))
    p.add_argument("--only", default=None, help="comma-separated figure names")
    a = p.parse_args()
    ctx = Ctx(a.out, a.data_out)
    status = {n: ("written" if FIGURES[n](ctx) else "skipped") for n in (a.only.split(",") if a.only else FIGURES)}
    print(json.dumps(status, indent=1))
    if MISSING:
        print("missing inputs:\n  " + "\n  ".join(sorted(set(MISSING))))


if __name__ == "__main__":
    main()
