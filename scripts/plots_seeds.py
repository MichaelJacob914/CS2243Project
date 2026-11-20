"""Multi-seed report figures: pi0.5 on LIBERO (exp12), pi0 on LIBERO-10 (exp9 + exp10), robomimic transport.

    python scripts/plots_seeds.py
    python scripts/plots_seeds.py --only pi05_curves,pi05_final

Inputs (<runs>/ = /n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs):
  <runs>/exp12/pi05_<suite>_t<task>_qf10_<method>_s<seed>/eval_summary.csv   (50 episodes per eval, budget 1M)
  results/exp12/base_eval_qf10_libero{10,90}.csv                              (untrained pi0.5, task strings)
  <runs>/exp9/pi0_t<task>_<method>_s0 and <runs>/exp10/pi0_t<task>_<method>_s{1,2}   (pi0, budget 1M)
  <runs>/exp7/zero_noise_baseline/task<t>.csv                                 (plain pi0, via plots_pi0)
  <runs>/adsrl/transport/<method>_s<seed>/eval_summary.csv                     (200 episodes per eval; cut at 9M)
     These are the CSVs written by the runs whose stdout is in logs/exp3/{w1_T*,x6_X*,x10_T_*}/<method>_s<seed>.log;
     the [eval det] lines in those logs match the CSV rows exactly up to 3M (checked 2026-11-10); later rows come from the 9M continuation runs.
  results/base_eval*/transport_K20/result.json                                (transport base policy, via plots)

Outputs: report/assets/figs/<name>.{svg,png} and report/assets/data/<name>.json. Missing runs are skipped and
printed, so the script can be rerun as runs progress. A run is "in progress" if its last det eval is below 98%
of its budget (1M for LIBERO, 9M for transport); in-progress runs are drawn hollow in the *_final figures.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import textwrap

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "pi0"))
import plots as P  # noqa: E402  (sets the shared rcParams)
import plots_pi0 as Q  # noqa: E402

plt = P.plt
from matplotlib.lines import Line2D  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402

RUNS = "/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs"
N_LAST = 3
SEEDS_LIBERO = [0, 1, 2]
SEEDS_T = [1, 2, 3]
BUDGET_LIBERO, BUDGET_T = 1_000_000, 9_000_000

C = {"k0": P.METHOD_COLORS["dsrl"], "k3": P.METHOD_COLORS["best_fixed"], "k5": "#8cc0c0",
     "adaptive": P.METHOD_COLORS["adaptive"], "adaptive_old": P.METHOD_COLORS["adaptive_subset"],
     "uniform": "#9a9993", "base": P.METHOD_COLORS["base"]}  # fmt: skip
LABEL = {"k0": "k=0 (DSRL)", "k3": "k=3", "k5": "k=5", "adaptive": "adaptive k (k-entropy fix)",
         "adaptive_old": "adaptive k (before fix)", "uniform": "uniform-random k"}  # fmt: skip
LIBERO_METHODS = {"k0": "k0_xtfull_lf3_abase_b2.5", "k3": "k3_xtfull_lf3_abase_b2.5",
                  "adaptive": "adaptive_rho0.5_xtfull_lf3_abase_kdetach_b2.5"}  # fmt: skip
PI05_TASKS = [("libero_10", 8), ("libero_90", 14), ("libero_90", 28), ("libero_90", 77)]
PI05_BASE = {("libero_10", 8): 0.74, ("libero_90", 14): 0.26, ("libero_90", 28): 0.50, ("libero_90", 77): 0.20}
PI0_TASKS = [4, 6, 8]
T_METHODS = {"k0": "fixed_k0", "k3": "fixed_k3", "k5": "fixed_k5", "adaptive": "adaptivedetach_rho0.5",
             "adaptive_old": "adaptive_rho0.5", "uniform": "uniformk"}  # fmt: skip
T_SEED_LINES = ["k0", "k3", "k5", "adaptive"]  # per-seed lines drawn; the others are drawn as a dashed mean only
MISSING: list[str] = []
BASE_LS = (0, (1, 1.4))  # dotted, as in plots_pi0
MTIMES: list[float] = []


class Ctx(Q.Ctx):
    pass


# ---------------------------------------------------------------------------------------------- loading
def load_det(path, max_steps=None):
    """Sorted det rows (env_steps, success) of one eval_summary.csv, or None if missing/empty."""
    if not os.path.exists(path):
        MISSING.append(path)
        return None
    MTIMES.append(os.path.getmtime(path))
    df = pd.read_csv(path)
    d = df[df["mode"] == "det"].sort_values("env_steps")
    if max_steps is not None:
        d = d[d["env_steps"] <= max_steps]
    if d.empty:
        MISSING.append(path + " (no det rows)")
        return None
    return d[["env_steps", "success"]].reset_index(drop=True)


def pi05_path(suite, t, m, s):
    return os.path.join(RUNS, "exp12", f"pi05_{suite}_t{t}_qf10_{LIBERO_METHODS[m]}_s{s}", "eval_summary.csv")


def pi0_path(t, m, s):
    rnd = "exp9" if s == 0 else "exp10"
    return os.path.join(RUNS, rnd, f"pi0_t{t}_{LIBERO_METHODS[m]}_s{s}", "eval_summary.csv")


def transport_path(m, s):
    return os.path.join(RUNS, "adsrl", "transport", f"{T_METHODS[m]}_s{s}", "eval_summary.csv")


def task_strings():
    out = {}
    for f in ("base_eval_qf10_libero10.csv", "base_eval_qf10_libero90.csv"):
        p = os.path.join(P.ROOT, "results", "exp12", f)
        if os.path.exists(p):
            for _, r in pd.read_csv(p).iterrows():
                out[(r["suite"], int(r["task_id"]))] = r["task"]
    return out


def suite_label(suite):
    return {"libero_10": "LIBERO-10", "libero_90": "LIBERO-90"}[suite]


def transport_base():
    b = P.load_base_eval(os.path.join(P.ROOT, "results"))
    r = b[(b["task"] == "transport") & (b["K"] == P.RECOMMENDED_K["transport"])] if not b.empty else b
    return None if r.empty else (float(r.iloc[0]["success"]), int(r.iloc[0]["n"]))


# ---------------------------------------------------------------------------------------------- helpers
def seed_mean(runs, grid):
    """Mean over the seeds that have an eval at each grid point (env steps rounded to `grid`)."""
    tab = {}
    for _, d in runs:
        for x, y in zip(d["env_steps"], d["success"]):
            tab.setdefault(int(round(x / grid) * grid), []).append(float(y))
    xs = sorted(tab)
    return np.array(xs), np.array([np.mean(tab[x]) for x in xs]), np.array([len(tab[x]) for x in xs])


def draw_seed_curves(ax, runs, color, grid, n_full, seed_lines=True, ls="-", zorder=3, marker=None):
    """Thin faint line per seed + thick mean. The mean is drawn full strength where every seed contributes
    and faded where fewer seeds have reached that step."""
    if not runs:
        return None, None
    if seed_lines:
        for _, d in runs:
            ax.plot(d["env_steps"], d["success"], color=color, lw=0.7, alpha=0.35, ls=ls, zorder=zorder - 1)
    x, y, n = seed_mean(runs, grid)
    full = n >= n_full
    h = None
    if full.any():
        last = np.nonzero(full)[0].max()
        (h,) = ax.plot(x[: last + 1], y[: last + 1], color=color, lw=2.0, ls=ls, zorder=zorder, marker=marker, ms=3)
        if last + 1 < len(x):
            ax.plot(x[last:], y[last:], color=color, lw=1.4, ls=ls, alpha=0.5, zorder=zorder, marker=marker, ms=3)
    else:
        (h,) = ax.plot(x, y, color=color, lw=1.4, ls=ls, alpha=0.5, zorder=zorder, marker=marker, ms=3)
    return h, {"env_steps": x.tolist(), "mean": y.tolist(), "n_seeds": n.tolist()}


TOO_EARLY: list[str] = []


def final_of(d, budget, tag):
    """Final = mean of the last N_LAST det evals; None (listed in TOO_EARLY) if the run has fewer than N_LAST
    evals after env step 0, e.g. a run that was just (re)started."""
    if (d["env_steps"] > 0).sum() < N_LAST:
        TOO_EARLY.append(f"{tag} (latest det eval at {int(d['env_steps'].iloc[-1])} steps)")
        return None
    last = d.iloc[-N_LAST:]
    return {"final": float(last["success"].mean()), "last_evals": last["success"].tolist(),
            "last_env_steps": last["env_steps"].astype(int).tolist(), "latest_step": int(d["env_steps"].iloc[-1]),
            "in_progress": bool(d["env_steps"].iloc[-1] < 0.98 * budget)}  # fmt: skip


def draw_final_dots(ax, xpos, rows, color):
    """Per-seed dots (hollow if in progress) + a mean tick with its value."""
    if not rows:
        return
    jit = np.linspace(-0.12, 0.12, len(rows)) if len(rows) > 1 else [0.0]
    for j, r in zip(jit, rows):
        mfc = "white" if r["in_progress"] else color
        ax.plot(xpos + j, r["final"], "o", ms=4.2, mfc=mfc, mec=color, mew=1.1, zorder=4)
    m = float(np.mean([r["final"] for r in rows]))
    ax.plot([xpos - 0.24, xpos + 0.24], [m, m], color=color, lw=2.2, zorder=5, solid_capstyle="butt")
    lab = f"{m:.2f}" + ("" if len(rows) >= 3 else f"\n(n={len(rows)})")
    ax.text(xpos + 0.28, m, lab, color=color, fontsize=7, va="center", ha="left", zorder=6)
    return m


def base_line(ax, y, label=None):
    ax.axhline(y, color=C["base"], lw=1.3, ls=BASE_LS, zorder=1, label=label)


def as_of():
    return datetime.datetime.fromtimestamp(max(MTIMES)).strftime("%Y-%m-%d %H:%M") if MTIMES else None


def legend_handles(keys, base_label):
    hs = [Line2D([], [], color=C[k], lw=2.0, ls=(0, (4, 2)) if k in ("adaptive_old", "uniform") else "-",
                 label=LABEL[k]) for k in keys]  # fmt: skip
    hs.append(Line2D([], [], color=C["base"], lw=1.3, ls=BASE_LS, label=base_label))
    return hs


# ---------------------------------------------------------------------------------------------- LIBERO curves
def libero_curves(ctx, name, panels, path_fn, seeds, base_fn, base_label, ylabel, figsize, nrows, ncols):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True, squeeze=False,
                             gridspec_kw={"wspace": 0.10, "hspace": 0.55 if nrows > 1 else 0.2})  # fmt: skip
    data = {"description": "Deterministic eval success (50 episodes per eval) vs env steps. Thin lines: one per seed; "
            "thick line: mean over the seeds that have an eval at that step (env steps rounded to 50k), faded "
            "where fewer than all seeds have reached it.", "panels": []}
    for ax, (key, title) in zip(axes.flat, panels):
        b = base_fn(key)
        if b is not None:
            base_line(ax, b)
        ent = {"panel": title, "base": b, "methods": {}}
        for m in ["k0", "k3", "adaptive"]:
            runs = [(s, d) for s in seeds if (d := load_det(path_fn(key, m, s))) is not None]
            _, mean = draw_seed_curves(ax, runs, C[m], 50_000, len(seeds), zorder=3 + 2 * (m == "adaptive"))
            ent["methods"][m] = {"seeds": {s: {"env_steps": d["env_steps"].tolist(), "success": d["success"].tolist()}
                                           for s, d in runs}, "mean": mean}  # fmt: skip
        ax.set_title(title, fontsize=7.8, loc="left")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, 1.02e6)
        ax.set_xlabel("Env steps")
        P.kfmt_steps(ax)
        P._style_ax(ax)
        data["panels"].append(ent)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    data["data_as_of"] = as_of()
    fig.legend(handles=legend_handles(["k0", "k3", "adaptive"], base_label), loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.06 if nrows > 1 else -0.14), handlelength=2.4, columnspacing=1.6)
    return P.save(ctx, fig, name, data, check_alignment=False)


def pi05_title(key, width):
    suite, t = key
    txt = TASK_TXT.get(key, "")
    return textwrap.fill(f"{suite_label(suite)} #{t}: {txt}", width)


TASK_TXT: dict = {}


def pi05_curves(ctx):
    TASK_TXT.update(task_strings())
    panels = [(k, pi05_title(k, 52)) for k in PI05_TASKS]
    return libero_curves(ctx, "pi05_curves", panels,
                         lambda key, m, s: pi05_path(key[0], key[1], m, s), SEEDS_LIBERO, PI05_BASE.get,
                         "untrained π0.5 (50 episodes)", "Det. success (50 ep / eval)", (7.2, 5.0), 2, 2)  # fmt: skip


def pi0_seeds_curves(ctx):
    base = Q.plain_pi0()
    panels = [(t, f"LIBERO-10 task {t}") for t in PI0_TASKS]
    return libero_curves(ctx, "pi0_seeds_curves", panels, pi0_path, SEEDS_LIBERO,
                         lambda t: base[t][0] if t in base else None, "plain π0 (200 episodes)",
                         "Det. success (50 ep / eval)", (7.2, 2.7), 1, 3)  # fmt: skip


# ---------------------------------------------------------------------------------------------- finals
def pi05_final(ctx):
    name = "pi05_final"
    TASK_TXT.update(task_strings())
    methods = ["k0", "k3", "adaptive"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.0), sharey=True, gridspec_kw={"wspace": 0.16})
    data = {"description": f"Final det success per seed = mean of its last {N_LAST} det evals (50 episodes each); "
            "bar = mean over seeds. Hollow = run still training (budget 1M env steps), read at its latest evals.",
            "rows": []}  # fmt: skip
    for ax, key in zip(axes, PI05_TASKS):
        base_line(ax, PI05_BASE[key])
        for i, m in enumerate(methods):
            rows = []
            for s in SEEDS_LIBERO:
                d = load_det(pi05_path(key[0], key[1], m, s))
                f = final_of(d, BUDGET_LIBERO, f"pi05 {key[0]} t{key[1]} {m} s{s}") if d is not None else None
                if f is not None:
                    rows.append({"suite": key[0], "task": key[1], "method": m, "seed": s, **f})
            mean = draw_final_dots(ax, i, rows, C[m])
            data["rows"] += rows
            data.setdefault("means", []).append({"suite": key[0], "task": key[1], "method": m, "mean": mean,
                                                 "n_seeds": len(rows)})  # fmt: skip
        ax.set_title(pi05_title(key, 26), fontsize=7.2, loc="left")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(["k=0", "k=3", "adaptive"])
        ax.set_xlim(-0.5, len(methods) - 0.3)
        ax.set_ylim(-0.02, 1.04)
        P._style_ax(ax)
    axes[0].set_ylabel(f"Det. success, mean of last {N_LAST} evals")
    hs = [Line2D([], [], marker="o", ls="", color=P.INK, ms=4.2, label="seed, finished (1M steps)"),
          Line2D([], [], marker="o", ls="", mfc="white", mec=P.INK, ms=4.2, label="seed, still training"),
          Line2D([], [], color=P.INK, lw=2.2, label="mean over seeds"),
          Line2D([], [], color=C["base"], lw=1.3, ls=BASE_LS, label="untrained π0.5")]  # fmt: skip
    fig.legend(handles=hs, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    data["data_as_of"] = as_of()
    data["excluded_too_early"] = [x for x in TOO_EARLY if x.startswith("pi05")]
    return P.save(ctx, fig, name, data, check_alignment=False)


def transport_runs(m):
    return [(s, d) for s in SEEDS_T if (d := load_det(transport_path(m, s), max_steps=BUDGET_T)) is not None]


def transport_seeds_curves(ctx):
    name = "transport_seeds_curves"
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    base = transport_base()
    if base:
        base_line(ax, base[0])
    data = {"description": "Robomimic transport, deterministic eval success (200 episodes per eval) vs env steps, cut "
            "at 9M. Thin lines: one per seed (seeds 1-3); thick: mean over the seeds that have reached that step. "
            "Adaptive before the fix and uniform-random k are drawn as dashed means only.",
            "base": base, "methods": {}}  # fmt: skip
    for m in ["uniform", "adaptive_old", "k5", "k3", "k0", "adaptive"]:
        runs = transport_runs(m)
        dashed = m in ("uniform", "adaptive_old")
        _, mean = draw_seed_curves(ax, runs, C[m], 1000, len(SEEDS_T), seed_lines=not dashed,
                                   ls=(0, (4, 2)) if dashed else "-", zorder=3 + 2 * (m == "adaptive"),
                                   marker="o")  # fmt: skip
        data["methods"][m] = {"seeds": {s: {"env_steps": d["env_steps"].tolist(), "success": d["success"].tolist()}
                                        for s, d in runs}, "mean": mean}  # fmt: skip
    ax.set_xlim(0, 9.15e6)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Env steps")
    ax.set_ylabel("Det. success (200 ep / eval)")
    ax.set_title("Robomimic transport, 3 seeds", loc="left")
    P.kfmt_steps(ax)
    P._style_ax(ax)
    bl = f"base policy ({base[0]:.3f}, n={base[1]})" if base else "base policy"
    ax.legend(handles=legend_handles(["k0", "k3", "k5", "adaptive", "adaptive_old", "uniform"], bl),
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, handlelength=2.2, fontsize=6.5, columnspacing=1.2, frameon=False)  # fmt: skip
    data["data_as_of"] = as_of()
    return P.save(ctx, fig, name, data, check_alignment=False)


def transport_seeds_final(ctx):
    name = "transport_seeds_final"
    methods = ["k0", "k3", "k5", "adaptive", "adaptive_old", "uniform"]
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    base = transport_base()
    if base:
        base_line(ax, base[0])
    data = {"description": f"Transport final det success per seed = mean of its last {N_LAST} det evals at <= 9M env "
            "steps (200 episodes each); bar = mean over seeds. Hollow = run still training, read at its latest evals.",
            "base": base, "rows": [], "means": []}  # fmt: skip
    for i, m in enumerate(methods):
        rows = [{"method": m, "seed": s, **f} for s, d in transport_runs(m)
                if (f := final_of(d, BUDGET_T, f"transport {m} s{s}")) is not None]
        mean = draw_final_dots(ax, i, rows, C[m])
        data["rows"] += rows
        data["means"].append({"method": m, "mean": mean, "n_seeds": len(rows)})
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(["k=0\n(DSRL)", "k=3", "k=5", "adaptive\n(fix)", "adaptive\n(before fix)", "uniform\nrandom k"])
    ax.set_xlim(-0.5, len(methods) - 0.3)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel(f"Det. success, last {N_LAST} evals ≤9M")
    ax.set_title("Robomimic transport, 3 seeds", loc="left")
    P._style_ax(ax)
    hs = [Line2D([], [], marker="o", ls="", color=P.INK, ms=4.2, label="seed at 9M"),
          Line2D([], [], marker="o", ls="", mfc="white", mec=P.INK, ms=4.2, label="seed still training"),
          Line2D([], [], color=P.INK, lw=2.2, label="mean"),
          Line2D([], [], color=C["base"], lw=1.3, ls=BASE_LS, label="base policy")]  # fmt: skip
    ax.legend(handles=hs, loc="center", ncol=2, fontsize=7)
    data["data_as_of"] = as_of()
    return P.save(ctx, fig, name, data, check_alignment=False)


# ---------------------------------------------------------------------------------------------- transport to 9M
BUDGET_T6 = 9_000_000
C6 = {**C, "k8": "#5b7db1", "k11": "#b07aa1"}  # single-seed fixed k=8 and k=11 (trained to 6M only)
LABEL6 = {**LABEL, "k8": "k=8 (seed 1, to 6M)", "k11": "k=11 (seed 1, to 6M)"}
T6_EXTRA = {"k8": "fixed_k8", "k11": "fixed_k11"}
T6_DASHED = ("uniform",)


def transport_runs6(m):
    path = lambda s: os.path.join(RUNS, "adsrl", "transport", f"{T6_EXTRA.get(m) or T_METHODS[m]}_s{s}",
                                  "eval_summary.csv")  # fmt: skip
    seeds = [1] if m in T6_EXTRA else SEEDS_T
    return [(s, d) for s in seeds if (d := load_det(path(s), max_steps=BUDGET_T6)) is not None]


def final_at(d, budget):
    """Mean of the last N_LAST det evals at <= budget, only if the run has an eval within 2% of `budget`."""
    d = d[d["env_steps"] <= budget]
    if d.empty or d["env_steps"].iloc[-1] < 0.98 * budget or (d["env_steps"] > 0).sum() < N_LAST:
        return None
    last = d.iloc[-N_LAST:]
    return {"value": float(last["success"].mean()), "last_evals": last["success"].tolist(),
            "last_env_steps": last["env_steps"].astype(int).tolist()}  # fmt: skip


T6_SMOOTH = 250_000  # half-width (env steps) of the moving average on the thick mean lines in transport_9M


def smooth_steps(x, y, half):
    """Moving average over evals within +-`half` env steps. Evals are denser before 1.6M (every 64k) than after,
    so the window is set in env steps, not in number of evals."""
    return np.array([y[np.abs(x - xi) <= half].mean() for xi in x])


def transport_9M(ctx):
    """(a) learning curves to 9M, (b) final success at 9M per method (bar = mean over seeds that reached 9M)."""
    name = "transport_9M"
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.2})
    base = transport_base()
    data = {"description": "Robomimic transport, deterministic eval success (200 episodes per eval). (a) Success vs "
            "env steps to 9M; thin lines one per seed, thick line the mean where >= 2 seeds have an eval at that step, smoothed by a moving average over "
            "evals within +-250k env steps "
            "(single-seed stretches drawn medium). (b) Per seed: mean of the last 3 det evals at <= 9M; bar = mean over the "
            "seeds whose latest eval is within 2% of 9M (n). Seeds still training are read at their latest 3 evals, drawn "
            "hollow and left out of the bar; a method with no seed at 9M gets a hatched bar at its in-progress mean. k=8 and k=11 are single-seed runs that stopped at 6M; their hatched bars are read at 6M. Adaptive k before the entropy fix is left out.", "base": base, "methods": {}, "panel_b": []}  # fmt: skip
    if base:
        for a in (ax, bx):
            base_line(a, base[0])
    # ---- (a)
    for m in ["k11", "k8", "uniform", "k5", "k3", "k0", "adaptive"]:
        runs = transport_runs6(m)
        if not runs:
            continue
        ls = (0, (4, 2)) if m in T6_DASHED else "-"
        faint = False
        z = 3 + 2 * (m == "adaptive")
        for _, d in runs:
            ax.plot(d["env_steps"], d["success"], color=C6[m], lw=0.7 if not faint else 1.0,
                    alpha=0.35 if not faint else 0.9, ls=ls, zorder=z - 1)  # fmt: skip
        x, y, n = seed_mean(runs, 1000)
        ys = smooth_steps(x, y, T6_SMOOTH)
        if not faint:
            for lo, hi, multi in _runs_of(n >= 2):
                sl = slice(lo, hi + 2 if hi + 1 < len(x) else hi + 1)  # bridge into the next stretch
                ax.plot(x[sl], ys[sl], color=C6[m], ls=ls, zorder=z, lw=2.0 if multi else 1.1,
                        alpha=1.0 if multi else 0.8)  # fmt: skip
        data["methods"][m] = {"seeds": {s: {"env_steps": d["env_steps"].astype(int).tolist(),
                                            "success": d["success"].tolist()} for s, d in runs},
                              "mean": {"env_steps": x.tolist(), "mean": y.tolist(), "mean_smoothed": ys.tolist(),
                                       "n_seeds": n.tolist()}}  # fmt: skip
    ax.set_xlim(0, 9.1e6)
    ax.set_ylim(-0.02, 1.0)
    ax.set_xticks(np.arange(0, 9.01e6, 1e6))
    ax.set_xlabel("Env steps")
    ax.set_ylabel("Det. success (200 ep / eval)")
    P.kfmt_steps(ax)
    P._style_ax(ax)
    P.panel_label(ax, "a")
    # ---- (b) final performance at 9M
    methods = ["k0", "k3", "k5", "k8", "k11", "uniform", "adaptive"]
    names = ["k=0\n(DSRL)", "k=3", "k=5", "k=8\n(to 6M)", "k=11\n(to 6M)", "random\nk", "adaptive\nk"]
    tr = mtransforms.blended_transform_factory(bx.transData, bx.transAxes)
    for i, m in enumerate(methods):
        c = C6[m]
        rows = []
        for s, d in transport_runs6(m):
            f6 = final_at(d, BUDGET_T6)
            latest = int(d["env_steps"].iloc[-1])
            cur = d.iloc[-N_LAST:]  # read-out for seeds still training: mean of their latest N_LAST evals
            rows.append({"method": m, "seed": s, "done_9M": f6 is not None, "latest_step": latest,
                         "value": f6["value"] if f6 else float(cur["success"].mean()),
                         "last_evals": f6["last_evals"] if f6 else cur["success"].tolist(),
                         "last_env_steps": f6["last_env_steps"] if f6 else cur["env_steps"].astype(int).tolist()})  # fmt: skip
        done = [r["value"] for r in rows if r["done_9M"]]
        ytop = max([r["value"] for r in rows], default=0.0) + 0.03  # value label sits above bar and dots
        prog = [r for r in rows if not r["done_9M"]]
        if done and not prog:  # solid bar only once every seed has reached the budget
            mu = float(np.mean(done))
            bx.bar(i, mu, width=0.62, color=c, alpha=0.85, lw=0, zorder=2)
            bx.text(i, max(mu + 0.015, ytop), f"{mu:.2f}", color=P.INK, fontsize=6.8, ha="center", va="bottom", zorder=6)
        else:  # some seed still training: hatched outline bar at the mean over all seeds at their latest evals
            mu = float(np.mean([r["value"] for r in rows])) if rows else None
            if mu is not None:
                bx.bar(i, mu, width=0.62, facecolor="none", edgecolor=c, hatch="////", lw=1.0, zorder=2)
                steps = min(r["latest_step"] for r in prog) / 1e6  # the least-trained seed
                bx.text(i, max(mu + 0.015, ytop), f"{mu:.2f}\n@{steps:.1f}M", color=c, fontsize=6.2,
                        ha="center", va="bottom", zorder=6, linespacing=1.0)  # fmt: skip
        jit = np.linspace(-0.17, 0.17, len(rows)) if len(rows) > 1 else [0.0]
        for j, r in zip(jit, rows):
            bx.plot(i + j, r["value"], "o", ms=3.8, mec=P.INK, mew=0.9,
                    mfc=P.INK if r["done_9M"] else "white", zorder=5)  # fmt: skip
        n_txt = f"n={len(done)}" + (f"\n(+{len(prog)})" if prog else "")
        bx.text(i, -0.025, n_txt, transform=tr, fontsize=6.0, color=P.INK2, ha="center", va="top", linespacing=1.0)
        bx.text(i, -0.105, names[i], transform=tr, fontsize=5.7, ha="center", va="top", color=P.INK, linespacing=1.0)
        data["panel_b"].append({"method": m, "mean_9M": float(np.mean(done)) if done else None, "n_seeds_9M": len(done),
                                "in_progress_mean": mu if prog else None, "rows": rows})  # fmt: skip
    bx.set_xticks([])
    bx.set_xlim(-0.6, len(methods) - 0.4)
    bx.set_ylim(0, 1.16)
    bx.set_yticks(np.arange(0, 1.01, 0.2))
    bx.set_ylabel(f"Det. success at 9M, mean of last {N_LAST} evals")
    P._style_ax(bx)
    P.panel_label(bx, "b")
    bl = f"base policy ({base[0]:.3f})" if base else "base policy"
    hs = [Line2D([], [], color=C6[k], lw=2.0 if k not in T6_EXTRA else 1.1, ls=(0, (4, 2)) if k in T6_DASHED else "-",
                 label=LABEL6[k]) for k in ["k0", "k3", "k5", "k8", "k11", "uniform", "adaptive"]]  # fmt: skip
    hs.append(Line2D([], [], color=C["base"], lw=1.3, ls=BASE_LS, label=bl))
    fig.legend(handles=hs, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.2), handlelength=2.4,
               columnspacing=1.4, fontsize=7)  # fmt: skip
    data["data_as_of"] = as_of()
    return P.save(ctx, fig, name, data, check_alignment=False)


def _runs_of(mask):
    """(start, end, value) of the maximal constant stretches of a boolean array."""
    out, lo = [], 0
    for j in range(1, len(mask) + 1):
        if j == len(mask) or mask[j] != mask[lo]:
            out.append((lo, j - 1, bool(mask[lo])))
            lo = j
    return out


FIGURES = {"pi05_curves": pi05_curves, "pi05_final": pi05_final, "pi0_seeds_curves": pi0_seeds_curves,
           "transport_seeds_curves": transport_seeds_curves, "transport_seeds_final": transport_seeds_final,
           "transport_9M": transport_9M}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(P.ROOT, "report", "assets", "figs"))
    p.add_argument("--data_out", default=os.path.join(P.ROOT, "report", "assets", "data"))
    p.add_argument("--only", default=None, help="comma-separated figure names")
    a = p.parse_args()
    ctx = Ctx(a.out, a.data_out)
    status = {n: ("written" if FIGURES[n](ctx) else "skipped") for n in (a.only.split(",") if a.only else FIGURES)}
    print(json.dumps(status, indent=1))
    print("data as of", as_of())
    if TOO_EARLY:
        print(f"left out of *_final (fewer than {N_LAST} evals after step 0):\n  " + "\n  ".join(TOO_EARLY))
    if MISSING:
        print("missing inputs:\n  " + "\n  ".join(sorted(set(MISSING))))


if __name__ == "__main__":
    main()
