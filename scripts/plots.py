"""Report figures for adaptive-step DSRL (notes/figures_plan.md), regenerated from files on disk only.

    python scripts/plots.py --runs_root logs/adsrl --out report/assets/figs
    python scripts/plots.py --only fig3_authority,fig_base_success

Inputs (wandb only for train_curves, read offline from the .wandb files and cached as train_metrics.csv):
  <runs_root>/<task>/<run>/eval_summary.csv + run_info.json   (train_adsrl.py runs; <run> may be nested,
                                                              e.g. <method>/seed1; also scripts/cem_noise.py runs)
  <runs_root>/<task>/<run>/eval_rollouts/step_<env_steps>[_det].npz
  results/authority/authority_<task>_K<K>_dataset.json         (scripts/authority.py)
  results/base_eval*/<task>_K<K>/result.json                   (scripts/base_eval.py)
  results/base_recheck/{recheck_per_seed,summary}.csv          (overrides base success for its (task, K))
  results/base_eval_wk/<task>.csv, results/authority_wscale/*_clip1_c*.json
  <runs_root>/<task>/<run>/wandb/offline-run-*/*.wandb          (train_curves -> <run>/train_metrics.csv)

Outputs: <out>/<name>.svg and <out>/<name>.png (2x: 192 dpi, so the PNG at half size matches the
SVG's CSS size), and <data_out>/<name>.json with the numbers drawn. Every figure function returns
the list of written paths, or None after printing what it is missing, so the script can be rerun
as results arrive.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = ["can", "square", "transport"]
TASK_LABEL = {"can": "Can", "square": "Square", "transport": "Transport"}
RECOMMENDED_K = {"can": 8, "square": 8, "transport": 20}  # DDIM steps K used for training (exp_1)
PNG_DPI = 192

# ---- one fixed method -> color map for every figure ------------------------------------------------
# Adaptive k is the only saturated accent; DSRL is neutral dark; the other methods are muted.
# Checked with the dataviz validator's checks (Machado-2009 CVD, OKLab dE x100) over ALL pairs of the
# five learning methods: worst CVD dE 11.6, worst normal-vision dE 20.1. Uniform-k gray is below 3:1
# contrast on white, so it is always dashed and direct-labeled.
METHOD_COLORS = {
    "adaptive": "#e8590c",         # saturated accent (ours)
    "adaptive_subset": "#f2a37a",  # same hue family, lighter (ablation only, drawn next to adaptive)
    "dsrl": "#2e2e2c",             # DSRL = k=0, neutral dark
    "best_fixed": "#489090",       # oracle best fixed k, muted teal
    "uniform_k": "#bdbcb5",        # uniform-random k, light gray, dashed
    "residual_sac": "#84486c",     # muted plum
    "base": "#8a8984",             # reference line (dashed)
    "cem": "#a8906a",              # reference line (dash-dot)
    "dppo": "#a3a29c",             # cited reference (dotted)
}
METHOD_LABEL = {
    "adaptive": "Adaptive k (ours)", "adaptive_subset": "Adaptive k, top-3 subset", "dsrl": "DSRL (k=0)",
    "best_fixed": "Best fixed k", "uniform_k": "Uniform-random k", "residual_sac": "Residual SAC",
    "base": "Base policy", "cem": "Constant-noise CEM", "dppo": "DPPO (cited)",
}  # fmt: skip
METHOD_LS = {"uniform_k": (0, (4, 2)), "base": (0, (3, 2)), "cem": (0, (5, 2, 1, 2)), "dppo": (0, (1, 2))}
METHOD_ORDER = ["dsrl", "best_fixed", "uniform_k", "residual_sac", "adaptive"]  # drawn last = on top
# Tasks (only in figures without methods, e.g. Fig 3a): validated all-pairs, CVD dE 9.1, normal 22.9.
TASK_COLORS = {"can": "#2a78d6", "square": "#1baf7a", "transport": "#eda100"}
TASK_MARKERS = {"can": "o", "square": "s", "transport": "D"}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab", "#104281", "#0d366b"]
AdsrlCSV_FIELDS = ["env_steps", "mode", "wall_s", "gpu_s", "success", "reward", "mean_ep_len_success",
                   "mean_a_minus_base", "k_entropy", "pi_k_entropy"]  # train_adsrl.AdsrlCallback.CSV_FIELDS
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})  # fmt: skip


# =====================================================================================
# I/O
# =====================================================================================
class Ctx:
    def __init__(self, runs_root, out, data_out, results, eval_mode="det", qa_dir=None):
        self.runs_root, self.out, self.data_out, self.results = runs_root, out, data_out, results
        self.eval_mode, self.qa_dir = eval_mode, qa_dir
        self._runs = self._runs_all = self._base = self._auth = None

    @property
    def runs_all(self) -> pd.DataFrame:
        """Every run, including the noise-bound sweep (steer.w_scale != 1)."""
        if self._runs_all is None:
            self._runs_all = load_runs(self.runs_root)
        return self._runs_all

    @property
    def runs(self) -> pd.DataFrame:
        """Runs at the standard noise bound (w_scale = 1); the bound sweep only appears in bound_train.
        Also drops the exp13 variants (equal strength, entropy, kick, trust region; plotted by plots_exp13.py), and on a
        task with adaptive runs using the k-entropy fix (k_entropy_bias=detach), the adaptive runs from before the fix,
        so the two versions are not averaged into one curve."""
        if self._runs is None:
            r = self.runs_all
            r = r[np.isclose(r["w_scale"], 1.0)] if "w_scale" in r else r
            if not r.empty:
                r = r[~r["run"].str.contains(EXP13_VARIANT)]
                fixed_tasks = set(r.loc[(r["method"] == "adaptive") & (r["k_entropy_bias"] == "detach"), "task"])
                old = (r["method"] == "adaptive") & (r["k_entropy_bias"] != "detach") & r["task"].isin(fixed_tasks)
                r = r[~old]
            self._runs = r
        return self._runs

    @property
    def base(self) -> pd.DataFrame:
        if self._base is None:
            self._base = load_base_eval(self.results)
        return self._base

    @property
    def authority(self) -> dict:
        if self._auth is None:
            self._auth = load_authority(os.path.join(self.results, "authority"))
        return self._auth


def missing(name, what):
    print(f"[skip] {name}: missing {what}")
    return None


EXP13_VARIANT = r"_(?:eq|ent005|kickzi|tr)_s\d+$"  # exp13 run-name suffixes, kept out of the main-report figures


def method_key(steer: dict) -> str:
    """Method of a train_adsrl.py run from its steer config (fixed k>0 runs are 'fixed')."""
    if steer["mode"] == "residual":
        return "residual_sac"
    if steer["mode"] == "cem":  # scripts/cem_noise.py: one fixed noise per task, drawn as a reference line
        return "cem"
    if steer["mode"] == "fixed":
        return "dsrl" if int(steer["k"]) == 0 else "fixed"
    if steer.get("freeze_k_uniform"):
        return "uniform_k"
    if steer.get("k_subset") is not None:
        return "adaptive_subset"
    return "adaptive"


def load_runs(runs_root: str) -> pd.DataFrame:
    """All eval_summary.csv rows under <runs_root>/<task>/<run>/, with run metadata columns."""
    frames = []
    for csv in sorted(glob.glob(os.path.join(runs_root, "*", "**", "eval_summary.csv"), recursive=True)):
        run_dir = os.path.dirname(csv)
        try:
            cfg = json.load(open(os.path.join(run_dir, "run_info.json")))["cfg"]
            resume = os.path.join(run_dir, "run_info_resume.json")
            if os.path.exists(resume):  # a resumed run may have a raised budget (train.total_timesteps)
                cfg["train"]["total_timesteps"] = json.load(open(resume))["cfg"]["train"]["total_timesteps"]
            df = pd.read_csv(csv)
        except (OSError, KeyError, ValueError) as e:
            print(f"[warn] cannot read {run_dir}: {e}")
            continue
        if df.empty:
            continue
        steer = cfg["steer"]
        task_dir = os.path.join(runs_root, os.path.relpath(run_dir, runs_root).split(os.sep)[0])
        df["task"], df["run"], df["run_dir"] = cfg["env_name"], os.path.relpath(run_dir, task_dir), run_dir
        df["method"] = method_key(steer)
        df["k"] = int(steer["k"]) if steer["mode"] == "fixed" else -1
        df["seed"], df["K"] = int(cfg["seed"]), int(cfg["model"]["ddim_steps"])
        df["rho"] = float(steer.get("rho", float("nan")))
        df["beta"] = float(steer["residual_beta"]) if steer["mode"] == "residual" else float("nan")
        df["total_timesteps"] = int(cfg["train"]["total_timesteps"])
        df["w_scale"] = float(steer.get("w_scale", 1.0) or 1.0)
        df["k_entropy_bias"] = str(steer.get("k_entropy_bias") or "")
        frames.append(df)
    if not frames:
        cols = AdsrlCSV_FIELDS + ["task", "run", "run_dir", "method", "k", "seed", "K", "rho", "beta", "total_timesteps"]
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


def load_base_eval(results: str) -> pd.DataFrame:
    """Base-policy success pooled over every results/base_eval*/<task>_K<K>/result.json (binomial SE)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(results, "base_eval*", "*_K*", "result.json"))):
        r = json.load(open(f))
        rows.append({"task": r["task"], "K": int(r["K"]), "n": int(r["n_episodes"]), "succ": r["success"] * r["n_episodes"],
                     "seed": r["seed"], "src": os.path.relpath(f, ROOT)})  # fmt: skip
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    g = df.groupby(["task", "K"]).agg(n=("n", "sum"), succ=("succ", "sum"), seeds=("seed", list), src=("src", list))
    g = g.reset_index()
    # The larger re-check (results/base_recheck, mode=base) replaces the base_eval estimate for its (task, K).
    rc = os.path.join(results, "base_recheck", "recheck_per_seed.csv")
    if os.path.exists(rc):
        r = pd.read_csv(rc)
        r = r[r["mode"] == "base"].assign(succ=lambda d: d["success"] * d["n"])
        for (t, K), d in r.groupby(["task", "K"]):
            g = g[~((g["task"] == t) & (g["K"] == int(K)))]
            g = pd.concat([g, pd.DataFrame([{"task": t, "K": int(K), "n": int(d["n"].sum()), "succ": d["succ"].sum(),
                                             "seeds": d["seed"].tolist(), "src": [os.path.relpath(rc, ROOT)]}])],
                          ignore_index=True)  # fmt: skip
    g["success"] = g["succ"] / g["n"]
    g["se"] = np.sqrt(g["success"] * (1 - g["success"]) / g["n"])
    return g.reset_index(drop=True)


def base_row(ctx, task):
    """Base success row at the training K for ``task`` (re-check takes precedence), or None."""
    b = ctx.base
    r = b[(b["task"] == task) & (b["K"] == RECOMMENDED_K[task])] if not b.empty else b
    return None if r.empty else r.iloc[0]


def load_recheck_summary(results: str) -> pd.DataFrame:
    """results/base_recheck/summary.csv (can only): base, tanh_k0 (bounded noise at k=0), zero_k1, ..."""
    f = os.path.join(results, "base_recheck", "summary.csv")
    return pd.read_csv(f).set_index("mode") if os.path.exists(f) else pd.DataFrame()


def load_base_wk(results: str) -> pd.DataFrame:
    """No-RL success per (task, k, w_mode) from results/base_eval_wk/<task>.csv (w_mode: zero | gauss)."""
    fs = [os.path.join(results, "base_eval_wk", f"{t}.csv") for t in TASKS]
    fs = [f for f in fs if os.path.exists(f)]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def load_authority(adir: str) -> dict:
    out = {}
    for f in sorted(glob.glob(os.path.join(adir, "authority_*_K*_*.json"))):
        d = json.load(open(f))
        out[(d["task"], int(d["K"]))] = {"std": np.asarray(d["mean_std_per_k"]), "dist": np.asarray(d["dist_to_base_per_k"]),
                                         "n_states": d.get("n_states_used"), "n_w": d.get("n_w"),
                                         "src": os.path.relpath(f, ROOT)}  # fmt: skip
    return out


def eval_rows(runs: pd.DataFrame, mode: str) -> pd.DataFrame:
    return runs[runs["mode"] == mode] if not runs.empty else runs


INTERIM = {}  # task -> earliest latest-eval step among that task's unfinished full-budget runs


def task_cutoffs(runs: pd.DataFrame, mode: str) -> dict:
    """Per task, the env step that every run with the task's largest budget has reached (capped at the budget)."""
    r = eval_rows(runs, mode)
    out = {}
    for t, rt in (r.groupby("task") if not r.empty else []):
        budget = int(rt["total_timesteps"].max())
        full = rt[rt["total_timesteps"] == budget]
        out[t] = min(budget, int(full.groupby("run_dir")["env_steps"].max().min()))
    return out


def final_rows(runs: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Last eval of each run at <= its task's largest budget. Runs with that budget are read at their latest eval
    even if unfinished (INTERIM records the earliest such step, for labels); smaller-budget runs count only if
    they reached their own budget."""
    r = eval_rows(runs, mode)
    if r.empty:
        return r
    cut = task_cutoffs(runs, mode)
    tmax = r.groupby("task")["total_timesteps"].max()
    full = r["total_timesteps"] == r["task"].map(tmax)
    keep = (full & (r["env_steps"] <= r["task"].map(tmax))) | ~full
    last = r[keep].sort_values("env_steps").groupby("run_dir").tail(1)
    ok = (last["total_timesteps"] == last["task"].map(tmax)) | (last["env_steps"] >= last["total_timesteps"])
    for t, c in cut.items():
        if c < 0.98 * tmax[t]:
            INTERIM[t] = c
    return last[ok].copy()


def norm_auc(curve: pd.DataFrame) -> float:
    c = curve.sort_values("env_steps")
    x, y = c["env_steps"].to_numpy(float), c["success"].to_numpy(float)
    if len(x) < 2 or x[-1] <= x[0]:
        return float("nan")
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def steps_to_frac(curve: pd.DataFrame, frac=0.8) -> float:
    """First env_steps at which success >= frac x the run's own best success. The env_steps=0 eval
    (the untrained actor) is excluded, so a run that only declines from init gets the first eval."""
    c = curve[curve["env_steps"] > 0].sort_values("env_steps")
    best = c["success"].max()
    if not best > 0:
        return float("nan")
    hit = c[c["success"] >= frac * best]
    return float(hit["env_steps"].iloc[0])


def mean_se(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), 0
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan"), len(x)


def oracle_fixed_k(runs: pd.DataFrame, task: str, mode: str):
    """k > 0 with the highest mean final success (None if no fixed-k>0 run finished)."""
    f = final_rows(runs, mode)
    f = f[(f["task"] == task) & (f["method"] == "fixed")]
    if f.empty:
        return None
    return int(f.groupby("k")["success"].mean().idxmax())


def best_residual_beta(runs: pd.DataFrame, task: str, mode: str):
    """Residual-SAC beta with the highest mean last-eval success on ``task`` (None if no residual runs)."""
    r = eval_rows(runs, mode)
    r = r[(r["task"] == task) & (r["method"] == "residual_sac")] if not r.empty else r
    if r.empty:
        return None
    last = r.sort_values("env_steps").groupby("run_dir").tail(1)
    return float(last.groupby("beta")["success"].mean().idxmax())


def cem_reference(runs: pd.DataFrame, task: str, mode: str):
    """Mean final success of the constant-noise CEM runs on ``task`` (None if there are none)."""
    r = eval_rows(runs, mode)
    r = r[(r["task"] == task) & (r["method"] == "cem")] if not r.empty else r
    if r.empty:
        return None
    return float(r.sort_values("env_steps").groupby("run_dir").tail(1)["success"].mean())


def curves_by_method(runs: pd.DataFrame, task: str, mode: str) -> dict:
    """{method: DataFrame(env_steps, mean, se, n)} with best fixed k folded in as 'best_fixed'."""
    r = eval_rows(runs, mode)
    r = r[r["task"] == task]
    out = {}
    kbest = oracle_fixed_k(runs, task, mode)
    for m in METHOD_ORDER:
        if m == "best_fixed":
            sub = r[(r["method"] == "fixed") & (r["k"] == kbest)] if kbest is not None else r.iloc[0:0]
        elif m == "adaptive":
            sub = r[(r["method"] == "adaptive") & (r["rho"] == 0.5)]
        elif m == "residual_sac":
            sub = r[(r["method"] == m) & (r["beta"] == best_residual_beta(runs, task, mode))]
        else:
            sub = r[r["method"] == m]
        if sub.empty:
            continue
        g = sub.groupby("env_steps")["success"]
        df = pd.DataFrame({"mean": g.mean(), "se": g.sem(), "n": g.count()}).reset_index()
        out[m] = (df, kbest if m == "best_fixed" else None, sorted(sub["seed"].unique().tolist()))
    return out


# =====================================================================================
# drawing helpers
# =====================================================================================
def _style_ax(ax, xgrid=False):
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    if xgrid:
        ax.grid(True, axis="x", color=GRID, lw=0.6)
    else:
        ax.grid(False, axis="x")
    ax.set_axisbelow(True)


def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom", ha="right", color=INK)


def line_with_band(ax, x, y, se, method, label=None, lw=1.8):
    c = METHOD_COLORS[method]
    ax.plot(x, y, color=c, lw=lw, ls=METHOD_LS.get(method, "-"), label=label, zorder=3 + (method == "adaptive"))
    if se is not None and np.any(np.isfinite(se)):
        se = np.nan_to_num(np.asarray(se, float))
        ax.fill_between(x, np.asarray(y) - se, np.asarray(y) + se, color=c, alpha=0.12, lw=0, zorder=2)


def hline_ref(ax, y, method, label=None):
    ax.axhline(y, color=METHOD_COLORS[method], lw=1.0, ls=METHOD_LS.get(method, "--"), zorder=1, label=label)


def kfmt_steps(ax, axis="x"):
    getattr(ax, f"{axis}axis").set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k"))


def save(ctx: Ctx, fig, name: str, data: dict, check_alignment=True):
    os.makedirs(ctx.out, exist_ok=True)
    os.makedirs(ctx.data_out, exist_ok=True)
    if ctx.qa_dir:
        qa(ctx, fig, name, check_alignment)
    svg, png = os.path.join(ctx.out, f"{name}.svg"), os.path.join(ctx.out, f"{name}.png")
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.04, metadata={"Date": None})
    fig.savefig(png, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.04)
    js = os.path.join(ctx.data_out, f"{name}.json")
    with open(js, "w") as f:
        json.dump({"figure": name, **data}, f, indent=1, default=_json_default)
    plt.close(fig)
    print(f"[ok] {name}: {svg}, {png}, {js}")
    return [svg, png, js]


def qa(ctx, fig, name, check_alignment):
    """Optional QA (--qa_dir): PDF for the text/collision audits and, for multi-panel figures, the
    render-time panel-alignment gate from the nature-figure skill (path in $NATURE_FIGURE_SCRIPTS)."""
    os.makedirs(ctx.qa_dir, exist_ok=True)
    fig.savefig(os.path.join(ctx.qa_dir, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.04)
    tools = os.environ.get("NATURE_FIGURE_SCRIPTS")
    if check_alignment and tools and len(fig.axes) > 1:
        sys.path.insert(0, tools)
        from audit_panel_alignment import require_matplotlib_panel_alignment

        fig.canvas.draw()
        try:
            rep = require_matplotlib_panel_alignment(fig, json_out=os.path.join(ctx.qa_dir, f"{name}.alignment.json"),
                                                     tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=False)  # fmt: skip
            print(f"[qa] {name} alignment: {rep.get('status', rep.get('verdict'))}")
        except SystemExit as e:  # the gate exits non-zero on FIX BEFORE DELIVERY
            print(f"[qa] {name} alignment gate failed: {e}")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _clean(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v


# =====================================================================================
# figures available from exp_1 (base eval + authority)
# =====================================================================================
def fig_base_success(ctx: Ctx):
    """Base-policy success per task at the DDIM K used for training, +-1 binomial SE."""
    name = "fig_base_success"
    b = ctx.base
    if b.empty:
        return missing(name, "results/base_eval*/<task>_K*/result.json")
    rows = []
    for t in TASKS:
        r = b[(b["task"] == t) & (b["K"] == RECOMMENDED_K[t])]
        if not r.empty:
            r = r.iloc[0]
            rows.append({"task": t, "K": int(r.K), "success": r.success, "se": r.se, "n": int(r.n), "sources": r.src})
    if not rows:
        return missing(name, f"base eval at K={RECOMMENDED_K}")
    fig, ax = plt.subplots(figsize=(3.4, 1.55))
    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows):
        ax.errorbar(r["success"], y, xerr=r["se"], fmt="o", ms=6, color=METHOD_COLORS["dsrl"], ecolor=INK2,
                    elinewidth=1.2, capsize=0, mec="white", mew=1.2, zorder=3)  # fmt: skip
        ax.text(r["success"] + r["se"] + 0.015, y, f"{r['success']:.3f} ± {r['se']:.3f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(ys, [f"{TASK_LABEL[r['task']]}  (K={r['K']}, n={r['n']})" for r in rows])
    ax.set_xlim(0, 0.8)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Base-policy success rate")
    ax.grid(False)  # every value is labeled; gridlines would cross the labels
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    data = {"description": "Base policy (deterministic DDIM from x_T ~ N(0, I)) success per task, pooled over eval "
            "seeds; SE is binomial. Same eval protocol as train_adsrl.py. Where results/base_recheck has a (task, K), "
            "its larger sample replaces the base_eval estimate (can: 1500 episodes, 3 seeds).", "rows": rows}  # fmt: skip
    return save(ctx, fig, name, data)


def app_transport_ddim(ctx: Ctx):
    """Appendix: transport base success vs number of DDIM steps K."""
    name = "app_transport_ddim"
    b = ctx.base
    b = b[b["task"] == "transport"] if not b.empty else b
    if b.empty:
        return missing(name, "results/base_eval*/transport_K*/result.json")
    # K must divide the 100-step training schedule; K=15 truncates it (100//15=6 -> t stops at 84).
    excluded = b[100 % b["K"] != 0]
    b = b[100 % b["K"] == 0].sort_values("K")
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(b["K"], b["success"], color=METHOD_COLORS["dsrl"], lw=1.4, zorder=2)
    ax.errorbar(b["K"], b["success"], yerr=b["se"], fmt="o", ms=6, color=METHOD_COLORS["dsrl"], ecolor=INK2,
                elinewidth=1.2, capsize=0, mec="white", mew=1.2, zorder=3)  # fmt: skip
    if 20 in b["K"].values:
        r = b[b["K"] == 20].iloc[0]
        ax.annotate("K=20, used for training", (20, r.success + r.se), textcoords="offset points", xytext=(-4, 16),
                    ha="center", fontsize=7, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7, shrinkA=1, shrinkB=2))  # fmt: skip
    ax.set_xscale("log")
    ax.set_xticks(b["K"].tolist(), [str(int(k)) for k in b["K"]])
    ax.set_xlim(b["K"].min() / 1.35, b["K"].max() * 1.25)  # room for the n labels inside the axes
    Ks = b["K"].tolist()
    for i, r in enumerate(b.itertuples()):
        above = i > 0 and r.K / Ks[i - 1] < 1.5  # crowded neighbor on the left: label above instead of below
        y, dy = (r.success + r.se, 4) if above else (r.success - r.se, -12)
        ax.annotate(f"n={int(r.n)}", (r.K, y), textcoords="offset points", xytext=(0, dy), ha="center",
                    fontsize=6.5, color=INK2)  # fmt: skip
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("DDIM steps K")
    ax.set_ylabel("Base success rate")
    ax.set_ylim(0, max(0.3, float((b["success"] + b["se"]).max()) + 0.05))
    ax.grid(False)  # n labels and the annotation would be crossed by gridlines
    rows = [{"K": int(r.K), "success": r.success, "se": r.se, "n": int(r.n), "eval_seeds": r.seeds, "sources": r.src}
            for _, r in b.iterrows()]  # fmt: skip
    data = {"description": "Transport base-policy success vs DDIM steps (pooled eval seeds, binomial SE).",
            "rows": rows, "excluded": [{"K": int(r.K), "success": r.success, "n": int(r.n),
                                        "reason": "K does not divide 100: DDIM schedule truncated (t stops at 84)"}
                                       for _, r in excluded.iterrows()]}  # fmt: skip
    return save(ctx, fig, name, data)


def fig3_authority(ctx: Ctx, transport_K=20):
    """Fig 3a: control strength (action std over w) and distance from the base vs normalized k, one
    line per task. Fig 3b/c (from trained runs) are drawn by fig3_distance when runs exist."""
    name = "fig3a_authority"
    auth = {t: ctx.authority.get((t, RECOMMENDED_K[t] if t != "transport" else transport_K)) for t in TASKS}
    auth = {t: a for t, a in auth.items() if a is not None}
    if not auth:
        return missing(name, "results/authority/authority_<task>_K<K>_dataset.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"wspace": 0.42})
    rows = []
    for t, a in auth.items():
        K = len(a["std"])
        x = np.arange(K) / (K - 1)
        for ax, key in zip(axes, ["std", "dist"]):
            ax.plot(x, a[key], color=TASK_COLORS[t], lw=1.8, zorder=3, label=f"{TASK_LABEL[t]} (K={K})")
            # markers only at k/(K-1) ~ 0, 1/7, ..., 1 so dense-K lines stay solid and comparable to K=8
            mk = sorted({int(round(q * (K - 1))) for q in np.linspace(0, 1, 8)})
            ax.plot(x[mk], a[key][mk], ls="none", marker=TASK_MARKERS[t], ms=4.5, color=TASK_COLORS[t],
                    mec="white", mew=0.8, zorder=4)  # fmt: skip
        for k in range(K):
            rows.append({"task": t, "K": K, "k": k, "k_norm": k / (K - 1), "action_std": a["std"][k],
                         "dist_to_base": a["dist"][k]})  # fmt: skip
    axes[0].set_ylabel("Action std over w  (mean per dim)")
    axes[1].set_ylabel("‖a − a_base‖  (mean over w)")
    for ax, lab in zip(axes, "ab"):
        ax.set_xlabel("Steering step k / (K − 1)")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(0, None)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0\n(DSRL)", "0.25", "0.5", "0.75", "1\n(last)"])
        _style_ax(ax)
        panel_label(ax, lab)
    handles = [Line2D([], [], color=TASK_COLORS[t], marker=TASK_MARKERS[t], mec="white", lw=1.8,
                      label=f"{TASK_LABEL[t]} (K={len(a['std'])})") for t, a in auth.items()]  # fmt: skip
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.98))
    data = {"description": "Authority diagnostic (scripts/authority.py, dataset states): for each steering step k, "
            "the per-dim std of the final action over w ~ N(0, I) with x_T fixed, and the mean ||a - a_base||. "
            "k is normalized by K-1 so tasks with different K share an axis.",
            "n_states": {t: a["n_states"] for t, a in auth.items()}, "n_w": {t: a["n_w"] for t, a in auth.items()},
            "sources": {t: a["src"] for t, a in auth.items()}, "rows": rows}  # fmt: skip
    return save(ctx, fig, name, data)


# =====================================================================================
# figures that need trained runs
# =====================================================================================
def fig2_sweep(ctx: Ctx):
    """Fixed-k sweep: final success vs k (seed dots, mean +- SE, base dashed) and env steps to 80%
    of the run's best success vs k; one column per task."""
    name = "fig2_sweep"
    runs = ctx.runs
    fx = runs[runs["method"].isin(["dsrl", "fixed"])] if not runs.empty else runs
    fin = final_rows(fx, ctx.eval_mode)
    tasks = [t for t in TASKS if not fin.empty and (fin["task"] == t).any()]
    if not tasks:
        return missing(name, f"finished fixed-k runs (steer.mode=fixed) under {ctx.runs_root}")
    fig, axes = plt.subplots(2, len(tasks), figsize=(2.3 * len(tasks) + 0.4, 4.0), squeeze=False,
                             gridspec_kw={"hspace": 0.45, "wspace": 0.35})  # fmt: skip
    rows = []
    ev = eval_rows(fx, ctx.eval_mode)
    for j, t in enumerate(tasks):
        f = fin[fin["task"] == t]
        K = int(f["K"].iloc[0])
        cutoff = dict(zip(f["run_dir"], f["env_steps"]))
        s80 = {rd: steps_to_frac(ev[(ev["run_dir"] == rd) & (ev["env_steps"] <= cutoff[rd])]) for rd in f["run_dir"]}
        f = f.assign(steps80=f["run_dir"].map(s80))
        for i, col in enumerate(["success", "steps80"]):
            ax = axes[i, j]
            rng = np.random.default_rng(0)
            ax.scatter(f["k"] + rng.uniform(-0.12, 0.12, len(f)), f[col], s=10, color=INK2, alpha=0.55, lw=0, zorder=2)
            g = f.groupby("k")[col]
            ks = np.array(sorted(g.groups))
            m = g.mean().reindex(ks).to_numpy()
            se = g.sem().reindex(ks).fillna(0).to_numpy()
            ax.errorbar(ks, m, yerr=se, color=METHOD_COLORS["dsrl"], lw=1.4, marker="o", ms=5, mec="white", mew=1,
                        elinewidth=1.2, capsize=0, zorder=3)  # fmt: skip
            if col == "success":
                bb = ctx.base[(ctx.base["task"] == t) & (ctx.base["K"] == K)] if not ctx.base.empty else []
                if len(bb):
                    hline_ref(ax, float(bb["success"].iloc[0]), "base")
                    ax.annotate("base", (ks.max(), float(bb["success"].iloc[0])), textcoords="offset points",
                                xytext=(2, 3), fontsize=6.5, color=INK2, ha="right")  # fmt: skip
                ax.set_ylim(0, 1.02)
                ax.set_title(f"{TASK_LABEL[t]} (K={K})" + (f", unfinished runs at {INTERIM[t] / 1e6:.2g}M" if t in INTERIM else ""))
            else:
                kfmt_steps(ax, "y")
                ax.set_ylim(0, None)
                ax.set_xlabel("Steering step k  (0 = DSRL)")
            ax.set_xticks(ks)
            _style_ax(ax)
        for _, r in f.iterrows():
            rows.append({"task": t, "K": K, "k": int(r.k), "seed": int(r.seed), "final_env_steps": int(r.env_steps),
                         "final_success": r.success, "steps_to_80pct_best": _clean(r.steps80), "run": r.run})  # fmt: skip
    axes[0, 0].set_ylabel("Final success rate")
    axes[1, 0].set_ylabel("Env steps to 80% of best")
    panel_label(axes[0, 0], "a")
    panel_label(axes[1, 0], "b")
    data = {"description": f"Fixed-k sweep, eval mode={ctx.eval_mode}. Final = last eval at total_timesteps.",
            "rows": rows}  # fmt: skip
    return save(ctx, fig, name, data)


def fig3_distance(ctx: Ctx):
    """Fig 3b/c: ||a - a_base|| at convergence vs k (from the final eval dump of each fixed-k run), and
    control strength vs final success per (task, k) colored by distance."""
    name = "fig3bc_distance"
    fin = final_rows(ctx.runs, ctx.eval_mode)
    fin = fin[fin["method"].isin(["dsrl", "fixed"])] if not fin.empty else fin
    if fin.empty:
        return missing(name, "finished fixed-k runs with eval_rollouts/*.npz")
    rows = []
    for _, r in fin.iterrows():
        sfx = "_det" if ctx.eval_mode == "det" else ""
        f = os.path.join(r.run_dir, "eval_rollouts", f"step_{int(r.env_steps)}{sfx}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        dist = float(np.linalg.norm(z["action"] - z["base_action"], axis=-1).mean())
        a = ctx.authority.get((r.task, int(r.K)))
        rows.append({"task": r.task, "K": int(r.K), "k": int(r.k), "seed": int(r.seed), "success": r.success,
                     "dist_to_base": dist, "authority_std": None if a is None else float(a["std"][int(r.k)])})  # fmt: skip
    if not rows:
        return missing(name, "final eval_rollouts npz dumps")
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"wspace": 0.42})
    for t in TASKS:
        d = df[df["task"] == t]
        if d.empty:
            continue
        K = int(d["K"].iloc[0])
        g = d.groupby("k")["dist_to_base"].agg(["mean", "sem"]).reset_index()
        axes[0].errorbar(g["k"] / (K - 1), g["mean"], yerr=g["sem"].fillna(0), color=TASK_COLORS[t], lw=1.6,
                         marker=TASK_MARKERS[t], ms=4.5, mec="white", mew=0.8, capsize=0, label=TASK_LABEL[t])  # fmt: skip
        dd = d.dropna(subset=["authority_std"]).groupby("k").agg(x=("authority_std", "first"), y=("success", "mean"),
                                                                 c=("dist_to_base", "mean"))  # fmt: skip
        sc = axes[1].scatter(dd["x"], dd["y"], c=dd["c"], cmap=matplotlib.colors.ListedColormap(SEQ_BLUE[1:]),
                             marker=TASK_MARKERS[t], s=36, edgecolor="white", lw=0.8, zorder=3)  # fmt: skip
    axes[0].set_xlabel("Steering step k / (K − 1)")
    axes[0].set_ylabel("‖a − a_base‖ at convergence")
    axes[0].legend(loc="upper right")
    axes[1].set_xlabel("Control strength (action std over w)")
    axes[1].set_ylabel("Final success (mean over seeds)")
    fig.colorbar(sc, ax=axes[1], label="‖a − a_base‖", shrink=0.85)
    for ax, lab in zip(axes, "bc"):
        _style_ax(ax)
        panel_label(ax, lab)
    data = {"description": "Executed-action distance from the base action at the final eval, per fixed-k run. "
            "The base model's denoising loss on executed actions is not computed yet (needs the model on GPU).",
            "rows": rows}  # fmt: skip
    return save(ctx, fig, name, data, check_alignment=False)


def fig4_curves(ctx: Ctx):
    """Main learning curves: success vs env steps, mean +- SE over seeds, one panel per task."""
    name = "fig4_curves"
    runs = ctx.runs
    tasks = [t for t in TASKS if not runs.empty and (runs["task"] == t).any()]
    if not tasks:
        return missing(name, f"eval_summary.csv files under {ctx.runs_root}")
    fig, axes = plt.subplots(1, len(tasks), figsize=(2.35 * len(tasks) + 0.5, 2.5), squeeze=False,
                             gridspec_kw={"wspace": 0.3})  # fmt: skip
    data_rows, seen = [], set()
    for ax, t in zip(axes[0], tasks):
        cm = curves_by_method(runs, t, ctx.eval_mode)
        for m in METHOD_ORDER:
            if m not in cm:
                continue
            df, kbest, seeds = cm[m]
            lab = METHOD_LABEL[m] + (f" (k={kbest})" if m == "best_fixed" else "")
            line_with_band(ax, df["env_steps"], df["mean"], df["se"], m, label=METHOD_LABEL[m])
            if m == "best_fixed":
                ax.annotate(f"k={kbest}", (df["env_steps"].iloc[-1], df["mean"].iloc[-1]), textcoords="offset points",
                            xytext=(3, 0), fontsize=6.5, color=INK2, va="center")  # fmt: skip
            seen.add(m)
            for _, r in df.iterrows():
                data_rows.append({"task": t, "method": m, "label": lab, "env_steps": int(r.env_steps), "mean": r["mean"],
                                  "se": _clean(r["se"]), "n_seeds": int(r.n), "seeds": seeds})  # fmt: skip
        K = int(runs[runs["task"] == t]["K"].iloc[0])
        bb = ctx.base[(ctx.base["task"] == t) & (ctx.base["K"] == K)] if not ctx.base.empty else []
        if len(bb):
            hline_ref(ax, float(bb["success"].iloc[0]), "base")
            seen.add("base")
        cem = cem_reference(runs, t, ctx.eval_mode)
        if cem is not None:
            hline_ref(ax, cem, "cem")
            seen.add("cem")
        ax.set_title(TASK_LABEL[t])
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Env steps")
        if not any(r["task"] == t and r["env_steps"] > 0 for r in data_rows):  # runs exist but have no eval past step 0 yet
            ax.set_xticks([])
            ax.text(0.5, 0.5, "no evals past step 0 yet", transform=ax.transAxes, ha="center", va="center",
                    fontsize=7, style="italic", color=INK2)  # fmt: skip
        else:
            kfmt_steps(ax)
        _style_ax(ax)
    axes[0, 0].set_ylabel("Success rate")
    handles = [Line2D([], [], color=METHOD_COLORS[m], ls=METHOD_LS.get(m, "-"), lw=1.8, label=METHOD_LABEL[m])
               for m in METHOD_ORDER[::-1] + ["base", "cem"] if m in seen]  # fmt: skip
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), bbox_to_anchor=(0.5, 0.99))
    data = {"description": f"Success vs env steps, mean +- SE over seeds (eval mode={ctx.eval_mode}). Best fixed k is "
            "the k>0 with the highest mean final success (oracle).", "rows": data_rows}  # fmt: skip
    return save(ctx, fig, name, data)


def summary_table(ctx: Ctx) -> pd.DataFrame:
    rows = []
    fin = final_rows(ctx.runs, ctx.eval_mode)
    ev = eval_rows(ctx.runs, ctx.eval_mode)
    cut = task_cutoffs(ctx.runs, ctx.eval_mode)
    for t in TASKS:
        kbest = oracle_fixed_k(ctx.runs, t, ctx.eval_mode)
        f = fin[fin["task"] == t] if not fin.empty else fin
        for _, r in f.iterrows():
            m = r.method
            if m == "fixed":
                if r.k != kbest:
                    continue
                m = "best_fixed"
            if m == "adaptive" and r.rho != 0.5:
                continue
            if m == "residual_sac" and r.beta != best_residual_beta(ctx.runs, t, ctx.eval_mode):
                continue
            # AUC over matched env steps (the task cutoff) so runs still training are not penalized for a short span
            curve = ev[(ev["run_dir"] == r.run_dir) & (ev["env_steps"] <= min(r.env_steps, cut[t]))]
            rows.append({"task": t, "method": m, "seed": int(r.seed), "k": int(r.k), "final_success": r.success,
                         "norm_auc": norm_auc(curve), "gpu_hours": r.gpu_s / 3600, "wall_hours": r.wall_s / 3600,
                         "ep_len_success": r.mean_ep_len_success, "env_steps": int(r.env_steps), "run": r.run})  # fmt: skip
    return pd.DataFrame(rows)


def fig5_summary(ctx: Ctx):
    """(a) final success and normalized AUC per method x task (seed dots + mean); (b) success vs GPU-hours."""
    name = "fig5_summary"
    s = summary_table(ctx)
    if s.empty:
        return missing(name, "finished runs (last eval at total_timesteps)")
    tasks = [t for t in TASKS if (s["task"] == t).any()]
    methods = [m for m in METHOD_ORDER + ["cem"] if (s["method"] == m).any()]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.55})
    for ax, col, xl in zip(axes[:2], ["final_success", "norm_auc"], ["Final success rate", "Normalized AUC"]):
        for i, t in enumerate(tasks):
            for j, m in enumerate(methods):
                d = s[(s["task"] == t) & (s["method"] == m)]
                if d.empty:
                    continue
                y = -(i * (len(methods) + 1) + j)
                ax.scatter(d[col], np.full(len(d), y), s=9, color=METHOD_COLORS[m], alpha=0.55, lw=0)
                ax.scatter(d[col].mean(), y, s=34, color=METHOD_COLORS[m], edgecolor="white", lw=1, zorder=3)
        ticks = [-(i * (len(methods) + 1) + (len(methods) - 1) / 2) for i in range(len(tasks))]
        ax.set_yticks(ticks, [TASK_LABEL[t] + (f"\n(unfinished\nat {INTERIM[t] / 1e6:.2g}M)" if t in INTERIM else "") for t in tasks])
        ax.set_xlim(0, 1.02)
        ax.set_xlabel(xl)
        ax.grid(False, axis="y")
        ax.grid(True, axis="x", color=GRID, lw=0.6)
        ax.tick_params(axis="y", length=0)
    ax = axes[2]
    for m in methods:
        d = s[s["method"] == m]
        ax.scatter(d["gpu_hours"], d["final_success"], s=18, color=METHOD_COLORS[m], edgecolor="white", lw=0.8,
                   marker="o", zorder=3)  # fmt: skip
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("GPU-hours (log)")
    ax.set_ylabel("Final success rate")
    ax.set_ylim(0, 1.02)
    _style_ax(ax)
    for ax, lab in zip(axes, "abc"):
        panel_label(ax, lab)
    handles = [Line2D([], [], ls="none", marker="o", ms=6, mec="white", color=METHOD_COLORS[m], label=METHOD_LABEL[m])
               for m in methods[::-1]]  # fmt: skip
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.98))
    data = {"description": "Per-run final success, normalized AUC (trapezoid over env steps / span) and GPU-hours "
            "(wall time / runs per GPU).", "rows": s.to_dict("records")}  # fmt: skip
    return save(ctx, fig, name, data, check_alignment=False)


def _adaptive_dumps(ctx, det):
    runs = ctx.runs
    a = runs[(runs["method"] == "adaptive")] if not runs.empty else runs
    out = {}
    for rd, g in (a.groupby("run_dir") if not a.empty else []):
        files = glob.glob(os.path.join(rd, "eval_rollouts", "step_*.npz"))
        files = [f for f in files if f.endswith("_det.npz") == det]
        steps = {int(re.search(r"step_(\d+)", f).group(1)): f for f in files}
        if steps:
            out[rd] = (g["task"].iloc[0], int(g["K"].iloc[0]), int(g["seed"].iloc[0]), steps)
    return out


def _is_main_adaptive(run_dir, task):
    """Transport has three adaptive variants; fig6 shows the reported one (k-entropy fix, standard noise)."""
    if task != "transport":
        return True
    return re.fullmatch(r"adaptivedetach_rho0\.5_s\d+", os.path.basename(run_dir.rstrip("/"))) is not None


def fig6_k_usage(ctx: Ctx):
    """(a) frequency of each k in stochastic evals vs env steps (stacked), (b) k vs normalized episode
    time at the final deterministic eval. Phase labels and the filmstrip (c) are not drawn here."""
    name = "fig6_k_usage"
    sto, det = _adaptive_dumps(ctx, det=False), _adaptive_dumps(ctx, det=True)
    if not sto and not det:
        return missing(name, "adaptive runs with eval_rollouts/*.npz")
    tasks = [t for t in TASKS if any(v[0] == t for v in {**sto, **det}.values())]
    fig, axes = plt.subplots(2, len(tasks), figsize=(2.4 * len(tasks) + 0.8, 4.2), squeeze=False,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.35})  # fmt: skip
    rows_a, rows_b = [], []
    for j, t in enumerate(tasks):
        K = None
        hist = {}
        for rd, (tt, K_, _seed, steps) in sto.items():
            if tt != t or not _is_main_adaptive(rd, t):
                continue
            K = K_
            for s, f in sorted(steps.items()):
                k = np.load(f)["k"]
                # transport runs resumed after a node failure eval at steps offset by up to ~0.1M from the other
                # runs; pool on a 256k grid (the transport eval interval), one value per run per grid point
                s = int(round(s / 256_000) * 256_000) if t == "transport" else s
                hist.setdefault(s, {})[rd] = np.bincount(k, minlength=K_) / len(k)
        hist = {s: list(v.values()) for s, v in hist.items()}
        ax = axes[0, j]
        if hist:
            n_max = max(len(v) for v in hist.values())
            xs = np.array(sorted(s for s, v in hist.items() if len(v) == n_max))  # only steps every run reached
            fr = np.stack([np.mean(hist[s], axis=0) for s in xs])  # [steps, K]
            cols = [matplotlib.colors.to_hex(c) for c in plt.get_cmap("Blues")(np.linspace(0.25, 0.95, K))]
            ax.stackplot(xs, fr.T, colors=cols, edgecolor="white", linewidth=0.6)
            ax.grid(False)
            for s, f_ in zip(xs, fr):
                rows_a.append({"task": t, "env_steps": int(s), "k_freq": f_.tolist(), "n_runs": len(hist[s])})
            sm = matplotlib.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(cols),
                                              norm=matplotlib.colors.Normalize(-0.5, K - 0.5))  # fmt: skip
            cb = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.03)
            cb.set_label("k", labelpad=1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Env steps")
        kfmt_steps(ax)
        ax.set_title(TASK_LABEL[t])
        ax = axes[1, j]
        mats = []
        for rd, (tt, K_, _seed, steps) in det.items():
            if tt != t or not _is_main_adaptive(rd, t):
                continue
            K = K_
            z = np.load(steps[max(steps)])
            n_chunk = int(z["chunk"].max()) + 1
            tb = np.minimum((z["chunk"] / n_chunk * 10).astype(int), 9)
            m = np.zeros((K_, 10))
            np.add.at(m, (z["k"], tb), 1)
            mats.append(m / np.maximum(m.sum(0, keepdims=True), 1))
        if mats:
            m = np.mean(mats, axis=0)
            im = ax.imshow(m, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=1, extent=(0, 1, -0.5, K - 0.5))
            fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03, label="P(k | time bin)")
            rows_b.append({"task": t, "K": K, "time_bins": 10, "p_k_given_time": m.tolist(), "n_runs": len(mats)})
        ax.set_xlabel("Normalized episode time")
        ax.grid(False)
    axes[0, 0].set_ylabel("Fraction of chunks")
    axes[1, 0].set_ylabel("Chosen k")
    panel_label(axes[0, 0], "a")
    panel_label(axes[1, 0], "b")
    data = {"description": "(a) k frequencies in the stochastic eval dumps (sampled k), averaged over adaptive "
            "runs (rho=any; transport: only adaptivedetach_rho0.5_s*, at steps all of them reached); (b) P(k | normalized time bin) at each run's final deterministic eval (argmax k). "
            "Phase labels from obs are pending.", "a": rows_a, "b": rows_b}  # fmt: skip
    return save(ctx, fig, name, data, check_alignment=False)


def fig7_ablations(ctx: Ctx):
    """Change in final success relative to adaptive k (rho=0.5, all k) for the ablations that exist."""
    name = "fig7_ablations"
    fin = final_rows(ctx.runs, ctx.eval_mode)
    if fin.empty or not (fin["method"] == "adaptive").any():
        return missing(name, "finished adaptive runs (reference for the deltas)")
    ab = {
        "adaptive_subset": ("k restricted to top-3", fin[fin["method"] == "adaptive_subset"]),
        "rho_0.25": ("ρ = 0.25", fin[(fin["method"] == "adaptive") & (fin["rho"] == 0.25)]),
    }
    ref = fin[(fin["method"] == "adaptive") & (fin["rho"] == 0.5)]
    rows = []
    for key, (lab, d) in ab.items():
        for t in TASKS:
            r = ref[ref["task"] == t]["success"]
            dd = d[d["task"] == t]
            if r.empty or dd.empty:
                continue
            for _, x in dd.iterrows():
                rows.append({"ablation": key, "label": lab, "task": t, "seed": int(x.seed), "delta": x.success - r.mean()})
    if not rows:
        return missing(name, "ablation runs (k_subset, rho=0.25); trust-region and best-of-N are not implemented yet")
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    keys = list(dict.fromkeys(df["ablation"]))
    w = 0.8 / len(TASKS)
    for i, key in enumerate(keys):
        for j, t in enumerate(TASKS):
            d = df[(df["ablation"] == key) & (df["task"] == t)]
            if d.empty:
                continue
            x = i + (j - 1) * w
            ax.bar(x, d["delta"].mean(), width=w * 0.85, color=TASK_COLORS[t], label=TASK_LABEL[t] if i == 0 else None)
            ax.scatter(np.full(len(d), x), d["delta"], s=8, color=INK, zorder=3)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(keys)), [df[df["ablation"] == k]["label"].iloc[0] for k in keys])
    ax.set_ylabel("Δ final success vs adaptive k")
    ax.legend(loc="best")
    _style_ax(ax)
    data = {"description": "Final success minus the mean final success of adaptive k (rho=0.5, all k), per seed.",
            "rows": rows, "pending": ["noise trust region on/off", "best-of-N at eval"]}  # fmt: skip
    return save(ctx, fig, name, data)


# =====================================================================================
# appendix figures that need trained runs
# =====================================================================================
def app_perk_curves(ctx: Ctx):
    """Learning curves for every fixed k (mean over seeds), sequential color by k."""
    name = "app_perk_curves"
    ev = eval_rows(ctx.runs, ctx.eval_mode)
    ev = ev[ev["method"].isin(["dsrl", "fixed"])] if not ev.empty else ev
    tasks = [t for t in TASKS if not ev.empty and (ev["task"] == t).any()]
    if not tasks:
        return missing(name, "fixed-k runs")
    fig, axes = plt.subplots(1, len(tasks), figsize=(2.4 * len(tasks) + 0.6, 2.4), squeeze=False,
                             gridspec_kw={"wspace": 0.45})  # fmt: skip
    rows = []
    for ax, t in zip(axes[0], tasks):
        d = ev[ev["task"] == t]
        K = int(d["K"].iloc[0])
        cols = plt.get_cmap("Blues")(np.linspace(0.3, 1.0, K))
        for k, g in d.groupby("k"):
            c = g.groupby("env_steps")["success"].agg(["mean", "count"]).reset_index()
            ax.plot(c["env_steps"], c["mean"], color=METHOD_COLORS["dsrl"] if k == 0 else cols[k], lw=1.8 if k == 0 else 1.3)
            rows += [{"task": t, "k": int(k), "env_steps": int(r.env_steps), "mean": r["mean"], "n_seeds": int(r["count"])}
                     for _, r in c.iterrows()]  # fmt: skip
        sm = matplotlib.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(cols), norm=matplotlib.colors.Normalize(-0.5, K - 0.5))
        fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.03, label="k (k=0 in black)")
        ax.set_title(TASK_LABEL[t])
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Env steps")
        kfmt_steps(ax)
        _style_ax(ax)
    axes[0, 0].set_ylabel("Success rate")
    return save(ctx, fig, name, {"description": "Per-k learning curves, mean over seeds.", "rows": rows}, check_alignment=False)


def app_k_entropy(ctx: Ctx):
    """Entropy of pi(k) (mean over eval states) and of the chosen-k histogram vs env steps, adaptive runs."""
    name = "app_k_entropy"
    r = ctx.runs
    a = r[r["method"].isin(["adaptive", "adaptive_subset"]) & (r["mode"] == "stoch")] if not r.empty else r
    if a.empty:
        return missing(name, "adaptive runs")
    tasks = [t for t in TASKS if (a["task"] == t).any()]
    fig, axes = plt.subplots(1, len(tasks), figsize=(2.4 * len(tasks) + 0.4, 2.4), squeeze=False, gridspec_kw={"wspace": 0.35})
    rows = []
    for ax, t in zip(axes[0], tasks):
        d = a[(a["task"] == t) & (a["method"] == "adaptive")]
        K = int(d["K"].iloc[0]) if not d.empty else None
        for col, ls in [("pi_k_entropy", "-"), ("k_entropy", (0, (3, 2)))]:
            for rd, g in d.groupby("run_dir"):
                g = g.sort_values("env_steps")
                ax.plot(g["env_steps"], g[col], color=METHOD_COLORS["adaptive"], ls=ls, lw=1.2, alpha=0.85)
                rows += [{"task": t, "run": os.path.basename(rd), "metric": col, "env_steps": int(x), "value": y}
                         for x, y in zip(g["env_steps"], g[col])]  # fmt: skip
        if K:
            ax.axhline(np.log(K), color=INK2, lw=0.8, ls=":")
            ax.annotate("log K", (0, np.log(K)), xycoords=("axes fraction", "data"), xytext=(2, 2),
                        textcoords="offset points", fontsize=6.5, color=INK2)  # fmt: skip
        ax.set_title(TASK_LABEL[t])
        ax.set_xlabel("Env steps")
        ax.set_ylim(0, None)
        kfmt_steps(ax)
        _style_ax(ax)
    axes[0, 0].set_ylabel("Entropy (nats)")
    handles = [Line2D([], [], color=METHOD_COLORS["adaptive"], lw=1.4, label="H[π(k|s)], mean over states"),
               Line2D([], [], color=METHOD_COLORS["adaptive"], lw=1.4, ls=(0, (3, 2)), label="H of chosen-k histogram")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.98))
    data = {"description": "Stochastic evals of adaptive runs. alpha_k is only logged to wandb, so it is not drawn.",
            "rows": rows}  # fmt: skip
    return save(ctx, fig, name, data)


def app_stability(ctx: Ctx, drop=0.2):
    """Per-seed curves per method and the count of runs ending > 20 pts below their peak."""
    name = "app_stability"
    ev = eval_rows(ctx.runs, ctx.eval_mode)
    if ev.empty:
        return missing(name, "runs")
    tasks = [t for t in TASKS if (ev["task"] == t).any()]
    fig, axes = plt.subplots(1, len(tasks), figsize=(2.35 * len(tasks) + 0.5, 2.4), squeeze=False, gridspec_kw={"wspace": 0.3})
    rows, seen = [], set()
    for ax, t in zip(axes[0], tasks):
        kbest = oracle_fixed_k(ctx.runs, t, ctx.eval_mode)
        for rd, g in ev[ev["task"] == t].groupby("run_dir"):
            m = g["method"].iloc[0]
            if m == "fixed":
                if g["k"].iloc[0] != kbest:
                    continue
                m = "best_fixed"
            if m not in METHOD_COLORS:
                continue
            g = g.sort_values("env_steps")
            ax.plot(g["env_steps"], g["success"], color=METHOD_COLORS[m], ls=METHOD_LS.get(m, "-"), lw=1.0, alpha=0.9)
            seen.add(m)
            peak, last = g["success"].max(), g["success"].iloc[-1]
            rows.append({"task": t, "method": m, "run": os.path.basename(rd), "seed": int(g["seed"].iloc[0]),
                         "peak": peak, "final": last, "collapsed": bool(last < peak - drop)})  # fmt: skip
        ax.set_title(TASK_LABEL[t])
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Env steps")
        kfmt_steps(ax)
        _style_ax(ax)
    axes[0, 0].set_ylabel("Success rate (per seed)")
    handles = [Line2D([], [], color=METHOD_COLORS[m], ls=METHOD_LS.get(m, "-"), lw=1.4, label=METHOD_LABEL[m])
               for m in METHOD_ORDER[::-1] if m in seen]  # fmt: skip
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.98))
    counts = pd.DataFrame(rows).groupby(["task", "method"])["collapsed"].agg(["sum", "count"]).reset_index()
    data = {"description": f"Runs whose last eval is more than {drop:.0%} below their peak.", "rows": rows,
            "counts": counts.to_dict("records")}  # fmt: skip
    return save(ctx, fig, name, data)


def app_ep_len(ctx: Ctx):
    """Mean length of successful episodes at the final eval per method x task."""
    name = "app_ep_len"
    s = summary_table(ctx)
    if s.empty:
        return missing(name, "finished runs")
    s = s.dropna(subset=["ep_len_success"])
    tasks = [t for t in TASKS if (s["task"] == t).any()]
    methods = [m for m in METHOD_ORDER if (s["method"] == m).any()]
    fig, axes = plt.subplots(1, len(tasks), figsize=(2.3 * len(tasks) + 0.6, 2.2), squeeze=False, gridspec_kw={"wspace": 0.6})
    for ax, t in zip(axes[0], tasks):
        for j, m in enumerate(methods):
            d = s[(s["task"] == t) & (s["method"] == m)]
            if d.empty:
                continue
            ax.scatter(d["ep_len_success"], np.full(len(d), -j), s=9, color=METHOD_COLORS[m], alpha=0.55, lw=0)
            ax.scatter(d["ep_len_success"].mean(), -j, s=34, color=METHOD_COLORS[m], edgecolor="white", lw=1, zorder=3)
        ax.set_yticks([-j for j in range(len(methods))], [METHOD_LABEL[m] for m in methods] if ax is axes[0, 0] else [])
        ax.set_title(TASK_LABEL[t])
        ax.set_xlabel("Env steps to success")
        ax.grid(False, axis="y")
        ax.grid(True, axis="x", color=GRID, lw=0.6)
        ax.tick_params(axis="y", length=0)
    data = {"description": "mean_ep_len_success at the final eval (successful episodes only).",
            "rows": s[["task", "method", "seed", "ep_len_success", "run"]].to_dict("records")}  # fmt: skip
    return save(ctx, fig, name, data)


# =====================================================================================
# figures that replace the report's tables
# =====================================================================================
WAVE0_ROWS = [  # (label, method, k, rho, beta, color, family gap before)
    ("DSRL (k=0)", "dsrl", 0, None, None, METHOD_COLORS["dsrl"], False),
    *[(f"Fixed k={k}", "fixed", k, None, None, METHOD_COLORS["best_fixed"], False) for k in range(1, 8)],
    ("Adaptive k, ρ=0.5 (ours)", "adaptive", -1, 0.5, None, METHOD_COLORS["adaptive"], True),
    ("Adaptive k, ρ=0.25", "adaptive", -1, 0.25, None, METHOD_COLORS["adaptive_subset"], False),
    ("Uniform-random k", "uniform_k", -1, None, None, METHOD_COLORS["uniform_k"], False),
    ("Residual SAC, β=0.1", "residual_sac", -1, None, 0.1, METHOD_COLORS["residual_sac"], True),
    ("Residual SAC, β=0.3", "residual_sac", -1, None, 0.3, METHOD_COLORS["residual_sac"], False),
    ("Constant-noise CEM", "cem", -1, None, None, METHOD_COLORS["cem"], False),
]


def _select_run(ev, method, k, rho, beta):
    d = ev[ev["method"] == method]
    if method in ("dsrl", "fixed"):
        d = d[d["k"] == k]
    if rho is not None:
        d = d[np.isclose(d["rho"], rho)]
    if beta is not None:
        d = d[np.isclose(d["beta"], beta)]
    return d


def can_results(ctx: Ctx, task="can", n_last=3):
    """Can, one panel: for each fixed step, success with no RL (hollow) -> after RL (filled); then the
    mixed-step methods and the two baselines as single points (seed mean, seeds shown small)."""
    name = "can_results"
    ev = eval_rows(ctx.runs, "det")
    ev = ev[ev["task"] == task] if not ev.empty else ev
    if ev.empty:
        return missing(name, f"{task} det evals")
    wk = load_base_wk(ctx.results)
    wk0 = wk[(wk["task"] == task) & (wk["w_mode"] == "zero")].set_index("k") if not wk.empty else pd.DataFrame()
    rs = load_recheck_summary(ctx.results)
    base = base_row(ctx, task)
    spec = [(f"k={k}" + ("  (DSRL)" if k == 0 else ""), "dsrl" if k == 0 else "fixed", k, None, None, False) for k in range(8)]
    spec += [("Adaptive k (ours)", "adaptive", -1, 0.5, None, True), ("Random k", "uniform_k", -1, None, None, False),
             ("Residual SAC", "residual_sac", -1, None, 0.3, True), ("Fixed-noise search", "cem", -1, None, None, False)]
    rows = []
    for lab, m, k, rho, beta, gap in spec:
        d = _select_run(ev, m, k, rho, beta)
        if d.empty:
            print(f"[warn] {name}: no run for {lab}")
            continue
        per_seed = []
        for (_, rd), g in d.groupby(["seed", "run_dir"]):
            g = g.sort_values("env_steps")
            if m != "cem" and g["env_steps"].max() < g["total_timesteps"].iloc[0]:
                continue  # unfinished run
            per_seed.append(float(g.tail(n_last)["success"].mean()))
        if not per_seed:
            continue
        ref = None
        if m == "dsrl" and "tanh_k0" in rs.index:
            ref = float(rs.loc["tanh_k0", "success"])
        elif m == "fixed" and k in wk0.index:
            ref = float(wk0.loc[k, "success"])
        rows.append({"label": lab, "method": m, "k": k, "gap": gap, "final": float(np.mean(per_seed)),
                     "per_seed": per_seed, "n_seeds": len(per_seed), "no_rl": ref})
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    y, ys = 0.0, []
    for r in rows:
        y -= 1.0 + (0.7 if r["gap"] else 0.0)
        ys.append(y)
    if base is not None:
        ax.axvline(base.success, color=METHOD_COLORS["base"], lw=0.9, ls=(0, (3, 2)), zorder=1)
        ax.text(base.success, 0.05, " base policy", ha="left", va="bottom", fontsize=6.5, color=INK2)
    for yy, r in zip(ys, rows):
        c = METHOD_COLORS["adaptive"] if r["method"] == "adaptive" else INK
        if r["no_rl"] is not None:
            ax.plot([r["no_rl"], r["final"]], [yy, yy], color="#c9c8c2", lw=1.6, zorder=2, solid_capstyle="butt")
            ax.plot(r["no_rl"], yy, "o", ms=5, mfc="white", mec=c, mew=1.1, zorder=3)
        if r["n_seeds"] > 1:
            ax.plot(r["per_seed"], [yy] * r["n_seeds"], "o", ms=2.6, color=c, alpha=0.45, mew=0, zorder=3)
        ax.plot(r["final"], yy, "o", ms=5.5, color=c, mec="white", mew=0.8, zorder=4)
        ax.text(max([r["final"]] + r["per_seed"]) + 0.03, yy, f"{r['final']:.2f}", va="center", fontsize=6.5, color=c)
    top = ys[0]
    first = rows[0]
    if first["no_rl"] is not None:
        ax.annotate("no RL", (first["no_rl"], top), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=6.5, color=INK2)
        ax.annotate("after RL", (first["final"], top), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=6.5, color=INK2)
    ax.set_yticks(ys, [r["label"] for r in rows])
    ax.set_ylim(min(ys) - 0.7, 0.8)
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Success rate (can, final)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    data = {"description": f"{task}, deterministic evals of 200 episodes. Final = mean of the last {n_last} evals per "
            "seed, then mean over finished seeds (fixed steps: 1 seed; adaptive rho=0.5, random k, residual beta=0.3: 3 "
            "seeds). No RL: w=0 at that k (results/base_eval_wk) for k>0; bounded noise tanh(N(0,I)) at k=0 "
            "(results/base_recheck). Fixed-noise search (CEM) has one eval at 576k steps.",
            "base": None if base is None else {"success": base.success, "se": base.se, "n": int(base.n)}, "rows": rows}
    return save(ctx, fig, name, data, check_alignment=False)


def norl_per_k(ctx: Ctx):
    """No-RL success vs steering step k, w=0 and w~N(0,I), one panel per task, base line +- SE."""
    name = "norl_per_k"
    wk = load_base_wk(ctx.results)
    if wk.empty:
        return missing(name, "results/base_eval_wk/<task>.csv")
    tasks = [t for t in TASKS if (wk["task"] == t).any()]
    fig, axes = plt.subplots(1, len(tasks), figsize=(7.0, 2.35), sharey=True, gridspec_kw={"wspace": 0.12})
    axes = np.atleast_1d(axes)
    W = {"zero": ("w = 0", INK, "o", "-"), "gauss": ("w ~ N(0, I)", "#a3a29c", "s", (0, (3, 2)))}
    rows = []
    for ax, t in zip(axes, tasks):
        d = wk[wk["task"] == t]
        K = RECOMMENDED_K[t]
        b = base_row(ctx, t)
        if b is not None:
            ax.axhspan(b.success - b.se, b.success + b.se, color=METHOD_COLORS["base"], alpha=0.15, lw=0, zorder=0)
            ax.axhline(b.success, color=METHOD_COLORS["base"], lw=1.0, ls=METHOD_LS["base"], zorder=1)
            ax.text(0.99, b.success - b.se - 0.012, f"base {b.success:.3f}", fontsize=6.5, color=INK2, va="top", ha="right",
                    transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData))  # fmt: skip
        for wm in ["gauss", "zero"]:
            lab, c, mk, ls = W[wm]
            g = d[d["w_mode"] == wm].sort_values("k")
            ax.errorbar(g["k"], g["success"], yerr=g["se"], color=c, ls=ls, lw=1.3 if wm == "zero" else 1.0, marker=mk,
                        ms=4.5 if wm == "zero" else 3.8, mfc=c if wm == "zero" else "white", mec=c, mew=1.0,
                        elinewidth=0.9, capsize=0, zorder=3 if wm == "zero" else 2)  # fmt: skip
            rows += [{"task": t, "K": K, "k": int(r.k), "w_mode": wm, "success": r.success, "se": r.se, "n": int(r.n)}
                     for r in g.itertuples()]  # fmt: skip
        ks = sorted(d["k"].unique())
        ax.set_xticks(ks if K <= 8 else [0, 5, 10, 15, K - 1])
        if K > 8:
            ax.set_xticks(ks, minor=True)
        ax.set_xlim(-0.06 * (K - 1), (K - 1) * 1.3)  # room for the base label at the right
        ax.set_xlabel(f"Steering step k  (K={K})")
        ax.set_title(f"{TASK_LABEL[t]}, n={int(d['n'].iloc[0])} per point", fontsize=8)
        _style_ax(ax)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Success rate, no RL")
    for ax, lab in zip(axes, "abc"):
        panel_label(ax, lab)
    handles = [Line2D([], [], color=W[w][1], ls=W[w][3], marker=W[w][2], mfc=W[w][1] if w == "zero" else "white",
                      mec=W[w][1], label=W[w][0]) for w in ["zero", "gauss"]]  # fmt: skip
    handles.append(Line2D([], [], color=METHOD_COLORS["base"], ls=METHOD_LS["base"], lw=1.0, label="Base policy ± SE"))
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.99))
    data = {"description": "Frozen base policy restarted at step k from its clean-action estimate with noise w (w=0 or "
            "w~N(0,I)), no RL; binomial SE. k=0 with w~N(0,I) is the base policy itself.",
            "base": {t: (lambda b: None if b is None else {"success": b.success, "se": b.se, "n": int(b.n)})(base_row(ctx, t))
                     for t in tasks}, "rows": rows}  # fmt: skip
    return save(ctx, fig, name, data)


def bound_sweep(ctx: Ctx):
    """Control strength (action std over w) vs k for noise bounds c in {1, 1.5, 2, 3}; reference = k=0, c=1."""
    name = "bound_sweep"
    files = sorted(glob.glob(os.path.join(ctx.results, "authority_wscale", "authority_*_clip1_c*.json")))
    if not files:
        return missing(name, "results/authority_wscale/*clip1*.json")
    A = {}
    for f in files:
        d = json.load(open(f))
        A.setdefault(d["task"], {})[float(d["w_scale"])] = (np.asarray(d["mean_std_per_k"]), d, os.path.relpath(f, ROOT))
    tasks = [t for t in TASKS if t in A]
    fig, axes = plt.subplots(1, len(tasks), figsize=(7.0, 2.4), sharey=True, gridspec_kw={"wspace": 0.12})
    axes = np.atleast_1d(axes)
    cs = [1.0, 1.5, 2.0, 3.0]
    ccol = dict(zip(cs, [SEQ_BLUE[2], SEQ_BLUE[3], SEQ_BLUE[5], SEQ_BLUE[7]]))
    rows = []
    for ax, t in zip(axes, tasks):
        ref = float(A[t][1.0][0][0])
        ax.axhline(ref, color=INK2, lw=0.9, ls=(0, (3, 2)), zorder=1)
        for c in cs:
            if c not in A[t]:
                continue
            s, d, src = A[t][c]
            K = len(s)
            ks = np.arange(K)
            ok = s >= ref - 1e-12
            ax.plot(ks, s, color=ccol[c], lw=1.4, zorder=2)
            ax.plot(ks[ok], s[ok], "o", ms=3.8, color=ccol[c], mec=ccol[c], zorder=3)
            ax.plot(ks[~ok], s[~ok], "o", ms=3.8, mfc="white", mec=ccol[c], mew=1.0, zorder=3)
            rows += [{"task": t, "K": K, "k": int(k), "bound_c": c, "action_std": float(v), "ge_reference": bool(o),
                      "n_states": d["n_states_used"], "n_w": d["n_w"], "src": src} for k, v, o in zip(ks, s, ok)]  # fmt: skip
        K = len(A[t][1.0][0])
        ax.set_xticks(list(range(K)) if K <= 8 else [0, 5, 10, 15, K - 1])
        ax.set_xlabel(f"Steering step k  (K={K})")
        ax.set_title(TASK_LABEL[t], fontsize=8)
        ax.set_yscale("log")
        _style_ax(ax)
    axes[0].set_ylabel("Control strength\n(action std over w, log scale)")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    axes[0].set_ylim(0.006, 0.6)
    for ax, lab in zip(axes, "abc"):
        panel_label(ax, lab)
    handles = [Line2D([], [], color=ccol[c], lw=1.4, marker="o", ms=3.8, label=f"bound ±{c:g}") for c in cs]
    handles.append(Line2D([], [], color=INK2, lw=0.9, ls=(0, (3, 2)), label="reference: step 0 at ±1"))
    handles.append(Line2D([], [], ls="none", marker="o", ms=3.8, mfc="white", mec=INK2, label="below the reference"))
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.99))
    data = {"description": "scripts/authority.py on 500 dataset states, 64 noise draws each: w ~ N(0,I) clipped to "
            "[-1, 1], then scaled by c. Control = per-dim std of the final action over w, averaged over states and "
            "dims. Reference = k=0 at c=1 (the current setting).", "rows": rows}  # fmt: skip
    return save(ctx, fig, name, data)


def bound_train(ctx: Ctx, task="can", n_last=3):
    """Noise-bound sweep, trained: final det success vs steering step k for bounds c in {1, 1.5, 2, 3}
    (c=1 = the Wave 0 runs), plus det learning curves per k."""
    name = "bound_train"
    ev = eval_rows(ctx.runs_all, "det")
    ev = ev[(ev["task"] == task) & (ev["seed"] == 1) & (ev["method"].isin(["dsrl", "fixed"]))] if not ev.empty else ev
    sweep_ks = sorted(ev.loc[~np.isclose(ev["w_scale"], 1.0), "k"].unique()) if not ev.empty else []
    if not sweep_ks:
        return missing(name, f"{task} bound-sweep det evals")
    cs = [1.0, 1.5, 2.0, 3.0]
    ccol = dict(zip(cs, [SEQ_BLUE[2], SEQ_BLUE[3], SEQ_BLUE[5], SEQ_BLUE[7]]))
    fig = plt.figure(figsize=(7.0, 2.7))
    gs = fig.add_gridspec(1, 1 + len(sweep_ks), width_ratios=[1.5] + [1] * len(sweep_ks), wspace=0.35)
    ax0 = fig.add_subplot(gs[0, 0])
    rows = []
    for c in cs:
        xs, ys, es = [], [], []
        for k in sweep_ks:
            d = ev[(ev["k"] == k) & np.isclose(ev["w_scale"], c)].sort_values("env_steps")
            if d.empty or d["env_steps"].max() < d["total_timesteps"].iloc[0]:
                continue
            last = d.tail(n_last)
            p = float(last["success"].mean())
            se = float(np.sqrt(p * (1 - p) / (200 * len(last))))
            xs.append(k); ys.append(p); es.append(se)
            rows.append({"task": task, "k": int(k), "bound_c": c, "final_success": p, "se": se,
                         "n_episodes": 200 * len(last), "run": d["run"].iloc[0]})
        if xs:
            off = (cs.index(c) - 1.5) * 0.12
            ax0.errorbar(np.array(xs) + off, ys, yerr=es, color=ccol[c], marker="o", ms=3.8, lw=1.3, capsize=0,
                         elinewidth=0.9, label=f"bound ±{c:g}")
    b = base_row(ctx, task)
    if b is not None:
        ax0.axhline(float(b["success"]), color=METHOD_COLORS["base"], lw=0.9, ls=(0, (3, 2)))
        ax0.text(sweep_ks[-1] + 0.3, float(b["success"]) + 0.015, "base policy", fontsize=6.5, color=INK2, ha="right")
    ax0.set_xticks(sweep_ks)
    ax0.set_xlabel("Steering step k")
    ax0.set_ylabel(f"Final success ({TASK_LABEL[task]}, det.)")
    ax0.set_ylim(-0.02, 1.02)
    ax0.legend(loc="lower left", fontsize=6.5, handlelength=1.2)
    _style_ax(ax0)
    panel_label(ax0, "a")
    for i, k in enumerate(sweep_ks):
        ax = fig.add_subplot(gs[0, 1 + i], sharey=ax0)
        for c in cs:
            d = ev[(ev["k"] == k) & np.isclose(ev["w_scale"], c)].sort_values("env_steps")
            if not d.empty:
                ax.plot(d["env_steps"] / 1e6, d["success"], color=ccol[c], lw=1.1)
        ax.set_title(f"k={k}", fontsize=8)
        ax.set_xlabel("Env steps (M)")
        ax.tick_params(labelleft=False)
        _style_ax(ax)
        if i == 0:
            panel_label(ax, "b")
    data = {"description": f"{task}, seed 1, deterministic eval (200 episodes each). Final = mean of the last "
            f"{n_last} evals; SE is binomial over the pooled episodes. Bound c scales the tanh-bounded noise w.",
            "rows": rows}
    return save(ctx, fig, name, data, check_alignment=False)


# ---- training curves from the offline wandb runs ----------------------------------------------------
TRAIN_KEYS = {"train/ep_rew_mean": "ep_rew_mean", "train/success_rate": "success_rate"}


def _parse_wandb_file(path):
    """History rows with a train key from one offline .wandb file (wandb's own record reader; never syncs)."""
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    ds = datastore.DataStore()
    ds.open_for_scan(path)
    out = []
    while True:
        try:
            data = ds.scan_data()
        except Exception:  # a run still writing can leave a partial record at the end
            break
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") != "history":
            continue
        d = {(i.key or "/".join(i.nested_key)): i.value_json for i in rec.history.item}
        if "env_steps" not in d or not any(k in d for k in TRAIN_KEYS):
            continue
        row = {"env_steps": int(float(d["env_steps"]))}
        for k, col in TRAIN_KEYS.items():
            row[col] = float(json.loads(d[k])) if k in d and d[k] not in ("NaN", "null") else float("nan")
        out.append(row)
    return out


def load_train_metrics(run_dir):
    """train/ep_rew_mean and train/success_rate vs env_steps for one run, cached in <run_dir>/train_metrics.csv
    (re-parsed when any offline .wandb file is newer than the cache)."""
    files = sorted(glob.glob(os.path.join(run_dir, "wandb", "offline-run-*", "*.wandb")))
    cache = os.path.join(run_dir, "train_metrics.csv")
    cols = ["env_steps"] + list(TRAIN_KEYS.values())
    if not files:
        return pd.DataFrame(columns=cols)
    if os.path.exists(cache) and os.path.getmtime(cache) >= max(os.path.getmtime(f) for f in files):
        return pd.read_csv(cache)
    rows = [r for f in files for r in _parse_wandb_file(f)]  # resumed runs: later files win on duplicate steps
    df = pd.DataFrame(rows, columns=cols).drop_duplicates("env_steps", keep="last").sort_values("env_steps")
    df.to_csv(cache, index=False)
    return df


TRAIN_GROUPS = [  # (label, method, k, rho, beta, color, linestyle, lw); fixed k>0 are added per task
    ("DSRL (k=0)", "dsrl", 0, None, None, METHOD_COLORS["dsrl"], "-", 1.6),
    ("Adaptive k, ρ=0.5 (ours)", "adaptive", -1, 0.5, None, METHOD_COLORS["adaptive"], "-", 1.8),
    ("Adaptive k, ρ=0.25", "adaptive", -1, 0.25, None, METHOD_COLORS["adaptive_subset"], "-", 1.4),
    ("Uniform-random k", "uniform_k", -1, None, None, "#9a9993", (0, (4, 2)), 1.4),
    ("Residual SAC, β=0.1", "residual_sac", -1, None, 0.1, METHOD_COLORS["residual_sac"], (0, (1, 1.5)), 1.4),
    ("Residual SAC, β=0.3", "residual_sac", -1, None, 0.3, METHOD_COLORS["residual_sac"], "-", 1.4),
]


def train_curves(ctx: Ctx, min_frac=0.5):
    """Training episode return and training success vs env steps, every method, one column per task."""
    name = "train_curves"
    runs = ctx.runs
    if runs.empty:
        return missing(name, "runs")
    meta = runs.drop_duplicates("run_dir")
    meta = meta[meta["method"] != "cem"]  # CEM has no RL training episodes
    tm = {rd: load_train_metrics(rd) for rd in meta["run_dir"]}
    fig, axes = plt.subplots(2, len(TASKS), figsize=(7.2, 4.3), squeeze=False,
                             gridspec_kw={"hspace": 0.32, "wspace": 0.28})  # fmt: skip
    rows, excluded, seen, status, n_runs = [], [], {}, {}, {}
    for j, t in enumerate(TASKS):
        mt = meta[meta["task"] == t]
        tmax = max([int(tm[rd]["env_steps"].max()) for rd in mt["run_dir"] if len(tm[rd])] or [0])
        rt = runs[(runs["task"] == t) & (runs["method"] != "cem")]
        ev_max = int(rt["env_steps"].max()) if not rt.empty else 0  # eval steps reach total_timesteps; train logs stop just short
        total = int(mt["total_timesteps"].max()) if not mt.empty else 0
        status[t] = {"max_env_steps": tmax, "total_timesteps": total}
        K = RECOMMENDED_K[t]
        fixed_ks = sorted(mt[mt["method"] == "fixed"]["k"].unique())
        kcols = plt.get_cmap("Blues")(np.linspace(0.35, 0.85, max(len(fixed_ks), 2)))
        groups = [(f"Fixed k={k}", "fixed", k, None, None, matplotlib.colors.to_hex(kcols[i]), "-", 0.9)
                  for i, k in enumerate(fixed_ks)] + TRAIN_GROUPS  # fmt: skip
        win = tmax / 25  # rolling mean over 4% of the plotted step range (at least 5 log points)
        for lab, m, k, rho, beta, col, ls, lw in groups:
            g = _select_run(mt, m, k, rho, beta)
            series = [(rd, tm[rd]) for rd in g["run_dir"] if len(tm[rd]) > 1]
            if not series:
                continue
            far = max(s["env_steps"].max() for _, s in series)
            keep = [(rd, s) for rd, s in series if s["env_steps"].max() >= min_frac * far]
            excluded += [{"task": t, "group": lab, "run": os.path.relpath(rd, ctx.runs_root), "max_env_steps": int(s["env_steps"].max()),
                          "reason": f"reached < {min_frac:.0%} of the furthest run in its group"}
                         for rd, s in series if s["env_steps"].max() < min_frac * far]  # fmt: skip
            lo = max(s["env_steps"].min() for _, s in keep)
            hi = min(s["env_steps"].max() for _, s in keep)
            step = float(np.median(np.diff(keep[0][1]["env_steps"])))
            grid = np.arange(lo, hi + 1, step)
            for i, (col_name, ylab) in enumerate([("ep_rew_mean", "ret"), ("success_rate", "succ")]):
                Y = []
                for _, s in keep:
                    sm = s.set_index("env_steps")[col_name].rolling(max(int(round(win / step)), 5), center=True, min_periods=1).mean()
                    Y.append(np.interp(grid, sm.index.to_numpy(float), sm.to_numpy(float)))
                Y = np.array(Y)
                mu = Y.mean(0)
                se = Y.std(0, ddof=1) / math.sqrt(len(Y)) if len(Y) > 1 else None
                line_with_band_c(axes[i, j], grid, mu, se, col, ls, lw, zorder=4 if m == "adaptive" else 3)
                for x, v, e in zip(grid[::5], mu[::5], (se[::5] if se is not None else [None] * len(mu[::5]))):
                    rows.append({"task": t, "group": lab, "metric": col_name, "env_steps": int(x), "mean": float(v),
                                 "se": _clean(None if e is None else float(e)), "n_runs": len(keep)})  # fmt: skip
            seen[lab] = (col, ls, lw if m != "fixed" else 1.2)
            n_runs[f"{t}: {lab}"] = len(keep)
        for i in range(2):
            ax = axes[i, j]
            _style_ax(ax)
            if tmax > 0:
                ax.set_xlim(0, tmax)
                kfmt_steps(ax)
            else:
                ax.set_xticks([])
                ax.set_yticklabels([])
                ax.text(0.5, 0.5, "no training episodes\nlogged yet", transform=ax.transAxes, ha="center", va="center",
                        fontsize=7, color=INK2, style="italic")  # fmt: skip
            if i == 1:
                ax.set_ylim(0, 1.02)
                ax.set_xlabel("Env steps")
        per_run = runs[(runs["task"] == t) & (runs["method"] != "cem")].groupby("run_dir")["env_steps"].max()
        own = runs[(runs["task"] == t) & (runs["method"] != "cem")].groupby("run_dir")["total_timesteps"].max()
        done = total > 0 and len(per_run) > 0 and bool((per_run >= own[per_run.index]).all())  # every run reached its own budget
        status[t]["complete"] = bool(done)
        axes[0, j].set_title(f"{TASK_LABEL[t]}: " + (f"done, {total / 1e6:g}M steps" if done else
                             f"some runs still going ({total / 1e6:g}M budget)"), fontsize=8)  # fmt: skip
    axes[0, 0].set_ylabel("Train episode return")
    axes[1, 0].set_ylabel("Train success rate")
    for j, t in enumerate(TASKS):
        if status[t]["max_env_steps"] == 0 and j > 0:
            axes[0, j].set_ylim(axes[0, j - 1].get_ylim())
    panel_label(axes[0, 0], "a")
    panel_label(axes[1, 0], "b")
    order = [lab for lab in seen if lab.startswith("Fixed")] + [g[0] for g in TRAIN_GROUPS if g[0] in seen]
    handles = [Line2D([], [], color=seen[lab][0], ls=seen[lab][1], lw=seen[lab][2], label=lab) for lab in order]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.91, 0.5), handlelength=2.4)
    data = {"description": "Training-time metrics from the offline wandb runs (train_adsrl.py log_train): "
            "train/ep_rew_mean (mean return of the episodes finished in the log window; reward is -1 per step until "
            "success) and train/success_rate (fraction of training envs that succeeded in the window). Rolling mean over "
            f"2% of each task's plotted range, then mean +- SE over seeds; runs below {min_frac:.0%} of the furthest "
            "seed in their group are excluded. Every 5th grid point is stored here.",
            "status": status, "excluded": excluded, "n_runs": n_runs, "rows": rows}  # fmt: skip
    return save(ctx, fig, name, data)


def line_with_band_c(ax, x, y, se, color, ls, lw, zorder=3):
    ax.plot(x, y, color=color, ls=ls, lw=lw, zorder=zorder)
    if se is not None and np.any(np.isfinite(se)):
        se = np.nan_to_num(np.asarray(se, float))
        ax.fill_between(x, y - se, y + se, color=color, alpha=0.15, lw=0, zorder=2)


FIGURES = {
    "can_results": can_results, "norl_per_k": norl_per_k, "bound_sweep": bound_sweep, "bound_train": bound_train, "train_curves": train_curves,
    "fig_base_success": fig_base_success, "fig2_sweep": fig2_sweep, "fig3_authority": fig3_authority,
    "fig3_distance": fig3_distance, "fig4_curves": fig4_curves, "fig5_summary": fig5_summary,
    "fig6_k_usage": fig6_k_usage, "fig7_ablations": fig7_ablations, "app_transport_ddim": app_transport_ddim,
    "app_perk_curves": app_perk_curves, "app_k_entropy": app_k_entropy, "app_stability": app_stability,
    "app_ep_len": app_ep_len,
}  # fmt: skip


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_root", default=os.path.join(ROOT, "logs", "adsrl"))
    p.add_argument("--out", default=os.path.join(ROOT, "report", "assets", "figs"))
    p.add_argument("--data_out", default=None, help="default: <out>/../data")
    p.add_argument("--results", default=os.path.join(ROOT, "results"))
    p.add_argument("--eval_mode", default="det", choices=["det", "stoch"])
    p.add_argument("--only", default=None, help="comma-separated figure functions")
    p.add_argument("--qa_dir", default=None, help="also write PDFs + alignment reports here")
    args = p.parse_args()
    data_out = args.data_out or os.path.join(os.path.dirname(os.path.abspath(args.out)), "data")
    ctx = Ctx(args.runs_root, args.out, data_out, args.results, args.eval_mode, args.qa_dir)
    names = args.only.split(",") if args.only else list(FIGURES)
    status = {}
    for n in names:
        res = FIGURES[n](ctx)
        status[n] = "written" if res else "skipped"
    print(json.dumps(status, indent=1))


if __name__ == "__main__":
    main()
