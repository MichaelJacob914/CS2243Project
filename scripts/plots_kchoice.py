"""Which denoising step k the adaptive policy picks on transport (K=20 DDIM), from the deterministic eval dumps.

    python scripts/plots_kchoice.py

Inputs: <runs>/adsrl/transport/<run>/eval_rollouts/step_<n>_det.npz (200 episodes x 100 chunks; argmax k per chunk)
  standard noise:          adaptivedetach_rho0.5_s{1,2,3}, latest det dump
  equal control strength:  adaptivedetach_rho0.5_eq_s1, latest det dump
Outputs: report/assets/figs/transport_k_choice.{svg,png}, report/assets/data/transport_k_choice.json.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plots_seeds as S  # noqa: E402  (shared rcParams, colors, paths)

P, plt = S.P, S.plt
TR = os.path.join(S.RUNS, "adsrl", "transport")
K = 20
TBINS = 10
STD = "#e8590c"  # adaptive accent used across the report
EQ = "#6a4c93"


def latest_det(run):
    fs = glob.glob(os.path.join(TR, run, "eval_rollouts", "step_*_det.npz"))
    if not fs:
        return None, None
    f = max(fs, key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
    return int(re.search(r"step_(\d+)", f).group(1)), np.load(f)


def summarize(z):
    k, ch = z["k"], z["chunk"]
    n_chunk = int(ch.max()) + 1
    tb = np.minimum(ch * TBINS // n_chunk, TBINS - 1)
    dev = np.linalg.norm(z["action"] - z["base_action"], axis=1)
    return {
        "k_freq": (np.bincount(k, minlength=K) / len(k)).tolist(),
        "mean_k_by_time": [float(k[tb == b].mean()) for b in range(TBINS)],
        "dev_by_k": [float(dev[k == j].mean()) if (k == j).sum() >= 20 else None for j in range(K)],
        "success": float(z["episode_success"][ch == 0].mean()),
    }


def main():
    out_fig = os.path.join(P.ROOT, "report", "assets", "figs")
    out_data = os.path.join(P.ROOT, "report", "assets", "data")
    std = {}
    for s in S.SEEDS_T:
        step, z = latest_det(f"adaptivedetach_rho0.5_s{s}")
        if z is not None:
            std[s] = {"env_steps": step, **summarize(z)}
    eq_step, zeq = latest_det("adaptivedetach_rho0.5_eq_s1")
    eq = {"env_steps": eq_step, **summarize(zeq)} if zeq is not None else None

    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.4), gridspec_kw={"wspace": 0.42, "width_ratios": [1.5, 1, 1]})
    ks = np.arange(K)

    ax = axs[0]
    fr = np.array([v["k_freq"] for v in std.values()])
    ax.bar(ks - 0.2, fr.mean(0), width=0.4, color=STD, alpha=0.85, label="standard noise (3 seeds)")
    for f in fr:
        ax.scatter(ks - 0.2, f, s=4, color=P.INK, lw=0, zorder=3, alpha=0.6)
    if eq:
        ax.bar(ks + 0.2, eq["k_freq"], width=0.4, color=EQ, alpha=0.85, label="equal control strength (seed 1)")
    ax.axhline(1 / K, color=P.INK2, lw=0.6, ls=(0, (2, 2)))
    ax.text(K - 0.5, 1 / K + 0.01, "uniform", fontsize=6, ha="right", color=P.INK2)
    ax.set_xticks([0, 3, 5, 8, 10, 15, 19])
    ax.set_xlabel("Chosen step k")
    ax.set_ylabel("Fraction of chunks")
    ax.legend(frameon=False, fontsize=6, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.set_ylim(0, 0.42)

    ax = axs[1]
    t = (np.arange(TBINS) + 0.5) / TBINS
    for v in std.values():
        ax.plot(t, v["mean_k_by_time"], color=STD, lw=0.7, alpha=0.4)
    ax.plot(t, np.mean([v["mean_k_by_time"] for v in std.values()], 0), color=STD, lw=2, marker="o", ms=2.5)
    if eq:
        ax.plot(t, eq["mean_k_by_time"], color=EQ, lw=2, marker="o", ms=2.5)
    ax.set_ylim(0, 19)
    ax.set_xlabel("Fraction of episode")
    ax.set_ylabel("Mean chosen k")

    ax = axs[2]
    for v in std.values():
        d = np.array([np.nan if x is None else x for x in v["dev_by_k"]])
        ax.plot(ks, d, color=STD, lw=0.9, alpha=0.6, marker="o", ms=1.8)
    if eq:
        d = np.array([np.nan if x is None else x for x in eq["dev_by_k"]])
        ax.plot(ks, d, color=EQ, lw=0.9, marker="o", ms=1.8)
    ax.set_xticks([0, 5, 10, 15, 19])
    ax.set_xlabel("Chosen step k")
    ax.set_ylabel("‖a − a_base‖ per chunk")
    ax.set_ylim(bottom=0)

    for a, lab in zip(axs, "abc"):
        P._style_ax(a)
        P.panel_label(a, lab)

    data = {
        "description": "Adaptive k on transport, argmax k per 8-step chunk in the latest deterministic eval "
        "(200 episodes x 100 chunks). (a) fraction of chunks per k; (b) mean chosen k in 10 bins of episode time; "
        "(c) mean L2 distance between the executed and the unsteered base action chunk (112-dim, normalized "
        "action space), grouped by chosen k (bins with <20 chunks dropped).",
        "standard": std,
        "equal_strength": eq,
    }
    os.makedirs(out_data, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(out_fig, f"transport_k_choice.{ext}"), bbox_inches="tight", dpi=200)
    json.dump(data, open(os.path.join(out_data, "transport_k_choice.json"), "w"), indent=1)
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    for s, v in std.items():
        print("std", s, v["env_steps"], "dev_by_k", np.array([np.nan if x is None else x for x in v["dev_by_k"]]))
    if eq:
        print("eq", eq["env_steps"], "dev_by_k", np.array([np.nan if x is None else x for x in eq["dev_by_k"]]))


if __name__ == "__main__":
    main()
