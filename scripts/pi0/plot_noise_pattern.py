"""Success vs injection step k per noise pattern (one panel per task), with plain pi0 (band) and w = 0 at k from
the zero-noise baseline for reference. Plots whatever task CSVs exist.

python scripts/pi0/plot_noise_pattern.py   # reads adsrl_runs/exp7/noise_pattern (+ zero_noise_baseline), writes png
"""

import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = "/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs/exp7"
D, Z = os.path.join(R, "noise_pattern"), os.path.join(R, "zero_noise_baseline")
STYLE = {"iid": ("C0", "iid N(0,I) 50x32"), "tiled": ("C1", "tiled N(0,I) row"),
         "agent": ("C3", "fresh agent (tiled)"), "agent_iid": ("C2", "fresh agent marginal, iid")}
files = sorted(glob.glob(os.path.join(D, "task*.csv")))
if files:
    fig, axes = plt.subplots(1, len(files), figsize=(4.4 * len(files), 3.6), sharey=True, squeeze=False)
    for ax, f in zip(axes[0], files, strict=True):
        rows = list(csv.DictReader(open(f)))
        task = rows[0]["task_id"]
        zf = os.path.join(Z, f"task{task}.csv")
        if os.path.exists(zf):
            zr = list(csv.DictReader(open(zf)))
            base = [r for r in zr if r["k"] == "base"]
            if base:
                b, bse = float(base[0]["success"]), float(base[0]["se"])
                ax.axhspan(b - bse, b + bse, color="0.88", lw=0)
                ax.axhline(b, color="0.4", lw=1.2, label=f"plain pi0 ({b:.2f}, n={base[0]['n']})")
            zk = sorted((r for r in zr if r["k"] != "base"), key=lambda r: int(r["k"]))
            ax.plot([int(r["k"]) for r in zk], [float(r["success"]) for r in zk], ":", color="k", lw=1,
                    label="w = 0 at k")
        for pat, (c, lab) in STYLE.items():
            pr = sorted((r for r in rows if r["pattern"] == pat), key=lambda r: int(r["k"]))
            if not pr:
                continue
            x = np.array([int(r["k"]) for r in pr]) + 0.08 * (list(STYLE).index(pat) - 1.5)
            y = np.array([float(r["success"]) for r in pr])
            se = np.array([float(r["se"]) for r in pr])
            ax.errorbar(x, y, yerr=se, fmt="o-", color=c, lw=1.4, ms=4, capsize=2, label=lab)
        ax.set_title(f"LIBERO-10 task {task} (n={rows[0]['n']} / point)")
        ax.set_xlabel("injection step k")
        ax.set_xticks([0, 3, 6, 8])
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("success rate (± SE)")
    axes[0][0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    out = os.path.join(D, "noise_pattern_vs_k.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)
