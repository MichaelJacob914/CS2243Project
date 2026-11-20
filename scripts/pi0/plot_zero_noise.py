"""Success vs injection step k for the zero-noise baseline (one panel per task), with plain pi0 as a line + SE band.

python scripts/pi0/plot_zero_noise.py   # reads/writes /n/netscratch/.../adsrl_runs/exp7/zero_noise_baseline
"""

import csv
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else "/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs/exp7/zero_noise_baseline"
files = sorted(glob.glob(os.path.join(D, "task*.csv")))
fig, axes = plt.subplots(1, len(files), figsize=(4.2 * len(files), 3.4), sharey=True, squeeze=False)
for ax, f in zip(axes[0], files, strict=True):
    rows = list(csv.DictReader(open(f)))
    task = rows[0]["task_id"]
    base = [r for r in rows if r["k"] == "base"]
    ks = sorted((r for r in rows if r["k"] != "base"), key=lambda r: int(r["k"]))
    x = np.array([int(r["k"]) for r in ks])
    y = np.array([float(r["success"]) for r in ks])
    se = np.array([float(r["se"]) for r in ks])
    if base:
        b, bse = float(base[0]["success"]), float(base[0]["se"])
        ax.axhspan(b - bse, b + bse, color="0.85", lw=0)
        ax.axhline(b, color="0.4", lw=1.2, label=f"plain pi0 ({b:.2f})")
    ax.fill_between(x, y - se, y + se, color="C0", alpha=0.2, lw=0)
    ax.plot(x, y, "o-", color="C0", lw=1.5, ms=4, label="w = 0 at step k")
    ax.set_title(f"LIBERO-10 task {task} (n={ks[0]['n'] if ks else '?'} / point)")
    ax.set_xlabel("injection step k (0 = replace x_T)")
    ax.set_xticks(range(10))
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
axes[0][0].set_ylabel("success rate (± SE)")
axes[0][0].legend(loc="lower right", fontsize=8)
fig.tight_layout()
out = os.path.join(D, "zero_noise_vs_k.png")
fig.savefig(out, dpi=150)
print("wrote", out)
