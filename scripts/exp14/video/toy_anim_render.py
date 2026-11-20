"""Render the animated toy figure (report/assets/anim/exp14_toy_window.{mp4,webm} + poster PNG).

Trajectories: results/exp14/anim/toy_anim_data.npz (toy_anim_data.py: real toy model and sampler).
Curves: the Stage 0 gate run shown in the report (results/exp14/stage0_gate2_P64_g7/stage0_curves.npz,
64 clean chunks x 16 redraws). Run with the video env (matplotlib + imageio-ffmpeg):

    /n/netscratch/olveczky_lab/Lab/charleszhang/videnv/bin/python scripts/exp14/video/toy_anim_render.py
"""

from __future__ import annotations

import os
import subprocess

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "report/assets/anim")
FPS = 10
DPI = 160
FIG_W, FIG_H = 9.0, 7.0

UP = ["#d95f02", "#f0a04b"]  # originals that go above the disc
DN = ["#1b6ca8", "#64a6d8"]  # originals that go below
POL_C, CTL_C = "#1b6ca8", "#7a7a7a"


def monotone(y):
    up = np.maximum.accumulate(y)
    lo = np.flip(np.minimum.accumulate(np.flip(y)))
    return 0.5 * (up + lo)


def crossing(x, y, c):
    i = int(np.argmax(y >= c))
    if i == 0:
        return x[0]
    return x[i - 1] + (c - y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1])


def window_steps(R):
    m = monotone(np.asarray(R, float))
    ks = np.arange(len(m), dtype=float)
    return crossing(ks, m, 0.1), crossing(ks, m, 0.9)


def scene(ax, view):
    ax.set_facecolor("white")
    ax.add_patch(plt.Circle((0, 0), 0.5, color="#e6e6e6", zorder=0))
    for r in (-1, 1):
        ax.plot([-1, 0, 1], [0, 0.8 * r, 0], color="#c8c8c8", lw=1, ls="--", zorder=0)
    ax.plot(1, 0, marker="*", color="#2a2a2a", ms=12, zorder=1)
    ax.set_xlim(*view[0])
    ax.set_ylim(*view[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#bbbbbb")


def main():
    d = np.load(os.path.join(ROOT, "results/exp14/anim/toy_anim_data.npz"))
    c = np.load(os.path.join(ROOT, "results/exp14/stage0_gate2_P64_g7/stage0_curves.npz"))
    L = len(d["levels"])
    phases = [
        dict(key="start", pos=(-1.0, 0.0), view=((-1.08, -0.22), (-0.55, 0.55)), Rp=c["start__cluster_label__R"], Rn=c["start__cluster_label__Rn"],
             title="At the start, the point must pick a route: above or below the grey obstacle",
             sel={"policy": [0, 2], "control": [0, 2]}, cols=["#d95f02", "#1b6ca8"],
             ylab="share of the route\nalready decided", count=True),
        dict(key="past", pos=(0.31, 0.55), view=((0.27, 0.8), (0.12, 0.65)), Rp=c["past_obstacle__pc1__R"], Rn=c["past_obstacle__pc1__Rn"],
             title="Later, on the upper route past the disc: no choice is left",
             sel={"policy": [0, 3], "control": [0, 3]}, cols=["#7b3294", "#1b9e77"],
             ylab="share of the chunk's main\ndirection already decided", count=False),
    ]
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1], hspace=0.5, wspace=0.08, left=0.12, right=0.97, top=0.83, bottom=0.1)
    ax_p, ax_c, ax_r = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    frames = []  # (phase index, k, extra) ; extra: 'intro' | 'step' | 'end'
    for pi in range(len(phases)):
        frames += [(pi, 0, "intro")] * 25
        for k in range(L):
            frames += [(pi, k, "step")] * (8 if pi == 0 else 5)
        frames += [(pi, L - 1, "end")] * 40

    def draw(pi, k, extra):
        ph = phases[pi]
        for ax in (ax_p, ax_c, ax_r):
            ax.clear()
        fig.texts.clear()
        fig.text(0.5, 0.955, ph["title"], ha="center", va="center", fontsize=12.5, weight="bold")
        if extra == "intro":
            sub = "Thick lines: two action chunks (8 moves each). Next: re-noise each to step k and let the sampler redraw it 32 times."
        else:
            sub = f"Each chunk re-noised to step k = {k} and redrawn 32 times (thin lines in their chunk's colour: " + \
                ("orange above, blue below)" if ph["count"] else "purple, green)")
        fig.text(0.5, 0.918, sub, ha="center", va="center", fontsize=10, color="#333333")
        for ax, arm, name in ((ax_p, "policy", "Diffusion policy"),
                              (ax_c, "control", "Gaussian control: a Gaussian fitted to the\npolicy's chunks, run through the same sampler")):
            scene(ax, ph["view"])
            ax.set_title(name, fontsize=10.5, pad=4)
            sel = ph["sel"][arm]
            clean = d[f"{ph['key']}__{arm}__clean"][sel]
            cols = ph["cols"]
            for p_ in range(clean.shape[0]):
                ax.plot(clean[p_, :, 0], clean[p_, :, 1], color=cols[p_], lw=7, alpha=0.35, zorder=2, solid_capstyle="round")
            if extra != "intro":
                tr = d[f"{ph['key']}__{arm}__traj"][k][sel]  # (2, M, H+1, 2)
                segs = tr.reshape(-1, *tr.shape[2:])
                lc = np.repeat(np.arange(tr.shape[0]), tr.shape[1])
                order = np.random.default_rng(k).permutation(len(segs))  # interleave colours so neither hides the other
                ax.add_collection(LineCollection(segs[order], colors=[cols[i] for i in lc[order]], lw=1.1, alpha=0.7, zorder=3))
            else:
                for p_ in range(clean.shape[0]):
                    ax.plot(clean[p_, :, 0], clean[p_, :, 1], color=cols[p_], lw=2.6, zorder=4, solid_capstyle="round")
            ax.plot(*ph["pos"], "o", color="#2a2a2a", ms=6, zorder=5)
        # curve panel
        ks = np.arange(L)
        ax = ax_r
        upto = -1 if extra == "intro" else k
        ax.plot(ks[: upto + 1], ph["Rp"][: upto + 1], "o-", color=POL_C, lw=2, ms=5, label="diffusion policy", zorder=3)
        ax.plot(ks[: upto + 1], ph["Rn"][: upto + 1], "s--", color=CTL_C, lw=1.6, ms=4.5, label="Gaussian control", zorder=2)
        if extra != "intro":
            ax.axvline(k, color="#d0d0d0", lw=6, zorder=0)
        if extra == "end":
            for R, col, y0, lab in ((ph["Rp"], POL_C, 1.13, "policy"), (ph["Rn"], CTL_C, 1.36, "control")):
                a, b = window_steps(R)
                ax.annotate("", (a, y0), (b, y0), arrowprops=dict(arrowstyle="<->", color=col, lw=1.8))
                ax.text(a, y0 + 0.035, f"{lab}: from 0.1 to 0.9 within {b - a:.0f} steps" + (" (about the same as the policy)" if lab == "control" and not ph["count"] else ""), color=col, va="bottom", fontsize=9.5)
        ax.set(xlim=(-0.5, 19.5), ylim=(-0.08, 1.5), xticks=range(0, 20, 2), yticks=[0, 0.5, 1])
        ax.set_xlabel("denoising step k that the chunk is re-noised to   (0 = pure noise, 19 = almost clean)", fontsize=10)
        ax.set_ylabel(ph["ylab"], fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="lower right", fontsize=10)
        ax.set_title(("64 chunks x 16 redraws per step.  0: redraws ignore their chunk (about half keep its route by chance).  1: all keep it."
                      if ph["count"] else "64 chunks x 16 redraws per step.  0: redraws ignore their chunk.  1: they all match it."),
                     fontsize=9, color="#444444", loc="left")

    canvas = fig.canvas
    canvas.draw()
    w, h = canvas.get_width_height()
    w2, h2 = w - w % 2, h - h % 2
    os.makedirs(OUT, exist_ok=True)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    raw = os.path.join(OUT, "_frames.rgb")
    with open(raw, "wb") as fh:
        for i, f in enumerate(frames):
            draw(*f)
            canvas.draw()
            img = np.asarray(canvas.buffer_rgba())[:h2, :w2, :3]
            fh.write(img.tobytes())
            if i == 25:  # first frame with redraws (k = 0)
                plt.imsave(os.path.join(OUT, "exp14_toy_window_poster.png"), img)
    for ext, codec in (("mp4", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "slow", "-movflags", "+faststart"]),
                       ("webm", ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "38", "-row-mt", "1"])):  # fmt: skip
        cmd = [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w2}x{h2}", "-r", str(FPS), "-i", raw, *codec,
               os.path.join(OUT, f"exp14_toy_window.{ext}")]  # fmt: skip
        subprocess.run(cmd, check=True, capture_output=True)
    os.remove(raw)
    # stills for review: one per k at the start phase, plus phase ends
    if os.environ.get("STILLS_DIR"):
        sd = os.environ["STILLS_DIR"]
        os.makedirs(sd, exist_ok=True)
        for tag, f in (("a_intro", (0, 0, "intro")), ("b_k00", (0, 0, "step")), ("c_k04", (0, 4, "step")), ("d_k06", (0, 6, "step")),
                       ("e_k09", (0, 9, "step")), ("f_k15", (0, 15, "step")), ("g_start_end", (0, L - 1, "end")),
                       ("h_past_k03", (1, 3, "step")), ("i_past_k10", (1, 10, "step")), ("j_past_end", (1, L - 1, "end"))):  # fmt: skip
            draw(*f)
            fig.savefig(os.path.join(sd, f"{tag}.png"), dpi=100, facecolor="white")
    for ext in ("mp4", "webm"):
        p = os.path.join(OUT, f"exp14_toy_window.{ext}")
        print(p, os.path.getsize(p) / 1e6, "MB")
    print("frames", len(frames), "size", w2, h2)


if __name__ == "__main__":
    main()
