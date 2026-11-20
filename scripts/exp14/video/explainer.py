"""3Blue1Brown-style explainer of the exp14 critical-windows report (manim Community 0.19+).

Data: results/exp14/anim/toy_anim_data.npz (toy redraws, real model and sampler), results/exp14/anim/video_data.json
(per-state widths and midpoints, extract_video_data.py), results/exp14/stage0_gate2_P64_g7/stage0_curves.npz (toy
curves shown in the report), results/exp14/analysis/analysis.json (transport numbers).

Render one scene (720p30):  manim -qm scripts/exp14/video/explainer.py S01Question
Each scene writes its captions with start times to results/exp14/anim/captions/<scene>.json.
"""

from __future__ import annotations

import json
import os
import textwrap

import numpy as np
from manim import (
    DOWN, LEFT, RIGHT, UL, UP, UR, Arrow, Axes, Create, DashedLine, Dot, FadeIn, FadeOut, GrowFromCenter,
    Indicate, Line, Rectangle, RoundedRectangle, Scene, Star, Text, Transform, VGroup, VMobject, WHITE, Write, config, rate_functions,
)  # fmt: skip

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
config.background_color = "#0f1218"
FONT = "Nimbus Sans"

D = np.load(os.path.join(ROOT, "results/exp14/anim/toy_anim_data.npz"))
C = np.load(os.path.join(ROOT, "results/exp14/stage0_gate2_P64_g7/stage0_curves.npz"))
_VP = os.path.join(ROOT, "results/exp14/anim/video_data.json")
V = json.load(open(_VP)) if os.path.exists(_VP) else {}
A = json.load(open(os.path.join(ROOT, "results/exp14/analysis/analysis.json")))
ABAR = 1 / (1 + np.exp(-np.asarray(D["logsnr"], float)))
AMIN, AMAX = np.asarray(D["amin"], float), np.asarray(D["amax"], float)

BLUE = "#58a6ff"
ORANGE = "#ff9f43"
GREY = "#9aa4b2"
GREEN = "#5fd38d"
RED = "#ff6b6b"
YELLOW = "#ffd166"
PURPLE = "#c39bff"
UP_COLS = [ORANGE, "#ffc58a"]
DN_COLS = [BLUE, "#a8cfff"]
PHASE_COLS = ["#4cc9f0", "#80ed99", "#ffd166", "#f78c6b", "#c39bff", "#9aa4b2"]
PHASE_NAMES = ["before contact", "open & grasp", "trash", "handover", "place"]


def T(s, size=28, color="#e8ecf2", **kw):
    return Text(s, font=FONT, font_size=size, color=color, **kw)


def Ax(x_range, y_range, x_length, y_length, xt=None, yt=None, xfmt="{:g}", yfmt="{:g}", size=18, x_ticks=True):
    """Axes with Text tick labels (no LaTeX on this cluster). xt / yt: tick values to label (None = none)."""
    ax = Axes(x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length, tips=False,
              axis_config={"color": "#8b95a5", "stroke_width": 2, "include_ticks": False})  # fmt: skip
    lab = VGroup()
    for v in xt or []:
        if x_ticks:
            lab.add(Line(ax.c2p(v, y_range[0]), ax.c2p(v, y_range[0]) + 0.08 * DOWN, color="#8b95a5", stroke_width=2))
        lab.add(T(xfmt.format(v), size=size, color="#c9d1dc").next_to(ax.c2p(v, y_range[0]), DOWN, buff=0.14))
    for v in yt or []:
        lab.add(Line(ax.c2p(x_range[0], v), ax.c2p(x_range[0], v) + 0.08 * LEFT, color="#8b95a5", stroke_width=2))
        lab.add(T(yfmt.format(v), size=size, color="#c9d1dc").next_to(ax.c2p(x_range[0], v), LEFT, buff=0.14))
    ax.add(lab)
    return ax


def TC(s, size=28, color="#e8ecf2", buff=0.12):
    """Multi-line text with each line centred."""
    return VGroup(*[T(line, size=size, color=color) for line in s.split("\n")]).arrange(DOWN, buff=buff)


def monotone(y):
    y = np.asarray(y, float)
    return 0.5 * (np.maximum.accumulate(y) + np.flip(np.minimum.accumulate(np.flip(y))))


def crossing(y, c):
    i = int(np.argmax(y >= c))
    if i == 0:
        return 0.0
    return i - 1 + (c - y[i - 1]) / (y[i] - y[i - 1])


def window_steps(R):
    m = monotone(R)
    return crossing(m, 0.1), crossing(m, 0.9)


class Base(Scene):
    """Scene with a bottom caption bar; captions are logged with their start time."""

    def setup(self):
        self._cap = None
        self._log = []

    def caption(self, s, wait=None, width=74):
        lines = textwrap.wrap(s, width)
        txt = T("\n".join(lines), size=24, line_spacing=0.9)
        txt.to_edge(DOWN, buff=0.32)
        bg = RoundedRectangle(corner_radius=0.12, width=txt.width + 0.5, height=txt.height + 0.3, fill_color="#000000",
                              fill_opacity=0.55, stroke_width=0).move_to(txt)  # fmt: skip
        new = VGroup(bg, txt)
        self._log.append({"t": round(float(self.renderer.time), 2), "text": s})
        if self._cap is None:
            self.play(FadeIn(new, shift=0.1 * UP), run_time=0.5)
        else:
            self.play(FadeOut(self._cap), FadeIn(new, shift=0.1 * UP), run_time=0.5)
        self._cap = new
        if wait:
            self.wait(wait)

    def clear_caption(self):
        if self._cap is not None:
            self.play(FadeOut(self._cap), run_time=0.4)
            self._cap = None

    def tear_down(self):
        d = os.path.join(ROOT, "results/exp14/anim/captions")
        os.makedirs(d, exist_ok=True)
        json.dump({"scene": type(self).__name__, "duration": float(self.renderer.time), "captions": self._log},
                  open(os.path.join(d, f"{type(self).__name__}.json"), "w"), indent=1)  # fmt: skip

    def fade_all(self, rt=0.6):
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=rt)
        self._cap = None


# ---------------------------------------------------------------------------------------------------------------
# toy scene helpers


def toy_axes(xr=(-1.15, 1.15), yr=(-0.95, 0.95), width=6.4):
    h = width * (yr[1] - yr[0]) / (xr[1] - xr[0])
    ax = Axes(x_range=[xr[0], xr[1], 1], y_range=[yr[0], yr[1], 1], x_length=width, y_length=h,
              tips=False, axis_config={"stroke_opacity": 0, "include_ticks": False})  # fmt: skip
    ax.view = (xr, yr)
    return ax


def _inside(p, v):
    return v[0][0] <= p[0] <= v[0][1] and v[1][0] <= p[1] <= v[1][1]


def clip_path(pts, v):
    """Keep a polyline that starts inside the view up to where it first leaves it (cut at the border)."""
    pts = np.asarray(pts, float)
    if v is None:
        return pts
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        if _inside(b, v):
            out.append(b)
            continue
        lo, hi = 0.0, 1.0
        for _ in range(20):
            m = 0.5 * (lo + hi)
            lo, hi = (m, hi) if _inside(a + m * (b - a), v) else (lo, m)
        out.append(a + lo * (b - a))
        break
    return np.array(out)


def clip_polygon(poly, v):
    """Sutherland-Hodgman clip of a polygon to the view rectangle."""
    (x0, x1), (y0, y1) = v
    edges = [(lambda p: p[0] >= x0, 0, x0), (lambda p: p[0] <= x1, 0, x1), (lambda p: p[1] >= y0, 1, y0), (lambda p: p[1] <= y1, 1, y1)]
    out = list(poly)
    for keep, ax_i, val in edges:
        src, out = out, []
        for i in range(len(src)):
            a, b = np.asarray(src[i - 1]), np.asarray(src[i])
            if keep(b):
                if not keep(a):
                    out.append(a + (val - a[ax_i]) / (b[ax_i] - a[ax_i]) * (b - a))
                out.append(b)
            elif keep(a):
                out.append(a + (val - a[ax_i]) / (b[ax_i] - a[ax_i]) * (b - a))
        if not out:
            return []
    return out


def toy_world(ax, routes=True, frame=True):
    v = getattr(ax, "view", None)
    th = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    circ = np.stack([0.5 * np.cos(th), 0.5 * np.sin(th)], 1)
    poly = clip_polygon(circ, v) if v else list(circ)
    g = VGroup()
    if len(poly) >= 3:
        disc = VMobject(fill_color="#3a4150", fill_opacity=1, stroke_width=0)
        disc.set_points_as_corners([ax.c2p(*p) for p in poly] + [ax.c2p(*poly[0])])
        g.add(disc)
    if routes:
        for s in (1, -1):
            for a, b in (((-1, 0), (0, 0.8 * s)), ((0, 0.8 * s), (1, 0))):
                seg = np.linspace(a, b, 300)
                keep = seg[[(_inside(p, v) if v else True) for p in seg]]
                if len(keep) >= 2:
                    g.add(DashedLine(ax.c2p(*keep[0]), ax.c2p(*keep[-1]), color="#5a6375", stroke_width=2, dash_length=0.08))
    if v is None or _inside((1, 0), v):
        g.add(Star(n=5, outer_radius=0.14, color=YELLOW, fill_opacity=1).move_to(ax.c2p(1, 0)))
    if v is not None and frame:
        g.add(Rectangle(width=ax.x_length, height=ax.y_length, stroke_color="#3a4150", stroke_width=2).move_to(ax.get_center()))
    return g


def path(ax, pts, color, width=2.0, opacity=1.0):
    pts = clip_path(pts, getattr(ax, "view", None))
    m = VMobject(stroke_color=color, stroke_width=width, stroke_opacity=opacity)
    m.set_points_as_corners([ax.c2p(*p) for p in pts])
    return m


def redraw_group(ax, traj, cols, width=1.3, opacity=0.45, seed=0):
    """traj (P, M, H+1, 2) -> VGroup of thin paths, coloured by original chunk, colours interleaved."""
    items = [(p, j) for p in range(traj.shape[0]) for j in range(traj.shape[1])]
    order = np.random.default_rng(seed).permutation(len(items))
    return VGroup(*[path(ax, traj[items[i][0], items[i][1]], cols[items[i][0]], width, opacity) for i in order])


def outcome_group(ax, traj1, width=1.6, opacity=0.6):
    """Redraws of one chunk (M, H+1, 2), coloured by where they end: orange above the axis, blue below."""
    return VGroup(*[path(ax, t, ORANGE if t[-1, 1] > 0 else BLUE, width, opacity) for t in traj1])


def clean_group(ax, clean, cols, width=6):
    return VGroup(*[path(ax, clean[p], cols[p], width) for p in range(len(clean))])


def noisy_traj(clean_pts, k, eps):
    """Clean positions (H+1, 2) -> positions of the chunk noised to DDIM step k (same schedule as the policy)."""
    v = np.diff(clean_pts, axis=0)
    a = 2 * (v - AMIN) / (AMAX - AMIN) - 1
    x = np.sqrt(ABAR[k]) * a + np.sqrt(1 - ABAR[k]) * eps
    vel = (x + 1) / 2 * (AMAX - AMIN) + AMIN
    return np.vstack([clean_pts[:1], clean_pts[0] + np.cumsum(vel, 0)])


def curve_axes(width=5.6, height=3.0, ylab="resolved fraction"):
    ax = Ax([0, 19, 1], [0, 1, 0.5], width, height, xt=[0, 5, 10, 15, 19], yt=[0, 0.5, 1], size=20)
    xl = T("denoising step k  (0 = pure noise, 19 = almost clean)", size=20, color="#c9d1dc").next_to(ax, DOWN, buff=0.12)
    yl = T(ylab, size=20, color="#c9d1dc").rotate(np.pi / 2).next_to(ax, LEFT, buff=0.12)
    return ax, VGroup(xl, yl)


def place_curve_axes(ax, lab, center):
    ax.move_to(center)
    lab[0].next_to(ax, DOWN, buff=0.12)
    lab[1].next_to(ax, LEFT, buff=0.12)


def curve_line(ax, R, color, dashed=False, upto=19):
    pts = [ax.c2p(k, float(np.clip(R[k], 0, 1))) for k in range(upto + 1)]
    m = VMobject(stroke_color=color, stroke_width=4)
    m.set_points_as_corners(pts)
    if dashed:
        from manim import DashedVMobject

        m = DashedVMobject(m, num_dashes=40)
    dots = VGroup(*[Dot(p, radius=0.045, color=color) for p in pts])
    return VGroup(m, dots)


# ---------------------------------------------------------------------------------------------------------------


class S01Question(Base):
    def construct(self):
        title = TC("When does a diffusion policy\nmake up its mind?", size=52, buff=0.2)
        self.play(Write(title), run_time=1.6)
        self.wait(1.2)
        self.play(title.animate.scale(0.55).to_edge(UP, buff=0.3))

        # an episode as a row of chunks
        chunks = VGroup(*[RoundedRectangle(corner_radius=0.08, width=1.5, height=0.7, stroke_color="#8b95a5", fill_color="#1c2230",
                                           fill_opacity=1) for _ in range(6)]).arrange(RIGHT, buff=0.18).shift(1.3 * UP)  # fmt: skip
        labels = VGroup(*[T(f"chunk {i + 1}", size=20).move_to(c) for i, c in enumerate(chunks)])
        ep = T("one robot episode = a sequence of action chunks", size=22, color=GREY).next_to(chunks, UP, buff=0.2)
        self.play(FadeIn(chunks, lag_ratio=0.1), FadeIn(labels, lag_ratio=0.1), FadeIn(ep))
        self.caption("A diffusion policy controls the robot one action chunk at a time.", wait=1.5)

        # 20 denoising steps under chunk 3
        steps = VGroup(*[Rectangle(width=0.42, height=0.42, stroke_width=1.5, stroke_color="#8b95a5", fill_opacity=1,
                                   fill_color=self._noise_col(i)) for i in range(20)]).arrange(RIGHT, buff=0.06).shift(0.5 * DOWN)  # fmt: skip
        nums = VGroup(*[T(str(i), size=14, color=GREY).next_to(s, DOWN, buff=0.08) for i, s in enumerate(steps)])
        l1 = Line(chunks[2].get_bottom(), steps[0].get_top(), color="#5a6375")
        l2 = Line(chunks[2].get_bottom(), steps[-1].get_top(), color="#5a6375")
        lab_n = T("pure noise", size=20, color=GREY).next_to(steps[0], LEFT, buff=0.15)
        lab_a = T("action", size=20, color=GREY).next_to(steps[-1], RIGHT, buff=0.15)
        self.play(Create(l1), Create(l2), FadeIn(steps, lag_ratio=0.05), FadeIn(nums), FadeIn(lab_n), FadeIn(lab_a), run_time=1.5)
        self.caption("Each chunk starts as random noise. The policy removes the noise in 20 denoising steps, numbered 0 to 19.", wait=2.5)

        arrow = Arrow(steps[0].get_bottom() + 1.3 * DOWN, steps[0].get_bottom() + 0.35 * DOWN, color=YELLOW, buff=0, stroke_width=6)
        alab = T("RL steers here", size=22, color=YELLOW).next_to(arrow, DOWN, buff=0.1)
        self.play(GrowFromCenter(arrow), FadeIn(alab))
        self.caption("DSRL trains an RL agent to steer the policy by choosing the noise it starts from, at step 0.", wait=2.5)
        tgt = steps[6].get_bottom()
        self.play(arrow.animate.move_to(tgt + 0.825 * DOWN), alab.animate.next_to(tgt + 1.3 * DOWN, DOWN, buff=0.1), run_time=1.2)
        kl = T("step k", size=22, color=YELLOW).next_to(alab, RIGHT, buff=0.3)
        self.play(FadeIn(kl))
        self.caption("Our variant runs the policy's own denoising up to a later step k, then swaps in the agent's noise there. The adaptive version picks k anew for every chunk.", wait=2.8)

        # different k per chunk
        self.play(*[FadeOut(m) for m in (steps, nums, l1, l2, lab_n, lab_a, arrow, alab, kl)])
        ks = [2, 9, 4, 14, 6, 11]
        bars = VGroup()
        for c, k in zip(chunks, ks):
            b = VGroup(*[Rectangle(width=0.06, height=0.3, stroke_width=0, fill_opacity=1, fill_color=YELLOW if i == k else "#2a3140")
                         for i in range(20)]).arrange(RIGHT, buff=0.01).next_to(c, DOWN, buff=0.15)  # fmt: skip
            bars.add(b)
        q = T("best step to steer at, chunk by chunk?", size=24, color=YELLOW).next_to(bars, DOWN, buff=0.35)
        self.play(FadeIn(bars, lag_ratio=0.15), FadeIn(q))
        self.caption("Picking k per chunk only helps if the best step to steer at changes from chunk to chunk.", wait=2.5)

        idea = TC("Image diffusion models commit to coarse choices\nwithin a narrow band of steps: a critical window.", size=26).shift(0.9 * DOWN)
        self.play(FadeIn(idea, shift=0.2 * UP))
        self.caption("If the policy has such windows and they move between chunks, picking k per chunk makes sense. "
                     "If they sit at one step, a fixed k is enough.", wait=3.5)  # fmt: skip
        self.fade_all()

    @staticmethod
    def _noise_col(i):
        from manim import interpolate_color, ManimColor

        return interpolate_color(ManimColor("#6b7280"), ManimColor("#2f6fb0"), i / 19)


class S02Toy(Base):
    def construct(self):
        ax = toy_axes(width=7.2).shift(0.45 * UP)
        world = toy_world(ax, frame=False)
        start = Dot(ax.c2p(-1, 0), radius=0.08, color=WHITE)
        self.play(FadeIn(world), FadeIn(start))
        self.caption("First, a toy problem with a known choice. A point moves from the white dot to the star, going above or below the disc.", wait=3)
        clean = D["start__policy__clean"]
        cols = UP_COLS + DN_COLS
        cg = clean_group(ax, clean, cols)
        self.play(Create(cg, lag_ratio=0.3), run_time=1.5)
        self.caption("A small diffusion policy, trained on such demos, outputs the next 8 moves as one chunk. "
                     "These are chunks it produced: some go above, some below.", wait=3.5)  # fmt: skip
        # zoom in on the start
        grp = VGroup(ax, world, start, cg)
        p0 = ax.c2p(-1, 0)
        self.play(grp.animate.scale(2.6, about_point=p0).shift(np.array([-3.6, 0.4, 0]) - p0), run_time=1.5)
        self.play(cg[1:].animate.set_stroke(opacity=0.15))
        eps = np.random.default_rng(4).standard_normal((8, 2))
        k_lab = T("step 0", size=30, color=YELLOW).to_corner(UP + RIGHT, buff=0.5)
        noisy = path(ax, noisy_traj(clean[0], 0, eps), ORANGE, 5)
        self.play(Transform(cg[0], noisy), FadeIn(k_lab))
        self.caption("How the policy makes a chunk: at step 0 the chunk is pure noise, and each denoising step removes some of it.", wait=1.0)
        for k in range(1, 20):
            nl = T(f"step {k}", size=30, color=YELLOW).move_to(k_lab)
            self.play(Transform(cg[0], path(ax, noisy_traj(clean[0], k, eps), ORANGE, 5 + 2 * k / 19)), Transform(k_lab, nl),
                      run_time=0.25, rate_func=rate_functions.linear)  # fmt: skip
        self.play(cg[1:].animate.set_stroke(opacity=1))
        self.caption("Somewhere between step 0 and step 19, the policy decides: above or below. The question is where.", wait=2.5)
        self.fade_all()


class S03Measure(Base):
    def construct(self):
        ax = toy_axes(xr=(-1.08, -0.38), yr=(-0.45, 0.45), width=3.6).move_to(4.6 * LEFT + 0.4 * UP)
        world = toy_world(ax)
        start = Dot(ax.c2p(-1, 0), radius=0.07, color=WHITE)
        clean = D["start__policy__clean"]
        traj = D["start__policy__traj"]  # (20, 4, 32, 9, 2)
        cols = UP_COLS + DN_COLS
        orig = path(ax, clean[0], ORANGE, 6)
        self.play(FadeIn(world), FadeIn(start), Create(orig))
        self.caption("How do we find where it decides? Take a finished chunk (here, one that goes above the disc) ...", wait=1.6)

        rec = VGroup(T("1. take a finished chunk", size=26), T("2. add noise back, up to step k", size=26),
                     T("3. let the policy denoise it again, 32 times", size=26), T("4. do the 32 redraws agree?", size=26)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(np.array([-1.6, 2.2, 0]), aligned_edge=UL)  # fmt: skip
        self.play(FadeIn(rec[0]))
        k = 5
        eps = np.random.default_rng(1).standard_normal((8, 2))
        ghost = orig.copy()
        self.play(FadeIn(rec[1]), Transform(ghost, path(ax, noisy_traj(clean[0], k, eps), ORANGE, 3, 0.9)), run_time=1.0)
        self.caption("... add noise back to it, up to some step k (here k = 5), so it is partly undone ...", wait=1.8)
        self.play(FadeOut(ghost), run_time=0.4)
        rd = outcome_group(ax, traj[k][0])
        self.play(FadeIn(rec[2]), Create(rd, lag_ratio=0.02), run_time=1.6)
        self.caption("... and let the policy denoise it again, 32 times, from step k to the end.", wait=1.5)
        self.play(FadeIn(rec[3]))
        same = (np.sign(traj[:, 0, :, -1, 1]) == np.sign(clean[0, -1, 1])).mean(-1)  # (20,)
        cnt = T(f"redraws that also go above: {same[k]:.0%}", size=28, color=ORANGE).next_to(rec, DOWN, buff=0.5, aligned_edge=LEFT)
        self.play(FadeIn(cnt))
        self.caption("Redraws are coloured by where they end: orange above, blue below. If they disagree, the route was still open at step k; if all agree, it was decided.", wait=2.5)

        klab = T(f"k = {k}", size=30, color=YELLOW).next_to(ax, UP, buff=0.15)
        self.play(FadeIn(klab))
        for kk in [0, 2, 4, 6, 8, 10, 13, 16, 19]:
            nl = T(f"k = {kk}", size=30, color=YELLOW).move_to(klab)
            nc = T(f"redraws that also go above: {same[kk]:.0%}", size=28, color=ORANGE).move_to(cnt, aligned_edge=LEFT)
            self.play(Transform(rd, outcome_group(ax, traj[kk][0])), Transform(klab, nl), Transform(cnt, nc), run_time=0.7)
            self.wait(0.5 if kk not in (0, 6, 19) else 1.2)
        self.caption("Noise back to step 0: the redraws split between the routes. Noise back to a late step: they keep the original route. (32 redraws, so the count is noisy.)", wait=2.5)

        # many chunks -> resolved fraction
        self.play(FadeOut(rec), FadeOut(cnt))
        allc = clean_group(ax, clean, cols, 5)
        self.play(Transform(orig, allc), Transform(rd, redraw_group(ax, traj[0], cols, 1.2, 0.4)),
                  Transform(klab, T("k = 0", size=30, color=YELLOW).move_to(klab)))  # fmt: skip
        self.caption("Now do this for many chunks, coloured by chunk (4 shown; the measurement uses 64), and ask how the variation in route splits up.", wait=2.5)
        R = C["start__cluster_label__R"]
        bar_w = 6.4
        bar_origin = np.array([-1.6, 2.3, 0])

        def bar(r):
            r = float(np.clip(r, 0, 1))
            b1 = Rectangle(width=max(bar_w * r, 0.001), height=0.45, fill_color=ORANGE, fill_opacity=1, stroke_width=0)
            b2 = Rectangle(width=max(bar_w * (1 - r), 0.001), height=0.45, fill_color="#4a5466", fill_opacity=1, stroke_width=0)
            b1.move_to(bar_origin, aligned_edge=LEFT)
            b2.next_to(b1, RIGHT, buff=0)
            return VGroup(b1, b2)

        b = bar(R[0])
        bl1 = T("between chunks", size=22, color=ORANGE).next_to(b, UP, buff=0.12).align_to(b, LEFT)
        bl2 = T("within one chunk's redraws", size=22, color=GREY).next_to(b, UP, buff=0.12).align_to(b, RIGHT)
        rv = T(f"resolved fraction = {max(R[0], 0):.2f}", size=28, color=YELLOW).next_to(b, DOWN, buff=0.25).align_to(b, LEFT)
        self.play(FadeIn(b), FadeIn(bl1), FadeIn(bl2), FadeIn(rv))
        self.caption("The resolved fraction is the share of that variation that lies between chunks. "
                     "0: nothing is decided yet. 1: the route is fully decided.", wait=3.5)  # fmt: skip
        cax, clab = curve_axes(width=6.0, height=1.9, ylab="resolved")
        place_curve_axes(cax, clab, np.array([1.9, -0.55, 0]))
        self.play(FadeIn(cax), FadeIn(clab))
        for kk in range(20):
            d = Dot(cax.c2p(kk, float(np.clip(R[kk], 0, 1))), radius=0.06, color=BLUE)
            anims = [Transform(rd, redraw_group(ax, traj[kk], cols, 1.2, 0.4)), Transform(b, bar(R[kk])),
                     Transform(klab, T(f"k = {kk}", size=30, color=YELLOW).move_to(klab)),
                     Transform(rv, T(f"resolved fraction = {max(R[kk], 0):.2f}", size=28, color=YELLOW).move_to(rv, aligned_edge=LEFT)),
                     FadeIn(d, scale=2)]  # fmt: skip
            self.play(*anims, run_time=0.45)
            if kk == 0:
                self.caption("Sweep k from 0 to 19 and plot the resolved fraction at each step.")
        self.play(Create(curve_line(cax, R, BLUE)[0]))
        a_, b_ = window_steps(R)
        self.caption(f"For this policy, the route goes from 10% to 90% decided between steps {a_:.0f} and {b_:.0f}.", wait=3)
        self.fade_all()


class S04Control(Base):
    def construct(self):
        cax, clab = curve_axes(width=6.4, height=3.3)
        place_curve_axes(cax, clab, np.array([2.5, 0.55, 0]))
        R, Rn = C["start__cluster_label__R"], C["start__cluster_label__Rn"]
        pol = curve_line(cax, R, BLUE)
        lp = T("diffusion policy", size=24, color=BLUE).move_to(cax.c2p(0.6, 0.97), aligned_edge=UL)
        self.play(FadeIn(cax), FadeIn(clab), FadeIn(pol), FadeIn(lp))
        self.caption("But the resolved fraction rises from 0 to 1 for any distribution, so a rise alone proves nothing.", wait=2.5)

        ax = toy_axes(xr=(-1.08, -0.3), yr=(-0.62, 0.62), width=3.0).move_to(5.0 * LEFT + 0.5 * UP)
        world = toy_world(ax)
        tr = D["start__control__traj"]
        cl = D["start__control__clean"]
        cols = UP_COLS + DN_COLS
        rg = redraw_group(ax, tr[0], cols, 1.1, 0.4)
        ttl = T("Gaussian control", size=24, color=GREY).next_to(ax, UP, buff=0.12)
        self.play(FadeIn(world), FadeIn(ttl), FadeIn(clean_group(ax, cl, cols, 4)), FadeIn(rg))
        self.caption("So we compare with a control: a Gaussian with the same mean and spread as the policy's chunks, "
                     "run through the same sampler code. It has no two routes to choose between.", wait=4)  # fmt: skip
        ctrl = curve_line(cax, Rn, GREY, dashed=True)
        lc = T("Gaussian control", size=24, color=GREY).next_to(lp, DOWN, buff=0.12).align_to(lp, LEFT)
        klab = T("k = 0", size=26, color=YELLOW).next_to(ax, DOWN, buff=0.15)
        self.add(klab)
        for kk in range(20):
            self.play(Transform(rg, redraw_group(ax, tr[kk], cols, 1.1, 0.4)), Transform(klab, T(f"k = {kk}", size=26, color=YELLOW).move_to(klab)),
                      FadeIn(ctrl[1][kk]), run_time=0.3)  # fmt: skip
        self.play(Create(ctrl[0]), FadeIn(lc))
        self.caption("The control's redraws fan out and settle gradually, and its curve rises slowly.", wait=2.5)
        a1, b1 = window_steps(R)
        a2, b2 = window_steps(Rn)
        up = cax.c2p(0, 1) - cax.c2p(0, 0)
        br1 = Line(cax.c2p(a1, 0) + 1.08 * up, cax.c2p(b1, 0) + 1.08 * up, color=BLUE, stroke_width=6)
        br2 = Line(cax.c2p(a2, 0) + 1.30 * up, cax.c2p(b2, 0) + 1.30 * up, color=GREY, stroke_width=6)
        t1 = T(f"policy: 10% to 90% within {b1 - a1:.0f} steps", size=20, color=BLUE).next_to(br1, UP, buff=0.06).align_to(br1, LEFT)
        t2 = T(f"control: 10% to 90% within {b2 - a2:.0f} steps", size=20, color=GREY).next_to(br2, UP, buff=0.06).align_to(br2, LEFT)
        self.play(Create(br1), Create(br2), FadeIn(t1), FadeIn(t2))
        self.caption(f"A critical window is a rise clearly steeper than the control's. From 10% to 90% decided: about {b1 - a1:.0f} steps "
                     f"for the policy, {b2 - a2:.0f} for the control.", wait=4)  # fmt: skip
        self.fade_all()

        # past the obstacle
        ax2 = toy_axes(xr=(0.27, 0.8), yr=(0.12, 0.65), width=3.2).move_to(4.9 * LEFT + 0.5 * UP)
        w2 = toy_world(ax2)
        P, Pn = C["past_obstacle__pc1__R"], C["past_obstacle__pc1__Rn"]
        cax2, clab2 = curve_axes(width=6.4, height=3.3)
        place_curve_axes(cax2, clab2, np.array([2.5, 0.55, 0]))
        trp = D["past__policy__traj"]
        colsp = ["#c39bff", GREEN, ORANGE, "#ff7eb6"]
        start = Dot(ax2.c2p(0.31, 0.55), radius=0.07, color=WHITE)
        rg2 = redraw_group(ax2, trp[0], colsp, 1.1, 0.4)
        ttl2 = T("policy, past the disc", size=24, color=BLUE).next_to(ax2, UP, buff=0.12)
        self.play(FadeIn(w2), FadeIn(start), FadeIn(rg2), FadeIn(ttl2), FadeIn(cax2), FadeIn(clab2))
        self.caption("Past the disc, no route is left to choose. The curve now tracks the chunk's main direction of variation; policy and control should match.", wait=1.5)
        p1, p2 = curve_line(cax2, P, BLUE), curve_line(cax2, Pn, GREY, dashed=True)
        klab = T("k = 0", size=26, color=YELLOW).next_to(ax2, DOWN, buff=0.15)
        self.add(klab)
        for kk in range(20):
            self.play(Transform(rg2, redraw_group(ax2, trp[kk], colsp, 1.1, 0.4)), Transform(klab, T(f"k = {kk}", size=26, color=YELLOW).move_to(klab)),
                      FadeIn(p1[1][kk]), FadeIn(p2[1][kk]), run_time=0.25)  # fmt: skip
        self.play(Create(p1[0]), Create(p2[0]))
        lg = VGroup(T("policy", size=22, color=BLUE), T("Gaussian control", size=22, color=GREY)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        lg.move_to(cax2.c2p(0.5, 0.95), aligned_edge=UL)
        self.play(FadeIn(lg))
        self.caption("They do: both settle gradually, and the two curves overlap.", wait=2.5)
        self.caption("The method passed this toy check in 8 of 9 repeats with fresh random seeds, all run before any robot data were collected.", wait=3)
        self.fade_all()


class S05Transport(Base):
    def construct(self):
        title = T("The real test: a two-arm transport task", size=40).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.caption("Now the robot. The policy is the frozen diffusion policy for robomimic transport: "
                     "two arms pass an object from one bin to another.", wait=3)  # fmt: skip
        chips = VGroup(*[VGroup(RoundedRectangle(corner_radius=0.15, width=2.05, height=0.6, fill_color=c, fill_opacity=0.25,
                                                 stroke_color=c), T(n, size=20, color=c)) for n, c in zip(PHASE_NAMES, PHASE_COLS)])  # fmt: skip
        for ch in chips:
            ch[1].move_to(ch[0])
        chips.arrange(RIGHT, buff=0.2).shift(2.1 * UP)
        arr = Arrow(chips.get_left() + 0.5 * DOWN, chips.get_right() + 0.5 * DOWN, color="#5a6375", buff=0, stroke_width=3)
        tl = T("task phase", size=20, color=GREY).next_to(arr, DOWN, buff=0.1)
        self.play(FadeIn(chips, lag_ratio=0.15), Create(arr), FadeIn(tl))
        self.caption("We measured 319 robot states, from demos and policy rollouts, labelled by task phase.", wait=2.5)
        props = VGroup(
            T("which group of actions (at the 77 states where actions form 2–3 groups)", size=24, color=RED),
            T("gripper on arm 0: open or closed", size=24, color=BLUE),
            T("gripper on arm 1: open or closed", size=24, color=ORANGE),
            T("critic value: a trained critic's score of the chunk", size=24, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.24).shift(0.85 * DOWN)
        ph = T("At each state, four properties of each chunk:", size=26).next_to(props, UP, buff=0.3).align_to(props, LEFT)
        self.play(FadeIn(ph))
        self.play(FadeIn(props, lag_ratio=0.3), run_time=2)
        self.caption("For each property we measured the resolved-fraction curve of the policy and of its Gaussian control: "
                     "128 chunks × 16 redraws at every step.", wait=3.5)  # fmt: skip
        self.caption("The tests and their thresholds were written down before any robot data were collected.", wait=2.5)
        self.fade_all()


class S06Finding1(Base):
    def construct(self):
        head = T("1. Are there windows?", size=36).to_edge(UP, buff=0.3)
        self.play(FadeIn(head))
        feats = [("cluster_label", "action group", RED), ("grip0", "gripper, arm 0", BLUE), ("grip1", "gripper, arm 1", ORANGE),
                 ("value", "critic value", GREEN)]  # fmt: skip
        r1 = A["rule1"]
        panels = VGroup()
        for f, name, col in feats:
            ax = Ax([0, 10, 5], [0, 10, 5], 2.4, 2.4, xt=[0, 5, 10], yt=[0, 5, 10], size=16)
            diag = DashedLine(ax.c2p(0, 0), ax.c2p(10, 10), color="#6b7280", stroke_width=2)
            ttl = T(name, size=22, color=col).next_to(ax, UP, buff=0.15)
            panels.add(VGroup(ax, diag, ttl))
        panels.arrange(RIGHT, buff=0.6).shift(0.85 * UP)
        xl = T("control's width (in log-SNR, a noise-level scale: step 0 is −5.5, step 19 is +7.4)", size=19, color="#c9d1dc").next_to(panels, DOWN, buff=0.12)
        plegend = VGroup(*[VGroup(Dot(radius=0.05, color=c), T(n, size=16, color=c)).arrange(RIGHT, buff=0.06)
                           for n, c in zip(PHASE_NAMES + ["episode start"], PHASE_COLS)]).arrange(RIGHT, buff=0.25).next_to(head, DOWN, buff=0.12)  # fmt: skip
        yl = T("policy's width", size=20, color="#c9d1dc").rotate(np.pi / 2).next_to(panels, LEFT, buff=0.2)
        self.play(FadeIn(panels), FadeIn(xl), FadeIn(yl))
        self.play(FadeIn(plegend))
        self.caption("Each dot is one state, coloured by task phase: the width of the policy's rise (up) against the control's (across). "
                     "Below the diagonal, the policy decides faster.", wait=3.5)  # fmt: skip
        self.caption("Widths here are measured on the log-SNR noise scale, not in steps: the distance from 10% to 90% decided.", wait=2.5)
        counts = VGroup()
        for (f, name, col), pnl in zip(feats, panels):
            ax = pnl[0]
            rows = np.array([r for r in V["width"][f] if np.isfinite(r[2]) and np.isfinite(r[3])])
            rows = rows[(rows[:, 2] <= 10) & (rows[:, 3] <= 10)]
            dots = VGroup(*[Dot(ax.c2p(r[2], r[3]), radius=0.03, color=PHASE_COLS[int(r[1])], fill_opacity=0.75) for r in rows])
            self.play(FadeIn(dots, lag_ratio=0.004), run_time=1.4)
            c = T(f"{r1[f]['policy_narrower']} of {r1[f]['n']} states\nclearly narrower", size=19, color=col).next_to(pnl, DOWN, buff=0.5)
            counts.add(c)
            self.play(FadeIn(c), run_time=0.4)
        self.caption(f"Action groups: {r1['cluster_label']['policy_narrower']} of 77 states have a clear window. "
                     f"Grippers: {r1['grip0']['policy_narrower']} and {r1['grip1']['policy_narrower']} of 319.", wait=3)  # fmt: skip
        self.caption("'Clearly narrower' means the policy's 95% interval for the width lies wholly below the control's. Gripper dots "
                     "appear only where the control's gripper ever flips.", wait=4)  # fmt: skip
        nn = VGroup(*[T(f"control vs control: {r1[f]['null_vs_null_narrower']}", size=17, color=GREY).next_to(c, DOWN, buff=0.1) for (f, _, _), c in zip(feats, counts)])
        self.play(FadeIn(nn))
        self.caption("Run the same test with a second Gaussian in place of the policy and almost nothing passes "
                     "(0 of 77 for action groups), so these are not false positives.", wait=3.5)  # fmt: skip
        self.play(Indicate(panels[3], color=GREEN, scale_factor=1.05))
        self.caption(f"The critic's value has no window: its dots sit on the diagonal, and only {r1['value']['policy_narrower']} of 319 "
                     "states pass, which does not beat the control-vs-control rate by the 10 points the test required.", wait=3.5)  # fmt: skip
        self.caption("So at many states the policy makes its discrete choices inside a narrow window. The critic's value has none.", wait=2)
        self.fade_all()


class S07Finding2(Base):
    def construct(self):
        head = T("2. Does the window move between phases?", size=36).to_edge(UP, buff=0.3)
        self.play(FadeIn(head))
        feats = [("cluster_label", "action group", RED), ("grip0", "gripper, arm 0", BLUE), ("grip1", "gripper, arm 1", ORANGE),
                 ("value", "critic value", GREEN)]  # fmt: skip
        panels = VGroup()
        for f, name, col in feats:
            ax = Ax([-0.5, 4.5, 1], [0, 19, 5], 2.5, 3.1, yt=[0, 5, 10, 15, 19], size=16)
            ttl = T(name, size=22, color=col).next_to(ax, UP, buff=0.12)
            panels.add(VGroup(ax, ttl))
        panels.arrange(RIGHT, buff=0.55).shift(0.75 * UP)
        yl = T("step where half is decided", size=20, color="#c9d1dc").rotate(np.pi / 2).next_to(panels, LEFT, buff=0.15)
        legend = VGroup(*[VGroup(Dot(radius=0.06, color=c), T(n, size=17, color=c)).arrange(RIGHT, buff=0.08)
                          for n, c in zip(PHASE_NAMES, PHASE_COLS)]).arrange(RIGHT, buff=0.3).next_to(panels, DOWN, buff=0.25)  # fmt: skip
        self.play(FadeIn(panels), FadeIn(yl), FadeIn(legend))
        self.caption("For every state we read off the step where each property is half decided, and group the states by task phase.", wait=2.5)
        rng = np.random.default_rng(0)
        meds = VGroup()
        for (f, name, col), pnl in zip(feats, panels):
            ax = pnl[0]
            rows = V["mid"][f]
            dots = VGroup()
            for p in range(5):
                m = [r[2] for r in rows if r[1] == p]
                xs = p + rng.uniform(-0.25, 0.25, len(m))
                dots.add(*[Dot(ax.c2p(x, y), radius=0.028, color=PHASE_COLS[p], fill_opacity=0.7) for x, y in zip(xs, m)])
                if m:
                    meds.add(Line(ax.c2p(p - 0.35, np.median(m)), ax.c2p(p + 0.35, np.median(m)), color=WHITE, stroke_width=5))
            self.play(FadeIn(dots, lag_ratio=0.003), run_time=1.2)
        self.play(Create(meds, lag_ratio=0.05), run_time=1.2)
        self.caption("White bars are phase medians. The discrete choices sit near step 5–6 in every phase. "
                     "The critic value settles late, near step 16–17, also in every phase.", wait=4)  # fmt: skip
        r2 = A["rule2"]
        gaps = VGroup(*[T(f"largest gap\n{r2[f]['stat_max_pair_diff_steps']:.2f} steps", size=18, color=col).next_to(pnl, DOWN, buff=0.2)
                        for (f, _, col), pnl in zip(feats, panels)])  # fmt: skip
        self.play(FadeOut(legend), FadeIn(gaps))
        self.caption("To count as moving, two phases had to differ by more than one step, with p < 0.01 (both needed). "
                     f"Action groups: {r2['cluster_label']['stat_max_pair_diff_steps']:.2f} steps; gripper arm 1: "
                     f"{r2['grip1']['stat_max_pair_diff_steps']:.2f}.", wait=4)  # fmt: skip
        self.caption(f"Gripper arm 0 shows a {r2['grip0']['stat_max_pair_diff_steps']:.1f}-step gap, but it comes from the place phase, "
                     f"where only 3 states had a curve, and p = {r2['grip0']['perm_p']:.3f} misses the bar.", wait=4)  # fmt: skip
        self.caption("So choosing the step by task phase has nothing to gain. States within a phase still spread out, so section 4 tests per-chunk choice directly.", wait=3)
        self.fade_all()


class S08Finding3(Base):
    def construct(self):
        head = T("3. Does what is still undecided predict the RL gain?", size=34).to_edge(UP, buff=0.3)
        self.play(FadeIn(head))
        ks = [0, 3, 5, 8, 11, 14, 16, 19]
        und = [A["rule3"]["plan"]["open_median"][k] for k in ks]
        gain = [A["rl_gains"][str(k)]["gain"] for k in ks]
        ax = Ax([0, 19, 1], [-0.1, 1.0, 0.5], 9, 4, xt=ks, yt=[0, 0.5, 1], size=20).shift(0.3 * UP)
        ax.add(DashedLine(ax.c2p(0, 0), ax.c2p(19, 0), color="#5a6375", stroke_width=1.5))
        xl = T("steering step k", size=22, color="#c9d1dc").next_to(ax, DOWN, buff=0.12)
        self.play(FadeIn(ax), FadeIn(xl))
        l1 = VGroup(*[Dot(ax.c2p(k, u), radius=0.07, color=GREEN) for k, u in zip(ks, und)])
        c1 = VMobject(stroke_color=GREEN, stroke_width=4).set_points_as_corners([ax.c2p(k, u) for k, u in zip(ks, und)])
        t1 = TC("share of the critic value\nstill undecided at step k", size=22, color=GREEN).move_to(ax.c2p(19, 1.02), aligned_edge=UR)
        self.play(Create(c1), FadeIn(l1), FadeIn(t1), run_time=1.6)
        self.caption("Steering at step k can only change what is still undecided at k. Green: the share of the critic's value still open, median of 319 states.", wait=3.5)  # fmt: skip
        l2 = VGroup(*[Dot(ax.c2p(k, g), radius=0.07, color=YELLOW) for k, g in zip(ks, gain)])
        c2 = VMobject(stroke_color=YELLOW, stroke_width=4).set_points_as_corners([ax.c2p(k, g) for k, g in zip(ks, gain)])
        t2 = TC("RL gain in success when\ntraining at fixed step k", size=22, color=YELLOW).next_to(t1, DOWN, buff=0.2)
        self.play(Create(c2), FadeIn(l2), FadeIn(t2), run_time=1.6)
        self.caption("Yellow: gain in success rate from RL at that fixed step after about 6M env steps (0.6 means +60 points over no RL).", wait=3)
        self.caption(f"The two rank the eight steps almost the same way (Spearman ρ = {A['rule3']['plan']['rho']:.2f}).", wait=2.5)
        late = Rectangle(width=ax.c2p(19.5, 0)[0] - ax.c2p(13, 0)[0], height=4.1, fill_color=RED, fill_opacity=0.12, stroke_width=0)
        late.move_to([(ax.c2p(13, 0)[0] + ax.c2p(19.5, 0)[0]) / 2, ax.get_center()[1], 0])
        early = Rectangle(width=ax.c2p(11.5, 0)[0] - ax.c2p(-0.5, 0)[0], height=4.1, fill_color=BLUE, fill_opacity=0.10, stroke_width=0)
        early.move_to([(ax.c2p(-0.5, 0)[0] + ax.c2p(11.5, 0)[0]) / 2, ax.get_center()[1], 0])
        self.play(FadeIn(late))
        self.caption("But the agreement comes from the late steps. From step 14 on, little is left to change, and the RL gain drops.", wait=3)
        self.play(FadeIn(early))
        self.caption("Among steps 0–11 the RL gain barely changes (0.52 to 0.64). So this explains why late steering fails, but it does not pick a best early step.", wait=3.5)
        self.fade_all()


class S09Finding4(Base):
    def construct(self):
        head = T("4. In the simulator, does one chunk change success?", size=34).to_edge(UP, buff=0.3)
        self.play(FadeIn(head))
        self.caption("So far we trusted the critic. Now real outcomes: at 20 states, regenerate only the first chunk "
                     "from step 4, 9, 12, 15 or 18 (32 noised chunks, 8 redraws each), then let the policy finish: 25,600 episodes.", wait=4)  # fmt: skip
        ks = [4, 9, 12, 15, 18]
        s4 = A["stage4"]["pooled"]
        ax = Ax([0, 19, 1], [0, 1, 0.5], 6.0, 3.4, xt=ks, yt=[0, 0.5, 1], size=20).shift(3.0 * LEFT + 0.2 * UP)
        xl = T("step where the chunk is regenerated", size=20, color="#c9d1dc").next_to(ax, DOWN, buff=0.12)
        yl = T("share of the outcome decided by this chunk", size=20, color="#c9d1dc").rotate(np.pi / 2).next_to(ax, LEFT, buff=0.12)
        self.play(FadeIn(ax), FadeIn(xl), FadeIn(yl))
        ep = T("25,600 episodes", size=24, color=GREY).next_to(ax, UP, buff=0.1)
        self.play(FadeIn(ep))
        rv = [max(v, 0) for v in s4["mean_R_value"]]
        rs = [max(v, 0) for v in s4["mean_R_success"]]
        cv = VMobject(stroke_color=GREEN, stroke_width=4).set_points_as_corners([ax.c2p(k, v) for k, v in zip(ks, rv)])
        cs = VMobject(stroke_color=YELLOW, stroke_width=4).set_points_as_corners([ax.c2p(k, v) for k, v in zip(ks, rs)])
        dv = VGroup(*[Dot(ax.c2p(k, v), radius=0.07, color=GREEN) for k, v in zip(ks, rv)])
        ds = VGroup(*[Dot(ax.c2p(k, v), radius=0.07, color=YELLOW) for k, v in zip(ks, rs)])
        lv = T("critic's value", size=22, color=GREEN).next_to(dv[-1], LEFT, buff=0.2).shift(0.25 * UP)
        ls = T("success (rate stays 0.47–0.49)", size=22, color=YELLOW).next_to(ds[-1], UP, buff=0.2).shift(1.2 * LEFT)
        self.play(Create(cs), FadeIn(ds), FadeIn(ls))
        self.caption("The success rate was 0.47–0.49 at every step, and the chunk drawn decided almost none of it (yellow, near 0).", wait=3.5)
        self.play(Create(cv), FadeIn(dv), FadeIn(lv))
        self.caption("Same split as before: between the 32 noised chunks against among their redraws, each run to the end. For the critic's value it is 17% to 87% (green).", wait=3)
        # AUC bars
        bx = Ax([0, 2, 1], [0.5, 1.0, 0.25], 3.4, 3.4, yt=[0.5, 0.75, 1.0], size=18).shift(3.8 * RIGHT + 0.2 * UP)
        bars = VGroup()
        for i, (v, lab, col) in enumerate(((s4["auc_pooled"], "between\nstates", BLUE), (s4["auc_within_mean"], "chunks within\none state", RED))):
            h = bx.c2p(0, v)[1] - bx.c2p(0, 0.5)[1]
            r = Rectangle(width=0.9, height=max(h, 0.01), fill_color=col, fill_opacity=0.85, stroke_width=0)
            r.move_to(bx.c2p(0.5 + i, 0.5), aligned_edge=DOWN)
            t = T(f"{v:.2f}", size=24, color=col).next_to(r, UP, buff=0.08)
            lt = T(lab, size=18, color="#c9d1dc").next_to(bx.c2p(0.5 + i, 0.5), DOWN, buff=0.15)
            bars.add(VGroup(r, t, lt))
        bt = T("critic's ranking of success (AUC)", size=20, color="#c9d1dc").next_to(bx, UP, buff=0.1)
        chance = DashedLine(bx.c2p(0, 0.5), bx.c2p(2, 0.5), color=GREY)
        self.play(FadeIn(bx), FadeIn(bt), FadeIn(chance))
        self.play(GrowFromCenter(bars[0]), GrowFromCenter(bars[1]))
        self.caption(f"The critic tells easy states from hard ones well (AUC {s4['auc_pooled']:.2f}), but ranks chunks within a state "
                     f"barely better than chance ({s4['auc_within_mean']:.2f}; 0.5 is chance).", wait=4)  # fmt: skip
        self.caption("Picking the critic's top quarter of chunks raised success by at most 3 points, and most at the earliest step.", wait=3)
        self.fade_all()


class S10Recommend(Base):
    def construct(self):
        head = T("What we recommend", size=40).to_edge(UP, buff=0.4)
        self.play(Write(head))
        bar = VGroup(*[Rectangle(width=0.42, height=0.42, stroke_width=1.5, stroke_color="#8b95a5", fill_opacity=1,
                                 fill_color=YELLOW if 3 <= i <= 8 else "#1c2230") for i in range(20)]).arrange(RIGHT, buff=0.06).shift(1.6 * UP)  # fmt: skip
        nums = VGroup(*[T(str(i), size=14, color=GREY).next_to(s, DOWN, buff=0.08) for i, s in enumerate(bar)])
        lab = T("steer at one fixed early step, k ≈ 3–8", size=28, color=YELLOW).next_to(bar, UP, buff=0.2)
        self.play(FadeIn(bar, lag_ratio=0.03), FadeIn(nums), FadeIn(lab))
        self.caption("Steer at one fixed early step, k ≈ 3–8: around the window, which sits at the same place in every phase, and where RL gains are high and flat.", wait=4)  # fmt: skip
        drop = T("drop the learned per-chunk choice of step", size=28, color=RED).shift(0.2 * UP)
        strike = Line(drop.get_left(), drop.get_right(), color=RED, stroke_width=3)
        self.play(FadeIn(drop))
        self.play(Create(strike))
        self.caption("Drop the learned per-chunk choice of step. The window does not move, and the critic cannot tell an adaptive agent which step to use.", wait=4)
        self.play(FadeOut(drop), FadeOut(strike))
        # RL results so far
        ax = Ax([0, 3, 1], [0, 1.0, 0.25], 6, 2.4, yt=[0, 0.5, 1.0], yfmt="{:.1f}", size=18).shift(0.35 * DOWN)
        data = [("fixed k = 3 or 5", 0.915, "0.91–0.92", YELLOW), ("adaptive k", 0.87, "0.87", PURPLE), ("k = 0 (DSRL)", 0.82, "0.82", GREY)]
        bars = VGroup()
        for i, (n, v, s, col) in enumerate(data):
            h = ax.c2p(0, v)[1] - ax.c2p(0, 0)[1]
            r = Rectangle(width=1.2, height=h, fill_color=col, fill_opacity=0.85, stroke_width=0).move_to(ax.c2p(0.5 + i, 0), aligned_edge=DOWN)
            bars.add(VGroup(r, T(s, size=22, color=col).next_to(r, UP, buff=0.08), T(n, size=18, color="#c9d1dc").next_to(ax.c2p(0.5 + i, 0), DOWN, buff=0.15)))
        yt = T("RL success at 8–9M env steps", size=20, color="#c9d1dc").next_to(ax, UP, buff=0.1)
        self.play(FadeOut(nums), bar.animate.scale(0.7).shift(0.4 * UP), lab.animate.scale(0.8).shift(0.55 * UP))
        self.play(FadeIn(ax), FadeIn(yt), *[GrowFromCenter(b) for b in bars])
        self.caption("Longer RL runs (8–9M env steps) agree: fixed k = 3 or 5 reach 0.91–0.92, against 0.87 for adaptive k (run before its entropy fix) and 0.82 at k = 0.", wait=4)
        self.play(*[FadeOut(m) for m in (ax, yt, bars)])
        lim = T("Main limit", size=30, color=RED).shift(0.2 * UP)
        lt = T("The simulator test changed only the first chunk.\nAn RL agent steers every chunk, and small effects could add up.\n"
               "Direct test, not yet run: let the critic pick every chunk\nof an episode at fixed k = 4, 9 or 15, and compare success.", size=24,
               line_spacing=0.9).next_to(lim, DOWN, buff=0.3)  # fmt: skip
        self.play(FadeIn(lim), FadeIn(lt, lag_ratio=0.1))
        self.caption("The main limit: the simulator test changed only the first chunk. That full test has not been run.", wait=4)
        self.fade_all()
        end = VGroup(TC("Where the policy has a discrete choice, it makes it in a window\nnear step 5–6, at the same step in every task phase.", size=30),
                     T("Steer at one fixed early step.", size=34, color=YELLOW)).arrange(DOWN, buff=0.6)  # fmt: skip
        self.play(FadeIn(end))
        self.wait(3)
        self.play(FadeOut(end))


SCENES = ["S01Question", "S02Toy", "S03Measure", "S04Control", "S05Transport", "S06Finding1", "S07Finding2", "S08Finding3",
          "S09Finding4", "S10Recommend"]  # fmt: skip
