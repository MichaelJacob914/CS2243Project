"""Animated demo: the same noise, steered once early / in the middle / late.

Goal (fixed for every sample and every step): end in the GREEN (counter-clockwise) lane of any
route, q(a) = log p(lane = outer | a).  One gradient nudge at step k, size
chosen per k on a validation half; success rates in the panel titles are on the
held-out half (2000 samples).  Dots: 300 held-out samples; the large marker is
one held-out example where early and late steering fail and intermediate
steering succeeds (chosen for illustration; the titles give the overall rates).

Outputs: demo_steering_timing.gif, demo_steering_timing_frames.png, demo_numbers.txt
"""
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import toy_intermediate_steering as T

OUTER = 2
SUB = 4  # interpolated frames per sampling step


def run():
    gen = torch.Generator().manual_seed(0)
    xK = torch.randn(T.N, 2, generator=gen)
    states = [None] * (T.K + 1)
    states[T.K] = xK
    for k in range(T.K, 0, -1):
        states[k - 1] = T.sample(states[k], k, k - 1)
    D = torch.stack([s.std(0) for s in states])

    def q(a):
        lr = T.log_resp(a)
        return torch.logsumexp(lr[:, T.LANE == OUTER], -1)

    def ok(a):
        return (T.LANE[T.assign(a)[0]] == OUTER) & T.on_support(a)

    val = torch.arange(T.N) < T.N // 2
    best_eps, test_rate = {}, {}
    for k in range(T.K + 1):
        hits = {e: ok(T.steer_once(states[k], k, q, D, e)) for e in T.EPS_GRID}
        e = max(T.EPS_GRID, key=lambda e: hits[e][val].double().mean().item())
        best_eps[k], test_rate[k] = e, hits[e][~val].double().mean().item()
    base = ok(states[0])[~val].double().mean().item()
    k_mid = max(range(2, T.K - 1), key=lambda k: test_rate[k])
    plan = {"early": T.K, "intermediate": k_mid, "late": 0}

    def trajectory(k_s):
        """Full path with the nudge at step k_s; returns list of (step, x) incl. the jump."""
        e = best_eps[k_s]
        x = states[T.K].clone()
        path = [(T.K, x)]
        for k in range(T.K, -1, -1):
            if k == k_s:
                xg = x.clone().requires_grad_(True)
                (g,) = torch.autograd.grad(q(T.sample(xg, k)).sum(), xg)
                Dg = D[k] * g
                x = (x + e * D[k] * Dg / Dg.norm(dim=-1, keepdim=True).clamp_min(1e-12)).detach()
                path.append((k, x))  # same step, after the nudge
            if k > 0:
                x = T.sample(x, k, k - 1)
                path.append((k - 1, x))
        return path

    paths = {name: trajectory(k) for name, k in plan.items()}
    final = {name: p[-1][1] for name, p in paths.items()}
    succ = {name: ok(final[name]) for name in plan}
    test_idx = torch.nonzero(~val).squeeze(1)
    pick = [int(i) for i in test_idx if not succ["early"][i] and succ["intermediate"][i] and not succ["late"][i]]
    n_pick = len(pick)
    gen2 = torch.Generator().manual_seed(1)
    show = test_idx[torch.randperm(len(test_idx), generator=gen2)[:300]]
    hl = pick[0]

    lines = [f"goal: end in the green lane (the counter-clockwise lane) of any route; fixed for all samples and steps",
             f"no steering: {base:.3f}",
             "test success of one nudge by step k (size tuned per k on the validation half):"]
    lines += [f"  k={k:2d}  {test_rate[k]:.3f}  (eps={best_eps[k]:g})" for k in range(T.K, -1, -1)]
    lines += [f"demo uses early k={T.K}, intermediate k={k_mid}, late k=0",
              f"held-out samples where early fails, intermediate succeeds, late fails: {n_pick} of {len(test_idx)}",
              f"(and where early succeeds / intermediate fails / late succeeds: "
              f"{int(((succ['early'] & ~succ['intermediate'] & succ['late'])[~val]).sum())})"]
    open("demo_numbers.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return paths, succ, show, hl, plan, test_rate, base


def render(paths, succ, show, hl, plan, test_rate, base):
    names = ["early", "intermediate", "late"]
    titles = {n: f"{n} steering (k = {plan[n]})\nsuccess {test_rate[plan[n]]:.0%}" for n in names}
    # frames: interpolate between consecutive path entries; a nudge is an extra entry at the same step
    n_entries = len(paths["early"])  # every path has K + 2 entries (one nudge)
    # align by step: build per-strategy arrays indexed by "time" t in [0, K + 1]
    seq = {}
    for n in names:
        xs = [x for _, x in paths[n]]
        frames, kind = [], []
        for i in range(len(xs) - 1):
            jump = paths[n][i][0] == paths[n][i + 1][0]
            for s in range(SUB):
                w = s / SUB
                frames.append((1 - w) * xs[i] + w * xs[i + 1])
                kind.append("nudge" if jump else "denoise")
        frames += [xs[-1]] * (3 * SUB)  # hold the end
        kind += ["end"] * (3 * SUB)
        seq[n] = (frames, kind)
    n_frames = len(seq["early"][0])

    theta = torch.linspace(0, 2 * np.pi, 60)
    circ = torch.stack([theta.cos(), theta.sin()], -1)
    GOOD, BAD, NEUTRAL, INK2 = "#1baf7a", "#e34948", "#3987e5", "#52514e"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    artists = []
    for ax, n in zip(axes, names):
        for m in range(len(T.MU)):
            c = T.MU[m] + 3 * T.S * circ
            outer = int(T.LANE[m]) == OUTER
            ax.fill(c[:, 0], c[:, 1], color=GOOD if outer else "#c3c2b7", alpha=0.25 if outer else 0.3, lw=0)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titles[n], fontsize=11)
        dots = ax.scatter([], [], s=7, color=NEUTRAL, alpha=0.6)
        trail, = ax.plot([], [], color="#0b0b0b", lw=1.2)
        star = ax.scatter([], [], s=140, marker="*", color="#0b0b0b", zorder=5)
        label = ax.text(0, -4.6, "", ha="center", fontsize=10, color=INK2)
        artists.append((dots, trail, star, label))
    fig.suptitle("Goal: end in the GREEN lane of any route.  Same noise in all three panels; one nudge each.",
                 fontsize=11)

    def update(f):
        out = []
        for (dots, trail, star, label), n in zip(artists, names):
            frames, kind = seq[n]
            x = frames[f]
            pts = x[show].numpy()
            dots.set_offsets(pts)
            if kind[f] == "end":
                dots.set_color([GOOD if s else BAD for s in succ[n][show]])
            else:
                dots.set_color(NEUTRAL)
            hist = torch.stack([fr[hl] for fr in frames[: f + 1]]).numpy()
            trail.set_data(hist[:, 0], hist[:, 1])
            star.set_offsets(frames[f][hl][None].numpy())
            if kind[f] == "end":
                star.set_color(GOOD if succ[n][hl] else BAD)
                label.set_text("success" if succ[n][hl] else "fails")
            elif kind[f] == "nudge":
                star.set_color("#eb6834")
                label.set_text("NUDGE")
            else:
                star.set_color("#0b0b0b")
                label.set_text("denoising...")
            out += [dots, trail, star, label]
        return out

    anim = FuncAnimation(fig, update, frames=n_frames, interval=80, blit=False)
    anim.save("demo_steering_timing.gif", writer=PillowWriter(fps=12))

    # static strip: start, just after each nudge, end
    fig2, ax2 = plt.subplots(3, 4, figsize=(14, 10.5))
    for r, n in enumerate(names):
        frames, kind = seq[n]
        j_nudge = kind.index("nudge")
        picks = [(0, "noise"), (j_nudge, "just before the nudge"), (j_nudge + SUB, "just after the nudge"),
                 (len(frames) - 1, "final action")]
        for c, (j, lab) in enumerate(picks):
            a = ax2[r, c]
            for m in range(len(T.MU)):
                cc = T.MU[m] + 3 * T.S * circ
                outer = int(T.LANE[m]) == OUTER
                a.fill(cc[:, 0], cc[:, 1], color=GOOD if outer else "#c3c2b7", alpha=0.25 if outer else 0.3, lw=0)
            pts = frames[j][show].numpy()
            col = [GOOD if s else BAD for s in succ[n][show]] if c == 3 else NEUTRAL
            a.scatter(pts[:, 0], pts[:, 1], s=6, color=col, alpha=0.7)
            a.scatter(*frames[j][hl].numpy(), s=120, marker="*", color="#0b0b0b", zorder=5)
            a.set_xlim(-5, 5)
            a.set_ylim(-5, 5)
            a.set_aspect("equal")
            a.set_xticks([])
            a.set_yticks([])
            if r == 0:
                a.set_title(lab, fontsize=10)
            if c == 0:
                a.set_ylabel(titles[n].replace("\n", "\n"), fontsize=9)
    fig2.tight_layout()
    fig2.savefig("demo_steering_timing_frames.png", dpi=130)


if __name__ == "__main__":
    render(*run())
    print("saved demo_steering_timing.gif, demo_steering_timing_frames.png")
