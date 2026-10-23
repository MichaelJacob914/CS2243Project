"""When does steering a diffusion model in the MIDDLE of sampling help?

A self-contained 2D toy (CPU, ~1 minute).  Run:  python toy_intermediate_steering.py

Data: 4 "routes" (clusters on a ring), each split into 3 "lanes" (sub-clusters
along the route's tangent).  Lane spacing differs per route (0.8 ... 1.4), as
real behaviour geometry usually depends on context.  The diffusion model is
exact (closed-form Gaussian-mixture denoiser), sampled with 20 deterministic
DDIM steps on a cosine schedule, so nothing is learned and every effect is due
to *when* we intervene.

Steering = ONE gradient nudge at step k (k = 20: initial noise, k = 0: final
action) of normalised size eps:
    x_k <- x_k + eps * D_k^2 g / ||D_k g||,   g = d q(sample(x_k)) / d x_k
eps is chosen per goal and per k from a small grid on half of the samples;
success is reported on the other half (paired noise across all k).
Three goals, at three scales:
    route    "go as far as possible in direction w"   (coarse choice)
    lane     "keep your route, move one lane over"     (mid-level choice)
    precise  "land on a point inside your own lane"    (fine correction)

Insight: each choice is made in its own window of the sampling process.  The
denoiser amplifies a nudge along direction e exactly when its Jacobian has an
eigenvalue > 1 along e, which for a mixture means the posterior is still
uncertain between alternatives separated by more than the current noise level.
Coarse choices open first, fine ones later.  Steer a choice while its window is
open and coarser choices are already settled: that is an intermediate step for
the lane goal.

Outputs toy_intermediate_steering.png and prints a table.
"""
import math

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
K = 20                       # DDIM steps
S = 0.12                     # lane width (std)
R = 4.0                      # route radius
SPACING = [0.8, 1.0, 1.2, 1.4]  # lane spacing per route
EPS_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]  # nudge sizes ||D_k^-1 u||, tuned per goal and k
N = 4000                     # samples (paired across all k)


# ------------------------------------------------------------------ data + exact sampler
def make_mixture():
    th = torch.tensor([45.0, 135.0, 225.0, 315.0]) * math.pi / 180
    centre = R * torch.stack([th.cos(), th.sin()], -1)
    tangent = torch.stack([-th.sin(), th.cos()], -1)
    mu, route, lane = [], [], []
    for i in range(4):
        for j, off in enumerate([-1, 0, 1]):
            mu.append(centre[i] + off * SPACING[i] * tangent[i])
            route.append(i)
            lane.append(j)
    return torch.stack(mu), torch.tensor(route), torch.tensor(lane), centre, tangent


MU, ROUTE, LANE, CENTRE, TANGENT = make_mixture()
t = torch.linspace(0, 1, K + 1)
f = lambda t: torch.cos(0.5 * math.pi * (t + 0.008) / 1.008) ** 2
ALPHA = (f(t) / f(torch.zeros(()))).clamp(0, 1).sqrt()       # x_t = alpha x_0 + sigma eps
SIGMA = (1 - ALPHA**2).clamp_min(0).sqrt()


def log_resp(x, a=1.0, s=0.0):
    """log p(component | x_t)."""
    var = a**2 * S**2 + s**2
    return torch.log_softmax(-0.5 * ((x[:, None] - a * MU[None]) ** 2).sum(-1) / var, -1)


def denoise(x, k):
    """Exact E[x_0 | x_k]."""
    a, s = ALPHA[k], SIGMA[k]
    r = log_resp(x, a, s).exp()
    cond = MU[None] + (a * S**2 / (a**2 * S**2 + s**2)) * (x[:, None] - a * MU[None])
    return (r[..., None] * cond).sum(1)


def sample(x, k_from, k_to=0):
    """Deterministic DDIM from step k_from down to k_to."""
    for k in range(k_from, k_to, -1):
        x0 = denoise(x, k)
        x = ALPHA[k - 1] * x0 + SIGMA[k - 1] * (x - ALPHA[k] * x0) / SIGMA[k]
    return x


def assign(a):
    m = log_resp(a).argmax(-1)
    return m, ROUTE[m]


def on_support(a):
    return (torch.cdist(a, MU).min(-1).values / S) <= 3.0


# ------------------------------------------------------------------ goals
def goals(a0, gen):
    """Per-sample goal q(a) and success test, defined from the unsteered action a0."""
    m0, r0 = assign(a0)
    th = 2 * math.pi * torch.rand(N, generator=gen)
    w_route = torch.stack([th.cos(), th.sin()], -1)
    best_route = (w_route @ CENTRE.T).argmax(-1)
    # lane: move one lane along +-tangent of the current route (inward at the edges)
    sign = torch.where(torch.rand(N, generator=gen) < 0.5, 1.0, -1.0)
    sign = torch.where(LANE[m0] == 0, 1.0, torch.where(LANE[m0] == 2, -1.0, sign))
    w_lane = sign[:, None] * TANGENT[r0]
    # precise: a point 1.5 lane-widths from the own lane centre
    th2 = 2 * math.pi * torch.rand(N, generator=gen)
    target = MU[m0] + 1.5 * S * torch.stack([th2.cos(), th2.sin()], -1)

    def route_logp(a):
        lr = log_resp(a)
        return torch.stack([torch.logsumexp(lr[:, ROUTE == i], -1) for i in range(4)], -1).gather(1, r0[:, None])[:, 0]

    return {
        "route": (lambda a: (a * w_route).sum(-1),
                  lambda a: (assign(a)[1] == best_route) & on_support(a)),
        "lane": (lambda a: (a * w_lane).sum(-1) + route_logp(a),
                 lambda a: (assign(a)[1] == r0) & on_support(a)
                 & ((MU[assign(a)[0]] * w_lane).sum(-1) > (MU[m0] * w_lane).sum(-1) + 1e-9)),
        "precise": (lambda a: -0.5 * ((a - target) ** 2).sum(-1),
                    lambda a: ((a - target).norm(dim=-1) <= 0.5 * S) & (assign(a)[0] == m0)),
    }


def fixed_goals(gen):
    """Goals that are fixed functions of the action alone (drawn independently of
    the sample, identical at every step k): classifier-style targets.
        lane_any    end in lane j* (inner / middle / outer), on any route
        lane_route  end in lane j* of route r*
        point       land within 0.5 lane-widths of a point c placed 1.5 lane-widths
                    from a random lane centre (any route): needs the right mode AND
                    precision.  point_mode scores the same steering by whether it
                    reached c's lane at all (the exploration half of the goal).
    """
    j_star = torch.randint(0, 3, (N,), generator=gen)
    c_star = torch.randint(0, 12, (N,), generator=gen)  # component = (route, lane)
    p_star = torch.randint(0, 12, (N,), generator=gen)
    th = 2 * math.pi * torch.rand(N, generator=gen)
    point = MU[p_star] + 1.5 * S * torch.stack([th.cos(), th.sin()], -1)
    q_point = lambda a: -0.5 * ((a - point) ** 2).sum(-1)

    def lane_logp(a):
        lr = log_resp(a)
        return torch.stack([torch.logsumexp(lr[:, LANE == j], -1) for j in range(3)], -1).gather(1, j_star[:, None])[:, 0]

    return {
        "lane_any": (lane_logp, lambda a: (LANE[assign(a)[0]] == j_star) & on_support(a)),
        "lane_route": (lambda a: log_resp(a).gather(1, c_star[:, None])[:, 0],
                       lambda a: (assign(a)[0] == c_star) & on_support(a)),
        "point": (q_point, lambda a: (a - point).norm(dim=-1) <= 0.5 * S),
        "point_mode": (q_point, lambda a: (assign(a)[0] == p_star) & on_support(a)),
    }


# ------------------------------------------------------------------ steering
def steer_once(xk, k, q, D, eps):
    """One gradient nudge of size ||D_k^-1 u|| = eps at step k, then finish sampling."""
    x = xk.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(q(sample(x, k)).sum(), x)
    Dg = D[k] * g
    u = eps * D[k] * Dg / Dg.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    with torch.no_grad():
        return sample(xk + u, k)


def decision_windows(states):
    """Fraction of samples whose denoiser Jacobian expands across routes / across lanes.

    For this mixture J_den = s^2/(s^2+tau^2) I + tau^2 Cov_post(mu)/(s^2+tau^2)^2 in
    data units (tau = sigma/alpha), so a nudge grows along e iff the posterior
    variance of e^T mu exceeds s^2 + tau^2.  Splitting Cov_post(mu) into between-
    route and within-route parts gives the route and lane windows.
    """
    out = {"route": [], "lane": []}
    for k in range(K + 1):
        tau2 = 0.0 if k == 0 else float((SIGMA[k] / ALPHA[k].clamp_min(1e-12)) ** 2)
        r = log_resp(states[k], ALPHA[k], SIGMA[k]).exp()
        mean = r @ MU
        cov = torch.einsum("nm,mi,mj->nij", r, MU, MU) - mean[:, :, None] * mean[:, None]
        Rw = torch.stack([r[:, ROUTE == i].sum(-1) for i in range(4)], -1)
        rm = torch.stack([(r[:, ROUTE == i] @ MU[ROUTE == i]) / Rw[:, i:i + 1].clamp_min(1e-300) for i in range(4)], 1)
        between = torch.einsum("nr,nri,nrj->nij", Rw, rm, rm) - mean[:, :, None] * mean[:, None]
        within = cov - between
        thr = S**2 + tau2
        out["route"].append(float((torch.linalg.eigvalsh(between)[:, -1] > thr).double().mean()))
        out["lane"].append(float((torch.linalg.eigvalsh(within)[:, -1] > thr).double().mean()))
    return out


def main():
    import sys
    fixed = "--fixed" in sys.argv
    gen = torch.Generator().manual_seed(0)
    xK = torch.randn(N, 2, generator=gen)
    states = [None] * (K + 1)
    states[K] = xK
    for k in range(K, 0, -1):
        states[k - 1] = sample(states[k], k, k - 1)
    a0 = states[0]
    D = torch.stack([s.std(0) for s in states])  # per-step latent scale from unguided rollouts
    G = fixed_goals(gen) if fixed else goals(a0, gen)

    ks = list(range(K, -1, -1))
    val = torch.arange(N) < N // 2  # tune eps on the first half, report the second
    res = {name: [] for name in G}
    for name, (q, ok) in G.items():
        for k in ks:
            hits = [ok(steer_once(states[k], k, q, D, e)) for e in EPS_GRID]
            best = max(range(len(EPS_GRID)), key=lambda i: hits[i][val].double().mean().item())
            res[name].append(hits[best][~val].double().mean().item())
    win = decision_windows(states)
    base = {name: ok(a0)[~val].double().mean().item() for name, (q, ok) in G.items()}

    print(f"test success of ONE nudge (size tuned per k on held-out half) by step k   (no steering: "
          + ", ".join(f"{n} {b:.2f}" for n, b in base.items()) + ")")
    print("goal      " + " ".join(f"k={k:<3d}" for k in ks) + "  best k")
    for name in G:
        row = np.array(res[name])
        print(f"{name:9s} " + " ".join(f"{v:5.2f}" for v in row) + f"  {ks[int(row.argmax())]} ({row.max():.2f})")

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    a = ax[0]
    traj = torch.stack(states)[:, :150]
    for i in range(150):
        a.plot(traj[:, i, 0], traj[:, i, 1], color="#9ec5f4", lw=0.5, alpha=0.7)
    col = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
    m, rr = assign(a0)
    for i in range(4):
        sel = rr == i
        a.scatter(a0[sel, 0][:600], a0[sel, 1][:600], s=2, color=col[i])
    a.set_aspect("equal")
    a.set_title("4 routes x 3 lanes; DDIM paths from noise (centre)", fontsize=10)
    a = ax[1]
    labels = [("route", "#2a78d6", "route goal (coarse)"), ("lane", "#eb6834", "lane goal (mid-level)"),
              ("precise", "#1baf7a", "precise goal (fine)")]
    if fixed:
        labels = [("lane_any", "#eb6834", "end in lane j*, any route"),
                  ("lane_route", "#6250d6", "end in lane j* of route r*"),
                  ("point", "#1baf7a", "hit a point in a random lane (mode + precision)"),
                  ("point_mode", "#2a78d6", "  ... reached that lane at all (mode only)")]
    for name, c, lab in labels:
        a.plot(ks, res[name], color=c, marker="o", ms=3.5, lw=2, label=lab)
    a.set_xlim(K + 0.5, -0.5)
    a.set_ylim(-0.02, 1.02)
    a.set_xlabel("intervention step k   (20 = noise  ->  0 = action)")
    a.set_ylabel("success after ONE nudge\n(size tuned per step)")
    a.set_title("each goal has its own best time to steer", fontsize=10)
    a.legend(fontsize=8, frameon=False, loc="upper right" if fixed else "lower left",
             bbox_to_anchor=None if fixed else (0.0, 0.07))
    a = ax[2]
    a.fill_between(ks, [win["route"][k] for k in ks], color="#2a78d6", alpha=0.35, step="mid", label="route still open")
    a.fill_between(ks, [win["lane"][k] for k in ks], color="#eb6834", alpha=0.35, step="mid", label="lane still open")
    a.set_xlim(K + 0.5, -0.5)
    a.set_xlabel("intervention step k")
    a.set_ylabel("fraction of samples where the denoiser\namplifies a nudge across that choice")
    a.set_title("decision windows (computed from the frozen model)", fontsize=10)
    a.legend(fontsize=8, frameon=False)
    for a in ax[1:]:
        a.grid(color="#e6e5e0")
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = "toy_intermediate_steering_fixed.png" if fixed else "toy_intermediate_steering.png"
    fig.savefig(out, dpi=160)
    print("saved", out)


if __name__ == "__main__":
    main()
