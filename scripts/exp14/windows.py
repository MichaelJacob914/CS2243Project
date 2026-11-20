"""exp14 core: re-noise-and-regenerate measurements for a DPPO diffusion policy (model-agnostic).

Used by both the Stage 0 toy control and the transport stages, so the pipeline is identical.

Noise levels. A level is a DDPM index t (0..T-1); x_t = sqrt(abar_t) a + sqrt(1 - abar_t) eps. The DDIM
steering step k (K = 20 over T = 100) takes input at t = ddim_t[k] = 95 - 5k. Level "k" below always
means that DDPM index. log-SNR = ln(abar_t / (1 - abar_t)).

Samplers.
  stochastic   DDPM ancestral with the training schedule from level t down to 0, posterior variance
               beta~_t, NO x0 clamp, NO noise clamp, NO std floor; the t = 0 step returns the mean.
  deterministic  DDIM (eta = 0) steps k..K-1 with the evaluation settings (x0 clamp as configured),
               via the snapshot's adsrl.sampler.ddim_step.

Estimators (per state, per level, per feature f with values in R^d; vector features use traces).
  resolved fraction  R = (S_b - W/M) / (S_b - W/M + W), S_b = sum_d Var_i(mean_j f_ijd),
                     W = sum_d mean_i Var_j(f_ijd) (ddof 1), P points i, M regenerations j.
  open fraction      O = mean_b sum_d Var_c(f_bcd) / sum_d Var(f_d) (ddof 1).

Gaussian null. N(mu, Sigma) fitted to samples. Its clean points are exact draws; everything after that runs
through the IDENTICAL sampler code as the policy (ddpm_regenerate for 2a, ddim_step with the same clamps for 2b),
with the network replaced by the null's closed-form epsilon-predictor (GaussEpsModel). GaussNull.posterior_sample
and ode_map (exact continuous-time quantities) are kept only for the closed-form tests.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# ----------------------------------------------------------------------------------------
# schedule helpers
# ----------------------------------------------------------------------------------------


def network(model):
    return getattr(model, "actor", model.network)


def abar(model) -> np.ndarray:
    return model.alphas_cumprod.detach().cpu().double().numpy()


def ddim_levels(model) -> np.ndarray:
    """DDPM index fed at each DDIM step k = 0..K-1 (noisiest first)."""
    return model.ddim_t.detach().cpu().numpy().astype(int)


def logsnr(model, ts) -> np.ndarray:
    a = abar(model)[np.asarray(ts)]
    return np.log(a / (1 - a))


def q_sample(model, a, t: int, eps):
    ab = float(model.alphas_cumprod[t])
    return math.sqrt(ab) * a + math.sqrt(1 - ab) * eps


# ----------------------------------------------------------------------------------------
# samplers
# ----------------------------------------------------------------------------------------


@torch.no_grad()
def ddpm_regenerate(model, cond, x_t, t_start: int, gen: torch.Generator | None = None):
    """Stochastic ancestral DDPM from x_t (at DDPM index t_start) to a clean action. No clipping, no floor."""
    net = network(model)
    B = x_t.shape[0]
    x = x_t
    c1, c2 = model.ddpm_mu_coef1, model.ddpm_mu_coef2
    sr, srm1 = model.sqrt_recip_alphas_cumprod, model.sqrt_recipm1_alphas_cumprod
    var = model.ddpm_var
    for t in range(int(t_start), -1, -1):
        t_b = torch.full((B,), t, device=x.device, dtype=torch.long)
        eps = net(x, t_b, cond=cond)
        x0 = sr[t] * x - srm1[t] * eps
        mu = c1[t] * x0 + c2[t] * x
        if t > 0:
            z = torch.randn(x.shape, device=x.device, generator=gen)
            x = mu + var[t].sqrt() * z
        else:
            x = mu
    return x


@torch.no_grad()
def ddpm_sample(model, cond, n_per_cond: int, gen=None):
    """Fresh stochastic samples from the top of the chain (t = T-1). cond['state']: (B, obs_dim)."""
    s = cond["state"]
    B = s.shape[0]
    s_rep = s.repeat_interleave(n_per_cond, 0)
    x = torch.randn((B * n_per_cond, model.horizon_steps, model.action_dim), device=s.device, generator=gen)
    out = ddpm_regenerate(model, {"state": s_rep}, x, model.denoising_steps - 1, gen)
    return out.view(B, n_per_cond, model.horizon_steps, model.action_dim)


@torch.no_grad()
def ddim_from_step(model, cond, x, k: int):
    """Deterministic DDIM from the input of step k (x at DDPM index ddim_t[k]) to the clean action."""
    from adsrl.sampler import ddim_step, num_ddim_steps

    K = num_ddim_steps(model)
    for j in range(int(k), K):
        x, *_ = ddim_step(model, x, j, cond)
        if model.final_action_clip_value is not None and j == K - 1:
            x = torch.clamp(x, -model.final_action_clip_value, model.final_action_clip_value)
    return x


@torch.no_grad()
def ddim_sample(model, cond, x_T):
    return ddim_from_step(model, cond, x_T, 0)


# ----------------------------------------------------------------------------------------
# Gaussian null
# ----------------------------------------------------------------------------------------


class GaussNull:
    """N(mu, Sigma) on flattened actions; exact posterior given x_t = sqrt(ab) a + sqrt(1-ab) eps."""

    def __init__(self, samples: np.ndarray):
        X = samples.reshape(len(samples), -1).astype(np.float64)
        self.shape = samples.shape[1:]
        self.mu = X.mean(0)
        C = np.cov(X, rowvar=False)
        lam, U = np.linalg.eigh(C)
        self.lam = np.clip(lam, 0, None)
        self.U = U

    def sample(self, n, rng):
        z = rng.standard_normal((n, len(self.mu)))
        return (self.mu + (z * np.sqrt(self.lam)) @ self.U.T).reshape(n, *self.shape)

    def posterior_sample(self, y, ab, n, rng):
        """y: (P, D) noisy points at abar = ab. Returns (P, n, *shape) posterior draws."""
        y = y.reshape(len(y), -1)
        u = (y - math.sqrt(ab) * self.mu) @ self.U  # coordinates of y - sqrt(ab) mu
        den = ab * self.lam + 1 - ab
        pm = math.sqrt(ab) * self.lam / den * u  # posterior mean of (a - mu) in eigbasis
        pv = self.lam * (1 - ab) / den
        z = rng.standard_normal((len(y), n, len(self.mu)))
        c = pm[:, None] + z * np.sqrt(pv)
        return (self.mu + c @ self.U.T).reshape(len(y), n, *self.shape)

    def ode_map(self, y, ab):
        """Probability-flow (DDIM, continuous time) map from x_t to the clean action, exact for a Gaussian."""
        y = y.reshape(len(y), -1)
        u = (y - math.sqrt(ab) * self.mu) @ self.U
        c = np.sqrt(self.lam / (ab * self.lam + 1 - ab)) * u
        return (self.mu + c @ self.U.T).reshape(len(y), *self.shape)


class _GaussEps(torch.nn.Module):
    """Exact epsilon-predictor of N(mu, Sigma): E[eps | x_t] = sqrt(1-ab) (ab Sigma + (1-ab) I)^-1 (x - sqrt(ab) mu),
    evaluated in Sigma's eigenbasis. Called like a dppo network: net(x (B, H, A), t (B,) DDPM index, cond=...)."""

    def __init__(self, null: "GaussNull", alphas_cumprod: torch.Tensor):
        super().__init__()
        dev = alphas_cumprod.device
        self.register_buffer("mu", torch.as_tensor(null.mu, dtype=torch.float32, device=dev))
        self.register_buffer("lam", torch.as_tensor(null.lam, dtype=torch.float32, device=dev))
        self.register_buffer("U", torch.as_tensor(null.U, dtype=torch.float32, device=dev))
        self.register_buffer("ac", alphas_cumprod.float())

    def forward(self, x, t, cond=None, **kwargs):
        B = x.shape[0]
        ab = self.ac[t].view(B, 1)
        u = (x.reshape(B, -1) - ab.sqrt() * self.mu) @ self.U
        e = (1 - ab).sqrt() * u / (ab * self.lam + 1 - ab)
        return (e @ self.U.T).view(x.shape)


class GaussEpsModel:
    """Stand-in for a dppo DiffusionModel with the same schedule, DDIM settings and clamps as `template`, whose
    network is the exact epsilon-predictor of the Gaussian null. Passing it to ddpm_regenerate / ddim_step runs
    the null through the identical sampler code as the policy."""

    _BUFFERS = ("betas", "alphas_cumprod", "ddpm_mu_coef1", "ddpm_mu_coef2", "ddpm_var", "sqrt_recip_alphas_cumprod",
                "sqrt_recipm1_alphas_cumprod", "ddim_t", "ddim_alphas", "ddim_alphas_prev", "ddim_sqrt_one_minus_alphas")  # fmt: skip
    _ATTRS = ("denoised_clip_value", "eps_clip_value", "final_action_clip_value", "use_ddim", "horizon_steps",
              "action_dim", "denoising_steps")  # fmt: skip

    def __init__(self, template, null: "GaussNull"):
        for n in self._BUFFERS:
            setattr(self, n, getattr(template, n))
        for n in self._ATTRS:
            setattr(self, n, getattr(template, n))
        self.network = _GaussEps(null, self.alphas_cumprod)
        self.null = null


# ----------------------------------------------------------------------------------------
# estimators
# ----------------------------------------------------------------------------------------


def point_summary(f):
    """f: (..., P, M, d) -> (fbar (..., P, d), wvar (..., P)) with wvar = sum_d Var_j (ddof 1)."""
    fbar = f.mean(-2)
    wvar = f.var(-2, ddof=1).sum(-1)
    return fbar, wvar


def resolved_from_summary(fbar, wvar, M: int):
    """fbar (..., P, d), wvar (..., P) -> R (...). Unbiased between-point variance, ICC form."""
    Sb = fbar.var(-2, ddof=1).sum(-1)
    W = wvar.mean(-1)
    between = Sb - W / M
    tot = between + W
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.where(tot > 1e-12, between / tot, np.nan)
    return R


def resolved_fraction(f):
    fbar, wvar = point_summary(f)
    return resolved_from_summary(fbar, wvar, f.shape[-2])


def bootstrap_resolved(fbar, wvar, M, n_boot, rng):
    """fbar (L, P, d), wvar (L, P). Resample points (same indices at every level). -> (n_boot, L)."""
    L, P = wvar.shape
    idx = rng.integers(0, P, size=(n_boot, P))
    fb = fbar[:, idx]  # (L, n_boot, P, d)
    wv = wvar[:, idx]  # (L, n_boot, P)
    return resolved_from_summary(fb, wv, M).T


def bootstrap_resolved_gram(fbar, wvar, M, n_boot, rng):
    """Same output as bootstrap_resolved (same rng draws), without materializing the resampled (L, n_boot, P, d)
    array: with resample counts c (n_boot, P), sum_d Var_resampled(fbar) = (c.n - c G c / P) / (P - 1), where
    n_i = ||fbar_i||^2 and G = fbar fbar^T per level."""
    L, P = wvar.shape
    idx = rng.integers(0, P, size=(n_boot, P))
    C = np.zeros((n_boot, P))
    np.add.at(C, (np.repeat(np.arange(n_boot), P), idx.ravel()), 1.0)
    fc = fbar - fbar.mean(axis=1, keepdims=True)  # centering per level (variance is shift-invariant; better conditioned)
    G = np.einsum("lpd,lqd->lpq", fc, fc)  # (L, P, P)
    nrm = np.einsum("lpp->lp", G)  # (L, P)
    cGc = np.stack([((C @ G[l]) * C).sum(-1) for l in range(L)])  # (L, n_boot)
    Sb = (np.einsum("bp,lp->lb", C, nrm) - cGc / P) / (P - 1)
    Wm = np.einsum("bp,lp->lb", C, wvar) / P
    between = Sb - Wm / M
    tot = between + Wm
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.where(tot > 1e-12, between / tot, np.nan)
    return R.T


def open_fraction(f, total_var):
    """f: (..., B, C, d); total_var scalar (sum_d Var over fresh samples)."""
    within = f.var(-2, ddof=1).sum(-1).mean(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total_var > 1e-12, within / total_var, np.nan)


def total_var(f):
    """f: (N, d) -> sum_d Var (ddof 1)."""
    return float(f.var(0, ddof=1).sum())


# ----------------------------------------------------------------------------------------
# curve statistics
# ----------------------------------------------------------------------------------------


def monotone(y):
    """Monotone (non-decreasing along the last axis) smoothing: average of the upper envelope
    (running max from the left) and the lower envelope (running min from the right). NaN -> NaN row."""
    up = np.maximum.accumulate(y, axis=-1)
    lo = np.flip(np.minimum.accumulate(np.flip(y, -1), axis=-1), -1)
    return 0.5 * (up + lo)


def crossing(x, y, c):
    """x (L,) increasing; y (..., L) non-decreasing. First crossing of level c, linearly interpolated.
    Returns (value, censored_low, never): value NaN if y[0] >= c (censored_low) or max y < c (never)."""
    y = np.asarray(y, dtype=float)
    above = y >= c
    has = above.any(-1)
    i = np.argmax(above, -1)
    cens = above[..., 0]
    i0 = np.clip(i - 1, 0, len(x) - 1)
    y0 = np.take_along_axis(y, i0[..., None], -1)[..., 0]
    y1 = np.take_along_axis(y, i[..., None], -1)[..., 0]
    x0, x1 = x[i0], x[i]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(y1 > y0, (c - y0) / (y1 - y0), 0.0)
    val = x0 + frac * (x1 - x0)
    bad = cens | ~has | np.isnan(y).any(-1)
    return np.where(bad, np.nan, val), cens, ~has


def curve_stats(x, R):
    """R (..., L) raw curves -> dict of midpoint / width (in units of x) after monotone smoothing."""
    m = monotone(R)
    lo, cens_lo, _ = crossing(x, m, 0.1)
    mid, _, _ = crossing(x, m, 0.5)
    hi, _, never_hi = crossing(x, m, 0.9)
    return {"mid": mid, "lo": lo, "hi": hi, "width": hi - lo, "cens_lo": cens_lo, "never_hi": never_hi}


def _sig(z):
    return 0.5 * (1 + np.tanh(0.5 * z))


def two_rise_test(x, R, dbic_min=10.0, w_min=0.2, sep_min=2.0):
    """Fit one logistic (m, s) and a two-logistic mixture (w, m1, s1, m2, s2) by least squares.
    Returns dict with bic1, bic2, dbic (= bic1 - bic2), params, preferred (bool)."""
    from scipy.optimize import least_squares

    ok = ~np.isnan(R)
    x, R = np.asarray(x)[ok], np.asarray(R)[ok]
    n = len(x)
    out = {"bic1": np.nan, "bic2": np.nan, "dbic": np.nan, "preferred": False, "p1": None, "p2": None}
    if n < 8:
        return out
    lo_b = [x.min() - 3, 0.05]
    hi_b = [x.max() + 3, 10.0]

    def f1(p):
        return _sig((x - p[0]) / p[1]) - R

    best1 = None
    for m0 in np.linspace(x.min(), x.max(), 5):
        for s0 in (0.5, 1.5):
            r = least_squares(f1, [m0, s0], bounds=(lo_b, hi_b))
            if best1 is None or r.cost < best1.cost:
                best1 = r

    def f2(p):
        w, m1, s1, m2, s2 = p
        return w * _sig((x - m1) / s1) + (1 - w) * _sig((x - m2) / s2) - R

    best2 = None
    grid = np.linspace(x.min(), x.max(), 5)
    for i, m1 in enumerate(grid):
        for m2 in grid[i + 1 :]:
            for w0 in (0.3, 0.5, 0.7):
                r = least_squares(
                    f2, [w0, m1, 0.7, m2, 0.7],
                    bounds=([0.0, lo_b[0], 0.05, lo_b[0], 0.05], [1.0, hi_b[0], 10.0, hi_b[0], 10.0]),
                )  # fmt: skip
                if best2 is None or r.cost < best2.cost:
                    best2 = r
    rss1 = max(2 * best1.cost, 1e-12)
    rss2 = max(2 * best2.cost, 1e-12)
    bic1 = n * np.log(rss1 / n) + 2 * np.log(n)
    bic2 = n * np.log(rss2 / n) + 5 * np.log(n)
    w, m1, s1, m2, s2 = best2.x
    pref = (bic1 - bic2 >= dbic_min) and (min(w, 1 - w) >= w_min) and (abs(m2 - m1) >= sep_min)
    out.update(bic1=bic1, bic2=bic2, dbic=bic1 - bic2, preferred=bool(pref), p1=best1.x.tolist(), p2=best2.x.tolist())
    return out


def width_test(width_pol_boot, width_null_boot, min_complete=0.95):
    """95% percentile intervals of the bootstrap widths. Narrower = policy upper < null lower. Each arm needs at
    least `min_complete` of its bootstrap replicates uncensored (finite width); otherwise the state x feature is
    marked censored and excluded (narrower = False), the same rule for both arms. Also a bootstrap p-value =
    P(policy width >= null width) over paired independent replicates (complete pairs only)."""
    fa, fb = np.isfinite(width_pol_boot), np.isfinite(width_null_boot)
    comp = (float(fa.mean()), float(fb.mean()))
    out = {"narrower": False, "censored": True, "p": np.nan, "pol_ci": (np.nan, np.nan), "null_ci": (np.nan, np.nan), "complete": comp}
    if comp[0] < min_complete or comp[1] < min_complete:
        return out
    a, b = width_pol_boot[fa], width_null_boot[fb]
    pci = np.percentile(a, [2.5, 97.5])
    nci = np.percentile(b, [2.5, 97.5])
    both = fa & fb if len(fa) == len(fb) else None
    p = float((width_pol_boot[both] >= width_null_boot[both]).mean()) if both is not None else np.nan
    out.update(narrower=bool(pci[1] < nci[0]), censored=False, p=p, pol_ci=tuple(pci), null_ci=tuple(nci))
    return out


# ----------------------------------------------------------------------------------------
# clustering (Stage 1)
# ----------------------------------------------------------------------------------------


def cluster_tests(actions, rng_seed=0, var_frac=0.9, dbic_min=10.0, dip_alpha=0.05):
    """actions (N, H, A). PCA to 90% variance, GMM 1/2/3 by BIC, dip test on PC1.
    Returns dict incl. a fitted labeler (pca mean/components + best multi-component GMM)."""
    import diptest
    from sklearn.mixture import GaussianMixture

    X = actions.reshape(len(actions), -1).astype(np.float64)
    mu = X.mean(0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    ev = S**2 / (len(X) - 1)
    cum = np.cumsum(ev) / ev.sum()
    nc = int(np.searchsorted(cum, var_frac) + 1)
    nc = min(nc, len(X) // 4)
    Z = Xc @ Vt[:nc].T
    bics, gmms = [], []
    for c in (1, 2, 3):
        g = GaussianMixture(c, covariance_type="full", n_init=5, random_state=rng_seed, reg_covar=1e-6).fit(Z)
        bics.append(g.bic(Z))
        gmms.append(g)
    best_multi = 1 if bics[1] <= bics[2] else 2
    dbic = bics[0] - bics[best_multi]
    dip, p = diptest.diptest(Z[:, 0])
    clustered = bool(dbic > dbic_min and p < dip_alpha)
    return {
        "n_pc": nc, "ev": ev[:10].tolist(), "bic": bics, "dbic": float(dbic), "n_comp": best_multi + 1,
        "dip": float(dip), "dip_p": float(p), "clustered": clustered,
        "labeler": {"mu": mu, "V": Vt[:nc], "gmm": gmms[best_multi]},
        "pcs": Vt[:5], "pc_ev": ev[:5],
    }  # fmt: skip


def cluster_label_onehot(labeler, actions):
    X = actions.reshape(len(actions), -1).astype(np.float64)
    Z = (X - labeler["mu"]) @ labeler["V"].T
    lab = labeler["gmm"].predict(Z)
    return np.eye(labeler["gmm"].n_components)[lab]
