"""exp14 per-state measurement runners shared by the toy control (Stage 0) and transport (Stages 1, 2a, 2b).

feat_fn(actions: np.ndarray (N, H, A)) -> dict[name, np.ndarray (N, d)] computes every feature on a batch
of clean actions at ONE state (it closes over the state, e.g. for the critic's obs).
"""

from __future__ import annotations

import numpy as np
import torch

import windows as W


def _feats_to_summary(feats_list, P, M):
    """feats_list: list over levels of dict name -> (P*M, d). -> name -> (fbar (L,P,d), wvar (L,P))."""
    out = {}
    for name in feats_list[0]:
        f = np.stack([fl[name].reshape(P, M, -1) for fl in feats_list]).astype(np.float64)  # (L,P,M,d)
        out[name] = W.point_summary(f)
    return out


@torch.no_grad()
def stage2a_policy(model, obs, clean, eps, levels, M, gen, feat_fn, keep_actions=False):
    """Resolved-fraction design with the stochastic sampler.
    obs (obs_dim,) tensor; clean (P, H, A) tensor of stochastic samples; eps (P, H, A) unit noise reused at
    every level; levels: DDPM indices. Returns (summary dict, regenerated actions (L,P,M,H,A) or None)."""
    P = clean.shape[0]
    cond = {"state": obs[None].expand(P * M, -1)}
    feats, acts = [], []
    for t in levels:
        x_t = W.q_sample(model, clean, int(t), eps)  # (P, H, A)
        x = x_t.repeat_interleave(M, 0)
        a = W.ddpm_regenerate(model, cond, x, int(t), gen)
        a_np = a.float().cpu().numpy()
        feats.append(feat_fn(a_np))
        if keep_actions:
            acts.append(a_np.reshape(P, M, *a_np.shape[1:]))
    return _feats_to_summary(feats, P, M), (np.stack(acts) if keep_actions else None)


@torch.no_grad()
def stage2a_null(null_model, P, eps, levels, M, gen, feat_fn, keep_actions=False):
    """Same design on the Gaussian null, by the IDENTICAL procedure as the policy: clean points from the same
    stochastic chain (ddpm_sample from t = T-1) and regenerations through the same ddpm_regenerate, with the
    null's closed-form epsilon-predictor (windows.GaussEpsModel). eps (P, H, A): unit noise reused per level."""
    obs = torch.zeros(1, device=eps.device)  # the closed-form epsilon ignores cond
    clean = W.ddpm_sample(null_model, {"state": obs[None]}, P, gen)[0]
    return stage2a_policy(null_model, obs, clean, eps, levels, M, gen, feat_fn, keep_actions)


@torch.no_grad()
def stage2b_from_base(model, obs, base, eps, ks, feat_fn, keep_actions=False):
    """Plan form of the open-fraction design from given base actions base (B, H, A):
    x = sqrt(ab_k) a_b + sqrt(1-ab_k) eps_bc, then deterministic DDIM steps k..K-1 (evaluation clamps).
    Returns name -> (L, B, C, d) [, actions (L, B, C, H, A)]."""
    B, C = eps.shape[:2]
    cond = {"state": obs[None].expand(B * C, -1)}
    ab = model.ddim_alphas
    base_rep = base.repeat_interleave(C, 0)
    eps_f = eps.reshape(B * C, *eps.shape[2:])
    res, acts = [], []
    for k in ks:
        a_k = float(ab[k])
        x = (a_k**0.5) * base_rep + ((1 - a_k) ** 0.5) * eps_f
        a = W.ddim_from_step(model, cond, x, int(k)).float().cpu().numpy()
        res.append({n: v.reshape(B, C, -1) for n, v in feat_fn(a).items()})
        if keep_actions:
            acts.append(a.reshape(B, C, *a.shape[1:]))
    out = {n: np.stack([d[n] for d in res]).astype(np.float64) for n in res[0]}
    return (out, np.stack(acts)) if keep_actions else out


@torch.no_grad()
def stage2b_policy(model, obs, x_T_base, eps, ks, feat_fn, steer_w=None, keep_actions=False):
    """Open-fraction design with the deterministic DDIM sampler.
    x_T_base (B, H, A): base x_T; base actions = DDIM(x_T_base). eps (B, C, H, A) unit noise reused per level.
    ks: DDIM step indices. Plan form via stage2b_from_base. If steer_w is given (B, C, H, A), also run the
    steering form steered_ddim_sample(x_T_b, k, w_bc). Returns (dict form -> name -> (L, B, C, d), base
    actions, optional actions)."""
    from adsrl.sampler import steered_ddim_sample

    B, C = eps.shape[:2]
    base = W.ddim_sample(model, {"state": obs[None].expand(B, -1)}, x_T_base)  # (B, H, A)
    out, acts = {}, {}
    r = stage2b_from_base(model, obs, base, eps, ks, feat_fn, keep_actions)
    if keep_actions:
        out["plan"], acts["plan"] = r
    else:
        out["plan"] = r
    if steer_w is not None:
        cond = {"state": obs[None].expand(B * C, -1)}
        xT_rep = x_T_base.repeat_interleave(C, 0)
        w = steer_w.reshape(B * C, *steer_w.shape[2:])
        res, sa = [], []
        for k in ks:
            kk = torch.full((B * C,), int(k), dtype=torch.long, device=xT_rep.device)
            a2 = steered_ddim_sample(model, cond, xT_rep, kk, w).float().cpu().numpy()
            res.append({n: v.reshape(B, C, -1) for n, v in feat_fn(a2).items()})
            if keep_actions:
                sa.append(a2.reshape(B, C, *a2.shape[1:]))
        out["steer"] = {n: np.stack([d[n] for d in res]).astype(np.float64) for n in res[0]}
        if keep_actions:
            acts["steer"] = np.stack(sa)
    return out, base.float().cpu().numpy(), (acts if keep_actions else None)


def open_curves(f_lbcd, tv):
    """(L, B, C, d) -> (L,) open fraction."""
    return W.open_fraction(f_lbcd, tv)


@torch.no_grad()
def stage2b_null(null_model, x_T_base, eps, ks, feat_fn):
    """Open fraction of the Gaussian null by the IDENTICAL procedure as the policy's plan form: base actions =
    the same DDIM sampler from x_T_base (B, H, A), then the same noising and DDIM path (same clamps), with the
    null's closed-form epsilon. eps (B, C, H, A) torch."""
    obs = torch.zeros(1, device=eps.device)
    return stage2b_policy(null_model, obs, x_T_base, eps, ks, feat_fn)[0]["plan"]


def curves_from_summary(summ, M, n_boot, rng):
    """summ: name -> (fbar, wvar). -> name -> {"R": (L,), "boot": (n_boot, L)}."""
    out = {}
    for name, (fbar, wvar) in summ.items():
        out[name] = {"R": W.resolved_from_summary(fbar, wvar, M), "boot": W.bootstrap_resolved(fbar, wvar, M, n_boot, rng)}
    return out


def energy_test(X, Y, n_perm=500, rng=None):
    """Two-sample energy-distance permutation test on flattened samples. Returns (stat, p)."""
    rng = rng or np.random.default_rng(0)
    Z = np.concatenate([X.reshape(len(X), -1), Y.reshape(len(Y), -1)]).astype(np.float64)
    n = len(X)
    D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))

    def stat(idx):
        a, b = idx[:n], idx[n:]
        return 2 * D[np.ix_(a, b)].mean() - D[np.ix_(a, a)].mean() - D[np.ix_(b, b)].mean()

    idx = np.arange(len(Z))
    s0 = stat(idx)
    cnt = sum(stat(rng.permutation(len(Z))) >= s0 for _ in range(n_perm))
    return float(s0), float((cnt + 1) / (n_perm + 1))
