"""Regression test for review item 1: the Stage 2a / 2b null must run the identical procedure as the policy.

Both arms get the SAME distribution: the policy arm is a GaussEpsModel (closed-form epsilon of N(mu, Sigma))
run through measure.stage2a_policy / stage2b_policy; the null arm is measure.stage2a_null / stage2b_null for
the same N(mu, Sigma). With no window anywhere, their R (and open-fraction) curves must agree within Monte
Carlo error at every level, for small (0.003-0.02) and large eigenvalues.

Also checks the closed-form epsilon itself: through the DDPM path, R at mid levels of a large-lambda direction
matches the closed form ab*lam/(ab*lam + 1 - ab).

    python scripts/exp14/test_null_path.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402,F401
import measure  # noqa: E402
import windows as W  # noqa: E402

H, A = 4, 2
LAM = np.array([2.0, 0.5, 0.02, 0.01, 0.005, 0.003, 0.003, 0.003])
P, M = 512, 16
TOL = 0.06  # closed-form check tolerance
ZMAX = 4.5  # max |difference| / combined bootstrap SE over 20 levels x 8 features (both arms identical in law)


def template_model():
    import adsrl  # noqa: F401
    from model.diffusion.diffusion import DiffusionModel
    from model.diffusion.mlp_diffusion import DiffusionMLP

    net = DiffusionMLP(action_dim=A, horizon_steps=H, cond_dim=1, mlp_dims=[8])
    return DiffusionModel(network=net, horizon_steps=H, obs_dim=1, action_dim=A, device="cpu", denoised_clip_value=1.0,
                          randn_clip_value=3, denoising_steps=100, predict_epsilon=True, use_ddim=True, ddim_steps=20)  # fmt: skip


def setup():
    rng = np.random.default_rng(0)
    D = H * A
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    mu = rng.uniform(-0.3, 0.3, D)
    samples = mu + (rng.standard_normal((200000, D)) * np.sqrt(LAM)) @ Q.T
    null = W.GaussNull(samples.reshape(-1, H, A))
    tm = template_model()
    gm = W.GaussEpsModel(tm, null)
    U = null.U  # columns = eigenvectors, ascending eigenvalue

    def feat(a):
        X = a.reshape(len(a), -1).astype(np.float64)
        Z = (X - null.mu) @ U
        return {f"e{i}_lam{null.lam[i]:.3f}": Z[:, i : i + 1] for i in range(D)}

    return rng, null, tm, gm, feat


def _new_api(fn):
    import inspect

    return list(inspect.signature(fn).parameters)[0] == "null_model"


def null_arm_2a(null, gm, levels, eps_t, rng, gen, feat):
    clean = torch.as_tensor(null.sample(P, rng), dtype=torch.float32)
    if _new_api(measure.stage2a_null):  # fixed API: identical procedure incl. clean points from the chain
        summ, _ = measure.stage2a_null(gm, P, eps_t, levels, M, gen, feat)
    else:  # pre-fix API: exact posterior draws
        ab = W.abar(gm)[levels]
        summ, _ = measure.stage2a_null(null, ab, P, M, eps_t.reshape(P, -1).numpy().astype(np.float64), rng, feat)
    return summ


def policy_arm_2b(gm, obs, base, eps, ks, feat):
    """Plan form of 2b written out here (no dependency on the code under test beyond ddim_from_step)."""
    B, C = eps.shape[:2]
    cond = {"state": obs[None].expand(B * C, -1)}
    out = []
    for k in ks:
        a_k = float(gm.ddim_alphas[k])
        x = (a_k**0.5) * base.repeat_interleave(C, 0) + ((1 - a_k) ** 0.5) * eps.reshape(B * C, H, A)
        a = W.ddim_from_step(gm, cond, x, int(k)).numpy()
        out.append({n: v.reshape(B, C, -1) for n, v in feat(a).items()})
    return {n: np.stack([d[n] for d in out]).astype(np.float64) for n in out[0]}


def null_arm_2b(gm, null, B, eps, ks, rng, feat, gen=None):
    if _new_api(measure.stage2b_null):
        return measure.stage2b_null(gm, torch.randn((B, H, A), generator=gen), eps, ks, feat)
    else:  # pre-fix API: continuous-time ODE map
        ab = gm.ddim_alphas.numpy()[ks]
        return measure.stage2b_null(null, ab, B, eps.reshape(B, eps.shape[1], -1).numpy().astype(np.float64), rng, feat)


def test_2a():
    rng, null, tm, gm, feat = setup()
    levels = W.ddim_levels(tm)
    gen = torch.Generator().manual_seed(1)
    obs = torch.zeros(1)
    clean_pol = W.ddpm_sample(gm, {"state": obs[None]}, P, gen)[0]  # policy arm: the policy's own chain
    eps_pol = torch.randn(clean_pol.shape, generator=gen)
    sp, _ = measure.stage2a_policy(gm, obs, clean_pol, eps_pol, levels, M, gen, feat)
    eps_null = torch.randn(clean_pol.shape, generator=gen)
    sn = null_arm_2a(null, gm, levels, eps_null, rng, gen, feat)
    worst = 0.0
    ab = W.abar(tm)[levels]
    brng = np.random.default_rng(5)
    for name in sp:
        Rp = W.resolved_from_summary(*sp[name], M)
        Rn = W.resolved_from_summary(*sn[name], M)
        se = np.sqrt(np.nanvar(W.bootstrap_resolved(*sp[name], M, 300, brng), 0) + np.nanvar(W.bootstrap_resolved(*sn[name], M, 300, brng), 0))
        z = np.abs(Rp - Rn) / (se + 0.005)  # 0.005 floor where both arms are pinned at 0 or 1
        print(f"2a {name}: max|dR| = {np.nanmax(np.abs(Rp - Rn)):.3f}, max |dR|/SE = {np.nanmax(z):.2f} at k={int(np.nanargmax(z))} "
              f"(R_pol k17-19 {np.round(Rp[17:], 2)}, R_null {np.round(Rn[17:], 2)})")  # fmt: skip
        worst = max(worst, float(np.nanmax(z)))
        lam = float(name.split("lam")[1])
        if lam >= 0.5:  # closed-form epsilon sanity at levels where the discretization is fine
            cf = ab * lam / (ab * lam + 1 - ab)
            e = np.abs(Rp[:13] - cf[:13]).max()
            assert e < TOL, (name, e)
    assert worst < ZMAX, f"policy path and null path disagree: max |dR|/SE = {worst:.2f}"


def test_2b():
    rng, null, tm, gm, feat = setup()
    gen = torch.Generator().manual_seed(2)
    B, C = 64, 32
    ks = list(range(20))
    obs = torch.zeros(1)
    xT1 = torch.randn((B, H, A), generator=gen)
    base = W.ddim_sample(gm, {"state": obs[None].expand(B, -1)}, xT1)  # policy arm: the policy's own DDIM sampler
    e1 = torch.randn((B, C, H, A), generator=gen)
    rp = policy_arm_2b(gm, obs, base, e1, ks, feat)
    e2 = torch.randn((B, C, H, A), generator=gen)
    rn = null_arm_2b(gm, null, B, e2, ks, rng, feat, gen)
    fresh = feat(W.ddim_sample(gm, {"state": torch.zeros(4096, 1)}, torch.randn((4096, H, A), generator=gen)).numpy())
    worst = 0.0
    for name in rp:
        tv = W.total_var(fresh[name])
        Op, On = W.open_fraction(rp[name], tv), W.open_fraction(rn[name], tv)
        # SE over base actions: O = mean_b (within-var_b) / tv
        wp, wn = rp[name].var(-2, ddof=1).sum(-1) / tv, rn[name].var(-2, ddof=1).sum(-1) / tv  # (L, B)
        se = np.sqrt(wp.var(1, ddof=1) / wp.shape[1] + wn.var(1, ddof=1) / wn.shape[1])
        z = np.abs(Op - On) / (se + 0.005)
        print(f"2b {name}: max|dO| = {np.nanmax(np.abs(Op - On)):.3f}, max |dO|/SE = {np.nanmax(z):.2f} at k={int(np.nanargmax(z))}")
        worst = max(worst, float(np.nanmax(z)))
    assert worst < ZMAX, worst


def test_width_test_censoring():
    rng = np.random.default_rng(3)
    a = rng.normal(3, 0.2, 1000)
    b = rng.normal(6, 0.5, 1000)
    assert W.width_test(a, b)["narrower"]
    a2 = a.copy()
    a2[:60] = np.nan  # 94% complete: must be censored, not silently dropped
    r = W.width_test(a2, b)
    assert r["censored"] and not r["narrower"], r
    a3 = a.copy()
    a3[:40] = np.nan  # 96% complete: fine
    assert not W.width_test(a3, b)["censored"]


if __name__ == "__main__":
    fails = []
    for t in (test_2a, test_2b, test_width_test_censoring):
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # report all, then fail
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
            fails.append(t.__name__)
    print("ALL OK" if not fails else f"FAILED: {fails}")
    sys.exit(1 if fails else 0)
