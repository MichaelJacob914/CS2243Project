"""Checks of the exp14 estimators against closed forms (CPU, seconds).

    python scripts/exp14/test_windows.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402,F401
import windows as W  # noqa: E402


def r_null(ab, lam):
    return ab * lam / (ab * lam + 1 - ab)


def test_closed_form_width():
    x = np.linspace(-15, 15, 3001)
    ab = 1 / (1 + np.exp(-x))
    R = r_null(ab, 1.0)
    st = W.curve_stats(x, R)
    assert abs(st["width"] - 2 * np.log(9)) < 0.02, st["width"]  # 4.39 log-SNR units
    assert abs(st["mid"]) < 0.02
    print("closed-form null width", st["width"])


def test_icc_matches_closed_form():
    rng = np.random.default_rng(0)
    lam = np.array([2.0, 0.3, 0.05, 0.01])
    D = len(lam)
    samples = rng.standard_normal((20000, D)) * np.sqrt(lam) + 0.3
    null = W.GaussNull(samples.reshape(-1, 1, D))
    x = np.linspace(-6, 7, 14)
    ab = 1 / (1 + np.exp(-x))
    P, M = 400, 16
    eps = rng.standard_normal((P, D))
    clean = null.sample(P, rng).reshape(P, D)
    for i, l in enumerate(null.lam):
        Rs = []
        for a in ab:
            y = np.sqrt(a) * clean + np.sqrt(1 - a) * eps
            post = null.posterior_sample(y, a, M, rng).reshape(P, M, D)
            proj = (post - null.mu) @ null.U[:, i]
            Rs.append(float(W.resolved_fraction(proj[..., None])))
        err = np.abs(np.array(Rs) - r_null(ab, l)).max()
        assert err < 0.06, (l, err, Rs)
        # open fraction under the exact ODE map is 1 - R for a linear feature
        B, C = 50, 32
        base = null.sample(B, rng).reshape(B, D)
        e = rng.standard_normal((B, C, D))
        ofr = []
        for a in ab:
            y = np.sqrt(a) * np.repeat(base, C, 0) + np.sqrt(1 - a) * e.reshape(B * C, D)
            out = null.ode_map(y, a).reshape(B, C, D)
            proj = ((out - null.mu) @ null.U[:, i])[..., None]
            ofr.append(float(W.open_fraction(proj, l)))
        err2 = np.abs(np.array(ofr) - (1 - r_null(ab, l))).max()
        assert err2 < 0.08, (l, err2)
        print(f"lambda={l:.3f}: max |R_hat - R_null| = {err:.3f}, max |O_hat - (1-R_null)| = {err2:.3f}")


def test_bootstrap_shapes_and_two_rise():
    rng = np.random.default_rng(1)
    L, P, M, d = 20, 16, 16, 3
    f = rng.standard_normal((L, P, M, d))
    fbar, wvar = W.point_summary(f)
    b = W.bootstrap_resolved(fbar, wvar, M, 200, rng)
    assert b.shape == (200, L)
    x = np.linspace(-5.5, 7.4, 20)
    two = 0.5 / (1 + np.exp(-(x + 3) / 0.3)) + 0.5 / (1 + np.exp(-(x - 4) / 0.3))
    t = W.two_rise_test(x, two + rng.normal(0, 0.01, 20))
    one = W.two_rise_test(x, 1 / (1 + np.exp(-(x - 1) / 1.0)) + rng.normal(0, 0.01, 20))
    assert t["preferred"] and not one["preferred"], (t, one)
    print("two-rise: plateau dBIC", round(t["dbic"], 1), "single dBIC", round(one["dbic"], 1))


def test_gram_bootstrap_equals_direct():
    rng = np.random.default_rng(7)
    for d in (1, 5, 112):
        f = rng.standard_normal((6, 64, 16, d)) * rng.uniform(0.1, 2, (6, 1, 1, 1)) + rng.standard_normal((6, 64, 1, d))
        fbar, wvar = W.point_summary(f)
        a = W.bootstrap_resolved(fbar, wvar, 16, 200, np.random.default_rng(3))
        b = W.bootstrap_resolved_gram(fbar, wvar, 16, 200, np.random.default_rng(3))
        assert np.allclose(a, b, atol=1e-9), (d, np.abs(a - b).max())
    print("gram bootstrap == direct bootstrap")


def test_crossing_censoring():
    x = np.arange(5.0)
    v, c, n = W.crossing(x, np.array([[0.2, 0.3, 0.5, 0.95, 1.0], [0.0, 0.0, 0.05, 0.06, 0.07]]), 0.1)
    assert np.isnan(v[0]) and c[0] and np.isnan(v[1]) and n[1]
    v, _, _ = W.crossing(x, np.array([0.0, 0.0, 0.2, 0.6, 1.0]), 0.5)
    assert abs(v - 2.75) < 1e-9


if __name__ == "__main__":
    test_closed_form_width()
    test_icc_matches_closed_form()
    test_bootstrap_shapes_and_two_rise()
    test_crossing_censoring()
    test_gram_bootstrap_equals_direct()
    print("ALL OK")
