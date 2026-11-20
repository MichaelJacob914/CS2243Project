"""Noise generators of scripts/pi0/noise_pattern_eval.py (CPU): shapes, tiling vs iid, that `agent` is the fresh
agent's own exploration distribution, and that `agent_iid` has the same per-element marginal."""

import os
import sys

import numpy as np
import pytest
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "pi0"))
import noise_pattern_eval as npe  # noqa: E402

import jax  # noqa: E402

H, D, B = npe.H, npe.D, 12


def _batch(n, seed=0):
    """What run_episodes hands act_fn with xt_full: pixels (n,64,64,3,1) uint8, state (n,8,1), x_T (n,1600)."""
    r = np.random.default_rng(seed)
    return {"pixels": r.integers(0, 256, (n, 64, 64, 3, 1), dtype=np.uint8),
            "state": r.standard_normal((n, 8, 1)).astype(np.float32),
            "x_T": r.standard_normal((n, H * D)).astype(np.float32)}


@pytest.fixture(scope="module", params=[0, 6])
def fresh(request):
    k = request.param
    agent, use_xt = npe.build_fresh_agent(k)
    return k, agent, use_xt, npe.make_agent_iid_sampler(agent)


def _draw(fresh, pattern, n_calls, batch, seed=1):
    k, agent, use_xt, sampler = fresh
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    agent.set_rng(key)
    f = npe.make_act_fn(pattern, k, rng, key=key, agent=agent, use_xt=use_xt, iid_sampler=sampler)
    out = [f(batch) for _ in range(n_calls)]
    for kk, _w, probs in out:
        assert probs is None and kk.shape == (batch["state"].shape[0],) and (kk == k).all()
    return np.stack([w for _, w, _ in out])  # (calls, B, H, D)


def _is_tiled(W):
    return np.array_equal(W, np.broadcast_to(W[..., :1, :], W.shape))


def test_conditions_and_parse():
    assert len(npe.CONDITIONS) == 16 and len(set(npe.CONDITIONS)) == 16
    assert npe.parse_cond("agent_iid_k8") == ("agent_iid", 8) and npe.parse_cond("iid_k0") == ("iid", 0)


def test_gaussian_patterns():
    rng = np.random.default_rng(0)
    for tiled in (True, False):
        W = np.stack([npe.gaussian_w(rng, B, tiled) for _ in range(200)])
        assert W.shape == (200, B, H, D) and W.dtype == np.float32
        assert _is_tiled(W) == tiled
        x = W[:, :, 0].ravel() if tiled else W.ravel()  # independent elements
        assert abs(x.mean()) < 0.03 and abs(x.std() - 1) < 0.03
        assert stats.kstest(x, "norm").pvalue > 1e-3
    W = np.stack([npe.gaussian_w(rng, B, False) for _ in range(200)])
    c = np.corrcoef(W[:, :, 0].ravel(), W[:, :, 1].ravel())[0, 1]
    assert abs(c) < 0.03  # iid across time


def test_fresh_agent_is_sweep_agent(fresh):
    k, agent, use_xt, _ = fresh
    from adsrl_pi0.agent import FixedKAgent

    assert isinstance(agent, FixedKAgent) and agent.k == k and use_xt == (k != 0)
    assert agent.action_chunk_shape == (1, D)
    L = agent.learner
    assert L.action_dim == D
    # the actor's TanhMultivariateNormalDiag with low/high = -/+ action_magnitude (1.0 in the sweep)
    loc, scale = npe.actor_dist_params(agent, jax.device_put(npe.agent_obs(_batch(4), use_xt)))
    assert loc.shape == scale.shape == (4, D)
    print(f"\n[k={k}] fresh agent pre-tanh loc: mean {loc.mean():.4f}, |loc| max {np.abs(loc).max():.4f}; "
          f"scale: mean {scale.mean():.4f}, range [{scale.min():.4f}, {scale.max():.4f}]")


def test_agent_pattern_is_agent_act(fresh):
    """`agent` = FixedKAgent.act(obs, deterministic=False) (the training/eval call), tiled over H."""
    k, agent, use_xt, _ = fresh
    batch = _batch(B)
    W = _draw(fresh, "agent", 3, batch, seed=5)
    assert W.shape == (3, B, H, D) and _is_tiled(W)
    agent.set_rng(jax.random.PRNGKey(5))
    for i in range(3):
        _, w_ref, _ = agent.act(jax.device_put(npe.agent_obs(batch, use_xt)), deterministic=False)
        np.testing.assert_array_equal(W[i, :, 0], np.asarray(w_ref))


def _std_normal_scores(fresh, W, batch):
    """(atanh(w) - loc) / scale for each element: exactly N(0,1) iff w ~ tanh(N(loc, scale)) (magnitude 1)."""
    _, agent, use_xt, _ = fresh
    loc, scale = npe.actor_dist_params(agent, jax.device_put(npe.agent_obs(batch, use_xt)))
    u = np.arctanh(np.clip(W.astype(np.float64), -1 + 1e-7, 1 - 1e-7))
    return (u - loc[None, :, None, :]) / scale[None, :, None, :]


def test_agent_patterns_match_actor_distribution_and_each_other(fresh):
    k, _, _, _ = fresh
    batch = _batch(B, seed=3)
    Wc = _draw(fresh, "agent", 300, batch, seed=11)  # (300, B, H, D), tiled
    Wd = _draw(fresh, "agent_iid", 25, batch, seed=12)  # (25, B, H, D), iid over H
    assert np.abs(Wc).max() < 1 and np.abs(Wd).max() < 1  # tanh bound, magnitude 1
    assert _is_tiled(Wc) and not _is_tiled(Wd)
    # (c) tiled: use one row per draw; (d): every element is its own draw
    zc = _std_normal_scores(fresh, Wc, batch)[:, :, 0]
    zd = _std_normal_scores(fresh, Wd, batch)
    for z, name in ((zc, "agent"), (zd, "agent_iid")):
        z = z.ravel()
        assert abs(z.mean()) < 0.03 and abs(z.std() - 1) < 0.03, (name, z.mean(), z.std())
        assert stats.kstest(z, "norm").pvalue > 1e-3, name
    # same per-element marginal (pooled over rows/dims, and per dim for a few dims)
    xc, xd = Wc[:, :, 0].ravel(), Wd.ravel()
    assert stats.ks_2samp(xc, xd).pvalue > 1e-3
    for j in (0, 7, 31):
        assert stats.ks_2samp(Wc[:, :, 0, j].ravel(), Wd[..., j].ravel()).pvalue > 1e-3
    # (d) independent across time: lag-1 correlation ~ 0 (centre by per-row/dim mean)
    Z = zd - zd.mean(axis=(0, 2), keepdims=True)
    c = np.corrcoef(Z[:, :, :-1].ravel(), Z[:, :, 1:].ravel())[0, 1]
    assert abs(c) < 0.03, c
    print(f"\n[k={k}] agent w: std {xc.std():.4f}, mean|w| {np.abs(xc).mean():.4f}, rms {np.sqrt((xc**2).mean()):.4f}; "
          f"agent_iid w: std {xd.std():.4f}; tanh(N(0,1)) std would be "
          f"{np.tanh(np.random.default_rng(0).standard_normal(10**6)).std():.4f}")


def test_w_stats():
    class E:
        pass

    e = E()
    rng = np.random.default_rng(0)
    e.w = [npe.gaussian_w(rng, 1, True)[0] for _ in range(5)]
    assert npe.w_stats([e])["w_tile_dev"] == 0
    e.w = [npe.gaussian_w(rng, 1, False)[0] for _ in range(5)]
    assert npe.w_stats([e])["w_tile_dev"] > 0.5
