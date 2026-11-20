"""Tests for the JAX hybrid (adaptive-k) SAC agent and the fixed-k wrapper (adsrl_pi0/agent.py). CPU is fine."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import adsrl_pi0  # noqa: E402

adsrl_pi0.setup_env()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax.core import frozen_dict  # noqa: E402

from adsrl_pi0.agent import (FixedKAgent, HybridPixelSACLearner, hybrid_actor_loss, restore_agent,  # noqa: E402
                             save_agent, squashed_sample)
from jaxrl2.networks.learned_std_normal_policy import TanhMultivariateNormalDiag  # noqa: E402

B, K, D = 8, 10, 32
KW = dict(encoder_type="small", hidden_dims=(32, 32), latent_dim=16, cnn_features=(8, 8), cnn_strides=(2, 2), num_qs=3,
          discount=0.999)


def _obs(n, xt=True, seed=0):
    rng = np.random.default_rng(seed)
    o = {"pixels": rng.integers(0, 256, (n, 64, 64, 3, 1), dtype=np.uint8),
         "state": rng.standard_normal((n, 8, 1)).astype(np.float32)}
    if xt:
        o["x_T"] = rng.standard_normal((n, D)).astype(np.float32)
    return o


def _batch(n, with_k=True, allowed=range(K)):
    rng = np.random.default_rng(1)
    b = {"observations": _obs(n, seed=2), "next_observations": _obs(n, seed=3),
         "actions": rng.uniform(-1, 1, (n, 1, D)).astype(np.float32),
         "rewards": -np.ones(n, np.float32), "masks": np.ones(n, np.float32),
         "discount": np.full(n, 0.999**20, np.float32)}
    if with_k:
        b["k"] = rng.choice(np.array(list(allowed)), n).astype(np.int32)
    return frozen_dict.freeze(b)


@pytest.fixture(scope="module")
def agent():
    return HybridPixelSACLearner(0, _obs(1), D, K, list(range(K)), rho=0.5, **KW)


def test_squashed_logprob_matches_distrax():
    key = jax.random.key(0)
    mean = jax.random.normal(key, (5, D))
    log_std = 0.3 * jax.random.normal(jax.random.key(1), (5, D))
    w, lp = squashed_sample(jax.random.key(2), mean, log_std, 1.0)
    # magnitude 1: the rescale layer is the identity; distrax cannot invert its Lambda bijector, so compare to tanh
    ref = TanhMultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std)).log_prob(w)
    np.testing.assert_allclose(lp, ref, rtol=1e-4, atol=1e-3)
    assert np.all(np.abs(w) <= 1)


def test_probs_sum_to_one_and_shapes(agent):
    log_pk, mean, log_std = agent._actor.apply_fn({"params": agent._actor.params}, _obs(B))
    assert log_pk.shape == (B, K) and mean.shape == (B, K, D) and log_std.shape == (B, K, D)
    np.testing.assert_allclose(np.exp(log_pk).sum(-1), 1.0, atol=1e-5)
    k, w, probs = agent.act(_obs(B))
    assert k.shape == (B,) and w.shape == (B, D) and probs.shape == (B, K)
    assert set(k.tolist()) <= set(range(K))
    np.testing.assert_allclose(probs.sum(-1), 1.0, atol=1e-5)
    kd, wd, _ = agent.act(_obs(B), deterministic=True)
    np.testing.assert_array_equal(kd, np.argmax(probs, -1))


def _grads(agent, alpha_k=0.1):
    def f(p):
        return hybrid_actor_loss(p, agent._actor.apply_fn, agent._critic.params, agent._critic.apply_fn, _obs(B),
                                 jax.random.key(0), 0.1, alpha_k, agent.allowed_k, K, 1.0, agent.freeze_k_uniform,
                                 "mean")[0]
    return jax.grad(f)(agent._actor.params)


def test_gradients_reach_both_heads(agent):
    g = _grads(agent)["network"]
    dense = {k: v for k, v in g.items() if k.startswith("Dense")}
    # HybridPolicy Dense_0 = k logits, Dense_1 = w mean, Dense_2 = w log_std
    for name in ["Dense_0", "Dense_1", "Dense_2"]:
        assert float(jnp.abs(dense[name]["kernel"]).sum()) > 0, name
    g_all = _grads(agent)
    enc = jax.tree_util.tree_leaves(g_all[next(k for k in g_all if "ncoder" in k)])
    assert sum(float(jnp.abs(x).sum()) for x in enc) > 0


def test_k_subset_and_onehot():
    allowed = [0, 3, 6]
    ag = HybridPixelSACLearner(0, _obs(1), D, K, allowed, rho=0.5, **KW)
    k, _, probs = ag.act(_obs(B))
    assert set(k.tolist()) <= set(allowed)
    assert np.allclose(probs[:, [1, 2, 4, 5, 7, 8, 9]], 0)
    np.testing.assert_allclose(ag.target_entropy_k, 0.5 * np.log(3))
    info = ag.update(_batch(B, allowed=allowed))
    assert np.isfinite(float(info["critic_loss"])) and np.isfinite(float(info["actor_loss"]))


def test_update_changes_params_and_temps(agent):
    before = jax.tree_util.tree_leaves(agent._actor.params)
    tk0 = float(agent._temp_k.apply_fn({"params": agent._temp_k.params}))
    for _ in range(3):
        info = agent.update(_batch(B))
    after = jax.tree_util.tree_leaves(agent._actor.params)
    assert any(float(jnp.abs(a - b).max()) > 0 for a, b in zip(before, after, strict=True))
    assert float(agent._temp_k.apply_fn({"params": agent._temp_k.params})) != tk0
    for key in ["critic_loss", "actor_loss", "entropy", "k_entropy", "temperature", "k_temperature"]:
        assert np.isfinite(float(info[key])), key


def test_freeze_uniform():
    ag = HybridPixelSACLearner(0, _obs(1), D, K, list(range(K)), freeze_k_uniform=True, **KW)
    _, _, probs = ag.act(_obs(B))
    np.testing.assert_allclose(probs, 1.0 / K, atol=1e-6)
    g = _grads(ag)["network"]
    assert float(jnp.abs(g["Dense_0"]["kernel"]).sum()) == 0


def test_fixed_k_agent_and_checkpoint(tmp_path):
    ag = FixedKAgent(0, K, 0, _obs(1, xt=False), np.zeros((1, 1, D), np.float32), **KW)
    k, w, probs = ag.act(_obs(B, xt=False))
    assert (k == 0).all() and w.shape == (B, D) and np.all(np.abs(w) <= 1)
    ag.update(_batch(B, with_k=False).copy({"observations": _obs(B, xt=False),
                                            "next_observations": _obs(B, xt=False, seed=4)}))
    save_agent(ag, str(tmp_path), 7)
    ag2 = FixedKAgent(0, K, 1, _obs(1, xt=False), np.zeros((1, 1, D), np.float32), **KW)
    assert restore_agent(ag2, str(tmp_path)) == 7
    for a, b in zip(jax.tree_util.tree_leaves(ag.learner._actor.params),
                    jax.tree_util.tree_leaves(ag2.learner._actor.params), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_hybrid_checkpoint_roundtrip(agent, tmp_path):
    save_agent(agent, str(tmp_path), 3)
    ag2 = HybridPixelSACLearner(5, _obs(1), D, K, list(range(K)), rho=0.5, **KW)
    assert restore_agent(ag2, str(tmp_path)) == 3
    k1, w1, _ = agent.act(_obs(B), deterministic=True)
    k2, w2, _ = ag2.act(_obs(B), deterministic=True)
    np.testing.assert_array_equal(k1, k2)
    np.testing.assert_allclose(w1, w2, atol=1e-6)


def test_broadcast_critic_matches_repeated(agent):
    """Q(s, k, w) over all heads in one broadcast call equals the plain per-row critic on repeated obs."""
    obs = _obs(B)
    kw = np.random.default_rng(0).uniform(-1, 1, (B, K, K + D)).astype(np.float32)
    apply = agent._critic.apply_fn
    q_b = apply({"params": agent._critic.params}, obs, jnp.asarray(kw)[:, :, None])  # [n_qs, B, K]
    obs_rep = {k: np.repeat(v, K, axis=0) for k, v in obs.items()}
    q_r = apply({"params": agent._critic.params}, obs_rep, jnp.asarray(kw.reshape(B * K, K + D)))
    np.testing.assert_allclose(np.asarray(q_b), np.asarray(q_r).reshape(-1, B, K), atol=1e-5)
