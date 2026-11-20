"""Tests for the steered pi0 flow sampler (adsrl_pi0/sampler.py). Run with .venv-pi0 inside a job.

Two model variants:
  dummy : randomly initialized Pi0 with the 'dummy' gemma variants in float32 (any device).
  ckpt  : the real pi0_libero checkpoint through the full SteeredPi0 pipeline (needs a GPU; skipped otherwise).
"""

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

from adsrl_pi0.sampler import NUM_STEPS, SteeredPi0, implied_noise_at, module_jit_fn, steered_sample_actions  # noqa: E402

B = 4
HAS_GPU = any(d.platform == "gpu" for d in jax.devices())


@pytest.fixture(scope="module")
def dummy():
    from openpi.models import pi0

    cfg = pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy", dtype="float32")
    model = cfg.create(jax.random.key(0))
    obs = cfg.fake_obs(B)
    obs = jax.tree.map(lambda x: x, obs)
    steer = module_jit_fn(steered_sample_actions, model, static_argnames=("num_steps",))
    plain = module_jit_fn(lambda m, o, noise: m.sample_actions(o, noise=noise), model)
    return model, obs, steer, plain


def _noise(seed, shape=(B, 50, 32)):
    return np.asarray(jax.random.normal(jax.random.key(seed), shape), np.float32)


def test_k0_equals_openpi_sample_actions(dummy):
    model, obs, steer, plain = dummy
    w, x_T = _noise(1), _noise(2)
    ref = np.asarray(plain(obs, w))
    out = np.asarray(steer(obs, x_T, np.zeros(B, np.int32), w))
    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=0)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("k", list(range(1, NUM_STEPS)))
def test_implied_noise_is_a_noop(dummy, k):
    """w = eps_hat_{k-1} (the model's implied noise on the unsteered path) reproduces the unsteered sample."""
    model, obs, steer, plain = dummy
    x_T = _noise(3)
    ref = np.asarray(plain(obs, x_T))
    imp = module_jit_fn(implied_noise_at, model, static_argnames=("k_target",))
    w = np.asarray(imp(obs, x_T, k_target=k))
    out = np.asarray(steer(obs, x_T, np.full(B, k, np.int32), w))
    # The re-noised state equals the Euler state up to float32 rounding (~1e-7); the dummy model still runs parts
    # of the network in bf16, which amplifies that over the remaining 10-k steps to ~1e-3 (observed max 8e-4).
    err = np.abs(out - ref).max()
    other = np.asarray(steer(obs, x_T, np.full(B, k, np.int32), _noise(4)))
    effect = np.abs(other - ref).max()  # what a different w does at the same k
    print(f"k={k}: max|steered(eps_hat) - unsteered| = {err:.2e}, effect of a random w = {effect:.2e}")
    assert err < 5e-3 and err < 1e-2 * effect, (err, effect)


@pytest.mark.parametrize("k", [0, 1, 4, 9])
def test_zero_w_renoise_state(dummy, k):
    """w = 0 at step k sets x_k = (1 - t_k) * a_hat_{k-1} (x_0 = 0 for k = 0); states before k are unchanged.
    Checked on the returned trajectory: (1 - t_k) a_hat_{k-1} = x_k(unsteered) - t_k eps_hat_{k-1}, since the
    unsteered x_k = t_k eps_hat_{k-1} + (1 - t_k) a_hat_{k-1}."""
    model, obs, steer, plain = dummy
    x_T = _noise(21)
    zero = np.zeros_like(x_T)
    traj_fn = module_jit_fn(steered_sample_actions, model, static_argnames=("num_steps", "return_trajectory"))
    out, traj = traj_fn(obs, x_T, np.full(B, k, np.int32), zero, return_trajectory=True)
    _, traj_plain = traj_fn(obs, x_T, np.zeros(B, np.int32), x_T, return_trajectory=True)  # k=0, w=x_T: unsteered
    traj, traj_plain = np.asarray(traj), np.asarray(traj_plain)
    np.testing.assert_array_equal(np.asarray(out), traj[-1])
    np.testing.assert_allclose(traj[:k], traj_plain[:k], atol=1e-6)  # identical before k
    if k == 0:
        np.testing.assert_array_equal(traj[0], 0.0)
    else:
        t_k = 1.0 - k / NUM_STEPS
        imp = module_jit_fn(implied_noise_at, model, static_argnames=("k_target",))
        eps_hat = np.asarray(imp(obs, x_T, k_target=k))
        np.testing.assert_allclose(traj[k], traj_plain[k] - t_k * eps_hat, atol=1e-5)
    assert np.isfinite(traj).all()


def test_mixed_k_batch_equals_per_k(dummy):
    model, obs, steer, plain = dummy
    x_T, w = _noise(5), _noise(6)
    ks = np.array([0, 3, 7, 9], np.int32)
    mixed = np.asarray(steer(obs, x_T, ks, w))
    for i, k in enumerate(ks):
        single = np.asarray(steer(obs, x_T, np.full(B, k, np.int32), w))
        np.testing.assert_allclose(mixed[i], single[i], atol=1e-6, rtol=0)


def test_later_k_moves_less(dummy):
    """Authority sanity check: spread of the final action over w shrinks as k grows (fixed x_T)."""
    model, obs, steer, plain = dummy
    x_T = _noise(7)
    spread = []
    for k in [0, 5, 9]:
        outs = [np.asarray(steer(obs, x_T, np.full(B, k, np.int32), _noise(100 + s))) for s in range(4)]
        spread.append(np.std(np.stack(outs), axis=0).mean())
    assert spread[0] > spread[1] > spread[2], spread


@pytest.fixture(scope="module")
def ckpt():
    if not HAS_GPU:
        pytest.skip("real pi0 checkpoint test needs a GPU")
    return SteeredPi0.load()


def _libero_obs(n, seed=0):
    rng = np.random.default_rng(seed)
    return [{"observation/state": rng.random(8).astype(np.float32),
             "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
             "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
             "prompt": "put both the alphabet soup and the tomato sauce in the basket"} for _ in range(n)]


def test_ckpt_k0_matches_policy_infer(ckpt):
    """k=0 through SteeredPi0 equals openpi Policy.infer(obs, noise=w), per example, in env action units."""
    obs = _libero_obs(3)
    w = _noise(11, (3, 50, 32))
    out = ckpt.infer(obs, _noise(12, (3, 50, 32)), np.zeros(3, np.int32), w)
    assert out.shape == (3, 50, 7)
    for i in range(3):
        ref = ckpt.policy.infer(obs[i], noise=jnp.asarray(w[i : i + 1]))["actions"]
        # same batch size (1) as openpi: should match to float precision
        single = ckpt.infer([obs[i]], _noise(15, (1, 50, 32)), np.zeros(1, np.int32), w[i : i + 1])[0]
        np.testing.assert_allclose(single, ref, atol=1e-5, rtol=0)
        # batch 3 (padded to 4) vs batch 1: the bf16 model is not batch-invariant (observed max 0.076 in env
        # units on H100, vs ~1 for a different noise); only a loose check here
        print(f"example {i}: max |batched - batch1| = {np.abs(out[i] - ref).max():.3f}")
        np.testing.assert_allclose(out[i], ref, atol=0.15, rtol=0)


def test_ckpt_mixed_batch_and_tiling(ckpt):
    obs = _libero_obs(4, seed=1)
    x_T, w = _noise(13, (4, 32)), _noise(14, (4, 32))
    ks = np.array([0, 2, 5, 8], np.int32)
    mixed = ckpt.infer(obs, x_T, ks, w)
    for i, k in enumerate(ks):
        single = ckpt.infer(obs, x_T, np.full(4, k, np.int32), w)
        np.testing.assert_allclose(mixed[i], single[i], atol=1e-5, rtol=0)


# ---- plan feature (exp9 --steer.obs_extra abase): the "prefix" half of a steered sample ----
def _features(model):
    from adsrl_pi0.sampler import plain_features

    return module_jit_fn(plain_features, model, static_argnames=("k", "num_steps"))


def test_plain_features_k0_is_plain_sample(dummy):
    model, obs, steer, plain = dummy
    x_T = _noise(31)
    np.testing.assert_allclose(np.asarray(_features(model)(obs, x_T, k=0)), np.asarray(plain(obs, x_T)), atol=1e-5)


@pytest.mark.parametrize("k", [1, 3, 9])
def test_plain_features_is_the_steered_samplers_a_hat(dummy, k):
    """Feature for k>0 = a_hat_{k-1} on the plain trajectory from x_T, i.e. the clean estimate the one-shot steered
    sampler re-noises at step k: with w = 0 its state at step k is (1 - t_k) a_hat_{k-1}. So prefix (feature) +
    suffix (the one-shot steered sampler, which recomputes the k plain steps) reproduce the one-shot sample."""
    model, obs, steer, plain = dummy
    x_T = _noise(32)
    traj_fn = module_jit_fn(steered_sample_actions, model, static_argnames=("num_steps", "return_trajectory"))
    _, traj = traj_fn(obs, x_T, np.full(B, k, np.int32), np.zeros_like(x_T), return_trajectory=True)
    t_k = np.float32(1.0)
    for _ in range(k):
        t_k = np.float32(t_k + np.float32(-1.0 / NUM_STEPS))
    ref = np.asarray(traj)[k] / (1.0 - t_k)
    out = np.asarray(_features(model)(obs, x_T, k=k))
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)
    assert np.abs(out - np.asarray(_features(model)(obs, _noise(33), k=k))).max() > 1e-2  # depends on x_T


def test_ckpt_plan_feature(ckpt):
    """SteeredPi0.plan_feature: (B, 50*7) real channels in normalized model space; k=0 equals the plain sample."""
    obs = _libero_obs(3, seed=2)
    x_T = _noise(41, (3, 50, 32))
    f0 = ckpt.plan_feature(obs, x_T, 0)
    assert f0.shape == (3, 350) and f0.dtype == np.float32
    plain_norm, _ = ckpt.sample_normalized(obs, x_T, np.zeros(3, np.int32), x_T)
    np.testing.assert_allclose(f0, np.asarray(plain_norm)[:, :, :7].reshape(3, -1), atol=2e-2)
    f3 = ckpt.plan_feature(obs, x_T, 3)
    assert f3.shape == (3, 350) and np.isfinite(f3).all()


def test_mixed_k_lowfreq_noise_equals_per_k(dummy):
    """exp9 adaptive: per-row k with per-row lowfreq noise (decoded from each row's x_T) = separate per-k calls."""
    from adsrl_pi0.rollout import lowfreq_noise

    model, obs, steer, plain = dummy
    x_T = _noise(51)
    z = np.random.default_rng(0).uniform(-2.5, 2.5, (B, 21)).astype(np.float32)
    w = lowfreq_noise(x_T, z, 3)
    ks = np.array([0, 3, 6, 9], np.int32)
    mixed = np.asarray(steer(obs, x_T, ks, w))
    for i, k in enumerate(ks):
        single = np.asarray(steer(obs, x_T, np.full(B, k, np.int32), w))
        np.testing.assert_allclose(mixed[i], single[i], atol=1e-6, rtol=0)
