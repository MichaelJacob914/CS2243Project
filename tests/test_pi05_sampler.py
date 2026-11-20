"""Tests for the steered flow sampler (adsrl_pi0/sampler.py) on pi0.5. Run with .venv-pi05 (upstream openpi) in a job.

Model variants:
  dummy  : randomly initialized Pi0Config(pi05=True) with the 'dummy' gemma variants in float32 (any device);
           also an upstream pi0 dummy (pi05=False) to cover the adarms_cond=None path of `_velocity`.
  ckpt   : the real pi05_libero checkpoint through the full SteeredPi0 pipeline (needs a GPU; skipped otherwise).
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

from adsrl_pi0.sampler import (  # noqa: E402
    NUM_STEPS,
    SteeredPi0,
    implied_noise_at,
    module_jit_fn,
    plain_features,
    steered_sample_actions,
)

B = 4
H = 10  # pi05_libero action_horizon
HAS_GPU = any(d.platform == "gpu" for d in jax.devices())


def _make(pi05: bool):
    from openpi.models import pi0_config

    cfg = pi0_config.Pi0Config(pi05=pi05, paligemma_variant="dummy", action_expert_variant="dummy", dtype="float32",
                               action_horizon=H)
    model = cfg.create(jax.random.key(0))
    obs = cfg.fake_obs(B)
    steer = module_jit_fn(steered_sample_actions, model, static_argnames=("num_steps",))
    # upstream signature: sample_actions(rng, observation, *, num_steps, noise)
    plain = module_jit_fn(lambda m, o, noise: m.sample_actions(jax.random.key(0), o, noise=noise), model)
    return model, obs, steer, plain


@pytest.fixture(scope="module")
def dummy():
    return _make(True)


@pytest.fixture(scope="module")
def dummy_pi0_upstream():
    return _make(False)


def _noise(seed, shape=(B, H, 32)):
    return np.asarray(jax.random.normal(jax.random.key(seed), shape), np.float32)


def test_model_is_pi05(dummy):
    model = dummy[0]
    assert model.pi05 and model.action_horizon == H and model.action_dim == 32
    out = model.embed_suffix(dummy[1], jnp.asarray(_noise(0)), jnp.ones(B))
    assert len(out) == 4 and out[3] is not None  # adaRMS time conditioning
    assert out[0].shape[1] == H  # no state token in the pi0.5 suffix


@pytest.mark.parametrize("which", ["dummy", "dummy_pi0_upstream"])
def test_k0_equals_openpi_sample_actions(which, request):
    model, obs, steer, plain = request.getfixturevalue(which)
    w, x_T = _noise(1), _noise(2)
    ref = np.asarray(plain(obs, w))
    out = np.asarray(steer(obs, x_T, np.zeros(B, np.int32), w))
    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=0)
    assert np.isfinite(out).all()
    assert np.abs(ref - np.asarray(plain(obs, x_T))).max() > 1e-2  # the noise matters


@pytest.mark.parametrize("k", list(range(1, NUM_STEPS)))
def test_implied_noise_is_a_noop(dummy, k):
    """w = eps_hat_{k-1} (the model's implied noise on the unsteered path) reproduces the unsteered sample."""
    model, obs, steer, plain = dummy
    x_T = _noise(3)
    ref = np.asarray(plain(obs, x_T))
    imp = module_jit_fn(implied_noise_at, model, static_argnames=("k_target",))
    w = np.asarray(imp(obs, x_T, k_target=k))
    out = np.asarray(steer(obs, x_T, np.full(B, k, np.int32), w))
    err = np.abs(out - ref).max()
    other = np.asarray(steer(obs, x_T, np.full(B, k, np.int32), _noise(4)))
    effect = np.abs(other - ref).max()
    print(f"k={k}: max|steered(eps_hat) - unsteered| = {err:.2e}, effect of a random w = {effect:.2e}")
    assert err < 5e-3 and err < 1e-2 * effect, (err, effect)


@pytest.mark.parametrize("k", [0, 1, 4, 9])
def test_zero_w_renoise_state(dummy, k):
    """w = 0 at step k sets x_k = (1 - t_k) a_hat_{k-1} = x_k(unsteered) - t_k eps_hat_{k-1}; states before k unchanged."""
    model, obs, steer, plain = dummy
    x_T = _noise(21)
    traj_fn = module_jit_fn(steered_sample_actions, model, static_argnames=("num_steps", "return_trajectory"))
    out, traj = traj_fn(obs, x_T, np.full(B, k, np.int32), np.zeros_like(x_T), return_trajectory=True)
    _, traj_plain = traj_fn(obs, x_T, np.zeros(B, np.int32), x_T, return_trajectory=True)
    traj, traj_plain = np.asarray(traj), np.asarray(traj_plain)
    np.testing.assert_array_equal(np.asarray(out), traj[-1])
    np.testing.assert_allclose(traj[:k], traj_plain[:k], atol=1e-6)
    if k == 0:
        np.testing.assert_array_equal(traj[0], 0.0)
    else:
        t_k = 1.0 - k / NUM_STEPS
        imp = module_jit_fn(implied_noise_at, model, static_argnames=("k_target",))
        eps_hat = np.asarray(imp(obs, x_T, k_target=k))
        np.testing.assert_allclose(traj[k], traj_plain[k] - t_k * eps_hat, atol=1e-5)


def test_mixed_k_batch_equals_per_k(dummy):
    model, obs, steer, plain = dummy
    x_T, w = _noise(5), _noise(6)
    ks = np.array([0, 3, 7, 9], np.int32)
    mixed = np.asarray(steer(obs, x_T, ks, w))
    for i, k in enumerate(ks):
        single = np.asarray(steer(obs, x_T, np.full(B, k, np.int32), w))
        np.testing.assert_allclose(mixed[i], single[i], atol=1e-6, rtol=0)


def test_later_k_moves_less(dummy):
    model, obs, steer, plain = dummy
    x_T = _noise(7)
    spread = []
    for k in [0, 5, 9]:
        outs = [np.asarray(steer(obs, x_T, np.full(B, k, np.int32), _noise(100 + s))) for s in range(4)]
        spread.append(np.std(np.stack(outs), axis=0).mean())
    assert spread[0] > spread[1] > spread[2], spread


def test_plain_features(dummy):
    model, obs, steer, plain = dummy
    feats = module_jit_fn(plain_features, model, static_argnames=("k", "num_steps"))
    x_T = _noise(31)
    np.testing.assert_allclose(np.asarray(feats(obs, x_T, k=0)), np.asarray(plain(obs, x_T)), atol=1e-5)
    traj_fn = module_jit_fn(steered_sample_actions, model, static_argnames=("num_steps", "return_trajectory"))
    for k in [1, 3, 9]:
        _, traj = traj_fn(obs, x_T, np.full(B, k, np.int32), np.zeros_like(x_T), return_trajectory=True)
        t_k = np.float32(1.0)
        for _ in range(k):
            t_k = np.float32(t_k + np.float32(-1.0 / NUM_STEPS))
        np.testing.assert_allclose(np.asarray(feats(obs, x_T, k=k)), np.asarray(traj)[k] / (1.0 - t_k),
                                   atol=1e-4, rtol=1e-4)


def test_mixed_k_lowfreq_noise_equals_per_k(dummy):
    from adsrl_pi0.rollout import lowfreq_noise

    model, obs, steer, plain = dummy
    x_T = _noise(51)
    z = np.random.default_rng(0).uniform(-2.5, 2.5, (B, 21)).astype(np.float32)
    w = lowfreq_noise(x_T, z, 3)
    assert w.shape == (B, H, 32)
    ks = np.array([0, 3, 6, 9], np.int32)
    mixed = np.asarray(steer(obs, x_T, ks, w))
    for i, k in enumerate(ks):
        single = np.asarray(steer(obs, x_T, np.full(B, k, np.int32), w))
        np.testing.assert_allclose(mixed[i], single[i], atol=1e-6, rtol=0)


# ---------------- real pi05_libero checkpoint ----------------
@pytest.fixture(scope="module")
def ckpt():
    if not HAS_GPU:
        pytest.skip("real pi0.5 checkpoint test needs a GPU")
    return SteeredPi0.load_model("pi05")


def _libero_obs(n, seed=0):
    rng = np.random.default_rng(seed)
    return [{"observation/state": rng.random(8).astype(np.float32),
             "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
             "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
             "prompt": "put both the alphabet soup and the tomato sauce in the basket"} for _ in range(n)]


def test_ckpt_shapes(ckpt):
    assert ckpt.model.pi05 and ckpt.horizon == H and ckpt.action_dim == 32


def test_ckpt_k0_matches_policy_infer(ckpt):
    """k=0 through SteeredPi0 == openpi Policy.infer(obs, noise=w) (upstream pi0.5 sample_actions), env units."""
    obs = _libero_obs(3)
    w = _noise(11, (3, H, 32))
    out = ckpt.infer(obs, _noise(12, (3, H, 32)), np.zeros(3, np.int32), w)
    assert out.shape == (3, H, 7)
    for i in range(3):
        ref = ckpt.policy.infer(obs[i], noise=np.asarray(w[i : i + 1]))["actions"]
        single = ckpt.infer([obs[i]], _noise(15, (1, H, 32)), np.zeros(1, np.int32), w[i : i + 1])[0]
        np.testing.assert_allclose(single, ref, atol=1e-5, rtol=0)
        other = ckpt.policy.infer(obs[i], noise=_noise(16 + i, (1, H, 32)))["actions"]
        print(f"example {i}: max |batched - batch1| = {np.abs(out[i] - ref).max():.3f}, "
              f"different noise = {np.abs(other - ref).max():.3f}")
        np.testing.assert_allclose(out[i], ref, atol=0.15, rtol=0)


@pytest.mark.parametrize("k", [1, 3, 5, 8])
def test_ckpt_implied_noise_is_a_noop(ckpt, k):
    """Real bf16 model, normalized model space: steering with w = eps_hat_{k-1} ~ unsteered sample."""
    obs = _libero_obs(4, seed=3)
    observation, _ = ckpt.prepare(obs)
    x_T = _noise(61, (4, H, 32))
    ref, _ = ckpt.sample_normalized(obs, x_T, np.zeros(4, np.int32), x_T)
    imp = module_jit_fn(implied_noise_at, ckpt.model, static_argnames=("k_target",))
    w = np.asarray(imp(observation, x_T, k_target=k))
    out, _ = ckpt.sample_normalized(obs, x_T, np.full(4, k, np.int32), w)
    other, _ = ckpt.sample_normalized(obs, x_T, np.full(4, k, np.int32), _noise(62, (4, H, 32)))
    ref, out, other = (np.asarray(x)[..., :7] for x in (ref, out, other))
    err, effect = np.abs(out - ref).max(), np.abs(other - ref).max()
    print(f"ckpt k={k}: max|steered(eps_hat) - unsteered| = {err:.3e}, random w effect = {effect:.3e}")
    assert err < 0.1 * effect, (err, effect)


def test_ckpt_mixed_batch_and_tiling(ckpt):
    obs = _libero_obs(4, seed=1)
    x_T, w = _noise(13, (4, 32)), _noise(14, (4, 32))
    ks = np.array([0, 2, 5, 8], np.int32)
    mixed = ckpt.infer(obs, x_T, ks, w)
    for i, k in enumerate(ks):
        single = ckpt.infer(obs, x_T, np.full(4, k, np.int32), w)
        np.testing.assert_allclose(mixed[i], single[i], atol=1e-5, rtol=0)


def test_ckpt_plan_feature(ckpt):
    obs = _libero_obs(3, seed=2)
    x_T = _noise(41, (3, H, 32))
    f0 = ckpt.plan_feature(obs, x_T, 0)
    assert f0.shape == (3, H * 7) and f0.dtype == np.float32
    plain_norm, _ = ckpt.sample_normalized(obs, x_T, np.zeros(3, np.int32), x_T)
    np.testing.assert_allclose(f0, np.asarray(plain_norm)[:, :, :7].reshape(3, -1), atol=2e-2)
    assert np.isfinite(ckpt.plan_feature(obs, x_T, 3)).all()
