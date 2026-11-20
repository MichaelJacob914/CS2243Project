"""Structured noise layout (--steer.w_layout structured), CPU.

With u = 1/sqrt(H) (1,...,1) in R^H, the pi0 noise is w = u z^T + (I - u u^T) x_T: the observed x_T with its
per-channel time-mean replaced by z / sqrt(H). Tests: the decode itself, that it is what pi0 receives in rollout
(train and eval paths), the training-script wiring, and that --action_magnitude sets the tanh bound.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import train_adsrl_pi0 as T  # noqa: E402  (light module; also runs adsrl_pi0.setup_env())

import jax  # noqa: E402

from adsrl_pi0.agent import FixedKAgent  # noqa: E402
from adsrl_pi0.rollout import run_episodes, structured_noise  # noqa: E402
from adsrl_pi0.sampler import SteeredPi0  # noqa: E402

H, D = 50, 32


# ---- the decode ----
def test_shapes_single_and_batched():
    rng = np.random.default_rng(0)
    assert structured_noise(rng.standard_normal((H, D)), rng.standard_normal(D)).shape == (H, D)
    assert structured_noise(rng.standard_normal((5, H, D)), rng.standard_normal((5, D))).shape == (5, H, D)


def test_time_mean_is_z_over_sqrt_h_and_residual_is_kept():
    rng = np.random.default_rng(1)
    x_T, z = rng.standard_normal((7, H, D)), rng.standard_normal((7, D))
    w = structured_noise(x_T, z)
    np.testing.assert_allclose(w.mean(1), z / np.sqrt(H), atol=1e-6)
    np.testing.assert_allclose(w - w.mean(1, keepdims=True), x_T - x_T.mean(1, keepdims=True), atol=1e-6)


def test_matches_projector_formula():
    rng = np.random.default_rng(2)
    x_T, z = rng.standard_normal((H, D)), rng.standard_normal(D)
    u = np.full((H, 1), 1 / np.sqrt(H))
    np.testing.assert_allclose(structured_noise(x_T, z), u @ z[None] + (np.eye(H) - u @ u.T) @ x_T, atol=1e-6)


def test_z_equal_to_scaled_mean_reproduces_x_T():
    rng = np.random.default_rng(3)
    x_T = rng.standard_normal((4, H, D)).astype(np.float32)
    z = np.sqrt(H) * x_T.mean(1)  # = u^T x_T
    np.testing.assert_allclose(structured_noise(x_T, z), x_T, atol=1e-5)


def test_batched_matches_per_sample():
    rng = np.random.default_rng(4)
    x_T, z = rng.standard_normal((6, H, D)), rng.standard_normal((6, D))
    wb = structured_noise(x_T, z)
    for i in range(6):
        np.testing.assert_allclose(wb[i], structured_noise(x_T[i], z[i]), atol=1e-6)


def test_gaussian_z_gives_iid_standard_normal_noise():
    rng = np.random.default_rng(5)
    n = 4000
    w = structured_noise(rng.standard_normal((n, H, D)), rng.standard_normal((n, D)))
    assert abs(w.mean()) < 0.01 and abs(w.std() - 1) < 0.01
    np.testing.assert_allclose(w.mean(0), 0, atol=0.08)  # per entry (std err 1/sqrt(4000) ~ 0.016)
    np.testing.assert_allclose(w.std(0), 1, atol=0.08)
    # full covariance over the 50 time steps of one channel ~ I (cross-time correlations would show up here)
    c = np.cov(w[:, :, 3].T)
    np.testing.assert_allclose(c, np.eye(H), atol=0.1)
    # and across channels at one time step
    np.testing.assert_allclose(np.cov(w[:, 10, :].T), np.eye(D), atol=0.1)


# ---- rollout: what pi0 receives ----
class FakeVenv:
    def __init__(self, n=3):
        self.n, self.max_steps, self.n_init_states = n, 40, 10

    @staticmethod
    def _obs():
        return {"pi0": {}, "pixels": np.zeros((64, 64, 3), np.uint8), "state": np.zeros(8, np.float32)}

    def reset(self, ids, idx):
        return [self._obs() for _ in ids]

    def step_chunk(self, ids, actions):
        return [(self._obs(), len(a), False) for a in actions]


class RecordingPi0:
    horizon, action_dim = H, D
    tile = SteeredPi0.tile

    def __init__(self):
        self.calls = []

    def infer(self, obs_list, x_T, k, w):
        self.calls.append((self.tile(x_T), np.asarray(k), self.tile(w)))
        return np.zeros((len(obs_list), H, 7), np.float32)


def _act_fn(record):
    rng = np.random.default_rng(9)

    def f(batch):
        B = batch["state"].shape[0]
        z = rng.uniform(-1, 1, (B, D)).astype(np.float32)
        record.append((batch["x_T"].copy(), z))
        return np.zeros(B, np.int32), z, None
    return f


@pytest.mark.parametrize("compare_base", [False, True])
def test_rollout_structured_feeds_decoded_noise(compare_base):
    pi0, rec = RecordingPi0(), []
    eps, _ = run_episodes(FakeVenv(), pi0, [0, 1, 2, 3], _act_fn(rec), 20, np.random.default_rng(0),
                          compare_base=compare_base, xt_full=True, w_layout="structured")
    assert len(pi0.calls) == len(rec) > 0
    for (x_in, k, w_in), (xt_obs, z) in zip(pi0.calls, rec, strict=True):
        B = len(z)
        x_T = xt_obs.reshape(B, H, D)
        np.testing.assert_allclose(x_in[:B], x_T)
        np.testing.assert_allclose(w_in[:B], structured_noise(x_T, z), atol=1e-6)
        if compare_base:  # base rows: k=0 and the plain x_T
            assert (k[B:] == 0).all()
            np.testing.assert_allclose(w_in[B:], x_T)
    # the buffer stores the agent's 32-d z, not the decoded noise
    assert all(np.asarray(w).shape == (D,) for e in eps for w in e.w)


def test_rollout_tiled_unchanged():
    pi0, rec = RecordingPi0(), []
    run_episodes(FakeVenv(), pi0, [0, 1], _act_fn(rec), 20, np.random.default_rng(0), xt_full=True)
    for (_, _, w_in), (_, z) in zip(pi0.calls, rec, strict=True):
        np.testing.assert_allclose(w_in, np.repeat(z[:, None], H, 1))


def test_rollout_structured_requires_full_xt():
    with pytest.raises(ValueError):
        run_episodes(FakeVenv(), RecordingPi0(), [0], _act_fn([]), 20, np.random.default_rng(0), xt_full=False,
                     w_layout="structured")


# ---- training script wiring ----
def test_parse_default_tiled_and_structured_flag():
    a = T.parse(["--task_id", "8", "--steer.xt_layout", "full"])
    assert a.w_layout == "tiled" and a.name == "pi0_t8_k0_xtfull_s0"
    a = T.parse(["--task_id", "8", "--steer.xt_layout", "full", "--steer.w_layout", "structured",
                 "--action_magnitude", "2.5"])
    assert a.w_layout == "structured" and a.action_magnitude == 2.5
    assert a.name == "pi0_t8_k0_xtfull_wstruct_b2.5_s0"


def test_parse_structured_requires_full_xt():
    with pytest.raises(SystemExit):
        T.parse(["--task_id", "8", "--steer.w_layout", "structured"])


def test_structured_k0_observes_x_T():
    """k=0 tiled drops x_T (it never affects the action); structured k=0 must keep it: Q(s, x_T, z)."""
    base = ["--task_id", "8", "--steer.xt_layout", "full"]
    assert not T.uses_xt_obs(T.parse(base))
    assert T.uses_xt_obs(T.parse(base + ["--steer.w_layout", "structured"]))
    assert T.uses_xt_obs(T.parse(base + ["--steer.k", "3"]))


@pytest.mark.parametrize("deterministic", [False, True])
def test_run_eval_uses_structured_noise(deterministic):
    """run_eval (stoch and det) decodes the agent's z with the run's layout."""
    obs1 = {"pixels": np.zeros((1, 64, 64, 3, 1), np.uint8), "state": np.zeros((1, 8, 1), np.float32),
            "x_T": np.zeros((1, H * D), np.float32)}
    KW = dict(encoder_type="small", hidden_dims=(32, 32), latent_dim=16, cnn_features=(8, 8), cnn_strides=(2, 2),
              num_qs=3)
    ag = FixedKAgent(0, 10, 0, obs1, np.zeros((1, 1, D), np.float32), **KW)
    rec = []

    def act_fn(batch):
        k, z, p = ag.act(jax.device_put(batch), deterministic=deterministic)
        rec.append((batch["x_T"].copy(), z))
        return k, z, p

    pi0 = RecordingPi0()
    T.run_eval(FakeVenv(), pi0, ag, act_fn, deterministic, seed=0, eval_index=0, eval_episodes=4, query_freq=20,
               rollout_kw=dict(xt_full=True, w_layout="structured"), train_rng=np.random.default_rng(0))
    for (_, _, w_in), (xt_obs, z) in zip(pi0.calls, rec, strict=True):
        B = len(z)
        np.testing.assert_allclose(w_in[:B], structured_noise(xt_obs.reshape(B, H, D), z), atol=1e-5)


# ---- the bound flag scales the tanh ----
@pytest.mark.parametrize("b", [1.0, 2.5])
def test_action_magnitude_sets_tanh_bound(b):
    obs = lambda n: {"pixels": np.zeros((n, 64, 64, 3, 1), np.uint8),  # noqa: E731
                     "state": np.zeros((n, 8, 1), np.float32)}
    KW = dict(encoder_type="small", hidden_dims=(32, 32), latent_dim=16, cnn_features=(8, 8), cnn_strides=(2, 2),
              num_qs=3, action_magnitude=b)
    ag = FixedKAgent(0, 10, 0, obs(1), np.zeros((1, 1, D), np.float32), **KW)
    _, z, _ = ag.act(obs(256))
    # at init the policy is ~ b * tanh(N(0, 1)): max |z| over 8192 draws is close to b, never above it
    assert np.abs(z).max() <= b + 1e-5 and np.abs(z).max() > 0.95 * b


# ---- control: --steer.force_xt_obs 1 (tiled k=0 noise, x_T only added to the observation) ----
XTOBS = ["--task_id", "8", "--steer.xt_layout", "full", "--steer.force_xt_obs", "1"]


def test_force_xt_obs_default_off():
    a = T.parse(["--task_id", "8", "--steer.xt_layout", "full"])
    assert a.force_xt_obs == 0 and not T.uses_xt_obs(a)
    assert "x_T" not in T.obs_spec(a)[0]


def test_force_xt_obs_flag():
    a = T.parse(XTOBS)
    assert a.w_layout == "tiled" and T.uses_xt_obs(a)
    assert a.name == "pi0_t8_k0_xtfull_xtobs_s0"
    shapes, dtypes = T.obs_spec(a)
    assert shapes["x_T"] == (H * D,) and dtypes["x_T"] == np.float32


def test_force_xt_obs_requires_full_xt():
    with pytest.raises(SystemExit):
        T.parse(["--task_id", "8", "--steer.force_xt_obs", "1"])


@pytest.mark.parametrize("flag", [0, 1])
def test_force_xt_obs_stored_transitions(flag):
    """With the flag the buffer's obs and next_obs carry the (1600,) x_T the chunk saw; without it, no x_T."""
    from adsrl_pi0.buffer import ChunkReplayBuffer

    a = T.parse(["--task_id", "8", "--steer.xt_layout", "full", "--steer.force_xt_obs", str(flag)])
    shapes, dtypes = T.obs_spec(a)
    shapes["pixels"] = (64, 64, 3, 1)
    buf = ChunkReplayBuffer(shapes, dtypes, D, 100)
    eps, _ = run_episodes(FakeVenv(), RecordingPi0(), [0, 1], _act_fn([]), 20, np.random.default_rng(0),
                          xt_full=True, w_layout=a.w_layout)
    n = T.add_episodes(buf, eps, 0.999**20, list(shapes))
    assert n == len(buf) == sum(len(e.w) for e in eps)
    obs, nxt = buf.data["observations"], buf.data["next_observations"]
    if not flag:
        assert "x_T" not in obs and "x_T" not in nxt
        return
    i = 0
    for e in eps:
        for t in range(len(e.w)):
            np.testing.assert_array_equal(obs["x_T"][i], e.obs[t]["x_T"])
            np.testing.assert_array_equal(nxt["x_T"][i], e.obs[t + 1]["x_T"])
            assert obs["x_T"][i].shape == (H * D,)
            i += 1


def test_force_xt_obs_noise_is_still_tiled_z():
    """Same z (and same x_T draws) -> pi0 gets exactly the same tiled noise with and without the flag; the agent
    built with the flag acts on the x_T-carrying obs."""
    def pi0_inputs(argv):
        a = T.parse(argv)
        pi0 = RecordingPi0()
        run_episodes(FakeVenv(), pi0, [0, 1, 2], _act_fn([]), 20, np.random.default_rng(0), compare_base=True,
                     xt_full=a.xt_layout == "full", w_layout=a.w_layout)
        return pi0.calls

    base = pi0_inputs(["--task_id", "8", "--steer.xt_layout", "full"])
    flag = pi0_inputs(XTOBS)
    assert len(base) == len(flag) > 0
    for (xb, kb, wb), (xf, kf, wf) in zip(base, flag, strict=True):
        np.testing.assert_array_equal(xb, xf)
        np.testing.assert_array_equal(kb, kf)
        np.testing.assert_array_equal(wb, wf)
        B = len(kb) // 2
        np.testing.assert_array_equal(wf[:B], np.repeat(wf[:B, :1], H, 1))  # steered rows: one row tiled

    a = T.parse(XTOBS)
    obs1 = {"pixels": np.zeros((1, 64, 64, 3, 1), np.uint8), "state": np.zeros((1, 8, 1), np.float32),
            "x_T": np.zeros((1, H * D), np.float32)}
    KW = dict(encoder_type="small", hidden_dims=(32, 32), latent_dim=16, cnn_features=(8, 8), cnn_strides=(2, 2),
              num_qs=3)
    ag = FixedKAgent(0, 10, 0, obs1, np.zeros((1, 1, D), np.float32), **KW)
    pi0, zs = RecordingPi0(), []

    def act_fn(batch):
        k, z, p = ag.act(jax.device_put(T.agent_input(batch, list(T.obs_spec(a)[0]))))
        zs.append(z)
        return k, z, p

    run_episodes(FakeVenv(), pi0, [0, 1], act_fn, 20, np.random.default_rng(0), xt_full=True, w_layout=a.w_layout)
    for (_, _, w_in), z in zip(pi0.calls, zs, strict=True):
        np.testing.assert_allclose(w_in, np.repeat(np.asarray(z)[:, None], H, 1))
