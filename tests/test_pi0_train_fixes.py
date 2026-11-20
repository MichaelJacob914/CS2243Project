"""Eval/resume behaviour of train_adsrl_pi0.py (CPU; fake LIBERO env and fake pi0).

1. Deterministic eval runs exactly one episode per benchmark init state, whatever --eval_episodes is.
2. An eval uses its own RNGs (numpy for x_T, JAX key for the agent's sampling) derived from (seed, eval index):
   it leaves the training RNG streams untouched and is reproducible.
3. Resume refuses to continue with a fresh agent when the agent checkpoint is missing or does not match state.pkl.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import train_adsrl_pi0 as T  # noqa: E402  (light module: heavy imports live inside main())

import jax  # noqa: E402

from adsrl_pi0.agent import HybridPixelSACLearner, save_agent  # noqa: E402
from adsrl_pi0.sampler import SteeredPi0  # noqa: E402

D, K, N_INIT = 32, 10, 50
KW = dict(encoder_type="small", hidden_dims=(32, 32), latent_dim=16, cnn_features=(8, 8), cnn_strides=(2, 2),
          num_qs=3)


class FakeVenv:
    """4 envs; every episode lasts 3 chunks of 20 steps and fails."""

    def __init__(self, n=4):
        self.n, self.max_steps, self.n_init_states = n, 60, N_INIT

    @staticmethod
    def _obs():
        return {"pi0": {}, "pixels": np.zeros((64, 64, 3), np.uint8), "state": np.zeros(8, np.float32)}

    def reset(self, ids, idx):
        return [self._obs() for _ in ids]

    def step_chunk(self, ids, actions):
        return [(self._obs(), len(a), False) for a in actions]


class FakePi0:
    horizon, action_dim = 50, 32
    tile = SteeredPi0.tile

    def infer(self, obs_list, x_T, k, w):
        return np.zeros((len(obs_list), 50, 7), np.float32)


def _sample_obs():
    return {"pixels": np.zeros((1, 64, 64, 3, 1), np.uint8), "state": np.zeros((1, 8, 1), np.float32),
            "x_T": np.zeros((1, D), np.float32)}


@pytest.fixture()
def agent():
    return HybridPixelSACLearner(0, _sample_obs(), D, K, list(range(K)), **KW)


def _eval(agent, deterministic, eval_index, train_rng, eval_episodes=20, seed=0):
    def act_fn(batch):
        return agent.act(jax.device_put(batch), deterministic=deterministic)

    return T.run_eval(FakeVenv(), FakePi0(), agent, act_fn, deterministic, seed=seed, eval_index=eval_index,
                      eval_episodes=eval_episodes, query_freq=20, rollout_kw=dict(xt_full=False),
                      train_rng=train_rng)


# ---- 1. deterministic eval: one episode per init state ----
@pytest.mark.parametrize("eval_episodes", [20, 50, 120])
def test_det_eval_one_episode_per_init_state(agent, eval_episodes):
    assert T.eval_init_indices(N_INIT, eval_episodes, deterministic=True) == list(range(N_INIT))
    eps = _eval(agent, True, 0, np.random.default_rng(0), eval_episodes=eval_episodes)
    assert sorted(e.init_idx for e in eps) == list(range(N_INIT))


def test_stoch_eval_uses_eval_episodes(agent):
    assert T.eval_init_indices(N_INIT, 120, deterministic=False) == [i % N_INIT for i in range(120)]
    assert len(_eval(agent, False, 0, np.random.default_rng(0), eval_episodes=20)) == 20


# ---- 2. eval RNG isolation and reproducibility ----
@pytest.mark.parametrize("deterministic", [False, True])
def test_eval_leaves_training_rngs_untouched(agent, deterministic):
    train_rng = np.random.default_rng(123)
    np_state = train_rng.bit_generator.state
    agent_rng = np.asarray(agent.get_rng()).copy()
    _eval(agent, deterministic, 3, train_rng)
    assert train_rng.bit_generator.state == np_state
    np.testing.assert_array_equal(np.asarray(agent.get_rng()), agent_rng)


def test_eval_is_reproducible_per_index(agent):
    def first_draws(eval_index):
        eps = _eval(agent, False, eval_index, np.random.default_rng(eval_index + 7))  # training rng differs
        eps = sorted(eps, key=lambda e: e.init_idx)
        return np.stack([e.obs[0]["x_T"] for e in eps]), np.stack([e.w[0] for e in eps])

    xt_a, w_a = first_draws(5)
    xt_b, w_b = first_draws(5)
    np.testing.assert_array_equal(xt_a, xt_b)  # same x_T noise
    np.testing.assert_array_equal(w_a, w_b)  # same stochastic agent samples
    xt_c, _ = first_draws(6)
    assert not np.array_equal(xt_a, xt_c)  # a different eval index draws different noise


# ---- 3. resume must not silently start a fresh agent ----
def test_resume_raises_without_agent_checkpoint(agent, tmp_path):
    with pytest.raises(FileNotFoundError):
        T.resume_agent(agent, str(tmp_path), expected_step=100)


def test_resume_raises_on_step_mismatch(agent, tmp_path):
    save_agent(agent, str(tmp_path), 7)
    with pytest.raises(RuntimeError):
        T.resume_agent(agent, str(tmp_path), expected_step=9)


def test_resume_ok(agent, tmp_path):
    save_agent(agent, str(tmp_path), 7)
    fresh = HybridPixelSACLearner(1, _sample_obs(), D, K, list(range(K)), **KW)
    assert T.resume_agent(fresh, str(tmp_path), expected_step=7) == 7
    for a, b in zip(jax.tree_util.tree_leaves(agent._actor.params), jax.tree_util.tree_leaves(fresh._actor.params),
                    strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
