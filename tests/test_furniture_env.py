"""FurnitureVecEnv (adsrl/furniture_env.py) on a fake batched sim with the FurnitureRLSimEnv interface.

Runs without IsaacGym. The real-sim check is scripts/furniture/smoke_furniture_env.py (GPU job).
"""

import math

import numpy as np
import pytest
import torch

from adsrl.env import XTAugmentWrapper
from adsrl.furniture_env import FurnitureVecEnv, denormalize, load_furniture_normalizer

N, H, A, OBS, RS, PARTS = 6, 8, 10, 58, 14, 42  # robot_state 14 -> 16 after the 6D-rotation map


def fake_robot_state_fn(rs):
    """Stands in for proprioceptive_quat_to_6d_rotation (14 -> 16 dims); keeps rs[:, 0] (= env step)."""
    return torch.cat([rs, torch.zeros(rs.shape[0], 2)], dim=-1)


class FakeSim:
    """Pair j of env i becomes assembled at sim step ``pair_at[i, j]`` and stays so (sticky, as
    FurnitureRLSimEnv.already_assembled); ``done`` = all pairs assembled. ``succ_at`` (one pair) is
    the 1-pair shorthand. robot_state[:, 0] holds the env step."""

    def __init__(self, num_envs, succ_at=None, pair_at=None):
        self.num_envs = num_envs
        self.device = "cpu"
        pair_at = torch.as_tensor(succ_at).reshape(-1, 1) if pair_at is None else torch.as_tensor(pair_at)
        self.pair_at = pair_at.float()
        self.already_assembled = torch.zeros(pair_at.shape, dtype=torch.bool)
        self.steps = torch.zeros(num_envs)
        self.actions = []
        self.n_resets = 0

    def _obs(self):
        rs = torch.zeros(self.num_envs, RS)
        rs[:, 0] = self.steps
        return {"robot_state": rs, "parts_poses": torch.full((self.num_envs, PARTS), 0.5)}

    def reset(self):
        self.steps.zero_()
        self.n_resets += 1
        return self._obs()

    def step(self, a):
        assert a.shape == (self.num_envs, A)
        self.actions.append(a.clone())
        self.steps += 1
        self.already_assembled = self.steps.unsqueeze(1) >= self.pair_at
        done = self.already_assembled.all(1, keepdim=True)
        return self._obs(), done.float(), done, {}


@pytest.fixture
def norm_path(tmp_path):
    p = tmp_path / "normalization.pth"
    sd = {
        "stats.observations.min": torch.zeros(OBS),
        "stats.observations.max": torch.full((OBS,), 100.0),
        "stats.actions.min": -torch.arange(1.0, A + 1),
        "stats.actions.max": torch.arange(1.0, A + 1) * 3,
    }
    torch.save(sd, p)
    return str(p)


def make(norm_path, succ_at=None, max_episode_steps=4 * H + 3, pair_at=None, partial_reward=False):
    sim = FakeSim(N, succ_at, pair_at)
    env = FurnitureVecEnv(sim, norm_path, act_steps=H, action_dim=A, max_episode_steps=max_episode_steps,
                          reward_offset=1.0, partial_reward=partial_reward, robot_state_fn=fake_robot_state_fn)  # fmt: skip
    return sim, env


def test_spaces_and_shapes(norm_path):
    sim, env = make(norm_path, [10**6] * N)
    assert env.num_envs == N
    assert env.observation_space.shape == (OBS,) and env.action_space.shape == (H * A,)
    assert np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)
    assert env.chunks_per_episode == math.ceil((4 * H + 3) / H) == 5
    obs = env.reset()
    assert obs.shape == (N, OBS) and obs.dtype == np.float32
    obs, rew, done, infos = env.step(np.zeros((N, H * A), np.float32))
    assert obs.shape == (N, OBS) and rew.shape == (N,) and done.shape == (N,) and done.dtype == bool
    assert len(infos) == N and not done.any()


def test_obs_normalization_and_clamp(norm_path):
    sim, env = make(norm_path, [10**6] * N)
    obs = env.reset()
    np.testing.assert_allclose(obs[:, 0], -1.0)  # env step 0 -> min
    np.testing.assert_allclose(obs[:, 16:], 2 * 0.5 / 100 - 1, rtol=1e-6)  # parts poses
    sim.steps.fill_(10_000 - 1)  # far above max -> clamped at 5
    obs, *_ = env.step(np.zeros((N, H * A), np.float32))
    np.testing.assert_allclose(obs[:, 0], 5.0)
    np.testing.assert_allclose(env.unnormalize_obs(obs[:, :1] * 0 + 1)[:, 0], 100.0)


def test_action_denormalization_and_order(norm_path):
    sim, env = make(norm_path, [10**6] * N)
    env.reset()
    a = np.random.default_rng(0).uniform(-1, 1, (N, H, A)).astype(np.float32)
    env.step(a.reshape(N, -1))
    assert len(sim.actions) == H
    stats = load_furniture_normalizer(norm_path, "cpu")["actions"]
    for i in range(H):
        torch.testing.assert_close(sim.actions[i], denormalize(torch.from_numpy(a[:, i]), stats))
    lo, hi = stats
    torch.testing.assert_close(denormalize(torch.full((A,), -1.0), stats), lo)
    torch.testing.assert_close(denormalize(torch.full((A,), 1.0), stats), hi)


def test_reward_success_and_auto_reset(norm_path):
    # env 0 never succeeds; env 1 at sim step 1; env 2 at step 8 (end of chunk 1); env 3 at step 9;
    # env 4 at step 16; env 5 at the last sim step of the episode (40)
    succ_at = [10**6, 1, 8, 9, 16, 40]
    sim, env = make(norm_path, succ_at)
    obs = env.reset()
    assert sim.n_resets == 1
    T = env.chunks_per_episode
    rews, all_obs = [], []
    for t in range(T):
        obs, rew, done, infos = env.step(np.zeros((N, H * A), np.float32))
        rews.append(rew)
        all_obs.append(obs)
        if t < T - 1:
            assert not done.any() and all(set(i) == {"chunk_success"} for i in infos)
    rews = np.stack(rews)  # (T, N)
    # expected: sum over the chunk's sim steps s of (1[s >= succ_at] - 1)
    for i, s_i in enumerate(succ_at):
        for t in range(T):
            steps = np.arange(t * H + 1, (t + 1) * H + 1)
            assert rews[t, i] == pytest.approx(float(np.sum((steps >= s_i).astype(float) - 1.0))), (i, t)
    # AdsrlCallback's success test: reward > -reward_offset <=> assembled during the whole chunk
    first = [int(np.argmax(rews[:, i] > -1)) if (rews[:, i] > -1).any() else -1 for i in range(N)]
    assert first == [-1, 0, 1, 1, 2, -1]
    # last chunk: every env done, terminal obs = last pre-reset obs, then the batch is reset
    assert done.all() and sim.n_resets == 2 and env.t == 0
    term = np.stack([info["terminal_observation"] for info in infos])
    np.testing.assert_allclose(term[:, 0], 2 * (T * H) / 100 - 1, rtol=1e-6)
    np.testing.assert_allclose(obs[:, 0], -1.0)  # first obs of the next episode
    assert [info["is_success"] for info in infos] == [False, True, True, True, True, True]
    np.testing.assert_allclose([info["episode"]["r"] for info in infos], rews.sum(0))
    assert all(info["episode"]["l"] == T for info in infos)
    # the next episode starts clean
    obs, rew, done, infos = env.step(np.zeros((N, H * A), np.float32))
    assert not done.any() and env.t == 1


def test_xt_augment_wrapper(norm_path):
    sim, env = make(norm_path, [3] * N, max_episode_steps=2 * H)
    venv = XTAugmentWrapper(env, H, A, seed=0)
    obs = venv.reset()
    assert obs.shape == (N, OBS + H * A)
    venv.step(np.zeros((N, H * A), np.float32))
    obs, rew, done, infos = venv.step(np.zeros((N, H * A), np.float32))
    assert done.all()
    for info in infos:
        assert info["terminal_observation"].shape == (OBS + H * A,)
    np.testing.assert_allclose(obs[:, 0], -1.0)


BIG = 10**6


def run_episode(env):
    env.reset()
    rews, flags = [], []
    for _ in range(env.chunks_per_episode):
        _, rew, done, infos = env.step(np.zeros((N, H * A), np.float32))
        rews.append(rew)
        flags.append([i["chunk_success"] for i in infos])
    assert done.all()
    return np.stack(rews), np.array(flags), infos


def test_one_pair_partial_reward_is_identical(norm_path):
    succ_at = [BIG, 1, 8, 9, 16, 40]
    r0, f0, _ = run_episode(make(norm_path, succ_at)[1])
    r1, f1, _ = run_episode(make(norm_path, succ_at, partial_reward=True)[1])
    np.testing.assert_array_equal(r0, r1)
    np.testing.assert_array_equal(f0, f1)


def test_partial_reward_and_success_flag(norm_path):
    # 2 pairs. env 0: nothing; env 1: one pair at step 1, never both; env 2: both at step 1;
    # env 3: pair A at 1, pair B at 12; env 4: one pair at 9; env 5: both at the last step (40)
    pair_at = [[BIG, BIG], [1, BIG], [1, 1], [1, 12], [9, BIG], [40, 40]]
    env = make(norm_path, pair_at=pair_at, partial_reward=True)[1]
    rews, flags, infos = run_episode(env)
    T = env.chunks_per_episode
    pa = np.array(pair_at, dtype=float)
    for i in range(N):
        for t in range(T):
            steps = np.arange(t * H + 1, (t + 1) * H + 1)
            frac = (steps[:, None] >= pa[i][None]).mean(1)
            assert rews[t, i] == pytest.approx(float(np.sum(frac - 1.0))), (i, t)
    # partial credit raises the reward but is never success: success = all pairs assembled
    assert rews[:, 1].max() == pytest.approx(-4.0) and not flags[:, 1].any()
    assert rews[1:, 4].max() == pytest.approx(-4.0) and not flags[:, 4].any()
    assert flags[:, 2].all()
    assert not flags[0, 3] and flags[1:, 3].all()  # both assembled at step 12 = in chunk 1
    assert not flags[:-1, 5].any() and flags[-1, 5]  # assembled at the last sim step of the episode
    assert [i["is_success"] for i in infos] == [False, False, True, True, False, True]


def test_callback_success_uses_flag_not_reward():
    """train_adsrl.chunk_success: furniture uses info["chunk_success"]; robomimic keeps reward > -offset."""
    import train_adsrl

    rew = np.array([-0.5, -4.0, 0.0, -8.0])  # -0.5 passes the reward test
    infos = [{"chunk_success": False}, {"chunk_success": False}, {"chunk_success": True}, {"chunk_success": False}]
    np.testing.assert_array_equal(train_adsrl.chunk_success(rew, infos, 1.0, use_flag=True), [False, False, True, False])
    np.testing.assert_array_equal(train_adsrl.chunk_success(rew, [{}] * 4, 1.0), [True, False, True, False])
