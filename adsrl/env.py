"""VecEnv wrapper that appends a fresh x_T ~ N(0, I) to every observation."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper


class XTAugmentWrapper(VecEnvWrapper):
    """Returns obs_aug = [obs, x_T.flatten()] with x_T ~ N(0, I) of size horizon*act_dim, resampled
    independently per env on every reset/step. `infos[i]["terminal_observation"]` is augmented too
    (with its own fresh x_T). The current x_T batch is kept in `self.x_T` (n_envs, H, A).
    """

    def __init__(self, venv, horizon: int, act_dim: int, seed=None):
        self.horizon = int(horizon)
        self.act_dim = int(act_dim)
        self.xt_dim = self.horizon * self.act_dim
        base = venv.observation_space
        assert isinstance(base, spaces.Box) and len(base.shape) == 1, base
        observation_space = spaces.Box(
            low=np.concatenate([base.low.astype(np.float32), np.full(self.xt_dim, -np.inf, np.float32)]),
            high=np.concatenate([base.high.astype(np.float32), np.full(self.xt_dim, np.inf, np.float32)]),
            dtype=np.float32,
        )
        super().__init__(venv, observation_space=observation_space)
        self.rng = np.random.default_rng(seed)
        self.x_T = None

    def _sample_xT(self, n: int) -> np.ndarray:
        return self.rng.standard_normal((n, self.xt_dim)).astype(np.float32)

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        x_T = self._sample_xT(obs.shape[0])
        self.x_T = x_T.reshape(-1, self.horizon, self.act_dim)
        return np.concatenate([np.asarray(obs, dtype=np.float32), x_T], axis=-1)

    def seed(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        return self.venv.seed(seed)

    def reset(self):
        return self._augment(self.venv.reset())

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for info in infos:
            term = info.get("terminal_observation") if isinstance(info, dict) else None
            if term is not None:
                term = np.asarray(term, dtype=np.float32)
                info["terminal_observation"] = np.concatenate([term, self._sample_xT(1)[0]], axis=-1)
        return self._augment(obs), rewards, dones, infos
