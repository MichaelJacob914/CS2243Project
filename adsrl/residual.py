"""Residual SAC in action space (method 7 in notes/plan_adaptive_dsrl.md).

The agent outputs a residual da in [-beta, beta]^(H*A) and the env executes
    a = clip(a_base(s, x_T) + da, low, high)
where a_base is the frozen diffusion policy's unsteered DDIM action for the chunk's x_T
(``SteeredBasePolicy.base_action``) and [low, high] are the env action bounds (the base model's
normalized [-1, 1] action space).

Observation: [s, a_base] instead of [s, x_T]. Both have H*A dims and both make the problem Markov
(a_base is a deterministic function of (s, x_T), and x_T affects nothing else), but with a_base the
actor and Q(s, da) see the action the residual is added to directly, instead of having to learn the
8-step DDIM map x_T -> a_base inside every network. This is also what residual-RL work on frozen
base policies does (the base action is part of the residual policy's input).

Split of work, mirroring DSRL-NA in this repo (the agent decodes noise into an executed action):
  - ``BaseActionObsWrapper`` only rewrites observations: [s, x_T] -> [s, a_base]. It passes the
    executed action through unchanged, and keeps the current x_T in ``self.x_T`` for eval dumps.
  - ``ResidualSAC`` is SB3 SAC whose action space is the residual box. ``_sample_action`` returns the
    composed action for the env and the scaled residual for the replay buffer, and
    ``predict_diffused`` returns the composed action (the interface ``AdsrlCallback`` uses).
The critic is Q(s, a_base, da); no sampler call is needed in any update.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.sac.sac import SAC

from adsrl.agent import FastReplayBuffer, _SteerMixin


def compose_action(a_base, delta, beta: float, low, high):
    """a = clip(a_base + clip(delta, -beta, beta), low, high). Works on numpy arrays or tensors."""
    if th.is_tensor(a_base):
        low = th.as_tensor(low, device=a_base.device, dtype=a_base.dtype)
        high = th.as_tensor(high, device=a_base.device, dtype=a_base.dtype)
        return th.clamp(a_base + th.clamp(delta, -beta, beta), low, high)
    return np.clip(np.asarray(a_base) + np.clip(delta, -beta, beta), low, high).astype(np.float32)


class BaseActionObsWrapper(VecEnvWrapper):
    """Wraps an ``XTAugmentWrapper``: obs [s, x_T] -> [s, a_base(s, x_T)] (also for
    ``infos[i]["terminal_observation"]``). Actions are passed through unchanged (already composed).
    ``self.x_T`` (n_envs, H, A) is the x_T behind the current observation."""

    def __init__(self, venv, steered):
        self.steered = steered
        self.obs_dim = int(steered.obs_dim)
        self.act_dim_flat = int(steered.horizon * steered.act_dim)
        inner = venv.observation_space
        act = venv.action_space
        assert inner.shape == (self.obs_dim + self.act_dim_flat,), (inner.shape, self.obs_dim, self.act_dim_flat)
        observation_space = spaces.Box(
            low=np.concatenate([inner.low[: self.obs_dim], act.low]).astype(np.float32),
            high=np.concatenate([inner.high[: self.obs_dim], act.high]).astype(np.float32),
            dtype=np.float32,
        )
        super().__init__(venv, observation_space=observation_space)
        self.x_T = None

    def _a_base(self, obs_aug: np.ndarray) -> np.ndarray:
        return self.steered.base_action(obs_aug).reshape(len(obs_aug), -1).cpu().numpy().astype(np.float32)

    def to_residual_obs(self, obs_aug: np.ndarray) -> np.ndarray:
        obs_aug = np.asarray(obs_aug, dtype=np.float32)
        return np.concatenate([obs_aug[:, : self.obs_dim], self._a_base(obs_aug)], axis=-1)

    def reset(self):
        obs = self.venv.reset()
        self.x_T = np.array(self.venv.x_T, copy=True)
        return self.to_residual_obs(obs)

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.x_T = np.array(self.venv.x_T, copy=True)
        term_idx = [i for i, info in enumerate(infos) if info.get("terminal_observation") is not None]
        if term_idx:
            terms = np.stack([infos[i]["terminal_observation"] for i in term_idx]).astype(np.float32)
            for i, t in zip(term_idx, self.to_residual_obs(terms)):
                infos[i]["terminal_observation"] = t
        return self.to_residual_obs(obs), rewards, dones, infos


class ResidualSAC(_SteerMixin, SAC):
    """SB3 SAC on residual actions da in [-beta, beta]^D (see module docstring).

    ``env`` must yield [s, a_base] observations (``BaseActionObsWrapper``); its action space is the
    executed-action box, used for clipping the composed action. ``state_dim`` is the length of s.
    """

    def __init__(
        self,
        policy,
        env,
        *,
        beta: float,
        state_dim: int,
        steered_policy=None,
        use_tf32: bool = False,
        compile_networks: bool = False,
        compile_mode: str = "default",
        replay_buffer_class=None,
        **kwargs,
    ):
        self._init_steer_common(steered_policy, use_tf32, compile_networks, compile_mode)
        self.beta = float(beta)
        assert self.beta > 0, beta
        self.state_dim = int(state_dim)
        self.last_k = -1  # no steering step; AdsrlCallback ignores negative k
        self.last_w: Optional[np.ndarray] = None  # the residual da of the last predict_diffused call
        self.last_probs = None
        super().__init__(
            policy, env, replay_buffer_class=replay_buffer_class or FastReplayBuffer, _init_setup_model=False, **kwargs
        )
        exec_space = self.action_space
        self.exec_low = exec_space.low.astype(np.float32)
        self.exec_high = exec_space.high.astype(np.float32)
        self.action_space = spaces.Box(-self.beta, self.beta, exec_space.shape, dtype=np.float32)
        self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._maybe_compile()

    def _modules_to_compile(self):
        return [self.critic, self.critic_target, self.actor]

    def _bounds(self):  # _SteerMixin's tensor scale/unscale use the residual box
        if not hasattr(self, "_act_low_t"):
            self._act_low_t = th.full(self.action_space.shape, -self.beta, device=self.device)
            self._act_high_t = th.full(self.action_space.shape, self.beta, device=self.device)
        return self._act_low_t, self._act_high_t

    def a_base(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)[:, self.state_dim :]

    def compose(self, obs: np.ndarray, delta: np.ndarray) -> np.ndarray:
        return compose_action(self.a_base(obs), delta, self.beta, self.exec_low, self.exec_high)

    def pop_k_counts(self):
        return None

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        super().train(gradient_steps, batch_size)
        self.last_train_stats = {k: v for k, v in self.logger.name_to_value.items() if k.startswith("train/")}

    def _sample_action(self, learning_starts: int, action_noise=None, n_envs: int = 1):
        """Env gets the composed action; the buffer gets the scaled residual (SB3 convention)."""
        unscaled_delta, buffer_action = super()._sample_action(learning_starts, action_noise, n_envs)
        return self.compose(self._last_obs, unscaled_delta), buffer_action

    def predict_diffused(self, observation, state=None, episode_start=None, deterministic: bool = False):
        delta, second = self.predict(observation, state, episode_start, deterministic)
        self.last_w = np.asarray(delta, dtype=np.float32).reshape(len(delta), -1)
        return self.compose(observation, self.last_w), second
