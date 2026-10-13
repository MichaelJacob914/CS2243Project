"""ALOHA-sim task wrapper matching openpi's example (third_party/openpi/examples/aloha_sim/env.py).

gym-aloha: episodes are cut at 300 steps; success = reaching the task's max reward (4), which also ends
the episode. The pi0_aloha_sim config injects its own prompt ("Transfer cube").
"""

import numpy as np


class AlohaTask:
    fps = 50  # gym-aloha control rate

    def __init__(self, task: str = "gym_aloha/AlohaTransferCube-v0", seed: int = 0, resize: int = 224):
        import gym_aloha  # noqa: F401 -- registers the gym_aloha/* environments
        import gymnasium

        self._gym = gymnasium.make(task, obs_type="pixels_agent_pos")
        self.prompt = task
        self.max_steps = self._gym.spec.max_episode_steps
        self._seed, self._resize = seed, resize
        self._img = None

    def reset(self, episode: int) -> dict:
        obs, _ = self._gym.reset(seed=self._seed * 100_000 + episode)  # deterministic per episode
        return self._policy_obs(obs)

    def step(self, action: np.ndarray) -> tuple[bool, bool, dict]:
        obs, _, terminated, truncated, info = self._gym.step(np.asarray(action))
        return terminated or truncated, bool(info.get("is_success", terminated)), self._policy_obs(obs)

    def frame(self) -> np.ndarray:
        return self._img

    def close(self) -> None:
        self._gym.close()

    def _policy_obs(self, obs: dict) -> dict:
        from openpi_client import image_tools

        self._img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(obs["pixels"]["top"], self._resize, self._resize)
        )
        return {"state": obs["agent_pos"], "images": {"cam_high": np.transpose(self._img, (2, 0, 1))}}
