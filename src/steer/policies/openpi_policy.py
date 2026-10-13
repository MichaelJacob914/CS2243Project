"""Thin adapter over an openpi policy: load by registry name, act with optional explicit noise."""

import numpy as np

from steer.policies import registry


class OpenPIPolicy:
    """Loads an openpi checkpoint in-process (needed for steering: openpi's websocket server drops `noise`).

    `act(obs, noise)` returns the action chunk. `noise` is the flow-matching starting point x_1 with
    shape `noise_shape` = (action_horizon, action_dim): (50, 32) for pi0, (10, 32) for pi0.5.
    `noise=None` lets openpi draw it from its own RNG.
    """

    def __init__(self, name: str, seed: int = 0):
        import jax
        from openpi.policies import policy_config
        from openpi.shared import download
        from openpi.training import config as openpi_config

        self.name = name
        self.spec = registry.get(name)
        cfg = openpi_config.get_config(self.spec.config)
        self._policy = policy_config.create_trained_policy(cfg, download.maybe_download(self.spec.checkpoint))
        # openpi seeds every policy with jax.random.key(0), so parallel jobs and repeated runs would
        # draw identical noise; reseed per run.
        self._policy._rng = jax.random.key(seed)
        self.noise_shape: tuple[int, int] = (cfg.model.action_horizon, cfg.model.action_dim)
        self.last_infer_ms = 0.0

    def act(self, obs: dict, noise: np.ndarray | None = None) -> np.ndarray:
        if noise is not None and noise.shape != self.noise_shape:
            raise ValueError(f"noise shape {noise.shape} != {self.noise_shape} for {self.name}")
        out = self._policy.infer(obs, noise=None if noise is None else noise.astype(np.float32))
        self.last_infer_ms = out["policy_timing"]["infer_ms"]
        return np.asarray(out["actions"])
