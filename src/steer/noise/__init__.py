"""Noise samplers: what the policy starts its flow-matching integration from at each query.

Modes (config `noise.mode`):
  default   openpi draws N(0, I) itself (published-numbers baseline)
  gaussian  we draw N(0, I) from our own seeded RNG; must match `default` (checks the steering wiring)
  fixed     one draw at the start of the run, reused for every query (steerability demo: one run per seed)
  actor     a learned noise policy (DSRL) -- not implemented yet

`repeated: true` draws a single (1, action_dim) row and tiles it across the chunk, as DSRL does for pi0;
otherwise every chunk step gets its own row.
"""

import numpy as np


class NoiseSampler:
    def __init__(self, mode: str, shape: tuple[int, int], seed: int = 0, repeated: bool = False, scale: float = 1.0):
        if mode not in ("default", "gaussian", "fixed", "actor"):
            raise ValueError(f"unknown noise mode {mode!r}")
        if mode == "actor":
            raise NotImplementedError("noise.mode=actor arrives with the DSRL code")
        self.mode, self.shape, self.repeated, self.scale = mode, tuple(shape), repeated, scale
        self._rng = np.random.default_rng(seed)
        self._fixed = self._draw() if mode == "fixed" else None

    def _draw(self) -> np.ndarray:
        horizon, dim = self.shape
        rows = self._rng.standard_normal((1 if self.repeated else horizon, dim))
        return self.scale * np.broadcast_to(rows, self.shape).astype(np.float32)

    def __call__(self, obs: dict | None = None) -> np.ndarray | None:
        if self.mode == "default":
            return None
        if self.mode == "fixed":
            return self._fixed
        return self._draw()


def make(cfg: dict | None, shape: tuple[int, int], seed: int) -> NoiseSampler:
    cfg = dict(cfg or {"mode": "default"})
    return NoiseSampler(cfg.pop("mode", "default"), shape, seed=cfg.pop("seed", seed), **cfg)
