"""Noise trust region: a Lagrangian bound on the actor's noise magnitude (after LP-DS, arXiv 2606.01151).

Constraint: E_{s, w ~ pi}[ ||w||^2 / D ] <= bound, with w in env noise units (D = H*A).
The actor minimizes  L_actor + lambda * (E||w||^2/D - bound)  and lambda >= 0 follows projected dual
ascent  lambda <- max(0, lambda + lr * (E||w||^2/D - bound)), using the batch estimate from the same
actor step. With w ~ N(0, I) clipped to the [-1, 1] noise box, E[w_i^2] ~ 0.29, so bound ~0.3 keeps
the noise about as concentrated as the prior; bound >= 1 is never active in the [-1, 1] box.

``lam`` is a 1-element tensor so it can be checkpointed and loaded in place like the temperatures.
"""

from __future__ import annotations

import torch as th


class NoiseTrustRegion:
    def __init__(self, bound: float, lr: float, init: float = 0.0, device="cpu"):
        assert bound > 0 and lr > 0 and init >= 0, (bound, lr, init)
        self.bound = float(bound)
        self.lr = float(lr)
        self.lam = th.full((1,), float(init), device=device)

    def penalty(self, sq_norm_per_dim: th.Tensor) -> th.Tensor:
        """lambda * (mean(sq_norm_per_dim) - bound); differentiable in the noise, lambda is fixed."""
        return self.lam.detach().squeeze() * (sq_norm_per_dim.mean() - self.bound)

    @th.no_grad()
    def update(self, mean_sq_norm_per_dim: float) -> None:
        self.lam.add_(self.lr * (float(mean_sq_norm_per_dim) - self.bound)).clamp_(min=0.0)


def sq_norm_per_dim(w: th.Tensor) -> th.Tensor:
    """||w||^2 / D over the last dim."""
    return w.pow(2).mean(-1)
