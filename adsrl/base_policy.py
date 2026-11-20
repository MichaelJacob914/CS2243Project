"""Steered wrapper around a frozen DPPO diffusion policy (counterpart of dsrl/utils.py
DPPOBasePolicyWrapper).

DPPOBasePolicyWrapper does no normalization of its own: observations arrive already normalized
by the robomimic_lowdim env wrapper, and the returned actions are in the model's normalized
[-1, 1]-ish action space (the env wrapper unnormalizes them). SteeredBasePolicy keeps that
contract: it passes `obs` straight through as cond["state"] and returns actions in the same space.

`w_scale` (config steer.w_scale) widens the noise bound: every steered call decodes w_scale * w, so
the agents keep acting, learning and distilling in the unscaled w space (the actor's [-1, 1] box)
while the sampler sees [-c, c]. `base_action` is unaffected (it injects x_T, not a steering w).

`param` (config steer.param): "absolute" (default) injects w_scale * w as the step-k noise; "kick" injects
eps_hat_k + w_scale * w, where eps_hat_k is the implied noise of the unsteered trajectory from x_T
(x_T itself at k=0), so w = 0 reproduces `base_action` at every k.

`w_scale_dims` (config steer.w_scale_dims, default None): an optional per-dimension scale of shape (H, A),
or (K, H, A) for one scale per steering step, applied on top of w_scale, so the sampler gets
w_scale * w_scale_dims * w (kick: eps_hat_k + that); with (K, H, A), sample i uses row k_i.
`load_w_scale_dims` reads it from a .npy or .json file: a 1-D vector of H*A values, or A values tiled over the
horizon; or a 2-D (K, A) array (tiled over the horizon) or 3-D (K, H, A) array. Every decode path goes through
`__call__`, so the agents see only the change in what w means.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from adsrl.sampler import num_ddim_steps, steered_ddim_sample

PARAMS = ("absolute", "kick")


def load_w_scale_dims(path, horizon: int, act_dim: int):
    """Per-dim noise scale from a .npy or .json file, or None for path=None. A 1-D vector of length H*A
    (row-major over (H, A)) or A (tiled over the horizon) gives (H, A). A 2-D (K, A) array (tiled over the
    horizon) or 3-D (K, H, A) array gives (K, H, A): one row per steering step k (SteeredBasePolicy checks K).
    A .json file may be a (nested) list or an object with key "w_scale_dims". All values must be > 0."""
    if path is None:
        return None
    if str(path).endswith(".json"):
        import json

        with open(path) as f:
            v = json.load(f)
        if isinstance(v, dict):
            v = v["w_scale_dims"]
    else:
        v = np.load(path)
    v = np.asarray(v, dtype=np.float32)
    if v.ndim in (2, 3):  # per-k: (K, A) or (K, H, A)
        if v.ndim == 2 and v.shape[1] == act_dim:
            v = np.repeat(v[:, None], horizon, 1)
        if v.shape[1:] != (horizon, act_dim):
            raise ValueError(f"per-k w_scale_dims has shape {v.shape}; need (K, {act_dim}) or (K, {horizon}, {act_dim})")
        if not np.all(np.isfinite(v) & (v > 0)):
            raise ValueError("w_scale_dims must be finite and > 0")
        return v
    v = v.reshape(-1)
    if v.size == act_dim:
        v = np.tile(v, horizon)
    if v.size != horizon * act_dim:
        raise ValueError(f"w_scale_dims has {v.size} values; need {act_dim} or {horizon * act_dim}")
    if not np.all(np.isfinite(v) & (v > 0)):
        raise ValueError("w_scale_dims must be finite and > 0")
    return v.reshape(horizon, act_dim)


class SteeredBasePolicy:
    def __init__(self, base_policy, obs_dim: int, horizon: int, act_dim: int, w_scale: float = 1.0,
                 param: str = "absolute", w_scale_dims=None):
        # Accept either the raw DiffusionModel or a DPPOBasePolicyWrapper around it.
        if not isinstance(base_policy, torch.nn.Module) and hasattr(base_policy, "base_policy"):
            base_policy = base_policy.base_policy
        self.base_policy = base_policy
        self.obs_dim = int(obs_dim)
        self.horizon = int(horizon)
        self.act_dim = int(act_dim)
        self.w_scale = float(w_scale)
        assert self.w_scale > 0, self.w_scale
        assert param in PARAMS, param
        self.param = param
        self.device = base_policy.ddim_alphas.device
        self.K = num_ddim_steps(base_policy)
        self.w_scale_dims = None  # (H, A) or per-k (K, H, A) tensor, or None
        if w_scale_dims is not None:
            d = torch.as_tensor(np.asarray(w_scale_dims, dtype=np.float32))
            if d.dim() == 3:
                if tuple(d.shape) != (self.K, self.horizon, self.act_dim):
                    raise ValueError(f"per-k w_scale_dims has shape {tuple(d.shape)}; need ({self.K}, {self.horizon}, {self.act_dim})")
            else:
                d = d.reshape(self.horizon, self.act_dim)
            assert bool((d > 0).all()), "w_scale_dims must be > 0"
            self.w_scale_dims = d

    def _to_tensor(self, x) -> Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x.to(device=self.device, dtype=torch.float32)

    def split(self, obs_aug):
        """obs_aug = [obs, x_T.flatten()] along the last dim -> (obs (B, obs_dim), x_T (B, H, A))."""
        obs_aug = self._to_tensor(obs_aug)
        obs = obs_aug[..., : self.obs_dim]
        x_T = obs_aug[..., self.obs_dim :].reshape(-1, self.horizon, self.act_dim)
        return obs, x_T

    @torch.no_grad()
    def __call__(self, obs_aug, k, w, return_x0: bool = False):
        """Actions (B, H, A) for steering step k with noise w (B, H*A) or (B, H, A); the sampler
        receives s * w (absolute) or eps_hat_k + s * w (kick), s = w_scale [* w_scale_dims]."""
        obs, x_T = self.split(obs_aug)
        B = obs.shape[0]
        if not torch.is_tensor(k):
            k = torch.as_tensor(np.asarray(k), dtype=torch.long)
        if k.dim() == 0:
            k = k.expand(B)
        w = self._to_tensor(w).reshape(B, self.horizon, self.act_dim)
        if self.w_scale != 1.0:  # skip at 1 so the default path is bit-identical
            w = w * self.w_scale
        k = k.to(self.device)
        if self.w_scale_dims is not None:  # None skips it, so the default path is bit-identical
            d = self.w_scale_dims.to(w.device)
            w = w * (d[k.to(w.device)] if d.dim() == 3 else d)  # per-k: sample i uses row k_i
        return steered_ddim_sample(
            self.base_policy, {"state": obs}, x_T, k, w, return_x0=return_x0,
            kick=self.param == "kick",
        )

    @torch.no_grad()
    def base_action(self, obs_aug) -> Tensor:
        """Unsteered deterministic DDIM action from the x_T stored in obs_aug."""
        obs, x_T = self.split(obs_aug)
        k = torch.zeros(obs.shape[0], dtype=torch.long, device=self.device)
        return steered_ddim_sample(self.base_policy, {"state": obs}, x_T, k, x_T)
