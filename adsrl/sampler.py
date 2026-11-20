"""Deterministic DDIM sampler for DPPO diffusion policies, with steering at a chosen step.

Matches the math of `DiffusionEval.p_mean_var` (deterministic=True, i.e. eta=0) and
`DiffusionModel.forward` (dsrl/dppo/model/diffusion/{diffusion,diffusion_eval}.py) op for op,
so k=0 reproduces upstream DSRL exactly.

Indexing (after upstream's flip, j = 0..K-1 runs from noisiest to cleanest):
    ddim_t[j]           diffusion timestep fed to the network at step j
    ddim_alphas[j]      alpha-bar of the *input* x at step j       (= alphas_cumprod[ddim_t[j]])
    ddim_alphas_prev[j] alpha-bar of the *output* of step j        (= ddim_alphas[j+1], 1.0 for j=K-1)

Steering at k: the input to step k is
    x = sqrt(ddim_alphas[k]) * x0_hat_{k-1} + sqrt(1 - ddim_alphas[k]) * w,
which is exactly the step k-1 DDIM update with its eps_hat replaced by w (since
ddim_alphas_prev[k-1] == ddim_alphas[k]). That is how it is implemented here, so passing
w = eps_hat_{k-1} reproduces unsteered DDIM bit for bit. For k=0, x = w (x_T is replaced).

Kick parameterization (``kick=True``, config steer.param=kick): w is a perturbation added to the
implied noise eps_hat_k, the noise that keeps the sampler on the unsteered trajectory from x_T:
eps_hat_k = eps_hat_{k-1} of the unsteered run for k >= 1 (the steps before k are unsteered, so this is
the eps the loop computes at step k-1) and eps_hat_0 = x_T. The injected noise is eps_hat_k + w, so
w = 0 gives the unsteered DDIM action bit for bit at every k.
"""

from __future__ import annotations

import torch
from torch import Tensor


def num_ddim_steps(model) -> int:
    """Number of DDIM steps K of a DPPO DiffusionModel."""
    assert model.use_ddim, "steered_ddim_sample requires a DDIM model (use_ddim=True)."
    return int(model.ddim_t.shape[0])


def _network(model):
    # DiffusionEval uses `self.actor` (same module as `self.network` when not fine-tuned).
    return getattr(model, "actor", model.network)


def _extract(a: Tensor, idx: Tensor, x_shape) -> Tensor:
    # Same as dppo model.diffusion.sampling.extract
    b = idx.shape[0]
    return a.gather(-1, idx).reshape(b, *((1,) * (len(x_shape) - 1)))


def ddim_step(model, x: Tensor, j: int, cond: dict):
    """One deterministic DDIM step at loop index j.

    Returns (mu, x0_hat, eps_hat, alpha_prev, dir_coef), where mu = sqrt(alpha_prev) * x0_hat
    + dir_coef * eps_hat is the next sample (before any final-action clip). x0_hat is the
    clamped clean estimate and eps_hat the epsilon recomputed from it, exactly as upstream.
    """
    B = x.shape[0]
    device = x.device
    t_b = torch.full((B,), int(model.ddim_t[j]), device=device, dtype=torch.long)
    index_b = torch.full((B,), j, device=device, dtype=torch.long)

    noise = _network(model)(x, t_b, cond=cond)
    alpha = _extract(model.ddim_alphas, index_b, x.shape)
    alpha_prev = _extract(model.ddim_alphas_prev, index_b, x.shape)
    sqrt_one_minus_alpha = _extract(model.ddim_sqrt_one_minus_alphas, index_b, x.shape)
    x_recon = (x - sqrt_one_minus_alpha * noise) / (alpha**0.5)
    if model.denoised_clip_value is not None:
        x_recon = x_recon.clamp(-model.denoised_clip_value, model.denoised_clip_value)
        noise = (x - alpha ** (0.5) * x_recon) / sqrt_one_minus_alpha
    if model.eps_clip_value is not None:
        noise = noise.clamp(-model.eps_clip_value, model.eps_clip_value)

    # eta = 0 (deterministic). Same expression as DiffusionEval.p_mean_var.
    etas = torch.zeros((B, 1, 1), device=device)
    sigma = (etas * ((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)) ** 0.5).clamp(
        min=1e-10
    )
    dir_coef = (1.0 - alpha_prev - sigma**2).clamp(min=0).sqrt()
    mu = (alpha_prev**0.5) * x_recon + dir_coef * noise
    return mu, x_recon, noise, alpha_prev, dir_coef


def steered_ddim_sample(
    model,
    cond: dict,
    x_T: Tensor,
    k,
    w: Tensor,
    return_x0: bool = False,
    kick: bool = False,
):
    """Deterministic DDIM from x_T, with the step-k input re-noised by w (or by eps_hat_k + w if kick).

    Args:
        model: DPPO DiffusionModel / DiffusionEval with use_ddim=True.
        cond: dict with "state" of shape (B, obs_dim) or (B, To, obs_dim).
        x_T: (B, H, A) initial noise.
        k: (B,) LongTensor (or int) of steering steps in [0, K-1].
        w: (B, H, A) steering noise.
        return_x0: also return x0_hat from every step, shape (B, K, H, A).
        kick: w is a perturbation of the implied noise eps_hat_k (x_T at k=0) instead of the noise.

    Returns:
        actions (B, H, A) [, x0_all (B, K, H, A)]
    """
    K = num_ddim_steps(model)
    B = x_T.shape[0]
    device = model.ddim_alphas.device
    x_T = x_T.to(device)
    w = w.to(device).reshape(x_T.shape)
    if not torch.is_tensor(k):
        k = torch.full((B,), int(k), dtype=torch.long)
    k = k.to(device=device, dtype=torch.long).reshape(B)
    if bool(((k < 0) | (k >= K)).any()):
        raise ValueError(f"k must be in [0, {K - 1}], got {k.tolist()}")
    k_b = k.view(B, *((1,) * (x_T.dim() - 1)))

    x = torch.where(k_b == 0, x_T + w if kick else w, x_T)
    x0_all = []
    for j in range(K):
        mu, x0, eps, alpha_prev, dir_coef = ddim_step(model, x, j, cond)
        if return_x0:
            x0_all.append(x0)
        # Steering at step k == j+1: replace this transition's eps_hat by w.
        # Always computed (no data-dependent branch, so the loop stays graph-capturable).
        mu_w = (alpha_prev**0.5) * x0 + dir_coef * (eps + w if kick else w)
        mu = torch.where(k_b == (j + 1), mu_w, mu)
        x = mu
        if model.final_action_clip_value is not None and j == K - 1:
            x = torch.clamp(x, -model.final_action_clip_value, model.final_action_clip_value)
    if return_x0:
        return x, torch.stack(x0_all, dim=1)
    return x
