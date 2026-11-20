"""CPU tests for the steered DDIM sampler, SteeredBasePolicy and XTAugmentWrapper.

Model variants:
  random_can   : randomly initialized DiffusionEval with the can config (T=20, K=8)
  random_k10   : random, T=100, K=10, final_action_clip_value=1.0, eps_clip_value=2.0
  can_ckpt     : the real can checkpoint (skipped if not downloaded)
"""

import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import adsrl  # noqa: E402  (adds dsrl/dppo to sys.path)
from adsrl.base_policy import SteeredBasePolicy  # noqa: E402
from adsrl.model_utils import abs_dsrl_path, build_model, load_task_cfg  # noqa: E402
from adsrl.sampler import ddim_step, num_ddim_steps, steered_ddim_sample  # noqa: E402

B = 6
ATOL = 1e-6


def _random_model(tmp_path, denoising_steps, ddim_steps, **extra):
    from model.diffusion.diffusion_eval import DiffusionEval
    from model.diffusion.mlp_diffusion import DiffusionMLP

    torch.manual_seed(0)
    obs_dim, H, A = 23, 4, 7
    net = DiffusionMLP(
        action_dim=A, horizon_steps=H, cond_dim=obs_dim, time_dim=16,
        mlp_dims=[64, 64, 64], residual_style=True,
    )
    # Scale up the output so x0 clamping is actually exercised.
    with torch.no_grad():
        for p in net.parameters():
            p.mul_(3.0)
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"ema": {f"network.{k}": v for k, v in net.state_dict().items()}}, ckpt)
    net2 = DiffusionMLP(
        action_dim=A, horizon_steps=H, cond_dim=obs_dim, time_dim=16,
        mlp_dims=[64, 64, 64], residual_style=True,
    )
    model = DiffusionEval(
        network_path=str(ckpt), ft_denoising_steps=0, network=net2, horizon_steps=H,
        obs_dim=obs_dim, action_dim=A, device="cpu", predict_epsilon=True,
        denoised_clip_value=1.0, randn_clip_value=3, denoising_steps=denoising_steps,
        use_ddim=True, ddim_steps=ddim_steps, controllable_noise=True, **extra,
    )
    return model.eval()


@pytest.fixture(params=["random_can", "random_k10", "can_ckpt"])
def model(request, tmp_path):
    if request.param == "random_can":
        return _random_model(tmp_path, 20, 8)
    if request.param == "random_k10":
        return _random_model(tmp_path, 100, 10, final_action_clip_value=1.0, eps_clip_value=2.0)
    cfg = load_task_cfg("can")
    if not os.path.exists(abs_dsrl_path(cfg.base_policy_path)):
        pytest.skip("can checkpoint not downloaded")
    return build_model(cfg, device="cpu")


def _inputs(model, seed=1):
    g = torch.Generator().manual_seed(seed)
    H, A = model.horizon_steps, model.action_dim
    state = torch.rand(B, model.obs_dim, generator=g) * 2 - 1
    x_T = torch.randn(B, H, A, generator=g)
    w = torch.randn(B, H, A, generator=g)
    return {"state": state}, x_T, w


@torch.no_grad()
def _plain_ddim(model, cond, x):
    """Unsteered DDIM, recording each step's input x and eps_hat."""
    K = num_ddim_steps(model)
    xs, eps = [], []
    for j in range(K):
        xs.append(x)
        mu, _, e, _, _ = ddim_step(model, x, j, cond)
        eps.append(e)
        x = mu
        if model.final_action_clip_value is not None and j == K - 1:
            x = x.clamp(-model.final_action_clip_value, model.final_action_clip_value)
    return x, xs, eps


@torch.no_grad()
def test_k0_matches_upstream_forward(model):
    cond, x_T, w = _inputs(model)
    ref = model({"state": cond["state"], "noise_action": w}, deterministic=True).trajectories
    k = torch.zeros(B, dtype=torch.long)
    out = steered_ddim_sample(model, cond, x_T, k, w)
    assert out.shape == x_T.shape
    assert torch.equal(out, ref) or torch.allclose(out, ref, atol=ATOL, rtol=0)
    # Unsteered from x_T also matches upstream with noise_action = x_T.
    ref0 = model({"state": cond["state"], "noise_action": x_T}, deterministic=True).trajectories
    plain, _, _ = _plain_ddim(model, cond, x_T)
    torch.testing.assert_close(plain, ref0, atol=ATOL, rtol=0)


@torch.no_grad()
def test_eps_hat_recovers_unsteered(model):
    cond, x_T, _ = _inputs(model)
    K = num_ddim_steps(model)
    plain, xs, eps = _plain_ddim(model, cond, x_T)
    for kk in range(K):
        # The direction feeding the input of step k: x_T itself for k=0, else eps_hat_{k-1}.
        w = x_T if kk == 0 else eps[kk - 1]
        k = torch.full((B,), kk, dtype=torch.long)
        out = steered_ddim_sample(model, cond, x_T, k, w)
        torch.testing.assert_close(out, plain, atol=ATOL, rtol=0, msg=f"k={kk}")


@torch.no_grad()
def test_injection_formula(model):
    """Input to step k is sqrt(ab_{t_k}) x0_hat_{k-1} + sqrt(1 - ab_{t_k}) w, ab_{t_k} = ddim_alphas[k]."""
    cond, x_T, w = _inputs(model)
    K = num_ddim_steps(model)
    for kk in range(1, K):
        k = torch.full((B,), kk, dtype=torch.long)
        _, x0_all = steered_ddim_sample(model, cond, x_T, k, w, return_x0=True)
        ab = model.ddim_alphas[kk]
        assert torch.allclose(ab, model.alphas_cumprod[model.ddim_t[kk]])
        x_k = ab.sqrt() * x0_all[:, kk - 1] + (1 - ab).sqrt() * w
        # x0_hat at step k must be what the model predicts from that input.
        _, x0_k, _, _, _ = ddim_step(model, x_k, kk, cond)
        torch.testing.assert_close(x0_all[:, kk], x0_k, atol=1e-5, rtol=0)


@torch.no_grad()
def test_mixed_k_batch(model):
    cond, x_T, w = _inputs(model)
    K = num_ddim_steps(model)
    k = torch.tensor([(3 * i + 1) % K for i in range(B)], dtype=torch.long)
    out, x0_all = steered_ddim_sample(model, cond, x_T, k, w, return_x0=True)
    assert x0_all.shape == (B, K, model.horizon_steps, model.action_dim)
    for kk in k.unique().tolist():
        ref, ref_x0 = steered_ddim_sample(
            model, cond, x_T, torch.full((B,), kk, dtype=torch.long), w, return_x0=True
        )
        m = k == kk
        torch.testing.assert_close(out[m], ref[m], atol=ATOL, rtol=0)
        torch.testing.assert_close(x0_all[m], ref_x0[m], atol=ATOL, rtol=0)
    # Different k should actually give different actions (steering has an effect).
    a0 = steered_ddim_sample(model, cond, x_T, torch.zeros(B, dtype=torch.long), w)
    a1 = steered_ddim_sample(model, cond, x_T, torch.ones(B, dtype=torch.long), w)
    assert not torch.allclose(a0, a1)


def test_k_out_of_range(model):
    cond, x_T, w = _inputs(model)
    with pytest.raises(ValueError):
        steered_ddim_sample(model, cond, x_T, num_ddim_steps(model), w)


def test_steered_base_policy(model):
    cond, x_T, w = _inputs(model)
    H, A, D = model.horizon_steps, model.action_dim, model.obs_dim
    pol = SteeredBasePolicy(model, D, H, A)
    obs_aug = torch.cat([cond["state"], x_T.reshape(B, -1)], dim=-1)
    obs, xt = pol.split(obs_aug)
    assert torch.equal(obs, cond["state"]) and torch.equal(xt, x_T)
    # k=0 with w is DSRL: same as DPPOBasePolicyWrapper(obs, w).
    a = pol(obs_aug.numpy(), 0, w.reshape(B, -1))
    with torch.no_grad():
        ref = model({"state": cond["state"], "noise_action": w}, deterministic=True).trajectories
    torch.testing.assert_close(a, ref, atol=ATOL, rtol=0)
    base = pol.base_action(obs_aug)
    with torch.no_grad():
        ref0 = model({"state": cond["state"], "noise_action": x_T}, deterministic=True).trajectories
    torch.testing.assert_close(base, ref0, atol=ATOL, rtol=0)
    k = torch.arange(B) % pol.K
    a2 = pol(obs_aug, k, w)
    assert a2.shape == (B, H, A)


def test_xt_augment_wrapper():
    import gymnasium
    from gymnasium import spaces
    from stable_baselines3.common.vec_env import DummyVecEnv

    from adsrl.env import XTAugmentWrapper

    obs_dim, H, A, n = 5, 4, 3, 3

    class Toy(gymnasium.Env):
        def __init__(self):
            self.observation_space = spaces.Box(-1, 1, (obs_dim,), np.float32)
            self.action_space = spaces.Box(-1, 1, (H * A,), np.float32)
            self.t = 0

        def reset(self, seed=None, options=None):
            self.t = 0
            return np.zeros(obs_dim, np.float32), {}

        def step(self, action):
            self.t += 1
            return np.full(obs_dim, self.t, np.float32), 0.0, self.t >= 2, False, {}

    venv = XTAugmentWrapper(DummyVecEnv([Toy] * n), H, A, seed=0)
    assert venv.observation_space.shape == (obs_dim + H * A,)
    obs = venv.reset()
    assert obs.shape == (n, obs_dim + H * A) and obs.dtype == np.float32
    np.testing.assert_array_equal(obs[:, :obs_dim], 0)
    assert np.allclose(venv.x_T.reshape(n, -1), obs[:, obs_dim:])
    prev = obs[:, obs_dim:].copy()
    acts = np.zeros((n, H * A), np.float32)
    obs, _, dones, infos = venv.step(acts)
    assert obs.shape == (n, obs_dim + H * A)
    assert not np.allclose(obs[:, obs_dim:], prev)  # fresh x_T
    assert not dones.any()
    obs2, _, dones, infos = venv.step(acts)
    assert dones.all()
    assert not np.allclose(obs2[:, obs_dim:], obs[:, obs_dim:])
    for i, info in enumerate(infos):
        term = info["terminal_observation"]
        assert term.shape == (obs_dim + H * A,)
        np.testing.assert_array_equal(term[:obs_dim], 2)
        assert not np.allclose(term[obs_dim:], obs2[i, obs_dim:])
    # x_T statistics look standard normal.
    xs = np.concatenate([venv.step(acts)[0][:, obs_dim:] for _ in range(300)])
    assert abs(xs.mean()) < 0.05 and abs(xs.std() - 1) < 0.05
    # Seeding is reproducible.
    v1 = XTAugmentWrapper(DummyVecEnv([Toy] * n), H, A, seed=7)
    v2 = XTAugmentWrapper(DummyVecEnv([Toy] * n), H, A, seed=7)
    np.testing.assert_array_equal(v1.reset(), v2.reset())
