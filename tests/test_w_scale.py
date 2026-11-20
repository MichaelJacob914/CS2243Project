"""steer.w_scale: SteeredBasePolicy hands the sampler w_scale * w on every agent path (CPU).

The real SteeredBasePolicy is used with a tiny fake base model; ``steered_ddim_sample`` is replaced by
a recorder so each test sees exactly the w that reaches the sampler.
"""

import os
import sys

import numpy as np
import pytest
import torch as th

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import adsrl  # noqa: E402,F401  (adds dsrl/dppo to sys.path)
import adsrl.base_policy as bp  # noqa: E402
from adsrl.agent import DSRLAdaptiveK, DSRLFixedK  # noqa: E402
from adsrl.base_policy import SteeredBasePolicy  # noqa: E402
from adsrl.best_of_n import agent_fns  # noqa: E402
from adsrl.sampler import steered_ddim_sample  # noqa: E402
from test_agent import A, H, K, OBS, agent_kwargs, make_env  # noqa: E402

C = 2.5


class FakeBase(th.nn.Module):
    """Just enough of a DPPO DiffusionModel for SteeredBasePolicy.__init__."""

    use_ddim = True

    def __init__(self):
        super().__init__()
        self.register_buffer("ddim_alphas", th.linspace(0.1, 0.9, K))
        self.register_buffer("ddim_t", th.arange(K))


@pytest.fixture
def rec(monkeypatch):
    """Records (k, w) of every sampler call; returns a cheap deterministic action."""
    calls = []

    def fake_sample(model, cond, x_T, k, w, return_x0=False, kick=False):
        calls.append((k.detach().cpu().clone(), w.detach().cpu().clone()))
        frac = ((k.float() + 1) / K).view(-1, 1, 1)
        return th.tanh(0.5 * x_T * (1 - frac) + 0.5 * w * frac + cond["state"].mean(-1).view(-1, 1, 1))

    monkeypatch.setattr(bp, "steered_ddim_sample", fake_sample)
    return calls


def make_agent(mode, **kw):
    steered = SteeredBasePolicy(FakeBase(), OBS, H, A)
    th.manual_seed(0)
    if mode == "fixed":
        model = DSRLFixedK("MlpPolicy", make_env(), k=2, **agent_kwargs(steered, **kw))
    else:
        model = DSRLAdaptiveK("MlpPolicy", make_env(), num_k=K, **agent_kwargs(steered, **kw))
    model.learn(total_timesteps=60)  # fill the buffer, take some updates
    return model, steered


def run_both_scales(model, steered, rec, fn, path):
    """Restore the same agent state + RNG, run ``fn`` at w_scale 1 and C; return both call lists."""
    ckpt = path / "state.pt"
    model.save_checkpoint(str(ckpt))
    out = []
    for c in (1.0, C):
        model.load_checkpoint(str(ckpt))
        np.random.seed(3)
        th.manual_seed(3)
        model.action_space.seed(3)  # upstream warm-up draws w with action_space.sample()
        steered.w_scale = c
        rec.clear()
        fn()
        out.append(list(rec))
    steered.w_scale = 1.0
    return out


def assert_scaled(calls1, callsc):
    assert len(calls1) == len(callsc) > 0
    for (k1, w1), (kc, wc) in zip(calls1, callsc):
        assert th.equal(k1, kc)
        th.testing.assert_close(wc, C * w1, rtol=1e-6, atol=0)


# --------------------------------------------------------------------------------------
# SteeredBasePolicy itself (real DPPO sampler)
# --------------------------------------------------------------------------------------
def test_w_scale_one_is_bit_identical(tmp_path):
    from test_sampler import B, _inputs, _random_model

    model = _random_model(tmp_path, 20, 8)
    cond, x_T, w = _inputs(model)
    obs_aug = th.cat([cond["state"], x_T.reshape(B, -1)], dim=-1)
    k = th.arange(B) % 8
    default = SteeredBasePolicy(model, 23, 4, 7)
    one = SteeredBasePolicy(model, 23, 4, 7, w_scale=1.0)
    assert default.w_scale == 1.0
    with th.no_grad():
        ref = steered_ddim_sample(model, {"state": cond["state"]}, x_T, k, w)
    assert th.equal(default(obs_aug, k, w), ref)
    assert th.equal(one(obs_aug, k, w), ref)
    assert th.equal(one.base_action(obs_aug), default.base_action(obs_aug))
    # c != 1 decodes c * w; base_action (x_T, not w) is unaffected
    two = SteeredBasePolicy(model, 23, 4, 7, w_scale=2.0)
    with th.no_grad():
        ref2 = steered_ddim_sample(model, {"state": cond["state"]}, x_T, k, 2.0 * w)
    assert th.equal(two(obs_aug, k, w), ref2)
    assert th.equal(two.base_action(obs_aug), default.base_action(obs_aug))


def test_steered_call_scales_w(rec):
    pol = SteeredBasePolicy(FakeBase(), OBS, H, A, w_scale=C)
    obs_aug = th.randn(3, OBS + H * A)
    w = th.rand(3, H * A) * 2 - 1
    pol(obs_aug, 1, w)
    th.testing.assert_close(rec[-1][1], C * w.reshape(3, H, A), rtol=0, atol=0)
    pol.base_action(obs_aug)
    th.testing.assert_close(rec[-1][1], obs_aug[:, OBS:].reshape(3, H, A), rtol=0, atol=0)  # x_T, unscaled


# --------------------------------------------------------------------------------------
# Every agent path
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_acting_scaled(rec, tmp_path, mode):
    model, steered = make_agent(mode)
    model._last_obs = model.env.reset()
    # policy acting
    assert_scaled(*run_both_scales(model, steered, rec, lambda: model._sample_action(0, n_envs=2), tmp_path))
    # warm-up acting (uniform w)
    assert_scaled(*run_both_scales(model, steered, rec, lambda: model._sample_action(10**9, n_envs=2), tmp_path))


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
@pytest.mark.parametrize("deterministic", [True, False])
def test_eval_scaled(rec, mode, deterministic):
    model, steered = make_agent(mode)
    steered.w_scale = C
    obs = model.env.reset()
    rec.clear()
    model.predict_diffused(obs, deterministic=deterministic)
    assert len(rec) == 1
    # last_w is the actor's (unscaled) w; the sampler got C * last_w
    th.testing.assert_close(rec[0][1], C * th.as_tensor(model.last_w).reshape(-1, H, A), rtol=1e-6, atol=0)
    assert np.abs(model.last_w).max() <= 1.0 + 1e-6


@pytest.mark.parametrize("mode,tr", [("fixed", None), ("fixed", dict(bound=0.3, lr=1e-3)), ("adaptive", None)])
def test_train_backup_and_distill_scaled(rec, tmp_path, mode, tr):
    """One train() call = target-backup decodes + noise-critic distillation decodes (w ~ N(0, I))."""
    model, steered = make_agent(mode, trust_region=tr)
    calls1, callsc = run_both_scales(model, steered, rec, lambda: model.train(gradient_steps=1, batch_size=32), tmp_path)
    assert len(calls1) == 3  # 1 target backup + noise_critic_grad_steps=2 distillation decodes
    assert_scaled(calls1, callsc)


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_distill_scaled_and_critic_input_unscaled(rec, mode, monkeypatch):
    """Distillation: the sampler gets C * w with w ~ N(0, I); Q^W is fed the unscaled w."""
    model, steered = make_agent(mode)
    steered.w_scale = C
    seen = []
    orig = model.critic_noise.forward
    monkeypatch.setattr(model.critic_noise, "forward", lambda obs, a: (seen.append(a.detach().clone()), orig(obs, a))[1])
    data = model.replay_buffer.sample(16)
    rec.clear()
    model.update_noise_critic(data)
    (k, w_s), = rec
    w_in = seen[-1][:, -H * A:]  # adaptive prepends onehot(k)
    th.testing.assert_close(w_s.reshape(16, -1), C * model._unscale_t(w_in), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_best_of_n_scaled(rec, mode):
    model, steered = make_agent(mode)
    steered.w_scale = C
    sample_fn, _ = agent_fns(model, steered)
    obs = th.as_tensor(model.env.reset())
    rec.clear()
    out = sample_fn(obs.repeat_interleave(3, 0))
    (k, w_s), = rec
    th.testing.assert_close(w_s.reshape(6, -1), C * out["w"], rtol=1e-6, atol=0)


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_checkpoint_round_trip(rec, tmp_path, mode):
    """Save at w_scale=C, rebuild a fresh agent around a w_scale=C policy, load: same decoded actions."""
    model, steered = make_agent(mode)
    steered.w_scale = C
    path = str(tmp_path / "ckpt.pt")
    model.save_checkpoint(path)
    steered2 = SteeredBasePolicy(FakeBase(), OBS, H, A, w_scale=C)
    cls = DSRLFixedK if mode == "fixed" else DSRLAdaptiveK
    extra = dict(k=2) if mode == "fixed" else dict(num_k=K)
    model2 = cls("MlpPolicy", make_env(), **extra, **agent_kwargs(steered2))
    model2.load_checkpoint(path)
    obs = model.env.reset()
    a1, _ = model.predict_diffused(obs, deterministic=True)
    a2, _ = model2.predict_diffused(obs, deterministic=True)
    np.testing.assert_array_equal(a1, a2)
    th.testing.assert_close(rec[-1][1], C * th.as_tensor(model2.last_w).reshape(-1, H, A), rtol=1e-6, atol=0)


# --------------------------------------------------------------------------------------
# Config plumbing
# --------------------------------------------------------------------------------------
def test_config_plumbing(tmp_path, monkeypatch):
    hydra = pytest.importorskip("hydra")
    import json

    from omegaconf import OmegaConf

    import train_adsrl

    assert train_adsrl.steer_w_scale(OmegaConf.create(dict(mode="fixed"))) == 1.0  # old configs
    assert train_adsrl.steer_w_scale(OmegaConf.create(dict(mode="adaptive", w_scale=2))) == 2.0
    assert train_adsrl.steer_w_scale(OmegaConf.create(dict(mode="residual", w_scale=1.0))) == 1.0
    with pytest.raises(ValueError):
        train_adsrl.steer_w_scale(OmegaConf.create(dict(mode="residual", w_scale=2.0)))
    for task in ("can", "square", "transport"):
        assert OmegaConf.load(os.path.join(ROOT, "cfg", "adsrl", f"{task}.yaml")).steer.w_scale == 1.0

    # render_rollouts / best_of_n_eval rebuild the sampler from run_info.json
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import render_rollouts

    import hydra.utils  # noqa: F401

    monkeypatch.setattr(hydra.utils, "instantiate", lambda cfg: FakeBase())
    (tmp_path / "checkpoint").mkdir()
    dims = dict(obs_dim=OBS, act_steps=H, action_dim=A, model=dict(device="cpu"), device="cpu")
    # new run, pre-w_scale run (no key), base-policy cfg (no steer section)
    for extra, want in ((dict(steer=dict(mode="fixed", k=2, w_scale=3.0)), 3.0),
                        (dict(steer=dict(mode="fixed", k=2)), 1.0), ({}, 1.0)):  # fmt: skip
        (tmp_path / "run_info.json").write_text(json.dumps({"cfg": {**dims, **extra}}))
        rc = render_rollouts.run_cfg(str(tmp_path / "checkpoint" / "latest.pt"), "cpu")
        assert render_rollouts.build_steered(rc).w_scale == want
