"""Tests for the baselines / eval add-ons: residual SAC, noise trust region, best-of-N, CEM update (CPU)."""

import os
import sys

import numpy as np
import pytest
import torch as th

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from test_agent import OBS, D, H, K, StubSteered, agent_kwargs, make_env  # noqa: E402

from adsrl.agent import DSRLAdaptiveK, DSRLFixedK  # noqa: E402
from adsrl.best_of_n import agent_fns, best_of_n  # noqa: E402
from adsrl.residual import BaseActionObsWrapper, ResidualSAC, compose_action  # noqa: E402
from adsrl.trust_region import NoiseTrustRegion, sq_norm_per_dim  # noqa: E402


# --------------------------------------------------------------------------------------
# Residual SAC
# --------------------------------------------------------------------------------------
def residual_kwargs(steered, **kw):
    base = agent_kwargs(steered, **kw)
    for key in ("noise_critic_grad_steps", "horizon", "act_dim"):
        base.pop(key)
    return base


def make_residual(beta=0.1, **kw):
    steered = StubSteered()
    env = BaseActionObsWrapper(make_env(), steered)
    return ResidualSAC("MlpPolicy", env, beta=beta, state_dim=OBS, **residual_kwargs(steered, **kw)), steered


def test_compose_action_bounds_and_composition():
    rng = np.random.default_rng(0)
    a_base = rng.uniform(-1, 1, (64, D)).astype(np.float32)
    delta = rng.uniform(-0.5, 0.5, (64, D)).astype(np.float32)
    beta = 0.1
    a = compose_action(a_base, delta, beta, -1.0, 1.0)
    assert a.min() >= -1 and a.max() <= 1
    assert np.all(np.abs(a - a_base) <= beta + 1e-6)
    inside = (np.abs(delta) <= beta) & (np.abs(a_base + delta) <= 1)
    assert np.allclose(a[inside], (a_base + delta)[inside])
    # torch path gives the same result
    at = compose_action(th.as_tensor(a_base), th.as_tensor(delta), beta, -1.0, 1.0)
    assert np.allclose(at.numpy(), a)


def test_base_action_obs_wrapper():
    steered = StubSteered()
    inner = make_env()
    env = BaseActionObsWrapper(inner, steered)
    assert env.observation_space.shape == (OBS + D,)
    obs = env.reset()
    obs_aug = np.concatenate([obs[:, :OBS], env.x_T.reshape(2, -1)], axis=-1)
    assert np.allclose(obs[:, OBS:], steered.base_action(obs_aug).reshape(2, -1).numpy(), atol=1e-6)
    # actions pass through unchanged
    act = np.full((2, D), 0.3, np.float32)
    for _ in range(10):  # DummyChunkEnv episodes last 10 chunks -> terminal obs is rewritten too
        obs, _, done, infos = env.step(act)
    assert done.all()
    for info in infos:
        assert info["terminal_observation"].shape == (OBS + D,)


def test_residual_sac_smoke_bounds_and_checkpoint(tmp_path):
    beta = 0.1
    model, _ = make_residual(beta)
    assert np.allclose(model.action_space.low, -beta) and np.allclose(model.action_space.high, beta)
    model.learn(total_timesteps=300)
    assert model._n_updates > 0 and np.isfinite(model.last_train_stats["train/critic_loss"])
    # stored residuals are SB3-scaled to [-1, 1]
    n = model.replay_buffer.pos
    assert np.abs(model.replay_buffer.actions[:n]).max() <= 1 + 1e-6

    obs = model.env.reset()
    for det in (True, False):
        act, _ = model.predict_diffused(obs, deterministic=det)
        a_base = obs[:, OBS:]
        assert act.shape == (2, D) and model.last_w.shape == (2, D)
        assert np.abs(model.last_w).max() <= beta + 1e-6
        assert np.allclose(act, np.clip(a_base + model.last_w, -1, 1), atol=1e-6)

    # rollout path: env gets the composed action, buffer gets the scaled residual
    model._last_obs = obs
    env_action, buf_action = model._sample_action(learning_starts=0, n_envs=2)
    delta = buf_action * beta
    assert np.allclose(env_action, np.clip(obs[:, OBS:] + delta, -1, 1), atol=1e-5)

    path = str(tmp_path / "ckpt.pt")
    model.save_checkpoint(path, extra={"x": 1})
    model2, _ = make_residual(beta)
    assert model2.load_checkpoint(path) == {"x": 1}
    for p1, p2 in zip(model.policy.parameters(), model2.policy.parameters()):
        assert th.equal(p1, p2)
    assert model2.replay_buffer.pos == model.replay_buffer.pos


def test_residual_eval_callback_and_warm_start(tmp_path):
    pytest.importorskip("hydra")
    from omegaconf import OmegaConf

    import train_adsrl

    cfg = OmegaConf.create(
        dict(
            act_steps=H, obs_dim=OBS, eval_interval=5, num_evals=2, deterministic_eval=True,
            checkpoint_every_evals=1, eval_dump_every=1, logdir=str(tmp_path),
            env=dict(n_envs=2, n_eval_envs=2, reward_offset=1, max_episode_steps=10 * H),
            steer=dict(mode="residual", k=0),
        )
    )  # fmt: skip
    model, steered = make_residual(0.2)
    train_adsrl.collect_base_rollouts(model, model.env, 5, steered, residual=True)
    assert np.all(model.replay_buffer.actions[:5] == 0)
    eval_env = BaseActionObsWrapper(make_env(), steered)
    cb = train_adsrl.AdsrlCallback(cfg, eval_env, steered, -np.ones(OBS), np.ones(OBS), None)
    cb.init_callback(model)
    cb.run_evals()
    model.learn(total_timesteps=20, callback=cb)
    cb.run_evals(final=True)
    rows = list(open(tmp_path / "eval_summary.csv"))
    assert rows[0].strip().split(",") == cb.CSV_FIELDS and len(rows) > 1
    dumps = sorted(os.listdir(tmp_path / "eval_rollouts"))
    z = np.load(tmp_path / "eval_rollouts" / dumps[0])
    assert z["x_T"].shape == z["action"].shape == z["base_action"].shape == (2 * 10, D)
    assert np.all(np.abs(z["action"] - z["base_action"]) <= 0.2 + 1e-5)  # |a - a_base| <= beta
    # base_action in the dump is a_base(s, x_T) for the dumped x_T
    obs_aug = np.concatenate([z["obs"], z["x_T"]], axis=-1)
    assert np.allclose(z["base_action"], steered.base_action(obs_aug).reshape(len(obs_aug), -1).numpy(), atol=1e-5)
    assert (z["k"] == -1).all()


def test_resolve_xt_inputs_residual():
    from omegaconf import OmegaConf

    import train_adsrl

    assert train_adsrl.resolve_xt_inputs(OmegaConf.create(dict(mode="residual", k=0, xT_inputs="auto"))) == "all"


# --------------------------------------------------------------------------------------
# Noise trust region
# --------------------------------------------------------------------------------------
def test_trust_region_multiplier_update():
    tr = NoiseTrustRegion(bound=0.3, lr=0.5)
    assert tr.lam.item() == 0.0
    tr.update(0.5)  # violation 0.2 -> lambda = 0.1
    assert np.isclose(tr.lam.item(), 0.1)
    tr.update(0.5)
    assert np.isclose(tr.lam.item(), 0.2)
    tr.update(0.1)  # slack 0.2 -> lambda = 0.1
    assert np.isclose(tr.lam.item(), 0.1)
    tr.update(0.0)  # would go negative -> projected to 0
    assert tr.lam.item() == 0.0


def test_trust_region_penalty_gradient():
    tr = NoiseTrustRegion(bound=0.3, lr=0.1, init=2.0)
    w = th.full((4, D), 0.5, requires_grad=True)
    pen = tr.penalty(sq_norm_per_dim(w))
    assert np.isclose(pen.item(), 2.0 * (0.25 - 0.3))
    pen.backward()
    # d/dw [lam * mean_b mean_d w^2] = lam * 2 w / (B * D): pushes |w| down
    assert th.allclose(w.grad, th.full_like(w, 2.0 * 2 * 0.5 / (4 * D)))
    assert tr.lam.grad is None


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_trust_region_agents(tmp_path, mode):
    steered = StubSteered()
    tr = dict(bound=1e-3, lr=0.05, init=0.0)  # tight bound: lambda must grow
    if mode == "fixed":
        make = lambda: DSRLFixedK("MlpPolicy", make_env(), k=1, trust_region=tr, **agent_kwargs(steered))  # noqa: E731
    else:
        make = lambda: DSRLAdaptiveK("MlpPolicy", make_env(), num_k=K, trust_region=tr, **agent_kwargs(steered))  # noqa: E731
    model = make()
    model.learn(total_timesteps=200)
    stats = model.last_train_stats
    assert stats["train/tr_lambda"] > 0 and np.isfinite(stats["train/actor_loss"])
    assert stats["train/tr_w_sq"] > 1e-3
    path = str(tmp_path / "ckpt.pt")
    model.save_checkpoint(path)
    model2 = make()
    model2.load_checkpoint(path)
    assert th.equal(model2.trust_region.lam, model.trust_region.lam)


def test_trust_region_off_by_default():
    steered = StubSteered()
    model = DSRLFixedK("MlpPolicy", make_env(), k=1, **agent_kwargs(steered))
    assert model.trust_region is None
    assert "trust_region.lam" not in model._get_torch_save_params()[1]


def test_trust_region_config_default_off():
    from omegaconf import OmegaConf

    for task in ("can", "square", "transport"):
        cfg = OmegaConf.load(os.path.join(ROOT, "cfg", "adsrl", f"{task}.yaml"))
        assert cfg.train.trust_region.enabled is False
        assert cfg.steer.residual_beta > 0


# --------------------------------------------------------------------------------------
# Best-of-N
# --------------------------------------------------------------------------------------
def test_best_of_n_argmax():
    B, N = 3, 5
    obs = th.arange(B, dtype=th.float32).view(B, 1).expand(B, 4).clone()
    target = th.tensor([0.2, -0.7, 0.9])

    def sample_fn(obs_rep):
        # candidate j of state b has action value (j - 2) / 2 and a tag b*100 + j
        j = th.arange(obs_rep.shape[0]) % N
        b = obs_rep[:, 0].long()
        return {"action": ((j - 2) / 2.0).view(-1, 1).expand(-1, D).clone(), "tag": b * 100 + j}

    def q_fn(obs_rep, action):
        return -(action[:, 0] - target[obs_rep[:, 0].long()]).abs()

    out = best_of_n(obs, N, sample_fn, q_fn)
    # closest of {-1, -0.5, 0, 0.5, 1} to 0.2, -0.7, 0.9
    assert out["idx"].tolist() == [2, 1, 4]
    assert out["tag"].tolist() == [2, 101, 204]
    assert out["q"].shape == (B, N)
    assert th.equal(out["q"].argmax(1), out["idx"])
    assert out["action"].shape == (B, D)


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_best_of_n_agent_fns(mode):
    steered = StubSteered()
    if mode == "fixed":
        model = DSRLFixedK("MlpPolicy", make_env(), k=2, **agent_kwargs(steered))
    else:
        model = DSRLAdaptiveK("MlpPolicy", make_env(), num_k=K, **agent_kwargs(steered))
    sample_fn, q_fn = agent_fns(model, steered)
    obs = th.as_tensor(model.env.reset())
    out = best_of_n(obs, 8, sample_fn, q_fn)
    assert out["action"].shape == (2, D) and out["w"].shape == (2, D) and out["k"].shape == (2,)
    # the chosen action is the decode of the chosen (k, w) and has the max Q among candidates
    a = steered(obs, out["k"], out["w"]).reshape(2, -1)
    assert th.allclose(a, out["action"], atol=1e-6)
    q_sel = q_fn(obs, out["action"])
    assert th.allclose(q_sel, out["q"].max(1).values, atol=1e-5)
    if mode == "fixed":
        assert (out["k"] == 2).all()


# --------------------------------------------------------------------------------------
# CEM
# --------------------------------------------------------------------------------------
def test_cem_update():
    pytest.importorskip("hydra")
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from cem_noise import cem_update

    cands = np.arange(12, dtype=float).reshape(6, 2)
    scores = np.array([0.1, 0.9, 0.5, 0.8, 0.0, 0.2])
    mu, sigma, elite = cem_update(cands, scores, 2, min_std=0.01)
    assert sorted(elite.tolist()) == [1, 3]
    assert np.allclose(mu, cands[[1, 3]].mean(0)) and np.allclose(sigma, cands[[1, 3]].std(0))
    _, sigma, _ = cem_update(cands, scores, 1, min_std=0.05)
    assert np.allclose(sigma, 0.05)
