"""Tests for adsrl/agent.py (CPU). A stub replaces SteeredBasePolicy so no checkpoint is needed."""

import os
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from adsrl.agent import DSRLAdaptiveK, DSRLFixedK, FastReplayBuffer, HybridActor  # noqa: E402
from adsrl.env import XTAugmentWrapper  # noqa: E402

OBS, H, A, K = 5, 2, 3, 4
D = H * A


class StubSteered:
    """Deterministic, cheap stand-in for SteeredBasePolicy with the same interface."""

    def __init__(self, obs_dim=OBS, horizon=H, act_dim=A, K=K):
        self.obs_dim, self.horizon, self.act_dim, self.K = obs_dim, horizon, act_dim, K
        self.k_seen = []

    def split(self, obs_aug):
        obs_aug = th.as_tensor(np.asarray(obs_aug) if not th.is_tensor(obs_aug) else obs_aug, dtype=th.float32)
        return obs_aug[:, : self.obs_dim], obs_aug[:, self.obs_dim :].reshape(-1, self.horizon, self.act_dim)

    @th.no_grad()
    def __call__(self, obs_aug, k, w):
        obs, x_T = self.split(obs_aug)
        B = obs.shape[0]
        k = th.as_tensor(k, dtype=th.long).reshape(-1).expand(B) if not th.is_tensor(k) or k.dim() == 0 else k
        self.k_seen.append(k.clone().cpu())
        w = th.as_tensor(w, dtype=th.float32).reshape(B, self.horizon, self.act_dim)
        frac = ((k.float() + 1) / self.K).view(B, 1, 1)
        return th.tanh(0.5 * x_T * (1 - frac) + 0.5 * w * frac + obs.mean(-1).view(B, 1, 1))

    def base_action(self, obs_aug):
        obs, x_T = self.split(obs_aug)
        return th.tanh(0.5 * x_T + obs.mean(-1).view(-1, 1, 1))


class DummyChunkEnv(gym.Env):
    """Reward is higher when the executed chunk is near 0.3; 10 chunks per episode."""

    def __init__(self):
        self.observation_space = spaces.Box(-1, 1, (OBS,), np.float32)
        self.action_space = spaces.Box(-1, 1, (D,), np.float32)
        self.t = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self.np_random.uniform(-1, 1, OBS).astype(np.float32), {}

    def step(self, action):
        self.t += 1
        err = float(np.mean((np.asarray(action) - 0.3) ** 2))
        reward = 0.0 if err < 0.02 else -1.0
        obs = self.np_random.uniform(-1, 1, OBS).astype(np.float32)
        return obs, reward, self.t >= 10, False, {}


def make_env(n=2):
    return XTAugmentWrapper(make_vec_env(DummyChunkEnv, n_envs=n, vec_env_cls=DummyVecEnv), H, A)


def agent_kwargs(steered, **kw):
    base = dict(
        learning_rate=1e-3,
        buffer_size=2000,
        learning_starts=1,
        batch_size=32,
        gradient_steps=2,
        noise_critic_grad_steps=2,
        ent_coef="auto",
        target_entropy=0.0,
        policy_kwargs=dict(
            net_arch=dict(pi=[32, 32], qf=[32, 32]),
            activation_fn=th.nn.Tanh,
            post_linear_modules=[th.nn.LayerNorm],
            n_critics=2,
        ),
        device="cpu",
        steered_policy=steered,
        horizon=H,
        act_dim=A,
    )
    base.update(kw)
    return base


def make_adaptive(**kw):
    steered = StubSteered()
    model = DSRLAdaptiveK("MlpPolicy", make_env(), num_k=K, **agent_kwargs(steered, **kw))
    return model, steered


# --------------------------------------------------------------------------------------
# Hybrid actor
# --------------------------------------------------------------------------------------
def test_hybrid_actor_shapes_and_probs():
    th.manual_seed(0)
    actor = HybridActor(OBS + D, D, K, [32, 32])
    obs = th.randn(7, OBS + D)
    out = actor.all_heads(obs)
    assert out["probs"].shape == (7, K)
    assert out["log_pk"].shape == (7, K)
    assert out["w"].shape == (7, K, D)
    assert out["log_prob_w"].shape == (7, K)
    th.testing.assert_close(out["probs"].sum(-1), th.ones(7))
    assert out["w"].abs().max() <= 1.0

    s = actor.sample(obs)
    assert s["k"].shape == (7,) and s["k"].dtype == th.long
    assert s["w"].shape == (7, D) and s["log_prob_w"].shape == (7,)
    assert ((s["k"] >= 0) & (s["k"] < K)).all()


def test_squashed_log_prob_matches_sb3():
    th.manual_seed(0)
    mean, log_std = th.randn(5, D), th.randn(5, D).clamp(-2, 1)
    th.manual_seed(1)
    w, lp = HybridActor._squashed_sample(mean, log_std)
    th.manual_seed(1)
    u = mean + log_std.exp() * th.randn_like(mean)
    dist = SquashedDiagGaussianDistribution(D)
    dist.proba_distribution(mean, log_std)
    th.testing.assert_close(w, th.tanh(u))
    th.testing.assert_close(lp, dist.log_prob(w, gaussian_actions=u), rtol=1e-4, atol=1e-4)


def test_deterministic_sample_is_argmax_and_mean():
    th.manual_seed(0)
    actor = HybridActor(OBS + D, D, K, [32])
    obs = th.randn(9, OBS + D)
    log_pk, mean, _ = actor(obs)
    s = actor.sample(obs, deterministic=True)
    assert th.equal(s["k"], log_pk.argmax(-1))
    th.testing.assert_close(s["w"], th.tanh(mean[th.arange(9), s["idx"]]))


def test_actor_loss_entropy_shapes_and_grads_reach_both_heads():
    model, _ = make_adaptive()
    obs = th.randn(16, OBS + D)
    loss, aux = model.actor_loss(obs, th.tensor(0.1), th.tensor(0.2))
    assert loss.dim() == 0
    assert aux["k_entropy"].shape == (16,) and aux["w_log_prob"].shape == (16,)
    assert (aux["k_entropy"] <= np.log(K) + 1e-5).all() and (aux["k_entropy"] >= 0).all()
    model.hybrid_actor.zero_grad()
    loss.backward()
    a = model.hybrid_actor
    for p in (a.k_logits.weight, a.mu.weight, a.log_std.weight, a.trunk[0].weight):
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_actor_loss_is_exact_sum_over_k():
    """With alpha_w = alpha_k = 0 the loss is -E_s sum_k pi(k) Q^W(s, k, w_k)."""
    model, _ = make_adaptive()
    obs = th.randn(4, OBS + D)
    th.manual_seed(3)
    loss, _ = model.actor_loss(obs, th.tensor(0.0), th.tensor(0.0))
    th.manual_seed(3)
    out = model.hybrid_actor.all_heads(obs)
    manual = 0.0
    for i in range(K):
        q = model._combine_q(model._q_noise(obs, th.full((4,), i), out["w"][:, i])).squeeze(-1)
        manual = manual - out["probs"][:, i] * q
    th.testing.assert_close(loss, manual.mean())


def test_freeze_k_uniform():
    model, steered = make_adaptive(freeze_k_uniform=True)
    a = model.hybrid_actor
    obs = th.randn(500, OBS + D)
    out = a.all_heads(obs)
    th.testing.assert_close(out["probs"], th.full((500, K), 1.0 / K))
    assert model.k_ent_coef_optimizer is None and model.log_k_ent_coef is None
    # k is uniform even in deterministic mode (argmax of a uniform is meaningless)
    ks = a.sample(obs, deterministic=True)["k"]
    assert set(ks.tolist()) == set(range(K))
    loss, _ = model.actor_loss(obs[:8], th.tensor(0.1), th.tensor(5.0))
    a.zero_grad()
    loss.backward()
    assert a.k_logits.weight.grad is None or a.k_logits.weight.grad.abs().sum() == 0
    assert a.mu.weight.grad.abs().sum() > 0


def test_k_subset():
    subset = [1, 3]
    model, steered = make_adaptive(k_subset=subset, rho=0.5)
    a = model.hybrid_actor
    obs = th.randn(300, OBS + D)
    out = a.all_heads(obs)
    assert out["probs"].shape == (300, 2) and out["w"].shape == (300, 2, D)
    assert set(a.sample(obs)["k"].tolist()) <= set(subset)
    assert np.isclose(model.target_entropy_k, 0.5 * np.log(2))
    # distillation samples k uniformly within the subset
    model.learn(total_timesteps=40)
    steered.k_seen.clear()
    data = model.replay_buffer.sample(256)
    model.update_noise_critic(data)
    ks = th.cat(steered.k_seen)
    assert set(ks.tolist()) == set(subset)
    assert set(np.nonzero(model.pop_k_counts())[0].tolist()) <= set(subset)


def test_fast_replay_buffer_after_wrap():
    obs_space = spaces.Box(-1, 1, (3,), np.float32)
    act_space = spaces.Box(-1, 1, (2,), np.float32)
    rb = FastReplayBuffer(10, obs_space, act_space, device="cpu", n_envs=2)
    for _ in range(12):
        rb.add(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 2)), np.zeros(2), np.zeros(2), [{}, {}])
    assert rb.full
    assert rb.sample(8).observations.shape == (8, 3)


# --------------------------------------------------------------------------------------
# End-to-end smoke tests
# --------------------------------------------------------------------------------------
def test_fixed_k_smoke_and_k_routing(tmp_path):
    steered = StubSteered()
    model = DSRLFixedK("MlpPolicy", make_env(), k=2, **agent_kwargs(steered))
    model.learn(total_timesteps=300)
    assert model.num_timesteps >= 300
    assert model._n_updates > 0
    ks = th.cat(steered.k_seen)
    assert len(ks) > 0 and (ks == 2).all()  # every diffusion call (rollout, backup, distill) used k=2
    assert np.isfinite(model.last_train_stats["train/critic_loss"])
    obs = model.env.reset()
    act, _ = model.predict_diffused(obs, deterministic=True)
    assert act.shape == (2, D) and model.last_w.shape == (2, D)

    path = str(tmp_path / "ckpt.pt")
    model.save_checkpoint(path, extra={"x": 1})
    model2 = DSRLFixedK("MlpPolicy", make_env(), k=2, **agent_kwargs(StubSteered()))
    assert model2.load_checkpoint(path) == {"x": 1}
    for p1, p2 in zip(model.policy.parameters(), model2.policy.parameters()):
        assert th.equal(p1, p2)
    assert model2.num_timesteps == model.num_timesteps
    assert model2.replay_buffer.pos == model.replay_buffer.pos


def test_adaptive_k_smoke(tmp_path):
    model, steered = make_adaptive()
    model.learn(total_timesteps=300)
    assert model.num_timesteps >= 300 and model._n_updates > 0
    stats = model.last_train_stats
    for key in ["train/critic_loss", "train/actor_loss", "train/noise_critic_loss", "train/k_entropy",
                "train/ent_coef", "train/k_ent_coef"]:  # fmt: skip
        assert key in stats and np.isfinite(stats[key]), key
    counts = model.pop_k_counts()
    assert counts.shape == (K,) and counts.sum() == model.num_timesteps
    assert model.pop_k_counts().sum() == 0
    assert np.isclose(model.last_mean_k_probs.sum(), 1.0)

    obs = model.env.reset()
    act, _ = model.predict_diffused(obs, deterministic=True)
    assert act.shape == (2, D)
    assert model.last_k.shape == (2,) and model.last_w.shape == (2, D) and model.last_probs.shape == (2, K)
    probs = model.hybrid_actor.all_heads(th.as_tensor(obs))["probs"]
    assert np.array_equal(model.last_k, probs.argmax(-1).numpy())

    # temperatures moved from their init
    assert not np.isclose(model.log_k_ent_coef.item(), 0.0)

    path = str(tmp_path / "ckpt.pt")
    model.save_checkpoint(path)
    model2, _ = make_adaptive()
    model2.load_checkpoint(path)
    for p1, p2 in zip(model.hybrid_actor.parameters(), model2.hybrid_actor.parameters()):
        assert th.equal(p1, p2)
    for p1, p2 in zip(model.critic_noise.parameters(), model2.critic_noise.parameters()):
        assert th.equal(p1, p2)
    assert th.equal(model.log_k_ent_coef, model2.log_k_ent_coef)
    # optimizer still bound to the (in-place loaded) temperature tensor
    assert model2.k_ent_coef_optimizer.param_groups[0]["params"][0] is model2.log_k_ent_coef
    model2.learn(total_timesteps=20, reset_num_timesteps=False)
    assert model2.num_timesteps >= model.num_timesteps + 20


def test_adaptive_never_calls_upstream_sampler():
    model, _ = make_adaptive()
    with pytest.raises(RuntimeError):
        model.diffusion_policy(None, None)


# --------------------------------------------------------------------------------------
# train_adsrl.py eval logging (dumps + CSV) with the dummy env
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_eval_callback_dumps(tmp_path, mode):
    pytest.importorskip("hydra")
    from omegaconf import OmegaConf

    import train_adsrl

    cfg = OmegaConf.create(
        dict(
            act_steps=H, obs_dim=OBS, eval_interval=5, num_evals=4, deterministic_eval=True,
            checkpoint_every_evals=1, eval_dump_every=2, logdir=str(tmp_path),
            env=dict(n_envs=2, n_eval_envs=2, reward_offset=1, max_episode_steps=10 * H),
            steer=dict(mode=mode, k=1),
        )
    )  # fmt: skip
    steered = StubSteered()
    if mode == "fixed":
        model = DSRLFixedK("MlpPolicy", make_env(), k=1, **agent_kwargs(steered))
    else:
        model = DSRLAdaptiveK("MlpPolicy", make_env(), num_k=K, **agent_kwargs(steered))
    cb = train_adsrl.AdsrlCallback(cfg, make_env(), steered, -np.ones(OBS), np.ones(OBS), None)
    cb.init_callback(model)
    cb.run_evals()  # eval 0: dumped
    model.learn(total_timesteps=24, callback=cb)  # evals at n_calls 5, 10 -> eval 2 dumped
    cb.run_evals(final=True)

    rows = list(open(tmp_path / "eval_summary.csv"))
    assert rows[0].strip().split(",") == cb.CSV_FIELDS
    assert len(rows) == 1 + 2 * 4  # 4 evals x (stoch, det)
    dumps = sorted(os.listdir(tmp_path / "eval_rollouts"))
    assert len(dumps) == 2 * 3, dumps  # evals 0, 2 and final
    z = np.load(tmp_path / "eval_rollouts" / dumps[0])
    n = 2 * 2 * 10  # rounds x envs x chunks
    assert z["obs_raw"].shape == (n, OBS) and z["action"].shape == (n, D) and z["base_action"].shape == (n, D)
    assert z["w"].shape == (n, D) and z["k"].shape == (n,) and z["episode"].max() == 3
    assert z["episode_success"].shape == (n,) and z["reward"].shape == (n,)
    if mode == "fixed":
        assert (z["k"] == 1).all() and "k_probs" not in z
    else:
        assert z["k_probs"].shape == (n, K)
    assert os.path.exists(tmp_path / "checkpoint" / "latest.pt")


def test_final_eval_not_duplicated_at_eval_boundary(tmp_path):
    """Regression: when training ends exactly on a periodic eval (or a finished run is resumed), the
    final eval used to run again at the same env_steps, duplicating CSV rows and overwriting the dump."""
    pytest.importorskip("hydra")
    from omegaconf import OmegaConf

    import train_adsrl

    cfg = OmegaConf.create(
        dict(
            act_steps=H, obs_dim=OBS, eval_interval=5, num_evals=2, deterministic_eval=True,
            checkpoint_every_evals=1, eval_dump_every=1, logdir=str(tmp_path),
            env=dict(n_envs=2, n_eval_envs=2, reward_offset=1, max_episode_steps=10 * H),
            steer=dict(mode="fixed", k=1),
        )
    )  # fmt: skip
    steered = StubSteered()
    model = DSRLFixedK("MlpPolicy", make_env(), k=1, **agent_kwargs(steered))
    cb = train_adsrl.AdsrlCallback(cfg, make_env(), steered, -np.ones(OBS), np.ones(OBS), None)
    cb.init_callback(model)
    cb.run_evals()
    model.learn(total_timesteps=20, callback=cb)  # 10 vec steps -> periodic evals at n_calls 5 and 10 (the end)
    cb.run_evals(final=True)  # same env_steps as the eval at n_calls 10: must be skipped
    # resuming a finished run: a fresh callback from the saved state must not re-evaluate either
    cb2 = train_adsrl.AdsrlCallback(cfg, make_env(), steered, -np.ones(OBS), np.ones(OBS), None, state=cb.state())
    cb2.init_callback(model)
    cb2.run_evals(final=True)

    rows = [r.strip().split(",") for r in open(tmp_path / "eval_summary.csv")][1:]
    keys = [(r[0], r[1]) for r in rows]
    assert len(keys) == len(set(keys)) == 2 * 3, keys  # evals at 0, 20, 40 env steps x (stoch, det)
    assert len(os.listdir(tmp_path / "eval_rollouts")) == 2 * 3


def test_final_eval_runs_when_last_eval_not_dumped(tmp_path):
    """The final eval must still run (and dump) if the eval at the same step was not dumped."""
    pytest.importorskip("hydra")
    from omegaconf import OmegaConf

    import train_adsrl

    cfg = OmegaConf.create(
        dict(
            act_steps=H, obs_dim=OBS, eval_interval=5, num_evals=2, deterministic_eval=False,
            checkpoint_every_evals=100, eval_dump_every=100, logdir=str(tmp_path),
            env=dict(n_envs=2, n_eval_envs=2, reward_offset=1, max_episode_steps=10 * H),
            steer=dict(mode="fixed", k=1),
        )
    )  # fmt: skip
    steered = StubSteered()
    model = DSRLFixedK("MlpPolicy", make_env(), k=1, **agent_kwargs(steered))
    cb = train_adsrl.AdsrlCallback(cfg, make_env(), steered, -np.ones(OBS), np.ones(OBS), None)
    cb.init_callback(model)
    cb.n_evals = 1  # so the eval at n_calls 5 is not a dump eval
    model.learn(total_timesteps=10, callback=cb)
    cb.run_evals(final=True)
    assert len(os.listdir(tmp_path / "eval_rollouts")) == 1



# --------------------------------------------------------------------------------------
# Which networks see x_T (steer.xT_inputs)
# --------------------------------------------------------------------------------------
def _xt_kwargs(steered, xt_inputs):
    kw = agent_kwargs(steered)
    kw["policy_kwargs"] = {**kw["policy_kwargs"], "xt_inputs": xt_inputs, "state_dim": OBS}
    return kw


def _depends_on_xt(fn):
    th.manual_seed(0)
    obs = th.randn(8, OBS + D)
    obs2 = obs.clone()
    obs2[:, OBS:] = th.randn(8, D)
    with th.no_grad():
        a, b = fn(obs), fn(obs2)
    a = a if isinstance(a, (tuple, list)) else [a]
    b = b if isinstance(b, (tuple, list)) else [b]
    return not all(th.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize(
    "xt_inputs,k,expect",
    [
        ("none", 0, dict(actor=False, critic=False, noise=False)),
        ("actor_noise_critic", 0, dict(actor=True, critic=False, noise=True)),
        ("actor_noise_critic", 2, dict(actor=True, critic=False, noise=True)),
        ("all", 2, dict(actor=True, critic=True, noise=True)),
    ],
)
def test_fixed_k_xt_inputs(xt_inputs, k, expect):
    from adsrl.agent import SteerSACPolicy

    model = DSRLFixedK(SteerSACPolicy, make_env(), k=k, **_xt_kwargs(StubSteered(), xt_inputs))
    act = th.rand(8, D) * 2 - 1
    assert _depends_on_xt(lambda o: model.actor(o, deterministic=True)) == expect["actor"]
    assert _depends_on_xt(lambda o: model.critic(o, act)) == expect["critic"]
    assert _depends_on_xt(lambda o: model.critic_target(o, act)) == expect["critic"]
    assert _depends_on_xt(lambda o: model.critic_noise(o, act)) == expect["noise"]
    model.learn(total_timesteps=8)  # trains end to end


def test_fixed_k_xt_inputs_none_rejected_at_k_gt_0():
    from adsrl.agent import SteerSACPolicy

    with pytest.raises(AssertionError):
        DSRLFixedK(SteerSACPolicy, make_env(), k=2, **_xt_kwargs(StubSteered(), "none"))


def test_adaptive_xt_inputs_actor_noise_critic(tmp_path):
    from adsrl.agent import SteerSACPolicy

    model = DSRLAdaptiveK(SteerSACPolicy, make_env(), num_k=K, **_xt_kwargs(StubSteered(), "actor_noise_critic"))
    act = th.rand(8, D) * 2 - 1
    kw = th.cat([th.nn.functional.one_hot(th.randint(0, K, (8,)), K).float(), act], -1)
    assert not _depends_on_xt(lambda o: model.critic(o, act))
    assert not _depends_on_xt(lambda o: model.critic_target(o, act))
    assert _depends_on_xt(lambda o: model.critic_noise(o, kw))
    assert _depends_on_xt(lambda o: model.actor(o)[0])
    model.learn(total_timesteps=8)
    # checkpoint round trip keeps the per-network input layout
    path = str(tmp_path / "ck.pt")
    model.save_checkpoint(path)
    model2 = DSRLAdaptiveK(SteerSACPolicy, make_env(), num_k=K, **_xt_kwargs(StubSteered(), "actor_noise_critic"))
    model2.load_checkpoint(path)
    obs = th.randn(4, OBS + D)
    with th.no_grad():
        assert th.equal(model.critic(obs, act[:4])[0], model2.critic(obs, act[:4])[0])


def test_adaptive_rejects_xt_inputs_none():
    from adsrl.agent import SteerSACPolicy

    with pytest.raises(AssertionError):
        DSRLAdaptiveK(SteerSACPolicy, make_env(), num_k=K, **_xt_kwargs(StubSteered(), "none"))


def test_resolve_xt_inputs():
    pytest.importorskip("hydra")
    from omegaconf import OmegaConf

    import train_adsrl

    def r(**kw):
        return train_adsrl.resolve_xt_inputs(OmegaConf.create(kw))

    assert r(mode="fixed", k=0, xT_inputs="auto") == "none"
    assert r(mode="fixed", k=4, xT_inputs="auto") == "actor_noise_critic"
    assert r(mode="adaptive", k=0, xT_inputs="auto") == "actor_noise_critic"
    assert r(mode="fixed", k=0, xT_inputs="all") == "all"
    with pytest.raises(AssertionError):
        r(mode="fixed", k=4, xT_inputs="none")
    with pytest.raises(AssertionError):
        r(mode="adaptive", k=0, xT_inputs="none")
