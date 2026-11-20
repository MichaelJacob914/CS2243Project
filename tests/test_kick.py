"""exp13 Phase 1 options (CPU): steer.param=kick, steer.actor_zero_init, train.ent_coef=auto_<init>,
train.early_eval_*.

Kick: the injected noise at step k is eps_hat_k + w_scale * Delta, where eps_hat_k is the implied noise of
the unsteered DDIM trajectory from x_T (x_T at k=0), so Delta = 0 must decode the base action exactly.
Sampler-level tests use the random DPPO models of tests/test_sampler.py; agent tests use the real
SteeredBasePolicy around the random K=8 model.
"""

import os
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import adsrl  # noqa: E402,F401  (adds dsrl/dppo to sys.path)
import adsrl.base_policy as bp  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from adsrl.agent import DSRLAdaptiveK, DSRLFixedK  # noqa: E402
from adsrl.base_policy import SteeredBasePolicy  # noqa: E402
from adsrl.best_of_n import agent_fns  # noqa: E402
from adsrl.env import XTAugmentWrapper  # noqa: E402
from adsrl.sampler import num_ddim_steps, steered_ddim_sample  # noqa: E402
from test_agent import agent_kwargs  # noqa: E402
from test_sampler import B, _inputs, _plain_ddim, _random_model, model  # noqa: E402,F401  (model = fixture)

OBS_R, H_R, A_R, K_R = 23, 4, 7, 8  # random_can model of test_sampler (T=20, K=8)
D_R = H_R * A_R


# --------------------------------------------------------------------------------------
# Sampler
# --------------------------------------------------------------------------------------
@th.no_grad()
def test_kick_zero_is_base_every_k(model):
    """Delta = 0 reproduces the unsteered DDIM action bit for bit at every k, and in a mixed-k batch."""
    cond, x_T, _ = _inputs(model)
    K = num_ddim_steps(model)
    plain, _, _ = _plain_ddim(model, cond, x_T)
    zero = th.zeros_like(x_T)
    for kk in range(K):
        out = steered_ddim_sample(model, cond, x_T, th.full((B,), kk, dtype=th.long), zero, kick=True)
        assert th.equal(out, plain), f"k={kk}"
    mixed = th.tensor([(3 * i + 1) % K for i in range(B)], dtype=th.long)
    assert th.equal(steered_ddim_sample(model, cond, x_T, mixed, zero, kick=True), plain)


@th.no_grad()
def test_kick_is_eps_hat_plus_delta(model):
    """kick(Delta) == absolute(w = eps_hat_k + Delta), with eps_hat_k defined as in
    test_sampler.test_eps_hat_recovers_unsteered (x_T at k=0, else eps_hat_{k-1})."""
    cond, x_T, delta = _inputs(model)
    delta = delta.clamp(-1, 1)
    K = num_ddim_steps(model)
    _, _, eps = _plain_ddim(model, cond, x_T)
    for kk in range(K):
        k = th.full((B,), kk, dtype=th.long)
        ref = x_T if kk == 0 else eps[kk - 1]
        a_abs = steered_ddim_sample(model, cond, x_T, k, ref + delta)
        a_kick = steered_ddim_sample(model, cond, x_T, k, delta, kick=True)
        th.testing.assert_close(a_kick, a_abs, atol=1e-6, rtol=0, msg=f"k={kk}")
        # and the kick actually steers
        assert not th.allclose(a_kick, steered_ddim_sample(model, cond, x_T, k, th.zeros_like(delta), kick=True))


@th.no_grad()
def test_kick_mixed_k_batch(model):
    cond, x_T, delta = _inputs(model)
    K = num_ddim_steps(model)
    k = th.tensor([(3 * i + 1) % K for i in range(B)], dtype=th.long)
    out = steered_ddim_sample(model, cond, x_T, k, delta, kick=True)
    for kk in k.unique().tolist():
        ref = steered_ddim_sample(model, cond, x_T, th.full((B,), kk, dtype=th.long), delta, kick=True)
        m = k == kk
        th.testing.assert_close(out[m], ref[m], atol=1e-6, rtol=0)


@th.no_grad()
def test_default_is_absolute(tmp_path):
    m = _random_model(tmp_path, 20, 8)
    cond, x_T, w = _inputs(m)
    k = th.arange(B) % 8
    assert th.equal(steered_ddim_sample(m, cond, x_T, k, w), steered_ddim_sample(m, cond, x_T, k, w, kick=False))
    pol = SteeredBasePolicy(m, OBS_R, H_R, A_R)
    assert pol.param == "absolute"
    obs_aug = th.cat([cond["state"], x_T.reshape(B, -1)], dim=-1)
    assert th.equal(pol(obs_aug, k, w), steered_ddim_sample(m, cond, x_T, k, w))
    with pytest.raises(AssertionError):
        SteeredBasePolicy(m, OBS_R, H_R, A_R, param="relative")


@th.no_grad()
@pytest.mark.parametrize("c", [1.0, 2.5])
def test_steered_policy_kick(tmp_path, c):
    """SteeredBasePolicy(param=kick): Delta = 0 is base_action at every k (and mixed k) for any w_scale;
    otherwise the sampler gets eps_hat_k + w_scale * Delta."""
    m = _random_model(tmp_path, 20, 8)
    cond, x_T, delta = _inputs(m)
    pol = SteeredBasePolicy(m, OBS_R, H_R, A_R, w_scale=c, param="kick")
    obs_aug = th.cat([cond["state"], x_T.reshape(B, -1)], dim=-1)
    base = pol.base_action(obs_aug)
    zero = th.zeros(B, D_R)
    for kk in range(pol.K):
        assert th.equal(pol(obs_aug, kk, zero), base), kk
    assert th.equal(pol(obs_aug, th.arange(B) % pol.K, zero), base)
    k = th.arange(B) % pol.K
    th.testing.assert_close(pol(obs_aug, k, delta), steered_ddim_sample(m, cond, x_T, k, c * delta, kick=True),
                            atol=0, rtol=0)  # fmt: skip


# --------------------------------------------------------------------------------------
# Agents with the real sampler
# --------------------------------------------------------------------------------------
class ChunkEnv(gym.Env):
    """obs/action sizes of the random model; reward 0 when the chunk is near 0.3, else -1; 5 chunks/episode."""

    def __init__(self):
        self.observation_space = spaces.Box(-1, 1, (OBS_R,), np.float32)
        self.action_space = spaces.Box(-1, 1, (D_R,), np.float32)
        self.t = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self.np_random.uniform(-1, 1, OBS_R).astype(np.float32), {}

    def step(self, action):
        self.t += 1
        reward = 0.0 if float(np.mean((np.asarray(action) - 0.3) ** 2)) < 0.05 else -1.0
        return self.np_random.uniform(-1, 1, OBS_R).astype(np.float32), reward, self.t >= 5, False, {}


def make_env_r(n=2):
    return XTAugmentWrapper(make_vec_env(ChunkEnv, n_envs=n, vec_env_cls=DummyVecEnv), H_R, A_R)


@pytest.fixture(scope="module")
def base_model(tmp_path_factory):
    return _random_model(tmp_path_factory.mktemp("rm"), 20, 8)


def make_real(base_model, mode, param="kick", zero_init=None, k=3, c=1.0, learn=0, seed=0, **kw):
    steered = SteeredBasePolicy(base_model, OBS_R, H_R, A_R, w_scale=c, param=param)
    th.manual_seed(seed)
    np.random.seed(seed)
    if zero_init is not None:
        kw["actor_zero_init"] = zero_init
    kwargs = agent_kwargs(steered, horizon=H_R, act_dim=A_R, **kw)
    if mode == "fixed":
        agent = DSRLFixedK("MlpPolicy", make_env_r(), k=k, **kwargs)
    else:
        agent = DSRLAdaptiveK("MlpPolicy", make_env_r(), num_k=K_R, **kwargs)
    agent.env.seed(seed)
    agent.action_space.seed(seed)
    if learn:
        agent.learn(total_timesteps=learn)
    return agent, steered


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_zero_init_gives_zero_mean(base_model, mode):
    agent, _ = make_real(base_model, mode, zero_init=True)
    obs = th.as_tensor(agent.env.reset())
    with th.no_grad():
        if mode == "fixed":
            head = agent.actor.mu
            th.testing.assert_close(agent.actor(obs, deterministic=True), th.zeros(obs.shape[0], D_R), atol=0, rtol=0)
        else:
            head = agent.hybrid_actor.mu
            _, mean, _ = agent.hybrid_actor(obs)
            assert th.equal(mean, th.zeros_like(mean))  # every Gaussian head
            assert agent.hybrid_actor.k_logits.weight.abs().sum() > 0  # k head untouched
            assert agent.hybrid_actor.log_std.weight.abs().sum() > 0
    assert th.equal(head.weight, th.zeros_like(head.weight)) and th.equal(head.bias, th.zeros_like(head.bias))
    # default: not zero
    agent2, _ = make_real(base_model, mode)
    head2 = agent2.actor.mu if mode == "fixed" else agent2.hybrid_actor.mu
    assert head2.weight.abs().sum() > 0


@pytest.mark.parametrize("mode,k_subset,k", [("fixed", None, kk) for kk in range(K_R)]
                         + [("adaptive", [kk], 0) for kk in range(K_R)] + [("adaptive", None, 0)])  # fmt: skip
def test_kick_zero_init_det_is_base(base_model, mode, k_subset, k):
    """Untrained kick + zero-init agent: det eval action == base action, for every k, fixed and adaptive."""
    extra = dict(k_subset=k_subset) if mode == "adaptive" else {}
    agent, steered = make_real(base_model, mode, zero_init=True, k=k, **extra)
    obs = agent.env.reset()
    a, _ = agent.predict_diffused(obs, deterministic=True)
    base = steered.base_action(obs).reshape(len(obs), -1).cpu().numpy()
    np.testing.assert_array_equal(a, base)
    assert np.all(agent.last_w == 0)


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_absolute_zero_init_is_not_base(base_model, mode):
    """Sanity: absolute + zero-init decodes w = 0, which is not the base action."""
    agent, steered = make_real(base_model, mode, param="absolute", zero_init=True, k=0,
                               **({"k_subset": [0]} if mode == "adaptive" else {}))  # fmt: skip
    obs = agent.env.reset()
    a, _ = agent.predict_diffused(obs, deterministic=True)
    assert not np.allclose(a, steered.base_action(obs).reshape(len(obs), -1).cpu().numpy())


@pytest.fixture
def spy(monkeypatch):
    """Records (k, w, kick) of every sampler call and runs the real sampler."""
    calls = []
    real = bp.steered_ddim_sample

    def wrapped(model, cond, x_T, k, w, return_x0=False, kick=False):
        calls.append((k.detach().cpu().clone(), w.detach().cpu().clone(), kick))
        return real(model, cond, x_T, k, w, return_x0=return_x0, kick=kick)

    monkeypatch.setattr(bp, "steered_ddim_sample", wrapped)
    return calls


@pytest.mark.parametrize("param", ["absolute", "kick"])
@pytest.mark.parametrize("mode,tr", [("fixed", None), ("fixed", dict(bound=0.3, lr=1e-3)), ("adaptive", None)])
def test_every_decode_path_uses_param(base_model, spy, param, mode, tr):
    """Acting (policy + warm-up uniform w), det/stoch eval, target backup, distillation and best-of-N all
    decode through the steered policy, so all see the configured parameterization."""
    agent, steered = make_real(base_model, mode, param=param, learn=20, trust_region=tr)
    agent._last_obs = agent.env.reset()
    spy.clear()
    agent._sample_action(0, n_envs=2)  # policy acting
    agent._sample_action(10**9, n_envs=2)  # warm-up: uniform w (kick: uniform Delta)
    assert spy[-1][1].abs().max() <= 1.0
    for det in (True, False):
        agent.predict_diffused(agent._last_obs, deterministic=det)
    agent.train(gradient_steps=1, batch_size=16)  # 1 backup + 2 distillation decodes
    sample_fn, _ = agent_fns(agent, steered)
    sample_fn(th.as_tensor(agent._last_obs).repeat_interleave(3, 0))
    assert len(spy) == 8
    assert all(kick == (param == "kick") for _, _, kick in spy)


@pytest.mark.parametrize("param", ["absolute", "kick"])
@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_distillation_noise(base_model, spy, monkeypatch, param, mode):
    """Q^W gets the actor-space w (kick: Delta), the sampler gets w_scale * it. Absolute: w ~ N(0, I);
    kick: Delta ~ Uniform[-1, 1]^D."""
    c = 2.0
    agent, _ = make_real(base_model, mode, param=param, c=c, learn=20)
    seen = []
    orig = agent.critic_noise.forward
    monkeypatch.setattr(agent.critic_noise, "forward", lambda o, a: (seen.append(a.detach().clone()), orig(o, a))[1])
    data = agent.replay_buffer.sample(512)
    spy.clear()
    agent.update_noise_critic(data)
    ((_, w_s, kick),) = spy
    w_in = seen[-1][:, -D_R:]  # adaptive prepends onehot(k)
    th.testing.assert_close(w_s.reshape(512, -1), c * agent._unscale_t(w_in), rtol=1e-6, atol=1e-6)
    assert kick == (param == "kick")
    if param == "kick":
        assert w_in.abs().max() <= 1.0 and abs(w_in.pow(2).mean().item() - 1 / 3) < 0.02  # uniform: E[u^2] = 1/3
    else:
        assert w_in.abs().max() > 2.0 and abs(w_in.pow(2).mean().item() - 1.0) < 0.05  # N(0, 1)


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_kick_checkpoint_round_trip(base_model, tmp_path, mode):
    agent, _ = make_real(base_model, mode, zero_init=True, learn=40)
    path = str(tmp_path / "ckpt.pt")
    agent.save_checkpoint(path)
    agent2, _ = make_real(base_model, mode, zero_init=True, seed=5)  # fresh agent, different init
    agent2.load_checkpoint(path)
    for name in ("actor", "critic", "critic_noise"):
        for (n1, p1), (n2, p2) in zip(getattr(agent, name).state_dict().items(), getattr(agent2, name).state_dict().items()):
            assert n1 == n2 and th.equal(p1, p2), n1
    obs = agent.env.reset()
    a1, _ = agent.predict_diffused(obs, deterministic=True)
    a2, _ = agent2.predict_diffused(obs, deterministic=True)
    np.testing.assert_array_equal(a1, a2)
    # training moved the actor off zero, so the det action is no longer exactly the base action
    assert agent.last_w is not None and np.abs(agent.last_w).max() > 0


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_defaults_match_explicit_absolute(base_model, mode):
    """No new kwargs == param=absolute, actor_zero_init=False: identical parameters after training."""
    a1, _ = make_real(base_model, mode, param="absolute", learn=30)
    a2, _ = make_real(base_model, mode, param="absolute", zero_init=False, learn=30)
    for name in ("actor", "critic", "critic_noise"):
        for p1, p2 in zip(getattr(a1, name).parameters(), getattr(a2, name).parameters()):
            assert th.equal(p1, p2)


# --------------------------------------------------------------------------------------
# Starting entropy
# --------------------------------------------------------------------------------------
def test_resolve_ent_coef():
    import train_adsrl

    assert train_adsrl.resolve_ent_coef(-1) == "auto"
    assert train_adsrl.resolve_ent_coef("auto") == "auto"
    assert train_adsrl.resolve_ent_coef("auto_0.05") == "auto_0.05"
    assert train_adsrl.resolve_ent_coef(0.1) == 0.1
    for bad in ("auto_", "auto_-1", "autox", "0.1x"):
        with pytest.raises(ValueError):
            train_adsrl.resolve_ent_coef(bad)
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(os.path.join(ROOT, "cfg", "adsrl", "transport.yaml"))
    assert train_adsrl.resolve_ent_coef(cfg.train.ent_coef) == "auto"
    cfg.merge_with_dotlist(["train.ent_coef=auto_0.05"])
    assert train_adsrl.resolve_ent_coef(cfg.train.ent_coef) == "auto_0.05"


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
@pytest.mark.parametrize("ent,want", [("auto", 1.0), ("auto_0.05", 0.05)])
def test_ent_init_reaches_model(base_model, mode, ent, want):
    agent, _ = make_real(base_model, mode, ent_coef=ent)
    assert agent.ent_coef_optimizer is not None
    assert abs(agent.log_ent_coef.exp().item() - want) < 1e-6
    if mode == "adaptive":
        assert abs(agent._alpha_w().item() - want) < 1e-6  # alpha_w of the hybrid actor
        assert abs(agent._alpha_k().item() - 1.0) < 1e-6  # steer.k_ent_coef=auto: alpha_k starts at 1.0
    agent.learn(total_timesteps=6)  # first train step logs the starting alpha
    assert abs(agent.last_train_stats["train/ent_coef"] - want) / want < 0.05


# --------------------------------------------------------------------------------------
# Early evals
# --------------------------------------------------------------------------------------
def test_early_eval_schedule_unit():
    import train_adsrl
    from omegaconf import OmegaConf

    assert train_adsrl.early_eval_schedule(OmegaConf.create({})) is None
    assert train_adsrl.early_eval_schedule(OmegaConf.create({"early_eval_every": None})) is None
    assert train_adsrl.early_eval_schedule(OmegaConf.create({"early_eval_every": 64000, "early_eval_until": 1600000})) == (64000, 1600000, 50)
    with pytest.raises(AssertionError):
        train_adsrl.early_eval_schedule(OmegaConf.create({"early_eval_every": 64000}))
    # transport: warm start ends at 640032, 32 env steps per vec step
    steps = [640032 + 32 * i for i in range(1, 40000)]
    fired = [s for s in steps if train_adsrl.early_eval_due(s, 32, 64000, 1600000)]
    assert fired == list(range(704000, 1600001, 64000))
    # step size that does not divide `every`: fires once per crossing, on the crossing step
    fired = [s for s in range(48, 400000, 48) if train_adsrl.early_eval_due(s, 48, 64000, 300000)]
    assert fired == [64032, 128016, 192000, 256032]


def _callback(tmp_path, state=None, early=True):
    import train_adsrl
    from omegaconf import OmegaConf

    train = {"early_eval_every": 64000, "early_eval_until": 1600000, "early_eval_episodes": 50} if early else {}
    cfg = OmegaConf.create(dict(
        act_steps=8, env=dict(n_envs=4, n_eval_envs=25, reward_offset=1, max_episode_steps=800),
        eval_interval=8000, num_evals=200, logdir=str(tmp_path), steer=dict(mode="fixed", k=3),
        checkpoint_every_evals=1, eval_dump_every=1, deterministic_eval=True, train=train,
    ))  # fmt: skip
    cb = train_adsrl.AdsrlCallback(cfg, None, None, None, None, None, state=state)
    cb.log_freq = 10**12

    class M:
        n_envs = 4

    cb.model = M()
    cb.locals = {"infos": [{}] * 4, "rewards": np.full(4, -1.0), "dones": np.zeros(4, bool)}
    log = []

    def evaluate(deterministic=False, dump=True, rounds=None):
        log.append(("eval", cb.env_steps, deterministic, dump, cb.eval_rounds if rounds is None else rounds))

    cb.evaluate = evaluate
    cb.save_checkpoint = lambda name="latest": log.append(("ckpt", cb.env_steps, cb.state()))
    return cb, log


def _run(cb, n):
    for _ in range(n):
        cb.n_calls += 1
        cb._on_step()


def test_early_eval_callback_and_resume(tmp_path):
    cb, log = _callback(tmp_path, state={"env_steps": 640032})
    _run(cb, 31000)  # to 1,632,032
    regular = [e for e in log if e[0] == "eval" and e[4] == 8]
    early = [e for e in log if e[0] == "eval" and e[4] == 2]
    assert [e[1] for e in regular] == [896032] * 2 + [1152032] * 2 + [1408032] * 2  # stoch + det
    assert [e[1] for e in early] == list(range(704000, 1600001, 64000))
    assert all(e[2] and not e[3] for e in early)  # det only, no dump
    assert cb.n_evals == 3  # early evals do not move the regular dump/checkpoint cadence
    ckpts = [e for e in log if e[0] == "ckpt"]
    assert [e[1] for e in ckpts] == [896032, 1152032, 1408032]

    # resume from the checkpoint at 1,152,032: same remaining schedule
    cb2, log2 = _callback(tmp_path, state=ckpts[1][2])
    _run(cb2, 31000 - 16000)
    early2 = [e[1] for e in log2 if e[0] == "eval" and e[4] == 2]
    assert early2 == [s for s in range(704000, 1600001, 64000) if s > 1152032]
    regular2 = [e[1] for e in log2 if e[0] == "eval" and e[4] == 8]
    assert regular2 == [1408032] * 2

    # off by default: only the regular schedule
    cb3, log3 = _callback(tmp_path, state={"env_steps": 640032}, early=False)
    _run(cb3, 31000)
    assert [e[1] for e in log3 if e[0] == "eval"] == [e[1] for e in regular]


def test_eval_csv_episodes_column(tmp_path):
    import csv

    cb, _ = _callback(tmp_path)
    row = dict(env_steps=1, mode="det", wall_s=0, gpu_s=0, success=0.5, reward=0, mean_ep_len_success=1,
               mean_a_minus_base=0, k_entropy=0, pi_k_entropy=0, episodes=50)  # fmt: skip
    cb.write_csv_row(row)
    cb.write_csv_row({**row, "episodes": 200})
    with open(tmp_path / "eval_summary.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["episodes"] for r in rows] == ["50", "200"]
    # a CSV from a run started before this column keeps its header and stays aligned
    old = tmp_path / "old"
    old.mkdir()
    header = cb.CSV_FIELDS[:-1]
    (old / "eval_summary.csv").write_text(",".join(header) + "\n")
    cb.logdir = str(old)
    cb.write_csv_row(row)
    with open(old / "eval_summary.csv") as f:
        lines = list(csv.reader(f))
    assert lines[0] == header and len(lines[1]) == len(header) and lines[1][4] == "0.5"


# --------------------------------------------------------------------------------------
# x_T routing in kick mode
# --------------------------------------------------------------------------------------
def test_kick_k0_keeps_xt_inputs():
    """Kick at k=0 decodes DDIM(x_T + Delta), so the actor and Q^W must see x_T: auto -> actor_noise_critic,
    and an explicit xT_inputs=none is rejected. Absolute k=0 keeps auto -> none."""
    import train_adsrl
    from omegaconf import OmegaConf

    def r(**kw):
        return train_adsrl.resolve_xt_inputs(OmegaConf.create(kw))

    assert r(mode="fixed", k=0) == "none"
    assert r(mode="fixed", k=0, param="absolute") == "none"
    assert r(mode="fixed", k=0, param="kick") == "actor_noise_critic"
    assert r(mode="fixed", k=3, param="kick") == "actor_noise_critic"
    with pytest.raises((AssertionError, ValueError), match="kick"):
        r(mode="fixed", k=0, param="kick", xT_inputs="none")


def test_fixed_agent_rejects_kick_with_xt_none(base_model):
    steered = SteeredBasePolicy(base_model, OBS_R, H_R, A_R, param="kick")
    kw = agent_kwargs(steered, horizon=H_R, act_dim=A_R)
    kw["policy_kwargs"] = {**kw["policy_kwargs"], "xt_inputs": "none", "state_dim": OBS_R}
    from adsrl.agent import SteerSACPolicy

    with pytest.raises(AssertionError, match="kick"):
        DSRLFixedK(SteerSACPolicy, make_env_r(), k=0, **kw)


# --------------------------------------------------------------------------------------
# steer.actor_log_std_init
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_actor_log_std_init(base_model, mode):
    agent, _ = make_real(base_model, mode, param="kick", zero_init=True, actor_log_std_init=-1.4)
    head = agent.actor.log_std if mode == "fixed" else agent.hybrid_actor.log_std
    assert th.equal(head.weight, th.zeros_like(head.weight))
    assert th.equal(head.bias, th.full_like(head.bias, -1.4))
    obs = th.as_tensor(agent.env.reset())
    with th.no_grad():
        if mode == "fixed":
            _, log_std, _ = agent.actor.get_action_dist_params(obs)
        else:
            _, _, log_std = agent.hybrid_actor(obs)  # every head
            assert agent.hybrid_actor.k_logits.weight.abs().sum() > 0
    th.testing.assert_close(log_std, th.full_like(log_std, -1.4), atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_actor_log_std_init_default_unchanged(base_model, mode):
    """None (default) == not passing the kwarg: identical parameters after training."""
    a1, _ = make_real(base_model, mode, learn=30)
    a2, _ = make_real(base_model, mode, learn=30, actor_log_std_init=None)
    for name in ("actor", "critic", "critic_noise"):
        for p1, p2 in zip(getattr(a1, name).parameters(), getattr(a2, name).parameters()):
            assert th.equal(p1, p2)
    head = a1.actor.log_std if mode == "fixed" else a1.hybrid_actor.log_std
    assert head.weight.abs().sum() > 0


def test_actor_log_std_init_overwritten_by_resume(base_model, tmp_path):
    a1, _ = make_real(base_model, "fixed", learn=20)
    path = str(tmp_path / "c.pt")
    a1.save_checkpoint(path)
    a2, _ = make_real(base_model, "fixed", actor_log_std_init=-1.4)
    a2.load_checkpoint(path)
    assert th.equal(a2.actor.log_std.bias, a1.actor.log_std.bias)


def test_actor_log_std_init_config():
    import train_adsrl  # noqa: F401
    from omegaconf import OmegaConf

    for task in ("can", "square", "transport"):
        assert OmegaConf.load(os.path.join(ROOT, "cfg", "adsrl", f"{task}.yaml")).steer.actor_log_std_init is None
