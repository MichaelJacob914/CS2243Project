"""exp9 (CPU): low-frequency noise layout (--steer.w_layout lowfreq) and the a_base plan feature
(--steer.obs_extra abase).

lowfreq: B = first m orthonormal DCT-II vectors over the 50 steps (mode 0 = time-mean direction u). For the 7 real
LIBERO channels w[:, c] = B z_c + (I - B B^T) x_T[:, c]; the 25 padding channels are x_T. z has m * 7 dims, laid
out mode-major: z.reshape(m, 7)[j, c] is mode j of channel c.
abase: the agent observes, instead of x_T, pi0's plan for this chunk's x_T (real channels, all 50 steps, normalized
model space): k=0 the plain sample from x_T, k>0 the clean estimate a_hat_{k-1} on the plain trajectory.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import train_adsrl_pi0 as T  # noqa: E402  (light module; also runs adsrl_pi0.setup_env())

import jax  # noqa: E402

from adsrl_pi0.buffer import ChunkReplayBuffer  # noqa: E402
from adsrl_pi0.rollout import N_REAL, dct_basis, decode_noise, lowfreq_noise, run_episodes, structured_noise  # noqa: E402
from adsrl_pi0.sampler import SteeredPi0  # noqa: E402

H, D = 50, 32


def test_n_real_is_libero_action_dim():
    assert N_REAL == 7


# ---- DCT basis and the lowfreq decode ----
@pytest.mark.parametrize("m", [1, 3, 8])
def test_dct_basis_orthonormal_mode0_is_mean(m):
    Bm = dct_basis(H, m)
    assert Bm.shape == (H, m)
    np.testing.assert_allclose(Bm.T @ Bm, np.eye(m), atol=1e-6)
    np.testing.assert_allclose(Bm[:, 0], np.full(H, 1 / np.sqrt(H)), atol=1e-7)
    if m > 1:  # mode 1 is the slowest cosine: monotone over time
        assert np.all(np.diff(Bm[:, 1]) < 0)


def _z_of(x_T, m):
    """z that reproduces x_T: B^T x_T on the real channels, mode-major flattened."""
    return np.einsum("tj,...tc->...jc", dct_basis(H, m), x_T[..., :N_REAL]).reshape(*x_T.shape[:-2], m * N_REAL)


def test_lowfreq_shapes_and_padding_channels_untouched():
    rng = np.random.default_rng(0)
    x_T, z = rng.standard_normal((5, H, D)), rng.standard_normal((5, 3 * N_REAL))
    w = lowfreq_noise(x_T, z, 3)
    assert w.shape == (5, H, D)
    np.testing.assert_allclose(w[..., N_REAL:], x_T[..., N_REAL:], atol=1e-6)
    assert lowfreq_noise(x_T[0], z[0], 3).shape == (H, D)


@pytest.mark.parametrize("m", [1, 3])
def test_lowfreq_projection_identities(m):
    rng = np.random.default_rng(1)
    x_T, z = rng.standard_normal((4, H, D)), rng.standard_normal((4, m * N_REAL))
    w = lowfreq_noise(x_T, z, m)
    Bm = dct_basis(H, m)
    # the low-frequency coefficients of w are z, the rest of x_T is kept
    np.testing.assert_allclose(_z_of(w, m), z, atol=1e-5)
    resid = lambda x: x[..., :N_REAL] - np.einsum("tj,...jc->...tc", Bm, _z_of(x, m).reshape(-1, m, N_REAL))  # noqa: E731
    np.testing.assert_allclose(resid(w), resid(x_T), atol=1e-5)


@pytest.mark.parametrize("m", [1, 3])
def test_lowfreq_z_equal_to_projection_reproduces_x_T(m):
    x_T = np.random.default_rng(2).standard_normal((3, H, D)).astype(np.float32)
    np.testing.assert_allclose(lowfreq_noise(x_T, _z_of(x_T, m), m), x_T, atol=1e-5)


def test_lowfreq_m1_matches_structured_on_real_channels():
    rng = np.random.default_rng(3)
    x_T, z = rng.standard_normal((4, H, D)), rng.standard_normal((4, D))
    w_lf = lowfreq_noise(x_T, z[:, :N_REAL], 1)
    w_st = structured_noise(x_T, z)
    np.testing.assert_allclose(w_lf[..., :N_REAL], w_st[..., :N_REAL], atol=1e-5)


def test_lowfreq_batched_matches_per_sample():
    rng = np.random.default_rng(4)
    x_T, z = rng.standard_normal((6, H, D)), rng.standard_normal((6, 3 * N_REAL))
    wb = lowfreq_noise(x_T, z, 3)
    for i in range(6):
        np.testing.assert_allclose(wb[i], lowfreq_noise(x_T[i], z[i], 3), atol=1e-6)


def test_lowfreq_gaussian_z_gives_iid_standard_normal_noise():
    rng = np.random.default_rng(5)
    n = 4000
    w = lowfreq_noise(rng.standard_normal((n, H, D)), rng.standard_normal((n, 3 * N_REAL)), 3)
    assert abs(w.mean()) < 0.01 and abs(w.std() - 1) < 0.01
    np.testing.assert_allclose(w.mean(0), 0, atol=0.08)
    np.testing.assert_allclose(w.std(0), 1, atol=0.08)
    np.testing.assert_allclose(np.cov(w[:, :, 2].T), np.eye(H), atol=0.1)  # real channel, over time
    np.testing.assert_allclose(np.cov(w[:, 7, :].T), np.eye(D), atol=0.1)  # across channels


def test_decode_noise_dispatch():
    rng = np.random.default_rng(6)
    x_T = rng.standard_normal((2, H, D)).astype(np.float32)
    np.testing.assert_array_equal(decode_noise("tiled", x_T, rng.standard_normal((2, D)), 3).shape, (2, D))
    z = rng.standard_normal((2, D))
    np.testing.assert_allclose(decode_noise("structured", x_T, z, 3), structured_noise(x_T, z))
    z = rng.standard_normal((2, 3 * N_REAL))
    np.testing.assert_allclose(decode_noise("lowfreq", x_T, z, 3), lowfreq_noise(x_T, z, 3))


# ---- training script wiring ----
BASE = ["--task_id", "8", "--steer.xt_layout", "full"]


def test_parse_names_and_dims():
    a = T.parse(BASE)
    assert a.obs_extra == "xt" and a.w_layout == "tiled" and T.action_dim(a) == D
    a = T.parse(BASE + ["--steer.k", "3", "--steer.w_layout", "lowfreq", "--steer.n_modes", "3",
                        "--steer.obs_extra", "abase", "--action_magnitude", "2.5"])
    assert T.action_dim(a) == 21
    assert a.name == "pi0_t8_k3_xtfull_lf3_abase_b2.5_s0"
    a = T.parse(BASE + ["--steer.w_layout", "structured", "--steer.obs_extra", "abase"])
    assert a.name == "pi0_t8_k0_xtfull_wstruct_abase_s0" and T.action_dim(a) == D


def test_parse_errors():
    with pytest.raises(SystemExit):  # lowfreq builds w from x_T: needs the full layout
        T.parse(["--task_id", "8", "--steer.w_layout", "lowfreq"])
    with pytest.raises(SystemExit):  # abase replaces x_T in the obs
        T.parse(BASE + ["--steer.obs_extra", "abase", "--steer.force_xt_obs", "1"])


@pytest.mark.parametrize("k", [0, 3])
def test_obs_spec_abase(k):
    a = T.parse(BASE + ["--steer.k", str(k), "--steer.obs_extra", "abase"])
    shapes, dtypes = T.obs_spec(a)
    assert set(shapes) == {"pixels", "state", "a_base"}
    assert shapes["a_base"] == (H * N_REAL,) and dtypes["a_base"] == np.float32
    assert not T.uses_xt_obs(a)


def test_obs_spec_xt_unchanged():
    assert set(T.obs_spec(T.parse(BASE))[0]) == {"pixels", "state"}
    assert T.obs_spec(T.parse(BASE + ["--steer.k", "3"]))[0]["x_T"] == (H * D,)


def test_lowfreq_agent_action_dim_and_target_entropy():
    a = T.parse(BASE + ["--steer.k", "3", "--steer.w_layout", "lowfreq", "--steer.n_modes", "3",
                        "--steer.obs_extra", "abase", "--action_magnitude", "2.5",
                        "--hidden_dims", "32"])
    ag = T.build_agent(a)
    assert ag.learner.action_dim == 21 and ag.learner.target_entropy == -10.5
    shapes, dtypes = T.obs_spec(a)
    obs = {k: np.zeros((4, *s), dtypes[k]) for k, s in shapes.items()}
    k, z, _ = ag.act(jax.device_put(obs))
    assert z.shape == (4, 21) and (k == 3).all() and np.abs(z).max() <= 2.5


# ---- rollout with the a_base feature ----
class FakeVenv:
    def __init__(self, n=3, ep_chunks=2):
        self.n, self.max_steps, self.n_init_states = n, 20 * ep_chunks, 10

    @staticmethod
    def _obs():
        return {"pi0": {}, "pixels": np.zeros((64, 64, 3), np.uint8), "state": np.zeros(8, np.float32)}

    def reset(self, ids, idx):
        return [self._obs() for _ in ids]

    def step_chunk(self, ids, actions):
        return [(self._obs(), len(a), False) for a in actions]


def fake_feature(x_T, k):
    """Deterministic stand-in for pi0's plan: a function of (x_T, k) only."""
    return (np.tanh(np.asarray(x_T, np.float32)[..., :N_REAL]) + k).reshape(len(x_T), -1)


class RecordingPi0:
    horizon, action_dim = H, D
    tile = SteeredPi0.tile

    def __init__(self):
        self.calls, self.feature_calls = [], []

    def infer(self, obs_list, x_T, k, w):
        self.calls.append((self.tile(x_T), np.asarray(k), self.tile(w)))
        return np.zeros((len(obs_list), H, 7), np.float32)

    def plan_feature(self, obs_list, x_T, k):
        x_T = self.tile(x_T)
        assert len(obs_list) == len(x_T)
        self.feature_calls.append(k)
        return fake_feature(x_T, k)


def _act_fn(record, dim):
    rng = np.random.default_rng(9)

    def f(batch):
        B = batch["state"].shape[0]
        z = rng.uniform(-1, 1, (B, dim)).astype(np.float32)
        record.append({key: v.copy() for key, v in batch.items()} | {"z": z})
        return np.full(B, 3, np.int32), z, None
    return f


@pytest.mark.parametrize("compare_base", [False, True])
def test_rollout_abase_lowfreq(compare_base):
    pi0, rec = RecordingPi0(), []
    eps, _ = run_episodes(FakeVenv(), pi0, [0, 1, 2, 3], _act_fn(rec, 21), 20, np.random.default_rng(0),
                          compare_base=compare_base, xt_full=True, w_layout="lowfreq", n_modes=3,
                          obs_extra="abase", feature_k=3)
    assert set(pi0.feature_calls) == {3}
    for (x_in, k, w_in), b in zip(pi0.calls, rec, strict=True):
        n = len(b["z"])
        x_T = b["x_T"].reshape(n, H, D)
        # the agent saw the plan for exactly this chunk's x_T; pi0 got the lowfreq noise built from that x_T
        np.testing.assert_allclose(b["a_base"], fake_feature(x_T, 3), atol=1e-6)
        np.testing.assert_allclose(x_in[:n], x_T)
        np.testing.assert_allclose(w_in[:n], lowfreq_noise(x_T, b["z"], 3), atol=1e-6)
    for e in eps:
        assert len(e.obs) == len(e.w) + 1
        for o in e.obs:  # every stored obs, including the final next_obs, carries its own x_T's plan
            assert o["a_base"].shape == (H * N_REAL,)
            np.testing.assert_allclose(o["a_base"], fake_feature(o["x_T"].reshape(1, H, D), 3)[0], atol=1e-6)


def test_rollout_xt_has_no_feature_call():
    pi0, rec = RecordingPi0(), []
    eps, _ = run_episodes(FakeVenv(), pi0, [0, 1], _act_fn(rec, D), 20, np.random.default_rng(0), xt_full=True)
    assert pi0.feature_calls == [] and all("a_base" not in o for e in eps for o in e.obs)


def test_add_episodes_abase_stores_feature_not_xt():
    a = T.parse(BASE + ["--steer.k", "3", "--steer.w_layout", "lowfreq", "--steer.obs_extra", "abase"])
    shapes, dtypes = T.obs_spec(a)
    buf = ChunkReplayBuffer(shapes, dtypes, T.action_dim(a), 100)
    eps, _ = run_episodes(FakeVenv(), RecordingPi0(), [0, 1], _act_fn([], 21), 20, np.random.default_rng(0),
                          **T.rollout_kw(a))
    n = T.add_episodes(buf, eps, 0.999**20, list(shapes))
    assert n == len(buf) > 0
    assert "x_T" not in buf.data["observations"]
    i = 0
    for e in eps:
        for t in range(len(e.w)):
            np.testing.assert_array_equal(buf.data["observations"]["a_base"][i], e.obs[t]["a_base"])
            np.testing.assert_array_equal(buf.data["next_observations"]["a_base"][i], e.obs[t + 1]["a_base"])
            np.testing.assert_array_equal(buf.data["actions"][i, 0], e.w[t])
            i += 1


def test_rollout_kw():
    a = T.parse(BASE + ["--steer.k", "3", "--steer.w_layout", "lowfreq", "--steer.n_modes", "2",
                        "--steer.obs_extra", "abase"])
    assert T.rollout_kw(a) == dict(xt_full=True, w_layout="lowfreq", n_modes=2, obs_extra="abase", feature_k=3)


@pytest.mark.parametrize("deterministic", [False, True])
def test_run_eval_abase_lowfreq(deterministic):
    a = T.parse(BASE + ["--steer.k", "3", "--steer.w_layout", "lowfreq", "--steer.obs_extra", "abase",
                        "--hidden_dims", "32"])
    ag = T.build_agent(a)
    keys = list(T.obs_spec(a)[0])
    rec = []

    def act_fn(batch):
        k, z, p = ag.act(jax.device_put(T.agent_input(batch, keys)), deterministic=deterministic)
        rec.append((batch["x_T"].copy(), batch["a_base"].copy(), z))
        return k, z, p

    pi0 = RecordingPi0()
    T.run_eval(FakeVenv(), pi0, ag, act_fn, deterministic, seed=0, eval_index=0, eval_episodes=4, query_freq=20,
               rollout_kw=T.rollout_kw(a), train_rng=np.random.default_rng(0))
    assert len(rec) == len(pi0.calls) > 0
    for (_, _, w_in), (xt, ab, z) in zip(pi0.calls, rec, strict=True):
        n = len(z)
        np.testing.assert_allclose(ab, fake_feature(xt.reshape(n, H, D), 3), atol=1e-6)
        np.testing.assert_allclose(w_in[:n], lowfreq_noise(xt.reshape(n, H, D), z, 3), atol=1e-5)


# ---- adaptive k with abase + lowfreq ----
AD = BASE + ["--steer.mode", "adaptive", "--steer.rho", "0.5", "--steer.w_layout", "lowfreq", "--steer.n_modes", "3",
             "--steer.obs_extra", "abase", "--action_magnitude", "2.5"]


def test_adaptive_abase_lowfreq_parse_and_spec():
    a = T.parse(AD)
    assert a.name == "pi0_t8_adaptive_rho0.5_xtfull_lf3_abase_b2.5_s0"
    assert T.action_dim(a) == 21
    shapes, _ = T.obs_spec(a)
    assert set(shapes) == {"pixels", "state", "a_base"} and shapes["a_base"] == (H * N_REAL,)
    # k is chosen after observing: the feature is always the k=0 one (plain sample), whatever --steer.k says
    assert T.rollout_kw(a)["feature_k"] == 0
    assert T.rollout_kw(T.parse(AD + ["--steer.k", "3"]))["feature_k"] == 0


def test_adaptive_default_unchanged():
    a = T.parse(BASE + ["--steer.mode", "adaptive", "--steer.rho", "0.5"])
    assert a.name == "pi0_t8_adaptive_rho0.5_xtfull_s0" and T.action_dim(a) == D
    shapes, _ = T.obs_spec(a)
    assert set(shapes) == {"pixels", "state", "x_T"} and shapes["x_T"] == (H * D,)
    assert T.rollout_kw(a) == dict(xt_full=True, w_layout="tiled", n_modes=3, obs_extra="xt", feature_k=0)
    ag = T.build_agent(T.parse(BASE + ["--steer.mode", "adaptive", "--hidden_dims", "32"]))
    assert ag.noise_dim == D and ag.target_entropy == -16.0 and ag.n_heads == 10


def _mixed_k_act_fn(record, dim):
    rng = np.random.default_rng(11)

    def f(batch):
        n = batch["state"].shape[0]
        k = rng.integers(0, 10, n).astype(np.int32)
        z = rng.uniform(-2.5, 2.5, (n, dim)).astype(np.float32)
        record.append({key: v.copy() for key, v in batch.items()} | {"z": z, "k": k})
        return k, z, np.full((n, 10), 0.1, np.float32)
    return f


@pytest.mark.parametrize("compare_base", [False, True])
def test_adaptive_rollout_feature_k0_and_per_row_decode(compare_base):
    a = T.parse(AD)
    pi0, rec = RecordingPi0(), []
    eps, _ = run_episodes(FakeVenv(n=4), pi0, list(range(6)), _mixed_k_act_fn(rec, 21), 20,
                          np.random.default_rng(0), compare_base=compare_base, **T.rollout_kw(a))
    assert set(pi0.feature_calls) == {0}
    for (x_in, k_in, w_in), b in zip(pi0.calls, rec, strict=True):
        n = len(b["z"])
        x_T = b["x_T"].reshape(n, H, D)
        np.testing.assert_allclose(b["a_base"], fake_feature(x_T, 0), atol=1e-6)  # k-independent feature
        np.testing.assert_array_equal(k_in[:n], b["k"])  # each row injected at its own k
        np.testing.assert_allclose(x_in[:n], x_T)
        np.testing.assert_allclose(w_in[:n], lowfreq_noise(x_T, b["z"], 3), atol=1e-6)  # row's z with row's x_T
    assert sum(len(e.k_probs) for e in eps) == sum(len(e.k) for e in eps) > 0


def test_adaptive_lowfreq_abase_agent_dims_and_update():
    from flax.core import frozen_dict

    a = T.parse(AD + ["--hidden_dims", "32"])
    ag = T.build_agent(a)
    assert ag.noise_dim == 21 and ag.target_entropy == -10.5 and ag.n_heads == 10
    shapes, dtypes = T.obs_spec(a)
    rng = np.random.default_rng(0)
    obs = lambda n: {k: (rng.integers(0, 256, (n, *s)).astype(dtypes[k]) if dtypes[k] == np.uint8  # noqa: E731
                         else rng.standard_normal((n, *s)).astype(dtypes[k])) for k, s in shapes.items()}
    k, w, probs = ag.act(jax.device_put(obs(5)))
    assert w.shape == (5, 21) and np.abs(w).max() <= 2.5 and probs.shape == (5, 10) and set(k) <= set(range(10))
    # actor: per-k heads of dim 21; critic: Q(s, a_base, onehot(k), w) with w 21-d
    heads = [v for p, v in jax.tree_util.tree_leaves_with_path(ag._actor.params) if "kernel" in str(p)]
    assert any(h.shape[-1] == 10 * 21 for h in heads)
    buf = ChunkReplayBuffer(shapes, dtypes, T.action_dim(a), 64)
    for _ in range(16):
        o, no = obs(1), obs(1)
        buf.insert({kk: v[0] for kk, v in o.items()}, {kk: v[0] for kk, v in no.items()},
                   rng.uniform(-2.5, 2.5, 21), int(rng.integers(10)), -1.0, 1.0, 0.98)
    batch = buf.sample(8, with_k=True)
    assert batch["actions"].shape == (8, 1, 21)
    info = ag.update(frozen_dict.freeze(batch))
    assert np.isfinite(float(info["critic_loss"])) and np.isfinite(float(info["actor_loss"]))


# ---- LIBERO worker crash evidence ----
def _die(conn):
    os._exit(3)


def _echo(conn):
    conn.send(conn.recv())


def test_vecenv_recv_reports_dead_worker(capsys):
    import multiprocessing as mp

    from adsrl_pi0.env import LiberoVecEnv

    ctx = mp.get_context("spawn")
    venv = LiberoVecEnv.__new__(LiberoVecEnv)  # no LIBERO: just the pipes/processes _recv uses
    venv.remotes, venv.procs = [], []
    for target in (_echo, _die):
        parent, child = ctx.Pipe()
        p = ctx.Process(target=target, args=(child,), daemon=True)
        p.start()
        child.close()
        venv.remotes.append(parent)
        venv.procs.append(p)
    venv.remotes[0].send("hi")
    assert venv._recv(0) == "hi"  # a live worker: unchanged
    with pytest.raises(EOFError):
        venv._recv(1)
    out = capsys.readouterr().out
    assert "worker 1" in out and "exitcode=3" in out


# ---- --hybrid.k_entropy_bias {keep, detach} ----
def _hyb_setup():
    from adsrl_pi0.agent import HybridPixelSACLearner

    obs1 = {"pixels": np.zeros((1, 64, 64, 3, 1), np.uint8), "state": np.zeros((1, 8, 1), np.float32)}
    KW = dict(encoder_type="small", hidden_dims=(32, 32), latent_dim=16, cnn_features=(8, 8), cnn_strides=(2, 2),
              num_qs=3)
    ag = HybridPixelSACLearner(0, obs1, 21, 10, list(range(10)), **KW)
    rng = np.random.default_rng(1)
    obs = {"pixels": rng.integers(0, 256, (8, 64, 64, 3, 1)).astype(np.uint8),
           "state": rng.standard_normal((8, 8, 1)).astype(np.float32)}
    return ag, obs


def _hyb_grads(ag, obs, alpha_w, detach=None):
    from adsrl_pi0.agent import hybrid_actor_loss

    kw = {} if detach is None else {"detach_k_w_entropy": detach}

    def f(p):
        return hybrid_actor_loss(p, ag._actor.apply_fn, ag._critic.params, ag._critic.apply_fn, obs,
                                 jax.random.key(0), alpha_w, 0.05, ag.allowed_k, 10, 2.5, False, "mean", **kw)

    (loss, _), g = jax.value_and_grad(f, has_aux=True)(ag._actor.params)
    g = g["network"]  # HybridPolicy: Dense_0 = k logits, Dense_1 = w mean, Dense_2 = w log_std
    return float(loss), {n: np.asarray(g[n]["kernel"]) for n in ["Dense_0", "Dense_1", "Dense_2"]}


def test_k_entropy_bias_detach_removes_w_entropy_from_logits():
    ag, obs = _hyb_setup()
    _, g_det = _hyb_grads(ag, obs, 0.3, detach=True)
    _, g_det0 = _hyb_grads(ag, obs, 0.0, detach=True)
    # detach: the categorical's gradient does not depend on alpha_w (no w-entropy contribution)
    np.testing.assert_allclose(g_det["Dense_0"], g_det0["Dense_0"], atol=1e-7)
    _, g_keep = _hyb_grads(ag, obs, 0.3, detach=False)
    _, g_keep0 = _hyb_grads(ag, obs, 0.0, detach=False)
    assert np.abs(g_keep["Dense_0"] - g_keep0["Dense_0"]).max() > 1e-6  # keep: it does


def test_k_entropy_bias_heads_and_default_unchanged():
    ag, obs = _hyb_setup()
    l_def, g_def = _hyb_grads(ag, obs, 0.3)
    l_keep, g_keep = _hyb_grads(ag, obs, 0.3, detach=False)
    l_det, g_det = _hyb_grads(ag, obs, 0.3, detach=True)
    assert l_def == l_keep and abs(l_det - l_keep) < 1e-6  # same forward value
    for n in g_def:
        np.testing.assert_array_equal(g_def[n], g_keep[n])  # default is keep
    for n in ["Dense_1", "Dense_2"]:  # per-k heads: same gradient (pi(k)-weighted alpha_w log pi(w_k|k) term)
        np.testing.assert_allclose(g_det[n], g_keep[n], atol=1e-7)


def test_k_entropy_bias_flag_wiring():
    a = T.parse(AD)
    assert a.k_entropy_bias == "keep"
    a = T.parse(AD + ["--hybrid.k_entropy_bias", "detach", "--hidden_dims", "32"])
    assert a.name == "pi0_t8_adaptive_rho0.5_xtfull_lf3_abase_kdetach_b2.5_s0"
    ag = T.build_agent(a)
    assert ag.k_entropy_bias == "detach"
    shapes, dtypes = T.obs_spec(a)
    obs = {k: np.zeros((6, *s), dtypes[k]) for k, s in shapes.items()}
    ls = ag.head_log_std(jax.device_put(obs))
    assert ls.shape == (10,) and np.isfinite(ls).all()


def test_head_log_std_logging_rng_isolated():
    """Per-k head log_std at eval reads buffer rows with its own RNG: the buffer's sampling RNG is untouched."""
    a = T.parse(AD + ["--hidden_dims", "32"])
    ag = T.build_agent(a)
    shapes, dtypes = T.obs_spec(a)
    buf = ChunkReplayBuffer(shapes, dtypes, 21, 50)
    o = {k: np.zeros(s, dtypes[k]) for k, s in shapes.items()}
    for _ in range(20):
        buf.insert(o, o, np.zeros(21), 0, -1.0, 1.0, 0.98)
    state = buf.rng.bit_generator.state
    ls = T.head_log_std_eval(ag, buf, seed=0, eval_index=3)
    assert buf.rng.bit_generator.state == state and ls.shape == (10,)
    assert T.head_log_std_eval(ag, ChunkReplayBuffer(shapes, dtypes, 21, 5), seed=0, eval_index=0) is None
