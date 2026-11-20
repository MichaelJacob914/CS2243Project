"""steer.w_scale_dims: a per-dimension noise scale applied on top of the scalar w_scale (CPU).

The sampler receives w_scale * w_scale_dims * w (elementwise over the (H, A) chunk). The file holds a vector
of length H*A, or of length A (per action dim, tiled over the horizon). Default None is bit-identical.
Per-k form: a (K, A) or (K, H, A) array; each sample's w is scaled by the row of its own k.
Agent paths reuse the recording fake sampler of tests/test_w_scale.py.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch as th

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import adsrl  # noqa: E402,F401
from adsrl.base_policy import SteeredBasePolicy, load_w_scale_dims  # noqa: E402
from adsrl.best_of_n import agent_fns  # noqa: E402
from adsrl.sampler import steered_ddim_sample  # noqa: E402
from test_agent import A, H, K, OBS  # noqa: E402
from test_w_scale import FakeBase, make_agent, rec  # noqa: E402,F401  (rec = fixture)

C = 2.5


def _vec(n, seed=0):
    return np.random.default_rng(seed).uniform(0.5, 2.0, n).astype(np.float32)


# --------------------------------------------------------------------------------------
# loader
# --------------------------------------------------------------------------------------
def test_load_w_scale_dims(tmp_path):
    v_a = _vec(A)
    np.save(tmp_path / "a.npy", v_a)
    out = load_w_scale_dims(str(tmp_path / "a.npy"), H, A)
    assert out.shape == (H, A) and np.array_equal(out, np.tile(v_a, (H, 1)))  # per action dim, tiled
    v_d = _vec(H * A, 1)
    (tmp_path / "d.json").write_text(json.dumps(v_d.tolist()))
    out = load_w_scale_dims(str(tmp_path / "d.json"), H, A)
    np.testing.assert_allclose(out, v_d.reshape(H, A), rtol=1e-7)
    (tmp_path / "obj.json").write_text(json.dumps({"w_scale_dims": v_a.tolist(), "k": 3}))
    np.testing.assert_allclose(load_w_scale_dims(str(tmp_path / "obj.json"), H, A), np.tile(v_a, (H, 1)), rtol=1e-7)
    np.save(tmp_path / "bad.npy", _vec(A + 1))
    with pytest.raises(ValueError):
        load_w_scale_dims(str(tmp_path / "bad.npy"), H, A)
    np.save(tmp_path / "neg.npy", -v_a)
    with pytest.raises(ValueError):
        load_w_scale_dims(str(tmp_path / "neg.npy"), H, A)
    assert load_w_scale_dims(None, H, A) is None


# --------------------------------------------------------------------------------------
# SteeredBasePolicy (real sampler)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("param", ["absolute", "kick"])
def test_policy_scales_per_dim(tmp_path, param):
    from test_sampler import B, _inputs, _random_model

    m = _random_model(tmp_path, 20, 8)
    cond, x_T, w = _inputs(m)
    Hm, Am = m.horizon_steps, m.action_dim
    obs_aug = th.cat([cond["state"], x_T.reshape(B, -1)], dim=-1)
    k = th.arange(B) % 8
    dims = np.tile(_vec(Am), (Hm, 1))
    pol = SteeredBasePolicy(m, m.obs_dim, Hm, Am, w_scale=C, param=param, w_scale_dims=dims)
    # same float op order as the policy: (w * w_scale) * w_scale_dims
    want = steered_ddim_sample(m, cond, x_T, k, (w * C) * th.as_tensor(dims), kick=param == "kick")
    assert th.equal(pol(obs_aug, k, w), want)
    # default None == no per-dim scale, bit for bit; base_action unaffected
    p0 = SteeredBasePolicy(m, m.obs_dim, Hm, Am, param=param)
    p1 = SteeredBasePolicy(m, m.obs_dim, Hm, Am, param=param, w_scale_dims=None)
    assert p0.w_scale_dims is None
    assert th.equal(p0(obs_aug, k, w), p1(obs_aug, k, w))
    assert th.equal(p0(obs_aug, k, w), steered_ddim_sample(m, cond, x_T, k, w, kick=param == "kick"))
    assert th.equal(pol.base_action(obs_aug), p0.base_action(obs_aug))


# --------------------------------------------------------------------------------------
# Every agent path (recording fake sampler)
# --------------------------------------------------------------------------------------
def _both(model, steered, rec, fn, path, dims, *more):
    """Same agent state + RNG; run fn without and with w_scale_dims (then with each of `more`, from the same
    saved state); return all call lists."""
    ckpt = path / "state.pt"
    model.save_checkpoint(str(ckpt))
    out = []
    for d in (None, dims, *more):
        model.load_checkpoint(str(ckpt))
        np.random.seed(3)
        th.manual_seed(3)
        model.action_space.seed(3)
        steered.w_scale_dims = None if d is None else th.as_tensor(d)
        rec.clear()
        fn()
        out.append(list(rec))
    steered.w_scale_dims = None
    return out


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_every_path_scaled_per_dim(rec, tmp_path, mode):
    model, steered = make_agent(mode)
    model._last_obs = model.env.reset()
    dims = np.tile(_vec(A), (H, 1))
    sample_fn, _ = agent_fns(model, steered)
    fns = [
        lambda: model._sample_action(0, n_envs=2),  # policy acting
        lambda: model._sample_action(10**9, n_envs=2),  # warm-up uniform w
        lambda: model.predict_diffused(model._last_obs, deterministic=True),
        lambda: model.predict_diffused(model._last_obs, deterministic=False),
        lambda: model.train(gradient_steps=1, batch_size=32),  # backup + distillation
        lambda: sample_fn(th.as_tensor(model._last_obs).repeat_interleave(3, 0)),  # best-of-N
    ]
    for fn in fns:
        plain, scaled = _both(model, steered, rec, fn, tmp_path, dims)
        assert len(plain) == len(scaled) > 0
        for (k1, w1), (k2, w2) in zip(plain, scaled):
            assert th.equal(k1, k2)
            th.testing.assert_close(w2, th.as_tensor(dims) * w1, rtol=1e-6, atol=0)


def test_checkpoint_independent_of_dims(rec, tmp_path):
    """w_scale_dims is decode-time config (not saved); a rebuilt policy with the same file decodes the same."""
    model, steered = make_agent("fixed")
    dims = th.as_tensor(np.tile(_vec(A), (H, 1)))
    steered.w_scale_dims = dims
    obs = model.env.reset()
    model.predict_diffused(obs, deterministic=True)
    th.testing.assert_close(rec[-1][1], dims * th.as_tensor(model.last_w).reshape(-1, H, A), rtol=1e-6, atol=0)


# --------------------------------------------------------------------------------------
# Config plumbing
# --------------------------------------------------------------------------------------
def test_config_plumbing(tmp_path, monkeypatch):
    pytest.importorskip("hydra")
    from omegaconf import OmegaConf

    import train_adsrl

    v = _vec(A)
    np.save(tmp_path / "v.npy", v)
    assert train_adsrl.steer_w_scale_dims(OmegaConf.create(dict(mode="fixed")), H, A) is None
    out = train_adsrl.steer_w_scale_dims(OmegaConf.create(dict(mode="fixed", w_scale_dims=str(tmp_path / "v.npy"))), H, A)
    assert np.array_equal(out, np.tile(v, (H, 1)))
    with pytest.raises(ValueError):
        train_adsrl.steer_w_scale_dims(OmegaConf.create(dict(mode="residual", w_scale_dims=str(tmp_path / "v.npy"))), H, A)
    for task in ("can", "square", "transport"):
        assert OmegaConf.load(os.path.join(ROOT, "cfg", "adsrl", f"{task}.yaml")).steer.w_scale_dims is None

    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import hydra.utils
    import render_rollouts

    monkeypatch.setattr(hydra.utils, "instantiate", lambda cfg: FakeBase())
    (tmp_path / "checkpoint").mkdir()
    dims_cfg = dict(obs_dim=OBS, act_steps=H, action_dim=A, model=dict(device="cpu"), device="cpu")
    for extra, want in ((dict(steer=dict(mode="fixed", k=2, w_scale_dims=str(tmp_path / "v.npy"))), np.tile(v, (H, 1))),
                        (dict(steer=dict(mode="fixed", k=2)), None), ({}, None)):  # fmt: skip
        (tmp_path / "run_info.json").write_text(json.dumps({"cfg": {**dims_cfg, **extra}}))
        rc = render_rollouts.run_cfg(str(tmp_path / "checkpoint" / "latest.pt"), "cpu")
        got = render_rollouts.build_steered(rc).w_scale_dims
        assert (got is None) if want is None else np.array_equal(got.numpy(), want)


# --------------------------------------------------------------------------------------
# Per-k scale: a file of shape (K, H, A) or (K, A) (tiled over H); each sample uses the row of its own k.
# A 1-D file keeps the single-vector meaning above.
# --------------------------------------------------------------------------------------


def _perk(n_k, seed=5):
    return np.random.default_rng(seed).uniform(0.5, 2.0, (n_k, H, A)).astype(np.float32)


def test_load_w_scale_dims_per_k(tmp_path):
    v_ka = _vec(K * A, 2).reshape(K, A)
    np.save(tmp_path / "ka.npy", v_ka)
    out = load_w_scale_dims(str(tmp_path / "ka.npy"), H, A)
    assert out.shape == (K, H, A) and np.array_equal(out, np.repeat(v_ka[:, None], H, 1))  # (K, A) tiled over H
    v_kha = _perk(K)
    np.save(tmp_path / "kha.npy", v_kha)
    out = load_w_scale_dims(str(tmp_path / "kha.npy"), H, A)
    assert out.shape == (K, H, A) and np.array_equal(out, v_kha)
    (tmp_path / "ka.json").write_text(json.dumps({"w_scale_dims": v_ka.tolist()}))
    np.testing.assert_allclose(load_w_scale_dims(str(tmp_path / "ka.json"), H, A), np.repeat(v_ka[:, None], H, 1))
    for bad in (np.ones((K, A + 1)), np.ones((K, H + 1, A)), -np.ones((K, A)), np.full((K, A), np.inf)):
        np.save(tmp_path / "bad.npy", bad.astype(np.float32))
        with pytest.raises(ValueError):
            load_w_scale_dims(str(tmp_path / "bad.npy"), H, A)


@pytest.mark.parametrize("param", ["absolute", "kick"])
def test_policy_scales_per_k(tmp_path, param):
    """Mixed-k batch: sample i gets w_scale * dims[k_i] * w_i. All-equal rows == the single-vector path, bitwise."""
    from test_sampler import B, _inputs, _random_model

    m = _random_model(tmp_path, 20, 8)
    cond, x_T, w = _inputs(m)
    Hm, Am = m.horizon_steps, m.action_dim
    Km = 8
    obs_aug = th.cat([cond["state"], x_T.reshape(B, -1)], dim=-1)
    k = th.tensor([0, 7, 3, 3, 5, 1])[:B]
    assert len(set(k.tolist())) > 1
    dims = np.random.default_rng(7).uniform(0.5, 2.0, (Km, Hm, Am)).astype(np.float32)
    pol = SteeredBasePolicy(m, m.obs_dim, Hm, Am, w_scale=C, param=param, w_scale_dims=dims)
    assert pol.K == Km
    want = steered_ddim_sample(m, cond, x_T, k, (w * C) * th.as_tensor(dims)[k], kick=param == "kick")
    assert th.equal(pol(obs_aug, k, w), want)
    # scalar k: every sample uses row k
    assert th.equal(pol(obs_aug, 3, w), steered_ddim_sample(m, cond, x_T, th.full((B,), 3), (w * C) * th.as_tensor(dims)[3],
                                                            kick=param == "kick"))  # fmt: skip
    # rows all equal -> bitwise the single (H, A) path
    one = np.tile(_vec(Am), (Hm, 1))
    p_single = SteeredBasePolicy(m, m.obs_dim, Hm, Am, w_scale=C, param=param, w_scale_dims=one)
    p_rows = SteeredBasePolicy(m, m.obs_dim, Hm, Am, w_scale=C, param=param, w_scale_dims=np.repeat(one[None], Km, 0))
    assert th.equal(p_rows(obs_aug, k, w), p_single(obs_aug, k, w))
    assert th.equal(pol.base_action(obs_aug), p_single.base_action(obs_aug))
    # wrong number of rows
    with pytest.raises(ValueError):
        SteeredBasePolicy(m, m.obs_dim, Hm, Am, w_scale_dims=dims[:-1])


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_every_path_scaled_per_k(rec, tmp_path, mode):
    """Actor action, warm-up, eval (det/stoch), target backup + noise-critic distillation, best-of-N: each sample's
    w is scaled by the row of its own k; with all-equal rows the recorded w equals the single-vector path bitwise."""
    model, steered = make_agent(mode)
    model._last_obs = model.env.reset()
    dims = _perk(K)
    one = np.tile(_vec(A), (H, 1))
    sample_fn, _ = agent_fns(model, steered)
    fns = [
        lambda: model._sample_action(0, n_envs=2),
        lambda: model._sample_action(10**9, n_envs=2),
        lambda: model.predict_diffused(model._last_obs, deterministic=True),
        lambda: model.predict_diffused(model._last_obs, deterministic=False),
        lambda: model.train(gradient_steps=1, batch_size=32),
        lambda: sample_fn(th.as_tensor(model._last_obs).repeat_interleave(3, 0)),
    ]
    mixed = False
    for fn in fns:
        rows = np.repeat(one[None], K, 0)
        plain, scaled, single, equal_rows = _both(model, steered, rec, fn, tmp_path, dims, one, rows)
        assert len(plain) == len(scaled) > 0
        for (k1, w1), (k2, w2) in zip(plain, scaled):
            assert th.equal(k1, k2)
            kk = k1.expand(w1.shape[0]) if k1.dim() == 0 else k1
            mixed |= len(set(kk.tolist())) > 1
            th.testing.assert_close(w2, th.as_tensor(dims)[kk] * w1, rtol=1e-6, atol=0)
        assert len(single) == len(equal_rows) > 0
        for (k1, w1), (k2, w2) in zip(single, equal_rows):
            assert th.equal(k1, k2) and th.equal(w1, w2)
    if mode == "adaptive":
        assert mixed, "adaptive paths should decode mixed-k batches"


def test_config_plumbing_per_k(tmp_path):
    pytest.importorskip("hydra")
    from omegaconf import OmegaConf

    import train_adsrl

    v = _vec(K * A, 3).reshape(K, A)
    np.save(tmp_path / "v.npy", v)
    out = train_adsrl.steer_w_scale_dims(OmegaConf.create(dict(mode="adaptive", w_scale_dims=str(tmp_path / "v.npy"))), H, A)
    assert out.shape == (K, H, A) and np.array_equal(out, np.repeat(v[:, None], H, 1))
    pol = SteeredBasePolicy(FakeBase(), OBS, H, A, w_scale_dims=out)
    assert tuple(pol.w_scale_dims.shape) == (K, H, A)
