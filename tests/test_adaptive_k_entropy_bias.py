"""steer.k_entropy_bias for DSRLAdaptiveK.actor_loss (port of the pi0 'detach' fix; CPU, stub sampler).

keep:   L = E sum_k pi(k)[a_w log pi(w_k|k) - Q_k] + a_k E sum_k pi(k) log pi(k)
detach: same value, but pi(k) in the a_w log pi(w_k|k) term is stop-gradiented, so the k-logits get the
        gradient of (-sum pi Q + a_k sum pi log pi) and the w heads get the gradient of the keep loss.
"""

import pytest
import torch as th

from test_agent import OBS, D, make_adaptive

ALPHA_W, ALPHA_K = th.tensor(0.7), th.tensor(0.3)
K_PARAMS = ("k_logits.weight", "k_logits.bias")
W_PARAMS = ("mu.weight", "mu.bias", "log_std.weight", "log_std.bias")


def _grads(model, loss_fn, names, seed=3):
    """Grads of loss_fn(model) w.r.t. the named hybrid-actor params, with fixed head-sampling noise."""
    actor = model.hybrid_actor
    actor.zero_grad()
    th.manual_seed(seed)
    loss_fn(model).backward()
    params = dict(actor.named_parameters())
    return {n: params[n].grad.clone() for n in names}


def _manual(model, obs, include_w_entropy):
    """Manual keep-formula loss; drop the a_w log pi(w|k) term if include_w_entropy is False."""
    out = model.hybrid_actor.all_heads(obs)
    B, Kp, _ = out["w"].shape
    obs_rep = obs.unsqueeze(1).expand(B, Kp, obs.shape[-1]).reshape(B * Kp, -1)
    k_rep = model.hybrid_actor.allowed_k.unsqueeze(0).expand(B, Kp).reshape(-1)
    q = model._combine_q(model._q_noise(obs_rep, k_rep, out["w"].reshape(B * Kp, D))).view(B, Kp)
    probs, log_pk, log_pw = out["probs"], out["log_pk"], out["log_prob_w"]
    loss = (-probs * q).sum(-1).mean() + ALPHA_K * (probs * log_pk).sum(-1).mean()
    if include_w_entropy:
        loss = loss + (probs * ALPHA_W * log_pw).sum(-1).mean()
    return loss


@pytest.fixture
def obs():
    th.manual_seed(0)
    return th.randn(16, OBS + D)


def _model(bias):
    th.manual_seed(1)
    model, _ = make_adaptive(k_entropy_bias=bias)
    return model


def _loss(obs):
    return lambda m: m.actor_loss(obs, ALPHA_W, ALPHA_K)[0]


def test_detach_k_logit_grads_ignore_w_entropy(obs):
    model = _model("detach")
    got = _grads(model, _loss(obs), K_PARAMS)
    ref = _grads(model, lambda m: _manual(m, obs, include_w_entropy=False), K_PARAMS)
    keep = _grads(model, lambda m: _manual(m, obs, include_w_entropy=True), K_PARAMS)
    for n in K_PARAMS:
        th.testing.assert_close(got[n], ref[n])
    # the dropped term actually matters here, so the test discriminates detach from keep
    assert not th.allclose(got["k_logits.weight"], keep["k_logits.weight"])


def test_detach_w_head_grads_and_value_match_keep(obs):
    model = _model("detach")
    got = _grads(model, _loss(obs), W_PARAMS)
    ref = _grads(model, lambda m: _manual(m, obs, include_w_entropy=True), W_PARAMS)
    for n in W_PARAMS:
        th.testing.assert_close(got[n], ref[n])
    th.manual_seed(3)
    v = _loss(obs)(model)
    th.manual_seed(3)
    th.testing.assert_close(v, _manual(model, obs, include_w_entropy=True))


def test_keep_matches_original_formula(obs):
    model = _model("keep")
    names = K_PARAMS + W_PARAMS + ("trunk.0.weight",)
    got = _grads(model, _loss(obs), names)
    ref = _grads(model, lambda m: _manual(m, obs, include_w_entropy=True), names)
    for n in names:
        th.testing.assert_close(got[n], ref[n])


def test_default_is_keep_and_bad_value_rejected():
    assert _model("keep").k_entropy_bias == "keep"
    th.manual_seed(1)
    assert make_adaptive()[0].k_entropy_bias == "keep"
    with pytest.raises(AssertionError):
        _model("bogus")
