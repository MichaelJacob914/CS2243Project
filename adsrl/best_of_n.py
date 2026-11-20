"""Best-of-N noise at eval: sample N candidate (k, w) per state, decode them, score each decoded action
chunk with the action critic Q(s, a), and execute the argmax (eval-time add-on for fixed-k and
adaptive agents; see scripts/best_of_n_eval.py). N = 1 is the agent's ordinary stochastic policy."""

from __future__ import annotations

from typing import Callable

import torch as th


@th.no_grad()
def best_of_n(obs: th.Tensor, n: int, sample_fn: Callable, q_fn: Callable) -> dict:
    """``sample_fn(obs_rep) -> dict`` of per-row tensors with at least "action" [B*N, D];
    ``q_fn(obs_rep, action) -> [B*N]``. ``obs_rep`` repeats each row of ``obs`` N times in a row
    (row b*N + j is candidate j of state b). Returns every sampled key at the argmax candidate
    ([B, ...]), plus "q" [B, N] (all candidate scores) and "idx" [B]."""
    B = obs.shape[0]
    obs_rep = obs.repeat_interleave(n, dim=0)
    cand = sample_fn(obs_rep)
    q = q_fn(obs_rep, cand["action"]).reshape(B, n)
    idx = q.argmax(dim=1)
    rows = th.arange(B, device=obs.device) * n + idx
    out = {key: v[rows] for key, v in cand.items()}
    out["q"], out["idx"] = q, idx
    return out


def agent_fns(model, steered):
    """(sample_fn, q_fn) for a DSRLFixedK or DSRLAdaptiveK agent. Candidates are drawn from the
    agent's stochastic policy (fixed k: w ~ pi(w|s); adaptive: k ~ pi(k|s), w ~ pi(w|s,k)) and
    scored with the same critic combination as the target backup (min or mean over critics)."""
    from adsrl.agent import DSRLAdaptiveK

    adaptive = isinstance(model, DSRLAdaptiveK)

    def sample_fn(obs_rep):
        if adaptive:
            out = model.hybrid_actor.sample(obs_rep, deterministic=False)
            k, w_scaled = out["k"], out["w"]
        else:
            w_scaled = model.actor(obs_rep, deterministic=False)
            k = th.full((obs_rep.shape[0],), model.k, dtype=th.long, device=obs_rep.device)
        w = model._unscale_t(w_scaled)
        a = steered(obs_rep, k, w).reshape(obs_rep.shape[0], -1).to(model.device, th.float32)
        return {"action": a, "k": k, "w": w}

    def q_fn(obs_rep, action):
        return model._combine_q(th.cat(model.critic(obs_rep, action), dim=1)).reshape(-1)

    return sample_fn, q_fn
