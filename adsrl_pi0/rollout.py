"""Batched LIBERO episodes with steered pi0: one pi0 call per chunk for all active envs.

Each chunk: a fresh x_T ~ N(0, I_32) per env (tiled over the 50-step horizon, the DSRL-pi0 noise layout) or
N(0, I) of shape (50, 32) (xt_full) is appended to the observation, plus (obs_extra='abase') pi0's plan for that
x_T; `act_fn(obs_batch) -> (k, w, probs)` picks the steering; the agent's w is decoded into pi0 noise
(`decode_noise`); pi0 samples the chunk; the first `query_freq` actions run in the env workers. Rewards follow
DSRL-pi0 (sparse -1 per chunk, 0 and mask 0 on the successful chunk).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Episode:
    init_idx: int
    obs: list = field(default_factory=list)  # per chunk dict(pixels (r,r,3,1) uint8, state (8,1), x_T (32,)), + final
    w: list = field(default_factory=list)
    k: list = field(default_factory=list)
    k_probs: list = field(default_factory=list)
    a_minus_base: list = field(default_factory=list)
    steps: int = 0
    success: bool = False

    def transitions(self, discount_chunk: float):
        n = len(self.w)
        rewards = -np.ones(n, np.float32)
        masks = np.ones(n, np.float32)
        if self.success:
            rewards[-1], masks[-1] = 0.0, 0.0
        for t in range(n):
            yield self.obs[t], self.obs[t + 1], self.w[t], self.k[t], rewards[t], masks[t], discount_chunk


def structured_noise(x_T, z):
    """w = u z^T + (I - u u^T) x_T with u = 1/sqrt(H) (1,...,1): x_T with each channel's time-mean replaced by
    z / sqrt(H). x_T (..., H, A), z (..., A) -> (..., H, A). If z ~ N(0, I) independently of x_T, w ~ iid N(0, I);
    z = sqrt(H) * mean_t x_T gives w = x_T."""
    x_T = np.asarray(x_T, np.float32)
    z = np.asarray(z, np.float32)
    return x_T - x_T.mean(-2, keepdims=True) + z[..., None, :] / np.sqrt(x_T.shape[-2])


N_REAL = 7  # real LIBERO action channels of pi0's 32 (openpi LiberoOutputs keeps actions[:, :7])


def dct_basis(n: int, m: int):
    """First m orthonormal DCT-II vectors over n steps, (n, m); column 0 is the constant 1/sqrt(n)."""
    t = np.arange(n)[:, None] + 0.5
    j = np.arange(m)[None, :]
    b = np.sqrt(2.0 / n) * np.cos(np.pi * t * j / n)
    b[:, 0] = 1.0 / np.sqrt(n)
    return b


def lowfreq_noise(x_T, z, m: int):
    """For each real channel c: w[:, c] = B z_c + (I - B B^T) x_T[:, c] with B = dct_basis(H, m); padding channels
    w[:, c] = x_T[:, c]. x_T (..., H, A), z (..., m * N_REAL) mode-major (z.reshape(m, N_REAL)[j, c] = mode j of
    channel c) -> (..., H, A). z ~ N(0, I) independent of x_T gives w ~ iid N(0, I); z = B^T x_T gives w = x_T."""
    x_T = np.asarray(x_T, np.float32)
    b = dct_basis(x_T.shape[-2], m).astype(np.float32)
    z = np.asarray(z, np.float32).reshape(*x_T.shape[:-2], m, N_REAL)
    real = x_T[..., :N_REAL]
    coef = np.einsum("tj,...tc->...jc", b, real)
    w = x_T.copy()
    w[..., :N_REAL] = real + np.einsum("tj,...jc->...tc", b, z - coef)
    return w


def decode_noise(w_layout: str, x_T, z, n_modes: int):
    """The pi0 noise for the agent's z: 'tiled' z as is (SteeredPi0 tiles a 32-d row over the horizon),
    'structured' structured_noise(x_T, z), 'lowfreq' lowfreq_noise(x_T, z, n_modes)."""
    if w_layout == "tiled":
        return z
    if w_layout == "structured":
        return structured_noise(x_T, z)
    if w_layout == "lowfreq":
        return lowfreq_noise(x_T, z, n_modes)
    raise ValueError(f"unknown w_layout {w_layout!r}")


def agent_obs(o, x_T, a_base=None):
    d = {"pixels": o["pixels"][..., None], "state": o["state"][:, None], "x_T": x_T.astype(np.float32)}
    if a_base is not None:
        d["a_base"] = np.asarray(a_base, np.float32)
    return d


def run_episodes(venv, pi0, init_indices, act_fn, query_freq: int, rng: np.random.Generator,
                 compare_base: bool = False, noise_dim: int = 32, xt_full: bool = False,
                 w_layout: str = "tiled", n_modes: int = 3, obs_extra: str = "xt", feature_k: int = 0):
    """Runs len(init_indices) episodes on `venv` (continuous refill of free env slots).

    act_fn(obs_batch dict of stacked arrays) -> (k (B,), w (B,32) or (B,H,32), probs (B,K) or None).
    compare_base: also sample the unsteered action from the same x_T (one doubled pi0 call: rows [k, w] and
    [0, x_T]) and record the mean per-step L2 |a - a_base| over the executed steps (env action units).
    xt_full: sample x_T as full (H, 32) Gaussian noise instead of one tiled row; the agent obs then carries it
    flattened (H*32).
    w_layout, n_modes: how the agent's w becomes pi0 noise (`decode_noise`, with this chunk's x_T); 'structured'
    and 'lowfreq' need xt_full. Episodes store the agent's w.
    obs_extra='abase': every obs (including each episode's final one) also gets 'a_base' =
    pi0.plan_feature(obs, its x_T, feature_k), (H * N_REAL,): one extra pi0 call per chunk batch. The obs always keeps
    x_T (the caller decides which keys the agent sees).
    Returns (episodes, timing dict).
    """
    if w_layout not in ("tiled", "structured", "lowfreq"):
        raise ValueError(f"unknown w_layout {w_layout!r}")
    if w_layout != "tiled" and not xt_full:
        raise ValueError(f"w_layout={w_layout!r} needs xt_full=True")
    if obs_extra not in ("xt", "abase"):
        raise ValueError(f"unknown obs_extra {obs_extra!r}")
    abase = obs_extra == "abase"
    queue = list(init_indices)
    eps: dict[int, Episode] = {}
    cur: dict[int, dict] = {}
    timing = {"pi0_s": 0.0, "pi0_calls": 0, "feat_s": 0.0, "act_s": 0.0, "env_s": 0.0, "env_steps": 0, "chunks": 0}
    done_eps = []

    def features(pi_in, x_T):
        t0 = time.perf_counter()
        f = pi0.plan_feature(pi_in, x_T, feature_k)
        timing["feat_s"] += time.perf_counter() - t0
        return f

    def start(slots):
        slots = slots[: len(queue)]
        if not slots:
            return
        idx = [queue.pop(0) for _ in slots]
        t0 = time.perf_counter()
        obs = venv.reset(slots, idx)
        timing["env_s"] += time.perf_counter() - t0
        for s, i, o in zip(slots, idx, obs, strict=True):
            eps[s] = Episode(init_idx=int(i))
            cur[s] = o

    start(list(range(venv.n)))
    while eps:
        ids = sorted(eps)
        B = len(ids)
        pi_in = [cur[s]["pi0"] for s in ids]
        x_T = rng.standard_normal((B, pi0.horizon, noise_dim) if xt_full else (B, noise_dim)).astype(np.float32)
        feat = features(pi_in, x_T) if abase else [None] * B
        ob = [agent_obs(cur[s], x_T[j].reshape(-1), feat[j]) for j, s in enumerate(ids)]
        batch = {key: np.stack([o[key] for o in ob]) for key in ob[0]}
        t0 = time.perf_counter()
        k, w, probs = act_fn(batch)
        timing["act_s"] += time.perf_counter() - t0
        k = np.asarray(k, np.int32).reshape(B)
        w_pi = decode_noise(w_layout, x_T, w, n_modes)
        t0 = time.perf_counter()
        if compare_base:
            x_T3 = pi0.tile(x_T)
            acts = pi0.infer(pi_in + pi_in, np.concatenate([x_T3, x_T3]), np.concatenate([k, np.zeros_like(k)]),
                             np.concatenate([pi0.tile(w_pi), x_T3]))
            acts, base = acts[:B], acts[B:]
        else:
            acts = pi0.infer(pi_in, x_T, k, w_pi)
        timing["pi0_s"] += time.perf_counter() - t0
        timing["pi0_calls"] += 1
        timing["chunks"] += B
        lens = [min(query_freq, venv.max_steps - eps[s].steps) for s in ids]
        t0 = time.perf_counter()
        results = venv.step_chunk(ids, [acts[j, :L] for j, L in enumerate(lens)])
        timing["env_s"] += time.perf_counter() - t0
        finished, final = [], []
        for j, (s, (o, n, succ)) in enumerate(zip(ids, results, strict=True)):
            e = eps[s]
            e.obs.append(ob[j])
            e.w.append(np.asarray(w[j], np.float32))
            e.k.append(int(k[j]))
            if probs is not None:
                e.k_probs.append(np.asarray(probs[j], np.float32))
            if compare_base:
                d = np.linalg.norm(acts[j, :n] - base[j, :n], axis=-1)
                e.a_minus_base.append(float(d.mean()))
            e.steps += n
            timing["env_steps"] += n
            cur[s] = o
            if succ or e.steps >= venv.max_steps:
                e.success = bool(succ)
                xt_dim = pi0.horizon * noise_dim if xt_full else noise_dim
                final.append((e, o, rng.standard_normal(xt_dim).astype(np.float32)))
                done_eps.append(e)
                del eps[s]
                finished.append(s)
        if final:
            if abase:
                xs = np.stack([x for _, _, x in final])
                f = features([o["pi0"] for _, o, _ in final], xs.reshape(len(final), pi0.horizon, noise_dim)
                             if xt_full else xs)
            for j, (e, o, x) in enumerate(final):
                e.obs.append(agent_obs(o, x, f[j] if abase else None))
        start(finished)
    return done_eps, timing


def summarize(episodes, num_k: int, query_freq: int) -> dict:
    succ = np.array([e.success for e in episodes], np.float64)
    ks = np.concatenate([np.asarray(e.k) for e in episodes]) if episodes else np.zeros(0, int)
    counts = np.bincount(ks, minlength=num_k)[:num_k]
    p = counts / max(counts.sum(), 1)
    k_ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    probs = [pp for e in episodes for pp in e.k_probs]
    pi_k_ent = float("nan")
    if probs:
        P = np.clip(np.stack(probs), 1e-12, 1)
        pi_k_ent = float(-(P * np.log(P)).sum(-1).mean())
    amb = [d for e in episodes for d in e.a_minus_base]
    lens = [e.steps for e in episodes if e.success]
    return {
        "success": float(succ.mean()) if len(succ) else float("nan"),
        "n_episodes": len(episodes),
        "mean_ep_len_success": float(np.mean(lens)) if lens else float("nan"),
        "mean_a_minus_base": float(np.mean(amb)) if amb else float("nan"),
        "k_entropy": k_ent,
        "pi_k_entropy": pi_k_ent,
        "k_hist": " ".join(str(int(c)) for c in counts),
    }
