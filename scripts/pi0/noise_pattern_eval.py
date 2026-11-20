"""Eval-only diagnostic: does the *pattern* of the steering noise w explain why fixed k>0 runs end below plain pi0?

No training. For each task and k in {0, 3, 6, 8}, pi0 is steered at step k (adsrl_pi0 steered sampler, the one the
agents use) with w drawn from one of four noise patterns:
  iid        w ~ N(0, I) of full size 50x32 (k=0: plain pi0; k>0: an in-distribution re-noise)
  tiled      one 32-d N(0, I) row repeated over the 50 timesteps
  agent      what a freshly initialised fixed-k agent explores with: `FixedKAgent.act(obs)` of the agent built
             exactly as train_adsrl_pi0.py builds it (sweep defaults, seed 0, xt_layout full; x_T in the agent obs
             for k>0), i.e. w = tanh(u), u ~ N(mean(s), std(s)) per dim, one 32-d row tiled over 50 steps
  agent_iid  same per-element distribution as `agent` (same actor, same obs), but 50 independent draws, one per
             timestep (separates "tiling" from "marginal/scale")
x_T is a full 50x32 Gaussian, fresh per chunk (as in the sweep's --steer.xt_layout full and zero_noise_eval.py).
Episode protocol and env seeding as zero_noise_eval.py (env seed 20261107, init state i % 50, training-eval
protocol); 100 episodes per condition = 2 per init state. Noise RNGs: numpy default_rng([seed, task, 100 + cond])
for x_T and the Gaussian patterns, JAX key fold_in(fold_in(PRNGKey(seed), task), 100 + cond) for the agent
patterns. |a - a_base| as in zero_noise_eval.py (steered vs unsteered action from the same x_T).
One CSV per task (one row per condition); rerunning skips finished conditions.

python scripts/pi0/noise_pattern_eval.py --task 8                # one task, all 16 conditions
python scripts/pi0/noise_pattern_eval.py --task 8 --dry_run 1    # print the plan and exit (no GPU, no env)
"""

import argparse
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import adsrl_pi0  # noqa: E402

adsrl_pi0.setup_env()

import numpy as np  # noqa: E402

OUT = "/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs/exp7/noise_pattern"
FIELDS = ["task_id", "cond", "pattern", "k", "success", "n", "se", "mean_a_minus_base", "w_rms", "w_abs_max",
          "w_tile_dev", "mean_ep_len_success", "wall_s", "gpu"]
PATTERNS = ["iid", "tiled", "agent", "agent_iid"]
KS = [0, 3, 6, 8]
CONDITIONS = [f"{p}_k{k}" for k in KS for p in PATTERNS]
H, D = 50, 32  # pi0 action horizon, action dim


def parse_cond(cond):
    p, k = cond.rsplit("_k", 1)
    assert p in PATTERNS, cond
    return p, int(k)


# ------------------------------------------------------------------------------------------------
# noise generators (tested on CPU in tests/test_noise_pattern.py); all return w of shape (B, H, D) float32
# ------------------------------------------------------------------------------------------------
def gaussian_w(rng, B, tiled, h=H, d=D):
    if tiled:
        return np.repeat(rng.standard_normal((B, 1, d)).astype(np.float32), h, axis=1)
    return rng.standard_normal((B, h, d)).astype(np.float32)


def build_fresh_agent(k, task=8, seed=0):
    """The fixed-k agent at step 0 of a sweep run: train_adsrl_pi0's own parse/build_agent with the sweep's
    settings (scripts/pi0/sweep_exp7.sh: seed 0, xt_layout full, everything else default)."""
    import train_adsrl_pi0 as tr

    a = tr.parse(["--task_id", str(task), "--steer.mode", "fixed", "--steer.k", str(k), "--steer.xt_layout", "full",
                  "--seed", str(seed), "--use_wandb", "0"])
    use_xt = not (a.mode == "fixed" and a.k == 0)  # as in train_adsrl_pi0.main
    return tr.build_agent(a, use_xt), use_xt


def agent_obs(batch, use_xt):
    return batch if use_xt else {kk: v for kk, v in batch.items() if kk != "x_T"}


def agent_tiled_w(agent, batch, use_xt, h=H):
    """Exactly the training/eval exploration call (train_adsrl_pi0 act_agent(False)), tiled as SteeredPi0.tile does."""
    import jax

    _, w, _ = agent.act(jax.device_put(agent_obs(batch, use_xt)), deterministic=False)
    w = np.asarray(w, np.float32)
    return np.repeat(w[:, None, :], h, axis=1)


def make_agent_iid_sampler(agent, h=H):
    """jit fn(obs, key) -> (B, h, D): h independent draws per row from the same actor distribution the agent samples
    from (jaxrl2 sample_actions_jit draws dist.sample(seed=key) once)."""
    import jax
    import jax.numpy as jnp

    actor = agent.learner._actor
    assert actor.batch_stats is None

    @jax.jit
    def f(params, obs, key):
        dist = actor.apply_fn({"params": params}, obs)
        return jnp.swapaxes(dist.sample(seed=key, sample_shape=(h,)), 0, 1)

    return lambda obs, key: np.asarray(f(actor.params, obs, key), np.float32)


def actor_dist_params(agent, obs):
    """Pre-tanh Gaussian (loc, scale) of the agent's actor, (B, D) each (for tests / reporting)."""
    actor = agent.learner._actor
    dist = actor.apply_fn({"params": actor.params}, obs)
    return np.asarray(dist.distribution.loc), np.asarray(dist.distribution.scale_diag)


def make_act_fn(pattern, k, rng, key=None, agent=None, use_xt=None, iid_sampler=None):
    """act_fn for adsrl_pi0.rollout.run_episodes: (k, w (B,H,D), None). Records nothing itself."""
    import jax

    state = {"key": key}

    def act_fn(batch):
        B = batch["state"].shape[0]
        if pattern in ("iid", "tiled"):
            w = gaussian_w(rng, B, tiled=pattern == "tiled")
        elif pattern == "agent":
            w = agent_tiled_w(agent, batch, use_xt)
        elif pattern == "agent_iid":
            state["key"], sub = jax.random.split(state["key"])
            w = iid_sampler(jax.device_put(agent_obs(batch, use_xt)), sub)
        else:
            raise ValueError(pattern)
        return np.full(B, k, np.int32), w, None

    return act_fn


def w_stats(eps):
    W = np.stack([w for e in eps for w in e.w])  # (chunks, H, D)
    return {"w_rms": round(float(np.sqrt((W**2).mean())), 4), "w_abs_max": round(float(np.abs(W).max()), 4),
            # mean |w_t - w_0| over timesteps: 0 for tiled patterns
            "w_tile_dev": round(float(np.abs(W - W[:, :1]).mean()), 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--conds", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--n_envs", type=int, default=12)
    p.add_argument("--query_freq", type=int, default=20)
    p.add_argument("--seed", type=int, default=20261107, help="env + noise seed (as zero_noise_eval.py)")
    p.add_argument("--agent_seed", type=int, default=0, help="seed of the fresh agent (sweep runs used 0)")
    p.add_argument("--out", default=OUT)
    p.add_argument("--dry_run", type=int, default=0)
    a = p.parse_args()
    path = os.path.join(a.out, f"task{a.task}.csv")
    done = {r["cond"] for r in csv.DictReader(open(path))} if os.path.exists(path) else set()
    todo = [c for c in a.conds if c not in done]
    print(f"task {a.task}: {len(todo)} conditions to run ({len(done)} already in {path}), {a.episodes} episodes "
          f"each, init states i % 50, env seed {a.seed}, agent seed {a.agent_seed}", flush=True)
    for c in todo:
        print(f"  {c}: pattern={parse_cond(c)[0]} k={parse_cond(c)[1]} rng=[{a.seed}, {a.task}, "
              f"{100 + CONDITIONS.index(c)}]", flush=True)
    if a.dry_run or not todo:
        return

    # heavy imports here: spawned LIBERO workers re-import this module
    import jax

    from adsrl_pi0.env import LiberoVecEnv
    from adsrl_pi0.rollout import run_episodes, summarize
    from adsrl_pi0.sampler import SteeredPi0

    os.makedirs(a.out, exist_ok=True)
    gpu = os.popen("nvidia-smi --query-gpu=name --format=csv,noheader -i 0").read().strip()
    t0 = time.time()
    pi0 = SteeredPi0.load()
    print(f"loaded pi0 in {time.time() - t0:.0f}s on {gpu}; jax {jax.__version__} {jax.devices()}", flush=True)
    venv = LiberoVecEnv(a.n_envs, a.task, seed=a.seed)
    idx = [i % venv.n_init_states for i in range(a.episodes)]
    agents = {}
    for cond in todo:
        pattern, k = parse_cond(cond)
        ci = 100 + CONDITIONS.index(cond)
        rng = np.random.default_rng([a.seed, a.task, ci])
        key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(a.seed), a.task), ci)
        agent = use_xt = sampler = None
        if pattern.startswith("agent"):
            if k not in agents:
                ag, ux = build_fresh_agent(k, a.task, a.agent_seed)
                agents[k] = (ag, ux, make_agent_iid_sampler(ag))
            agent, use_xt, sampler = agents[k]
            agent.set_rng(key)  # `agent` draws from the agent's own key (params never change: no updates here)
        act_fn = make_act_fn(pattern, k, rng, key=key, agent=agent, use_xt=use_xt, iid_sampler=sampler)
        t1 = time.time()
        eps, _ = run_episodes(venv, pi0, idx, act_fn, a.query_freq, rng, compare_base=True, xt_full=True)
        s = summarize(eps, 10, a.query_freq)
        n = s["n_episodes"]
        row = {"task_id": a.task, "cond": cond, "pattern": pattern, "k": k, "success": round(s["success"], 4),
               "n": n, "se": round(float(np.sqrt(s["success"] * (1 - s["success"]) / n)), 4),
               "mean_a_minus_base": round(s["mean_a_minus_base"], 4), **w_stats(eps),
               "mean_ep_len_success": round(s["mean_ep_len_success"], 1), "wall_s": round(time.time() - t1),
               "gpu": gpu}
        print(row, flush=True)
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                wr.writeheader()
            wr.writerow(row)
    venv.close()


if __name__ == "__main__":
    main()
