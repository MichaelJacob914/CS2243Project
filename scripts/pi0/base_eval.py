"""Base pi0 success on LIBERO tasks (no RL), with timing.

python scripts/pi0/base_eval.py --tasks 0 1 2 --episodes 50 --n_envs 10 --noise full --out results/exp7/base_eval.csv
python scripts/pi0/base_eval.py --model pi05 --suite libero_90 --tasks 0 1 --episodes 20 --out results/exp11/x.csv  (.venv-pi05)

noise=full: x ~ N(0, I) of shape (50, 32) (standard openpi sampling). noise=tiled: one N(0, I_32) row repeated
over the horizon (the DSRL-pi0 noise layout that the steering agents use for x_T and w).
Episode i uses benchmark init state i % 50, after openpi's 10-step settle wait; max 520 steps; pi0 queried
every --query_freq steps (DSRL-pi0 uses 20). --model pi05: horizon 10, default query_freq 5 (openpi's LIBERO eval);
the 50s above become 10.
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import adsrl_pi0  # noqa: E402

adsrl_pi0.setup_env()

import numpy as np  # noqa: E402


def main():
    # heavy imports here, not at module level: spawned LIBERO workers re-import this module
    from adsrl_pi0.env import LiberoVecEnv
    from adsrl_pi0.rollout import run_episodes, summarize
    from adsrl_pi0.sampler import SteeredPi0

    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--n_envs", type=int, default=10)
    p.add_argument("--noise", nargs="+", default=["full"],
                   help="full | tiled | k<K>-<full|tiled>[-wfull]: steer at step K with w ~ N(0, I) (tiled 32-d row, or full 50x32 with -wfull), x_T full/tiled")
    p.add_argument("--model", choices=["pi0", "pi05"], default="pi0")
    p.add_argument("--query_freq", type=int, default=None, help="default: 20 (pi0), 5 (pi05)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.query_freq is None:
        a.query_freq = {"pi0": 20, "pi05": 5}[a.model]

    t0 = time.time()
    pi0 = SteeredPi0.load_model(a.model)
    assert a.query_freq <= pi0.horizon, (a.query_freq, pi0.horizon)
    print(f"loaded {a.model} (horizon {pi0.horizon}) in {time.time() - t0:.1f}s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fields = ["suite", "task_id", "task", "noise", "query_freq", "episodes", "success", "mean_ep_len_success",
              "env_steps", "wall_s", "pi0_calls", "pi0_s_per_call", "mean_batch", "env_s", "sim_s_per_step",
              "env_steps_per_s", "n_envs", "gpu"]
    new = not os.path.exists(a.out)
    gpu = os.popen("nvidia-smi --query-gpu=name --format=csv,noheader -i 0").read().strip()
    for task_id in a.tasks:
        venv = LiberoVecEnv(a.n_envs, task_id, suite=a.suite, seed=a.seed)
        for noise in a.noise:
            rng = np.random.default_rng(a.seed * 1000 + task_id)

            if noise in ("full", "tiled"):
                k_eval, xt_full = 0, False
            else:
                parts = noise[1:].split("-")
                k_eval, xt_full = int(parts[0]), parts[1] == "full"
            w_full = noise == "full" or noise.endswith("-wfull")

            def act_fn(batch, w_full=w_full, rng=rng, k_eval=k_eval):
                B = batch["state"].shape[0]
                w = rng.standard_normal((B, pi0.horizon, 32) if w_full else (B, 32)).astype(np.float32)
                return np.full(B, k_eval, np.int32), w, None

            t1 = time.time()
            idx = [i % venv.n_init_states for i in range(a.episodes)]
            eps, tm = run_episodes(venv, pi0, idx, act_fn, a.query_freq, rng, xt_full=xt_full)
            wall = time.time() - t1
            s = summarize(eps, 10, a.query_freq)
            sim_t, sim_n = venv.sim_timing()
            row = {"suite": a.suite, "task_id": task_id, "task": venv.prompt, "noise": noise,
                   "query_freq": a.query_freq, "episodes": len(eps), "success": round(s["success"], 4),
                   "mean_ep_len_success": round(s["mean_ep_len_success"], 1), "env_steps": tm["env_steps"],
                   "wall_s": round(wall, 1), "pi0_calls": tm["pi0_calls"],
                   "pi0_s_per_call": round(tm["pi0_s"] / max(tm["pi0_calls"], 1), 4),
                   "mean_batch": round(tm["chunks"] / max(tm["pi0_calls"], 1), 2), "env_s": round(tm["env_s"], 1),
                   "sim_s_per_step": round(sim_t / max(sim_n, 1), 5),
                   "env_steps_per_s": round(tm["env_steps"] / wall, 1), "n_envs": a.n_envs, "gpu": gpu}
            print(row, flush=True)
            with open(a.out, "a", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=fields)
                if new:
                    wr.writeheader()
                    new = False
                wr.writerow(row)
        venv.close()


if __name__ == "__main__":
    main()
