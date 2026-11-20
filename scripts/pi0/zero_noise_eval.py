"""Eval-only baseline: pi0 with zero noise injected at flow step k (w = 0), vs plain pi0, on LIBERO tasks.

Per chunk: x_T ~ N(0, I) of full size 50x32, fresh for every chunk. Steering at k with w = 0 (adsrl_pi0 steered
sampler, the same one the agents use): plain Euler up to step k, then x_k = (1 - t_k) * a_hat_{k-1} (x_0 = 0 for
k = 0), then plain Euler. Condition "base" is plain pi0 (standard N(0, I) noise, k = 0, w = x_T).
Episode protocol = training evals: benchmark init states, 10 settle steps, 520-step cap, pi0 queried every 20 steps.
Episode i uses init state i % 50 (200 episodes = 4 per init state, each with different x_T).
|a - a_base|: mean per-step L2 over executed steps between the steered action and the unsteered action from the
same x_T (same pi0 call, doubled batch). Seeds: numpy default_rng([20261107, task, condition index]), independent
of the sweep. Writes one CSV per task, appending a row per condition; rerunning skips conditions already present.

python scripts/pi0/zero_noise_eval.py --tasks 8 4 6 --episodes 200 --n_envs 12
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

OUT = "/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs/exp7/zero_noise_baseline"
FIELDS = ["task_id", "k", "success", "n", "se", "mean_a_minus_base", "mean_ep_len_success", "wall_s", "gpu"]
CONDITIONS = ["base"] + [str(k) for k in range(10)]


def main():
    # heavy imports here: spawned LIBERO workers re-import this module
    from adsrl_pi0.env import LiberoVecEnv
    from adsrl_pi0.rollout import run_episodes, summarize
    from adsrl_pi0.sampler import SteeredPi0

    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, nargs="+", default=[8, 4, 6])
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--n_envs", type=int, default=12)
    p.add_argument("--query_freq", type=int, default=20)
    p.add_argument("--seed", type=int, default=20261107)
    p.add_argument("--out", default=OUT)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gpu = os.popen("nvidia-smi --query-gpu=name --format=csv,noheader -i 0").read().strip()
    t0 = time.time()
    pi0 = SteeredPi0.load()
    print(f"loaded pi0 in {time.time() - t0:.0f}s on {gpu}", flush=True)

    for task in a.tasks:
        path = os.path.join(a.out, f"task{task}.csv")
        done = {r["k"] for r in csv.DictReader(open(path))} if os.path.exists(path) else set()
        todo = [c for c in CONDITIONS if c not in done]
        if not todo:
            continue
        venv = LiberoVecEnv(a.n_envs, task, seed=a.seed)
        idx = [i % venv.n_init_states for i in range(a.episodes)]
        for cond in todo:
            rng = np.random.default_rng([a.seed, task, CONDITIONS.index(cond)])
            base = cond == "base"
            k = 0 if base else int(cond)

            def act_fn(batch, rng=rng, base=base, k=k):
                B = batch["state"].shape[0]
                w = rng.standard_normal((B, 50, 32)).astype(np.float32) if base else np.zeros((B, 50, 32), np.float32)
                return np.full(B, k, np.int32), w, None

            t1 = time.time()
            eps, _ = run_episodes(venv, pi0, idx, act_fn, a.query_freq, rng, compare_base=not base, xt_full=True)
            s = summarize(eps, 10, a.query_freq)
            n = s["n_episodes"]
            row = {"task_id": task, "k": cond, "success": round(s["success"], 4), "n": n,
                   "se": round(float(np.sqrt(s["success"] * (1 - s["success"]) / n)), 4),
                   "mean_a_minus_base": round(s["mean_a_minus_base"], 4) if not base else "",
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
