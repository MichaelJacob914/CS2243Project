"""Smoke test + micro-benchmarks for pi0 + LIBERO on one GPU.

One LIBERO-10 episode (task --task_id, init state 0, base pi0 with full Gaussian noise, query every 20 steps) in
this process, then: LIBERO step time and agentview/wrist render time, pi0 call time vs batch size (k=0 and a
mixed-k batch), device memory. Appends a JSON line to --out.
"""

import argparse
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import adsrl_pi0  # noqa: E402

adsrl_pi0.setup_env()

import jax  # noqa: E402
import numpy as np  # noqa: E402

from adsrl_pi0.env import LIBERO_DUMMY_ACTION, process_obs  # noqa: E402
from adsrl_pi0.sampler import SteeredPi0  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    res = {"job": os.environ.get("SLURM_JOB_ID"), "partition": os.environ.get("SLURM_JOB_PARTITION"),
           "gpu": os.popen("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i 0").read().strip(),
           "jax": jax.__version__, "devices": [str(d) for d in jax.devices()]}
    print(res, flush=True)

    t0 = time.time()
    pi0 = SteeredPi0.load()
    res["load_s"] = round(time.time() - t0, 1)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(a.task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    t0 = time.time()
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(a.task_id)[0])
    res["env_create_s"] = round(time.time() - t0, 1)
    for _ in range(10):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    rng = np.random.default_rng(0)
    pi0_t, step_t, proc_t = [], [], []
    success, t = False, 0
    while t < 520 and not success:
        t1 = time.perf_counter()
        o = process_obs(obs, str(task.language), 64)
        proc_t.append(time.perf_counter() - t1)
        t1 = time.perf_counter()
        acts = pi0.infer([o["pi0"]], rng.standard_normal((1, 50, 32)), np.zeros(1, np.int32),
                         rng.standard_normal((1, 50, 32)))
        pi0_t.append(time.perf_counter() - t1)
        for act in acts[0, :20]:
            t1 = time.perf_counter()
            obs, _, done, _ = env.step(act.tolist())
            step_t.append(time.perf_counter() - t1)
            t += 1
            if done:
                success = True
                break
    res.update(episode_success=success, episode_steps=t, pi0_first_call_s=round(pi0_t[0], 2),
               pi0_call_b1_s=round(float(np.median(pi0_t[1:])), 4), env_step_s=round(float(np.median(step_t)), 5),
               obs_process_s=round(float(np.median(proc_t)), 5))

    sim = env.env.sim
    rt = {}
    for cam in ["agentview", "robot0_eye_in_hand"]:
        sim.render(width=256, height=256, camera_name=cam)
        t1 = time.perf_counter()
        for _ in range(50):
            sim.render(width=256, height=256, camera_name=cam)
        rt[cam] = (time.perf_counter() - t1) / 50
    res["render_s_per_camera"] = {k: round(v, 5) for k, v in rt.items()}
    # physics only: robosuite steps the sim control_freq-many substeps per env.step
    t1 = time.perf_counter()
    for _ in range(50):
        sim.step()
    res["sim_substep_s"] = round((time.perf_counter() - t1) / 50, 6)
    env.close()

    o = process_obs(obs, str(task.language), 64)["pi0"]
    calls = {}
    for b in a.batches:
        for name, ks in [("k0", np.zeros(b, np.int32)), ("mixed", np.arange(b, dtype=np.int32) % 10)]:
            x = rng.standard_normal((b, 32)).astype(np.float32)
            pi0.infer([o] * b, x, ks, x)  # compile
            ts = []
            for _ in range(5):
                t1 = time.perf_counter()
                pi0.infer([o] * b, x, ks, x)
                ts.append(time.perf_counter() - t1)
            calls[f"b{b}_{name}"] = round(float(np.median(ts)), 4)
        print(b, calls, flush=True)
    res["pi0_call_s"] = calls
    try:
        ms = jax.devices()[0].memory_stats()
        res["peak_gb"] = round(ms.get("peak_bytes_in_use", 0) / 1e9, 2)
    except Exception as e:  # noqa: BLE001
        res["peak_gb"] = str(e)
    print(json.dumps(res), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
