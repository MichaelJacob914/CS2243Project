"""Run a config: tasks x episodes, one JSON line per episode, a few videos, and enough metadata to reproduce.

Resumable: episodes already in this shard's results file are skipped, so a job cut off by a time limit
continues where it stopped when resubmitted with the same command.
"""

import datetime
import json
import os
import pathlib
import platform
import subprocess
import time

from steer import noise
from steer.envs import make_task
from steer.eval.rollout import run_episode


def task_ids(cfg: dict) -> list:
    if cfg["env"] == "aloha":
        return [cfg.get("task", "gym_aloha/AlohaTransferCube-v0")]
    tasks = cfg.get("tasks", "all")
    if tasks == "all":
        from steer.envs.libero import suite_size

        return list(range(suite_size(cfg["suite"])))
    return list(tasks)


def task_kwargs(cfg: dict, task) -> dict:
    if cfg["env"] == "aloha":
        return {"task": task, "seed": cfg["seed"]}
    return {"suite": cfg["suite"], "task_id": task, "seed": cfg["seed"]}


def git_commits() -> dict:
    repo = pathlib.Path(__file__).resolve().parents[3]

    def head(path):
        try:
            out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
            dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"], capture_output=True, text=True)
            return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
        except OSError:
            return None

    return {"repo": head(repo), "openpi": head(repo / "third_party/openpi"), "LIBERO": head(repo / "third_party/LIBERO")}


def run(cfg: dict, out_dir: pathlib.Path, shard: tuple[int, int] = (0, 1)) -> None:
    import imageio

    from steer.policies.openpi_policy import OpenPIPolicy

    i, n = shard
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"results_shard{i:03d}of{n:03d}.jsonl"
    done = set()
    if results_path.exists():
        rows = map(json.loads, results_path.read_text().splitlines())
        done = {(r["task"], r.get("noise_seed"), r["episode"]) for r in rows}

    meta = {
        "config": cfg,
        "shard": [i, n],
        "git": git_commits(),
        "host": platform.node(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / f"meta_shard{i:03d}of{n:03d}.json").write_text(json.dumps(meta, indent=2))

    policy = OpenPIPolicy(cfg["policy"], seed=cfg["seed"] * 1000 + i)
    replan = cfg.get("replan_steps") or policy.spec.replan_steps
    videos = out_dir / "videos"

    # Work units are (task, noise seed). `noise.seeds: [..]` sweeps several fixed noises in one job;
    # shards split the units, so a sweep on a single task still parallelises.
    noise_cfg = dict(cfg.get("noise") or {"mode": "default"})
    seeds = noise_cfg.pop("seeds", None)
    units = [(task, s) for task in task_ids(cfg) for s in (seeds if seeds is not None else [None])]

    env, env_task = None, object()
    for task, nseed in units[i::n]:
        todo = [ep for ep in range(cfg["episodes"]) if (task, nseed, ep) not in done]
        if not todo:
            continue
        if task != env_task:
            if env is not None:
                env.close()
            env, env_task = make_task(cfg["env"], **task_kwargs(cfg, task)), task
        tid = task if isinstance(task, int) else 0
        ncfg = noise_cfg if nseed is None else {**noise_cfg, "seed": nseed}
        sampler = noise.make(ncfg, policy.noise_shape, seed=cfg["seed"] * 1000 + tid)
        for ep in todo:
            record = ep < cfg.get("videos_per_task", 0)
            t0 = time.time()
            res, frames = run_episode(policy, env, sampler, ep, replan, record=record)
            row = {"task": task, "noise_seed": nseed, "prompt": env.prompt, "episode": ep, **res,
                   "wall_s": round(time.time() - t0, 1)}  # fmt: skip
            with results_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if frames:
                videos.mkdir(exist_ok=True)
                tag = "" if nseed is None else f"_noise{nseed:03d}"
                name = f"task{tid:03d}{tag}_ep{ep:03d}_{'success' if res['success'] else 'failure'}.mp4"
                imageio.mimwrite(videos / name, frames, fps=env.fps)
    if env is not None:
        env.close()
