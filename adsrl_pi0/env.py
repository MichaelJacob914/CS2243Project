"""LIBERO environments for pi0 steering, run in worker processes that execute whole action chunks.

Episode protocol (same as openpi's LIBERO eval, examples/libero/main.py): reset, set one of the task's
benchmark init states, wait `num_steps_wait` steps with a no-op action so objects settle, then run the policy
for at most `max_steps` steps. Observation processing is copied from dsrl_pi0/examples/train_utils_sim.py
(agentview flipped, 224x224 resize-with-pad for pi0, a `resize_image` copy for the SAC encoder, 8-d state).
"""

from __future__ import annotations

import faulthandler
import math
import multiprocessing as mp
import os
import pathlib
import time

import numpy as np

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def _quat2axisangle(quat):
    quat = np.array(quat, dtype=np.float64)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def process_obs(obs, prompt: str, resize_image: int) -> dict:
    """Raw LIBERO obs -> {'pi0': openpi input dict, 'pixels': (r,r,3) uint8, 'state': (8,) float32}."""
    import PIL.Image
    from openpi_client import image_tools

    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    state = np.concatenate(
        (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    ).astype(np.float32)
    pi0 = {
        "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224)),
        "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224)),
        "observation/state": state,
        "prompt": prompt,
    }
    pixels = np.array(PIL.Image.fromarray(img).resize((resize_image, resize_image)))
    return {"pi0": pi0, "pixels": pixels, "state": state}


def _worker(remote, suite: str, task_id: int, resolution: int, resize_image: int, seed: int, num_steps_wait: int):
    faulthandler.enable()  # crash evidence: dump the Python stack to stderr on SIGSEGV/SIGABRT/... (no other effect)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_suite = benchmark.get_benchmark_dict()[suite]()
    task = task_suite.get_task(task_id)
    init_states = task_suite.get_task_init_states(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    prompt = str(task.language)
    t_sim = 0.0
    n_sim = 0
    while True:
        cmd, arg = remote.recv()
        if cmd == "info":
            remote.send({"prompt": prompt, "n_init_states": len(init_states)})
        elif cmd == "reset":
            env.reset()
            obs = env.set_init_state(init_states[int(arg)])
            for _ in range(num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            remote.send(process_obs(obs, prompt, resize_image))
        elif cmd == "chunk":
            # run up to len(actions) steps; stop at success. Returns (obs, n_steps, success)
            actions = arg
            success = False
            n = 0
            t0 = time.perf_counter()
            for a in actions:
                obs, _, done, _ = env.step(np.asarray(a, np.float64).tolist())
                n += 1
                if done:
                    success = True
                    break
            t_sim += time.perf_counter() - t0
            n_sim += n
            remote.send((process_obs(obs, prompt, resize_image), n, success))
        elif cmd == "timing":
            remote.send((t_sim, n_sim))
        elif cmd == "close":
            env.close()
            remote.close()
            return
        else:
            raise ValueError(cmd)


class LiberoVecEnv:
    """n independent LIBERO envs of one task, each in its own process; chunks are executed in the workers.

    The caller tracks episode bookkeeping (steps used, done flags); see `adsrl_pi0.rollout`.
    """

    def __init__(self, n_envs: int, task_id: int, suite: str = "libero_10", seed: int = 0, resolution: int = 256,
                 resize_image: int = 64, num_steps_wait: int = 10):
        ctx = mp.get_context("spawn")
        self.n = n_envs
        self.max_steps = MAX_STEPS[suite]
        self.remotes, self.procs = [], []
        for i in range(n_envs):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, suite, task_id, resolution, resize_image, seed + 1000 * i,
                                                  num_steps_wait), daemon=True)
            p.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(p)
        self.remotes[0].send(("info", None))
        info = self.remotes[0].recv()
        self.prompt, self.n_init_states = info["prompt"], info["n_init_states"]

    def reset(self, ids, init_idx):
        for i, s in zip(ids, init_idx, strict=True):
            self.remotes[i].send(("reset", int(s)))
        return [self._recv(i) for i in ids]

    def _recv(self, i):
        """remotes[i].recv(); on EOFError (worker i died) print which worker and its exit code, then re-raise."""
        try:
            return self.remotes[i].recv()
        except EOFError:
            p = self.procs[i]
            p.join(timeout=5)
            print(f"[LiberoVecEnv] EOFError from worker {i} (pid {p.pid}): alive={p.is_alive()} exitcode={p.exitcode} "
                  f"(negative = killed by that signal)", flush=True)
            raise

    def step_chunk(self, ids, actions):
        """actions: list of (L_i, 7) arrays for env ids. Returns list of (obs, n_steps, success)."""
        for i, a in zip(ids, actions, strict=True):
            self.remotes[i].send(("chunk", np.asarray(a)))
        return [self._recv(i) for i in ids]

    def sim_timing(self):
        for r in self.remotes:
            r.send(("timing", None))
        t = [r.recv() for r in self.remotes]
        return sum(x[0] for x in t), sum(x[1] for x in t)

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()


def libero10_task_names():
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(pathlib.Path(__file__).resolve().parents[1] / ".libero"))
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    return [suite.get_task(i).language for i in range(suite.n_tasks)]
