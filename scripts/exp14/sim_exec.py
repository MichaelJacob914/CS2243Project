"""exp14 simulator executor (CPU): executes action chunks from saved transport states.

  stage1  each of the 256 Stage 1 samples per state, one chunk from the saved state   (act_dir/s1_XXX.npz: samp)
  stage3  the Stage 2a regenerations (policy + null replicate 1) and the Stage 2b copies (plan + steering forms)
          of the Stage 3 subset, one chunk each                                        (act_dir/s3_XXX.npz)
  stage4  the Stage 4 regenerations: first chunk = the regenerated action, then the base policy (DDIM-20,
          fresh x_T per chunk, CPU) until success or the 800-step budget from the state's time t
                                                                                      (act_dir/s4_XXX.npz)

One-chunk features (transport_env.short_horizon): object displacement (payload, lid handle, trash; 9-d),
eef displacement (6-d), and per gripper the object it approached most (0 payload, 1 lid handle, 2 trash).

Outputs go to out_dir/<stage>/..., one npz per task; existing outputs are skipped (resumable).

    python scripts/exp14/sim_exec.py --stage stage1 --states results/exp14/states/states.npz \
        --act_dir /n/netscratch/.../exp14/actions --out_dir results/exp14/sim --n_workers 64
    python scripts/exp14/sim_exec.py --stage stage3 ... --time_only 200     # time 200 chunks, write nothing
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

_SIM = None
_POLICY = None
_STATES = None


def _init(states_path):
    global _SIM, _STATES
    import transport_env as TE

    _SIM = TE.TransportSim()
    _STATES = dict(np.load(states_path))


def _policy():
    global _POLICY
    if _POLICY is None:
        import torch

        torch.set_num_threads(1)
        from collect_states import load_policy_cpu

        _POLICY = load_policy_cpu()
    return _POLICY


def exec_chunks(i, chunks):
    """chunks (N, H, A) normalized, all from state i. Returns dict of per-chunk features."""
    import transport_env as TE

    st, gr, ij = _STATES["sim_state"][i], _STATES["grip"][i], _STATES["init_joint"][i]
    n = len(chunks)
    out = {"disp": np.zeros((n, 9), np.float32), "eef_disp": np.zeros((n, 6), np.float32),
           "toward0": np.zeros(n, np.int8), "toward1": np.zeros(n, np.int8), "succ": np.zeros(n, bool)}  # fmt: skip
    for j in range(n):
        _, raw0 = _SIM.restore(st, gr, init_joint=ij)
        _, raw1, succ, _ = _SIM.step_chunk(np.clip(chunks[j], -1.0, 1.0))  # Design change 5: execute the clipped chunk
        f = TE.short_horizon(raw0, raw1)
        out["disp"][j] = f["disp"][0]
        out["eef_disp"][j] = f["eef_disp"][0]
        out["toward0"][j] = f["toward0"][0]
        out["toward1"][j] = f["toward1"][0]
        out["succ"][j] = succ
    return out


def run_episode(i, first_chunk, seed):
    """Restore state i, execute first_chunk, then the base policy until success or 800 - t total steps."""
    import torch

    import transport_env as TE
    import windows as W

    budget = TE.MAX_STEPS - int(_STATES["t"][i])
    assert budget >= 8, f"state {i}: t = {int(_STATES['t'][i])} leaves {budget} < 8 env steps (review B2; Stage 4 uses rollout states only)"
    model = _policy()
    g = torch.Generator().manual_seed(int(seed))
    _SIM.restore(_STATES["sim_state"][i], _STATES["grip"][i], init_joint=_STATES["init_joint"][i])
    obs, _, succ, n = _SIM.step_chunk(np.clip(first_chunk, -1.0, 1.0))  # Design change 5
    steps = n
    while not succ and steps < budget:
        x_T = torch.randn((1, 8, 14), generator=g)
        a = W.ddim_sample(model, {"state": torch.as_tensor(obs)[None]}, x_T)[0].numpy()
        obs, _, succ, n = _SIM.step_chunk(np.clip(a, -1.0, 1.0))
        steps += n
    return bool(succ), int(steps)


def task(args):
    kind, i, key, sl, out_path, extra = args
    t0 = time.time()
    if kind == "stage4":
        d = np.load(extra["file"])
        acts = d["actions"][sl[0], sl[1]]  # (M4, H, A) for (level, point)
        res = [run_episode(i, a, 7_000_000 + 100_000 * i + 1000 * sl[0] + 10 * sl[1] + r) for r, a in enumerate(acts)]
        out = {"success": np.array([r[0] for r in res]), "steps": np.array([r[1] for r in res])}
    else:
        d = np.load(extra["file"])
        arr = d[key]
        if extra.get("n_points") is not None:  # 2a: first N points (i.i.d., so a random subset), all regenerations
            arr = arr[:, : extra["n_points"]]
        if extra.get("n_copies") is not None:  # 2b: all bases, first N copies
            arr = arr[:, :, : extra["n_copies"]]
        shape = arr.shape[:-2]
        out = exec_chunks(i, arr.reshape(-1, *arr.shape[-2:]))
        out = {k: v.reshape(*shape, *v.shape[1:]) for k, v in out.items()}
    if out_path is None:
        return time.time() - t0, None
    np.savez(out_path + ".tmp.npz", **out)
    os.replace(out_path + ".tmp.npz", out_path)
    return time.time() - t0, out_path


def build_tasks(args):
    tasks = []
    od = os.path.join(args.out_dir, args.stage)
    os.makedirs(od, exist_ok=True)
    if args.stage == "stage1":
        for f in sorted(glob.glob(os.path.join(args.act_dir, "s1_*.npz"))):
            i = int(os.path.basename(f)[3:6])
            tasks.append(("stage1", i, "samp", None, os.path.join(od, f"s1_{i:03d}.npz"), {"file": f}))
    elif args.stage == "stage3":
        for f in sorted(glob.glob(os.path.join(args.act_dir, "s3_*.npz"))):
            i = int(os.path.basename(f)[3:6])
            for key in args.s3_keys.split(","):
                npts = args.s3_points if key in ("a2a", "a2a_null") else None
                ncop = args.s3_copies if key in ("a2b_plan", "a2b_steer") else None
                tasks.append(("stage3", i, key, None, os.path.join(od, f"s3_{i:03d}_{key}.npz"), {"file": f, "n_points": npts, "n_copies": ncop}))
    elif args.stage == "stage4":
        for f in sorted(glob.glob(os.path.join(args.act_dir, "s4_*.npz"))):
            i = int(os.path.basename(f)[3:6])
            L, P4 = np.load(f)["actions"].shape[:2]
            for lv in range(L):
                for pt in range(P4):
                    tasks.append(("stage4", i, None, (lv, pt), os.path.join(od, f"s4_{i:03d}_{lv}_{pt:02d}.npz"), {"file": f}))
    return [t for t in tasks if not os.path.exists(t[4])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["stage1", "stage3", "stage4"])
    p.add_argument("--states", required=True)
    p.add_argument("--act_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_workers", type=int, default=32)
    # Design change 6 subsample is the default (review C3): a plain call cannot launch the ~740 core-h version
    p.add_argument("--s3_points", type=int, default=16, help="Stage 3: use only the first N 2a points (same for policy and null)")
    p.add_argument("--s3_copies", type=int, default=8, help="Stage 3: use only the first N 2b copies per base")
    p.add_argument("--s3_keys", default="a2a,a2a_null,a2b_plan", help="Stage 3: which action sets to execute")
    p.add_argument("--time_only", type=int, default=None, help="time this many chunks (stage1/3) or episodes (stage4) in one process")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    logf = open(os.path.join(args.out_dir, f"log_{args.stage}.txt"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log(f"=== sim_exec {time.ctime()} {vars(args)} {common.env_info()}")
    tasks = build_tasks(args)
    log(f"{len(tasks)} tasks to run")
    if args.time_only:
        _init(args.states)
        kind, i, key, sl, _, extra = tasks[0]
        t0 = time.time()
        if kind == "stage4":
            d = np.load(extra["file"])["actions"]
            for r in range(args.time_only):
                s, n = run_episode(i, d[0, 0, r % d.shape[2]], 123 + r)
                log(f"episode {r}: success={s} steps={n} t={time.time() - t0:.1f}s")
            per = (time.time() - t0) / args.time_only
            log(f"TIMING stage4: {per:.2f} s per episode (state {i}, t={int(_STATES['t'][i])})")
        else:
            arr = np.load(extra["file"])[key].reshape(-1, 8, 14)[: args.time_only]
            exec_chunks(i, arr)
            per = (time.time() - t0) / len(arr)
            log(f"TIMING {kind}: {per * 1000:.1f} ms per chunk (incl. restore); {per / 8 * 1000:.2f} ms per env step")
        return
    import multiprocessing as mp

    t0 = time.time()
    done = 0
    with mp.get_context("spawn").Pool(args.n_workers, initializer=_init, initargs=(args.states,)) as pool:
        for dt, path in pool.imap_unordered(task, tasks, chunksize=1):
            done += 1
            if done % max(1, len(tasks) // 50) == 0 or done == len(tasks):
                log(f"{done}/{len(tasks)} tasks, last {dt:.1f}s, elapsed {time.time() - t0:.0f}s")
    with open(os.path.join(args.out_dir, f"done_{args.stage}.json"), "w") as f:
        json.dump({"tasks": len(tasks), "seconds": time.time() - t0, "workers": args.n_workers}, f)
    log(f"=== {args.stage} done: {len(tasks)} tasks in {time.time() - t0:.0f}s with {args.n_workers} workers")


if __name__ == "__main__":
    main()
