"""exp14 state collection (CPU): base-policy rollouts, demo states, env checks, and the 330-state selection.

    python scripts/exp14/collect_states.py --out_dir results/exp14/states --n_workers 48 --n_episodes 192

Outputs (out_dir):
  rollouts.npz   every chunk boundary of every rollout: obs, obs_raw, sim_state, episode, chunk, t, episode_success
  demos.npz      every 8th step of every demo (+ t = 0): obs_raw (dataset), sim_state, demo, t
  checks.json    demo obs vs env obs after restore; restore determinism; action layout
  states.npz     the 330 selected states (obs as the policy sees it in the env after restoring the state)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

N_PER_PHASE_SRC = 30
N_AMBIG_SRC = 15


def load_policy_cpu():
    import torch

    torch.set_num_threads(1)
    from adsrl.model_utils import build_model, load_task_cfg

    cfg = load_task_cfg("transport", os.path.join(common.PROJECT, "cfg", "adsrl", "transport.yaml"))
    return build_model(cfg, device="cpu", ddim_steps=20)


def rollout_worker(args):
    wid, seeds = args
    import torch

    import transport_env as TE
    import windows as W

    torch.set_num_threads(1)
    model = load_policy_cpu()
    sim = TE.TransportSim()
    rows = []
    g = torch.Generator().manual_seed(10_000 + wid)
    for ep_seed in seeds:
        obs, raw = sim.reset_seeded(ep_seed)
        t, succ, ep_rows = 0, False, []
        while t < TE.MAX_STEPS:
            ep_rows.append(dict(obs=obs, obs_raw=raw, sim_state=sim.get_state(), grip=sim.get_grip(), episode=ep_seed, chunk=t // 8, t=t))
            s = torch.as_tensor(obs)[None]
            x_T = torch.randn((1, 8, 14), generator=g)
            a = W.ddim_sample(model, {"state": s}, x_T)[0].numpy()
            obs, raw, succ, n = sim.step_chunk(a)
            t += n
            if succ:
                break
        for r in ep_rows:
            r["episode_success"] = succ
        rows.extend(ep_rows)
        print(f"[w{wid}] episode seed {ep_seed}: success={succ} steps={t}", flush=True)
    return rows


def demo_rows(n_substeps, stride=8):
    """Every `stride`-th step of every demo; gripper current_action reconstructed by replaying actions[:t]."""
    import h5py

    import transport_env as TE

    rows = []
    with h5py.File(common.DATASET, "r") as f:
        for name in sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1])):
            g = f["data"][name]
            raw = np.concatenate([g["obs"][k][()].reshape(len(g["states"]), -1) for k in TE.LOW_DIM_KEYS], axis=1)
            st = g["states"][()]
            grips = TE.replay_grip_traj(g["actions"][()], n_substeps)  # grips[t] = state after actions[:t]
            for t in range(0, len(st), stride):
                rows.append(dict(obs_raw=raw[t], sim_state=st[t], grip=grips[t],
                                 demo=int(name.split("_")[1]), t=t, demo_len=len(st)))  # fmt: skip
    return rows


def checks(sim, demos, rng, n=200):
    """Demo obs vs env obs after restore; gripper-state replay vs live env; restore + step determinism; action
    layout (gripper dims 6 / 13, arm dims 0 / 7)."""
    import h5py

    import transport_env as TE

    norm = np.load(common.NORM)
    out = {}
    idx = rng.choice(len(demos), size=min(n, len(demos)), replace=False)
    d_raw, d_norm = [], []
    for i in idx:
        r = demos[i]
        obs_env, raw_env = sim.restore(r["sim_state"], r["grip"])
        d_raw.append(np.abs(raw_env - r["obs_raw"]))
        d_norm.append(np.abs(obs_env - sim.norm(r["obs_raw"])))
    d_raw, d_norm = np.array(d_raw), np.array(d_norm)
    out["demo_vs_env_raw_maxabs"] = float(d_raw.max())
    out["demo_vs_env_raw_maxabs_per_dim"] = d_raw.max(0).round(6).tolist()
    out["demo_vs_env_norm_maxabs"] = float(d_norm.max())
    out["demo_vs_env_norm_p99"] = float(np.percentile(d_norm, 99))
    out["n_checked"] = len(idx)
    # dppo train.npz states vs dataset obs normalized (are the demos the policy's training data?)
    tr = np.load(os.path.join(common.DSRL, "dppo", "log", "robomimic", "transport", "train.npz"))
    trs = tr["states"].astype(np.float64)
    if np.abs(trs).max() > 1.5:
        trs = 2 * ((trs - norm["obs_min"]) / (norm["obs_max"] - norm["obs_min"] + 1e-6) - 0.5)
    sub = np.stack([sim.norm(demos[i]["obs_raw"]) for i in idx[:50]])
    nn = np.array([np.abs(trs - s).max(1).min() for s in sub])
    out["demo_obs_nearest_train_npz_maxabs_median"] = float(np.median(nn))
    out["train_npz_n"] = int(len(trs))
    out["norm_obs_range_demo_states"] = [float(sub.min()), float(sub.max())]
    # gripper current_action: replay formula vs the live env, executing demo 0's actions from its first state
    with h5py.File(common.DATASET, "r") as f:
        g = f["data"]["demo_0"]
        st0, acts = g["states"][0], g["actions"][()]
    sim.restore(st0, np.zeros(4))
    errs = []
    rep = TE.replay_grip_traj(acts[:300], sim.n_substeps)
    for t, a in enumerate(acts[:300]):
        sim.renv.step(a)
        errs.append(np.abs(sim.get_grip() - rep[t + 1]).max())
    out["grip_replay_vs_live_maxabs"] = float(max(errs))
    out["n_substeps"] = sim.n_substeps
    # determinism: restore + same chunk twice (incl. a chunk run in between, to dirty the hidden state)
    r0 = demos[int(idx[0])]
    chunk = rng.uniform(-0.5, 0.5, (8, 14)).astype(np.float32)
    other = rng.uniform(-1, 1, (8, 14)).astype(np.float32)
    sim.restore(r0["sim_state"], r0["grip"])
    _, r1, _, _ = sim.step_chunk(chunk)
    s1 = sim.get_state()
    sim.step_chunk(other)
    sim.restore(r0["sim_state"], r0["grip"])
    _, r2, _, _ = sim.step_chunk(chunk)
    s2 = sim.get_state()
    out["determinism_raw_maxabs"] = float(np.abs(r1 - r2).max())
    out["determinism_state_maxabs"] = float(np.abs(s1 - s2).max())
    # same without restoring the gripper state (documents the bug this fixes)
    sim.restore(r0["sim_state"], r0["grip"])
    sim.step_chunk(other)
    od = sim.renv.reset_to({"states": np.asarray(r0["sim_state"], np.float64)})
    _, r3, _, _ = sim.step_chunk(chunk)
    out["determinism_without_grip_restore_raw_maxabs"] = float(np.abs(r1 - r3).max())
    # save / restore round trip
    _, raw0 = sim.restore(r0["sim_state"], r0["grip"])
    s_saved, g_saved = sim.get_state(), sim.get_grip()
    sim.step_chunk(chunk)
    _, raw_back = sim.restore(s_saved, g_saved)
    out["save_restore_raw_maxabs"] = float(np.abs(raw_back - raw0).max())
    # action layout, from a demo's first state with both grippers' internal state at 0
    st = demos[0]["sim_state"]
    lay = {}
    for dim in (6, 13):
        _, rb = sim.restore(st, np.zeros(4))
        c = np.zeros((8, 14), np.float32)
        c[:, dim] = 1.0
        _, ra, _, _ = sim.step_chunk(c)
        fb, fa = TE.feats(rb), TE.feats(ra)
        lay[f"dim{dim}=+1"] = {"g0_gap_change": float(fa["g0_gap"][0] - fb["g0_gap"][0]), "g1_gap_change": float(fa["g1_gap"][0] - fb["g1_gap"][0]),
                               "grip_state": sim.get_grip().round(3).tolist()}  # fmt: skip
    for dim in (0, 7):
        _, rb = sim.restore(st, np.zeros(4))
        c = np.zeros((8, 14), np.float32)
        c[:, dim] = 1.0
        _, ra, _, _ = sim.step_chunk(c)
        fb, fa = TE.feats(rb), TE.feats(ra)
        lay[f"dim{dim}=+1"] = {"eef0_disp": (fa["eef0"] - fb["eef0"])[0].round(4).tolist(), "eef1_disp": (fa["eef1"] - fb["eef1"])[0].round(4).tolist()}
    out["action_layout"] = lay
    # timing: restore + one chunk, and one policy call (CPU, 1 thread)
    import time as _t

    import torch

    t0 = _t.time()
    for _ in range(30):
        sim.restore(r0["sim_state"], r0["grip"])
        sim.step_chunk(chunk)
    out["sec_per_restore_plus_chunk"] = (_t.time() - t0) / 30
    model = load_policy_cpu()
    import windows as W

    o = torch.as_tensor(sim.restore(r0["sim_state"], r0["grip"])[0])[None]
    t0 = _t.time()
    for _ in range(10):
        W.ddim_sample(model, {"state": o}, torch.randn(1, 8, 14))
    out["sec_per_policy_call_cpu"] = (_t.time() - t0) / 10
    out["robot_action_dims"] = [int(r.action_dim) for r in sim.rs.robots]
    return out


def select(roll, demos, sim, rng, log):
    import transport_env as TE

    # phase labels
    ep = roll["episode"]
    first = {e: i for i, (e, c) in enumerate(zip(ep, roll["chunk"])) if c == 0}
    r0 = roll["obs_raw"][[first[e] for e in ep]]
    ph_r = TE.phase_of(roll["obs_raw"], r0, roll["t"])
    d_raw = np.stack([d["obs_raw"] for d in demos])
    d_demo = np.array([d["demo"] for d in demos])
    d_t = np.array([d["t"] for d in demos])
    d_first = {dm: i for i, (dm, t) in enumerate(zip(d_demo, d_t)) if t == 0}
    d_r0 = d_raw[[d_first[dm] for dm in d_demo]]
    ph_d = TE.phase_of(d_raw, d_r0, d_t)
    counts = {"rollout": np.bincount(ph_r + 1, minlength=6).tolist(), "demo": np.bincount(ph_d + 1, minlength=6).tolist()}
    log(f"phase counts (index 0 = unlabeled, then {TE.PHASES}): {counts}")

    def pick(pool_idx, groups, n, cap=2):
        """n indices from pool, at most `cap` per group (episode / demo), random order."""
        pool_idx = rng.permutation(pool_idx)
        used, out = {}, []
        for c in (cap, 10**9):
            for i in pool_idx:
                if len(out) >= n:
                    break
                if i in out or used.get(groups[i], 0) >= c:
                    continue
                out.append(int(i))
                used[groups[i]] = used.get(groups[i], 0) + 1
        return out

    sel = []  # (source, index, phase)
    shortfall = {}
    for p in range(5):
        r_idx = pick(np.where(ph_r == p)[0], ep, N_PER_PHASE_SRC)
        d_idx = pick(np.where(ph_d == p)[0], d_demo, N_PER_PHASE_SRC)
        if len(r_idx) < N_PER_PHASE_SRC:  # fill from demos
            extra = [i for i in pick(np.where(ph_d == p)[0], d_demo, 2 * N_PER_PHASE_SRC) if i not in d_idx]
            d_idx += extra[: N_PER_PHASE_SRC - len(r_idx)]
            shortfall[TE.PHASES[p]] = {"rollout_short_by": N_PER_PHASE_SRC - len(r_idx)}
        if len(d_idx) < 2 * N_PER_PHASE_SRC - len(r_idx):
            extra = [i for i in pick(np.where(ph_r == p)[0], ep, 2 * N_PER_PHASE_SRC) if i not in r_idx]
            r_idx += extra[: 2 * N_PER_PHASE_SRC - len(r_idx) - len(d_idx)]
            shortfall.setdefault(TE.PHASES[p], {})["demo_short"] = True
        sel += [(0, i, p) for i in r_idx] + [(1, i, p) for i in d_idx]
    amb_r = pick(np.where(roll["t"] == 0)[0], ep, N_AMBIG_SRC, cap=1)
    amb_d = pick(np.where(d_t == 0)[0], d_demo, N_AMBIG_SRC, cap=1)
    sel += [(0, i, 5) for i in amb_r] + [(1, i, 5) for i in amb_d]
    log(f"selected {len(sel)} states; shortfall {shortfall}")

    out = {k: [] for k in ["obs", "obs_raw", "sim_state", "grip", "source", "phase", "group", "t", "episode_success", "restore_obs_maxabs"]}
    for src, i, p in sel:
        if src == 0:
            st, gr, raw_rec, t, g, es = roll["sim_state"][i], roll["grip"][i], roll["obs_raw"][i], roll["t"][i], roll["episode"][i], roll["episode_success"][i]
        else:
            st, gr, raw_rec, t, g, es = demos[i]["sim_state"], demos[i]["grip"], demos[i]["obs_raw"], demos[i]["t"], demos[i]["demo"], True
        obs, raw = sim.restore(st, gr)  # what the policy sees in our env at this state
        out["obs"].append(obs)
        out["obs_raw"].append(raw)
        out["sim_state"].append(st)
        out["grip"].append(gr)
        out["source"].append(src)
        out["phase"].append(p)
        out["group"].append(g)
        out["t"].append(t)
        out["episode_success"].append(es)
        out["restore_obs_maxabs"].append(float(np.abs(raw - raw_rec).max()))
    res = {k: np.array(v) for k, v in out.items()}
    return res, counts, shortfall, ph_r, ph_d


def add_init_joint(out_dir, sim, log):
    """Review C1: each state's OSC nullspace target = its episode's own t = 0 arm qpos (rollouts: the episode's
    chunk-0 sim_state; demos: the demo's states[0]). Adds key init_joint (330, 14) to states.npz (backup kept)."""
    import shutil

    import h5py

    path = os.path.join(out_dir, "states.npz")
    S = dict(np.load(path))
    if "init_joint" in S:
        log("states.npz already has init_joint")
        return
    shutil.copy(path, path.replace(".npz", "_before_init_joint.npz"))
    roll = np.load(os.path.join(out_dir, "rollouts.npz"))
    first = {int(e): i for i, (e, c) in enumerate(zip(roll["episode"], roll["chunk"])) if c == 0}
    ij = []
    with h5py.File(common.DATASET, "r") as f:
        for i in range(len(S["phase"])):
            g = int(S["group"][i])
            st0 = roll["sim_state"][first[g]] if S["source"][i] == 0 else f["data"][f"demo_{g}"]["states"][0]
            ij.append(sim.arm_qpos(st0))
    S["init_joint"] = np.array(ij)
    # sanity: for t = 0 states, init_joint equals the state's own arm qpos
    t0 = np.where(S["t"] == 0)[0]
    err = max(np.abs(S["init_joint"][i] - sim.arm_qpos(S["sim_state"][i])).max() for i in t0)
    np.savez_compressed(path, **S)
    log(f"init_joint added for {len(ij)} states; t=0 states self-consistency max |d| = {err:.2e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_workers", type=int, default=48)
    p.add_argument("--n_episodes", type=int, default=192)
    p.add_argument("--seed_base", type=int, default=140_000)
    p.add_argument("--steps", default="rollouts,demos,checks,select")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    logf = open(os.path.join(args.out_dir, "log.txt"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log(f"=== collect_states {time.ctime()} {vars(args)} {common.env_info()}")
    import multiprocessing as mp

    import transport_env as TE

    steps = args.steps.split(",")
    rng = np.random.default_rng(14)
    t0 = time.time()
    rpath = os.path.join(args.out_dir, "rollouts.npz")
    if "rollouts" in steps:
        seeds = [args.seed_base + i for i in range(args.n_episodes)]
        chunks = [(w, seeds[w :: args.n_workers]) for w in range(args.n_workers)]
        with mp.get_context("spawn").Pool(args.n_workers) as pool:
            res = pool.map(rollout_worker, chunks)
        rows = [r for rr in res for r in rr]
        L = max(len(r["sim_state"]) for r in rows)
        roll = {k: np.array([r[k] for r in rows]) for k in ["obs", "obs_raw", "grip", "episode", "chunk", "t", "episode_success"]}
        roll["sim_state"] = np.stack([r["sim_state"] for r in rows])
        np.savez_compressed(rpath, **roll)
        succ = np.array([roll["episode_success"][roll["episode"] == e][0] for e in np.unique(roll["episode"])])
        log(f"rollouts: {len(np.unique(roll['episode']))} episodes, success {succ.mean():.3f}, {len(rows)} chunk states, state len {L}, {time.time() - t0:.0f}s")
    roll = dict(np.load(rpath))
    sim = TE.TransportSim()
    demos = demo_rows(sim.n_substeps)
    log(f"demos: {len(demos)} rows from {len(set(d['demo'] for d in demos))} demos")
    if "checks" in steps:
        t1 = time.time()
        ch = checks(sim, demos, rng)
        with open(os.path.join(args.out_dir, "checks.json"), "w") as f:
            json.dump(ch, f, indent=1)
        log(f"checks ({time.time() - t1:.0f}s): " + json.dumps({k: v for k, v in ch.items() if "per_dim" not in k}))
    if "select" in steps:
        res, counts, shortfall, ph_r, ph_d = select(roll, demos, sim, rng, log)
        np.savez_compressed(os.path.join(args.out_dir, "states.npz"), **res)
        np.save(os.path.join(args.out_dir, "rollout_phases.npy"), ph_r)
        np.save(os.path.join(args.out_dir, "demo_phases.npy"), ph_d)
        with open(os.path.join(args.out_dir, "selection.json"), "w") as f:
            json.dump({"phase_counts": counts, "shortfall": shortfall, "n": int(len(res["phase"])),
                       "per_phase_source": {f"{ph}_{s}": int(((res['phase'] == ph) & (res['source'] == s)).sum()) for ph in range(6) for s in (0, 1)},
                       "restore_obs_maxabs_max": float(res["restore_obs_maxabs"].max())}, f, indent=1)  # fmt: skip
        log(f"states.npz written: {len(res['phase'])} states, restore max|raw diff| {res['restore_obs_maxabs'].max():.2e}")
    if "init_joint" in steps:
        add_init_joint(args.out_dir, sim, log)
    log(f"=== done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
