"""Restore tests for the exp14 simulator executor (review items B1, C1, B2). Needs a compute node (robosuite).

B1  restore + chunk on a FRESHLY constructed env (and right after an env reset) must equal a repeat restore + chunk
    (the OSC controller caches eef pose / Jacobian; a stale cache shifts the first chunk by 26-63 mm).
C1  restore must set the OSC nullspace target controller.initial_joint to the episode's own t = 0 arm qpos, so the
    result does not depend on what the worker reset last.
B2  run_episode must refuse a state with less than one chunk of budget left (800 - t < 8).

    python scripts/exp14/test_restore.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402
import transport_env as TE  # noqa: E402

P = common.PROJECT
ACT = os.path.join(common.SCRATCH, "actions")
S = dict(np.load(os.path.join(P, "results/exp14/states/states.npz")))
EEF = np.r_[0:3, 9:12]
OBJ = np.r_[18:21, 25:28, 32:35]
STATES = [70, 76, 165, 238, 183]  # low repeat-noise states from the reviewer's run first
TOL_M = 1e-6  # 1 micrometre; BLAS-level repeat noise after one chunk is <= 3e-7 m


def init_joint_of(i):
    if "init_joint" in S:
        return S["init_joint"][i]
    return None


def restore(sim, i):
    ij = init_joint_of(i)
    try:
        return sim.restore(S["sim_state"][i], S["grip"][i], init_joint=ij)
    except TypeError:  # pre-fix API
        return sim.restore(S["sim_state"][i], S["grip"][i])


def run(sim, i, chunks):
    restore(sim, i)
    out = []
    for c in chunks:
        _, r1, _, _ = sim.step_chunk(np.clip(c, -1, 1))
        out.append(r1)
    return np.stack(out)


def dmax(a, b, c):
    return float(max(np.abs(a[c, EEF] - b[c, EEF]).max(), np.abs(a[c, OBJ] - b[c, OBJ]).max()))


def test_fresh_env_and_after_reset():
    fails = []
    for i in STATES:
        chunks = np.load(os.path.join(ACT, f"s1_{i:03d}.npz"))["samp"][:1]
        sim = TE.TransportSim()  # fresh env: controller cache from the construction-time reset
        A = run(sim, i, chunks)
        B = run(sim, i, chunks)
        np.random.seed(1000 + i)
        sim.renv.reset()  # a reset in between (as a worker does when it moves between tasks)
        C = run(sim, i, chunks)
        d_ab, d_ac = dmax(A, B, 0), dmax(A, C, 0)
        print(f"state {i}: fresh-vs-repeat {d_ab * 1000:.3g} mm, after-reset-vs-repeat {d_ac * 1000:.3g} mm")
        if d_ab > TOL_M or d_ac > TOL_M:
            fails.append((i, d_ab, d_ac))
    assert not fails, f"first chunk after construction / reset differs: {fails}"


def test_initial_joint_restored():
    sim = TE.TransportSim()
    fails = []
    for i in STATES:
        restore(sim, i)
        ij = np.concatenate([r.controller.initial_joint for r in sim.rs.robots])
        want = init_joint_of(i)
        if want is None or np.abs(ij - want).max() > 1e-12:
            fails.append(i)
    assert not fails, f"controller.initial_joint not set from the episode's t=0 qpos at states {fails}"
    # 5-chunk equality across a reset that re-randomizes initial_joint (low-noise states)
    for i in STATES[:3]:
        chunks = np.load(os.path.join(ACT, f"s1_{i:03d}.npz"))["samp"][:5]
        A = run(sim, i, chunks)
        np.random.seed(5000 + i)
        sim.renv.reset()
        B = run(sim, i, chunks)
        d = dmax(A, B, 4)
        print(f"state {i}: 5 chunks across a reset {d * 1000:.3g} mm")
        assert d < 1e-5, (i, d)


def test_budget_guard():
    import sim_exec

    sim_exec._init(os.path.join(P, "results/exp14/states/states.npz"))
    late = [i for i in range(len(S["t"])) if TE.MAX_STEPS - int(S["t"][i]) < 8]
    assert late, "no late state to test with"
    try:
        sim_exec.run_episode(late[0], np.zeros((8, 14), np.float32), 0)
    except AssertionError as e:
        print(f"budget guard OK at state {late[0]} (t = {S['t'][late[0]]}): {e}")
        return
    raise AssertionError(f"run_episode accepted state {late[0]} with t = {S['t'][late[0]]}")


if __name__ == "__main__":
    fails = []
    for t in (test_fresh_env_and_after_reset, test_initial_joint_restored, test_budget_guard):
        try:
            t()
            print(f"PASS {t.__name__}", flush=True)
        except Exception as e:
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}", flush=True)
            fails.append(t.__name__)
    print("ALL OK" if not fails else f"FAILED: {fails}")
    sys.exit(1 if fails else 0)
