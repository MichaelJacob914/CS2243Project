"""Regression test: analyze.py must skip excluded (diverged) states in Stage 3 and Stage 4 (memorization() uses
the same include filter but is not covered here, since it needs the training data),
the same way they are excluded everywhere else (job 59860365 failed with KeyError: 72 in stage3).

Synthetic dirs: sim/stage3 and act_dir hold files for an included state (5) and an excluded state (72);
res_by_i holds only state 5.

    python scripts/exp14/test_analyze_exclusion.py   (CPU, seconds; needs sklearn from the exp14 pylib)
"""

from __future__ import annotations

import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze  # noqa: E402

L, P, M = 20, 4, 3
INCLUDED, EXCLUDED = 5, 72


def sim3(rng):
    return {"disp": rng.normal(size=(L, P, M, 9)).astype(np.float32), "eef_disp": rng.normal(size=(L, P, M, 6)).astype(np.float32),
            "toward0": rng.integers(0, 3, (L, P, M)).astype(np.int8), "toward1": rng.integers(0, 3, (L, P, M)).astype(np.int8),
            "succ": np.zeros((L, P, M), bool)}  # fmt: skip


def make_dirs(root, rng):
    sim, act = os.path.join(root, "sim"), os.path.join(root, "act")
    for d in (os.path.join(sim, "stage3"), os.path.join(sim, "stage4"), os.path.join(sim, "stage1"), act):
        os.makedirs(d)
    ks = np.array([4, 9, 12, 15, 18])
    for i in (INCLUDED, EXCLUDED):
        np.savez(os.path.join(sim, "stage3", f"s3_{i:03d}_a2a.npz"), **sim3(rng))
        np.savez(os.path.join(sim, "stage3", f"s3_{i:03d}_a2a_null.npz"), **sim3(rng))
        np.savez(os.path.join(act, f"s4_{i:03d}.npz"), q=rng.normal(size=(5, 2, 8)), ks=ks)
        for lv in range(5):
            for pt in range(2):
                np.savez(os.path.join(sim, "stage4", f"s4_{i:03d}_{lv}_{pt:02d}.npz"),
                         success=rng.random(8) < 0.5, steps=np.full(8, 100))  # fmt: skip
    res_by_i = {INCLUDED: {"i": INCLUDED, "phase": 0, "pol": {"value": {"R": np.linspace(0, 1, 21)}}}}
    return sim, act, res_by_i


SCRATCH = os.environ.get("TMPDIR_EXP14", "/n/netscratch/olveczky_lab/Lab/charleszhang/exp14/tmp")


def fresh(name):
    """A fresh directory under exp14 scratch (TemporaryDirectory cleanup fails on Lustre and masks errors)."""
    root = os.path.join(SCRATCH, name)
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    return root


def test_stage3_skips_excluded():
    rng = np.random.default_rng(0)
    if True:
        sim, act, res_by_i = make_dirs(fresh("t_stage3"), rng)
        out = analyze.stage3(res_by_i, sim, rng, np.linspace(-5.5, 7.4, 20))
        rows, curves = out[0], out[1]
        skipped = out[2] if len(out) > 2 else None
        assert {r["i"] for r in rows} == {INCLUDED}, {r["i"] for r in rows}
        assert set(curves) == {INCLUDED}, set(curves)
        assert skipped == [EXCLUDED], skipped
        print(f"stage3: rows only for {INCLUDED}, skipped {skipped}")


def test_stage4_skips_excluded():
    rng = np.random.default_rng(1)
    if True:
        sim, act, res_by_i = make_dirs(fresh("t_stage4"), rng)
        out = analyze.stage4(sim, act, res_by_i, None)
        assert set(out["states"]) == {INCLUDED}, set(out["states"])
        assert out.get("skipped_states") == [EXCLUDED], out.get("skipped_states")
        assert out["pooled"]["n_episodes"] == 5 * 2 * 8, out["pooled"]["n_episodes"]
        print(f"stage4: states {sorted(out['states'])}, skipped {out['skipped_states']}, pooled episodes {out['pooled']['n_episodes']}")


if __name__ == "__main__":
    fails = []
    for t in (test_stage3_skips_excluded, test_stage4_skips_excluded):
        try:
            t()
            print(f"PASS {t.__name__}", flush=True)
        except Exception as e:
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}", flush=True)
            fails.append(t.__name__)
    print("ALL OK" if not fails else f"FAILED: {fails}")
    sys.exit(1 if fails else 0)
