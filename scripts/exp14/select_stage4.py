"""exp14 Stage 4 design (Design change 7): 20 ROLLOUT states (4 per phase; rollouts have t < 800, so the 800-step
budget is never exhausted at the start) with the largest spread of Q^A over their Stage 1 samples (outcome
undecided), excluding diverged states; and 5 DDIM levels where the median value R curve of those states is
closest to 0.1 / 0.3 / 0.5 / 0.7 / 0.9 (ties / duplicates move to the nearest unused step).

    python scripts/exp14/select_stage4.py --stage12 results/exp14/stage12 --states results/exp14/states/states.npz \
        --act_dir /n/netscratch/.../exp14/actions --out results/exp14/stage4_design.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402,F401


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage12", required=True)
    p.add_argument("--states", required=True)
    p.add_argument("--act_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per_phase", type=int, default=4)
    p.add_argument("--min_budget", type=int, default=80, help="require 800 - t >= this many env steps")
    args = p.parse_args()
    S = np.load(args.states)
    res = {}
    for f in sorted(glob.glob(os.path.join(args.stage12, "state_*.pkl"))):
        r = pickle.load(open(f, "rb"))
        s1 = np.load(os.path.join(args.act_dir, f"s1_{r['i']:03d}.npz"))["samp"]
        diverged = (not np.isfinite(s1).all()) or np.abs(s1).max() > 10 or np.isnan(np.asarray(r["pol"]["pc1"]["R"][:20])).any()
        res[r["i"]] = {"phase": r["phase"], "source": r["source"], "q_std": r["q1_std"], "diverged": diverged,
                       "R_value": np.asarray(r["pol"]["value"]["R"][:20]), "t": int(S["t"][r["i"]])}  # fmt: skip
    sel = []
    for ph in range(5):
        cand = [i for i, v in res.items() if v["phase"] == ph and v["source"] == 0 and not v["diverged"] and 800 - v["t"] >= args.min_budget]
        cand.sort(key=lambda i: -res[i]["q_std"])
        sel += cand[: args.per_phase]
    med = np.nanmedian(np.stack([res[i]["R_value"] for i in sel]), 0)
    ks = []
    for target in (0.1, 0.3, 0.5, 0.7, 0.9):
        order = np.argsort(np.abs(med - target))
        ks.append(int(next(k for k in order if k not in ks)))
    ks = sorted(ks)
    out = {"states": sorted(int(i) for i in sel), "ks": ks, "median_value_R": med.tolist(),
           "per_state": {int(i): {"phase": res[i]["phase"], "t": res[i]["t"], "q_std": res[i]["q_std"]} for i in sel}}  # fmt: skip
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"states": out["states"], "ks": ks, "median_value_R_at_ks": [round(float(med[k]), 3) for k in ks]}))


if __name__ == "__main__":
    main()
