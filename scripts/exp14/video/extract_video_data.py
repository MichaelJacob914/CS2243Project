"""Small JSON of per-state numbers for the explainer video (widths vs control, midpoints by phase, example curves).

Reads results/exp14/stage12/state_*.pkl and analysis.json the same way replot.py does (drops diverged states);
writes results/exp14/anim/video_data.json. Run with the project .venv.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import analyze as A  # noqa: E402
import common  # noqa: E402

ROOT = common.PROJECT


def main():
    a = json.load(open(os.path.join(ROOT, "results/exp14/analysis/analysis.json")))
    bad = {d["i"] for d in a["diverged_states"]}
    res = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results/exp14/stage12/state_*.pkl"))):
        with open(f, "rb") as fh:
            r = pickle.load(fh)
        if r["i"] not in bad:
            res.append(r)
    assert len(res) == a["n_states"]
    out = {"width": {}, "mid": {}, "examples": {}}
    for f in A.PRIMARY:
        rows = []
        for r in res:
            if f in r["pol"]:
                wp, wn = np.nanmedian(r["pol"][f]["boot_width"]), np.nanmedian(r["null"][f]["boot_width"])
                rows.append([int(r["i"]), int(r["phase"]), float(wn), float(wp)])
        out["width"][f] = rows
        out["mid"][f] = [list(x) for x in A.midpoints_k(res, f) if np.isfinite(x[2])]
    # example action-cluster curves: states where the policy is narrower, by phase
    r1 = {}
    import csv

    with open(os.path.join(ROOT, "results/exp14/analysis/rule1_rows.csv")) as fh:
        for row in csv.DictReader(fh):
            if row["feature"] == "cluster_label":
                r1[int(row["i"])] = row["narrower"] == "True"
    for r in res:
        if "cluster_label" in r["pol"] and r1.get(int(r["i"])):
            out["examples"][str(r["i"])] = {"phase": int(r["phase"]), "pol": np.asarray(r["pol"]["cluster_label"]["R"][:20]).tolist(),
                                            "null": np.asarray(r["null"]["cluster_label"]["R"][:20]).tolist(),
                                            "pol_val": np.asarray(r["pol"]["value"]["R"][:20]).tolist(),
                                            "null_val": np.asarray(r["null"]["value"]["R"][:20]).tolist()}  # fmt: skip
    out["logsnr"] = A.logsnr_levels().tolist()
    os.makedirs(os.path.join(ROOT, "results/exp14/anim"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results/exp14/anim/video_data.json"), "w"))
    print({f: (len(out["width"][f]), len(out["mid"][f])) for f in A.PRIMARY}, len(out["examples"]))


if __name__ == "__main__":
    main()
