"""Redraw the exp14 report figures from saved analysis outputs, without rerunning the analysis.

    python scripts/exp14/replot.py

Reads results/exp14/stage12/state_*.pkl, results/exp14/analysis/{analysis.json, rule1_rows.csv}; drops the
diverged states listed in analysis.json, as analyze.py does; writes report/assets/figs/exp14_*.png via analyze.figures.
"""

from __future__ import annotations

import ast
import csv
import glob
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze as A  # noqa: E402
import common  # noqa: E402

ROOT = common.PROJECT
OUT = os.path.join(ROOT, "results", "exp14", "analysis")


def _cell(v):
    if v in ("True", "False"):
        return v == "True"
    try:
        return float(v)
    except ValueError:
        return ast.literal_eval(v.replace("nan", "None")) if v.startswith("(") else v


def main():
    a = json.load(open(os.path.join(OUT, "analysis.json")))
    bad = {d["i"] for d in a["diverged_states"]}
    res = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "exp14", "stage12", "state_*.pkl"))):
        with open(f, "rb") as fh:
            r = pickle.load(fh)
        if r["i"] not in bad:
            res.append(r)
    assert len(res) == a["n_states"], (len(res), a["n_states"])
    with open(os.path.join(OUT, "rule1_rows.csv")) as fh:
        r1rows = [{k: _cell(v) for k, v in row.items()} for row in csv.DictReader(fh)]
    for row in r1rows:
        row["phase"] = int(row["phase"])
    figs = os.path.join(ROOT, "report", "assets", "figs")
    A.figures(res, r1rows, a["rule2"], a["rule3"], {int(k): v for k, v in a["rl_gains"].items()}, a["stage4"], figs, A.logsnr_levels())
    print("wrote", sorted(glob.glob(os.path.join(figs, "exp14_*.png"))))


if __name__ == "__main__":
    main()
