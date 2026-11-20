"""exp13 Phase 0B: pool the paired rollout arms and the per-state headroom over uniformk seeds 1-3.

    python scripts/exp13_pool_seeds.py   # writes results/exp13/pooled_seeds.json
"""

import csv
import json
import os
from math import comb

import numpy as np

R = "results/exp13"
ARMS = ["i_allk_N32", "ii_k0_N640", "iii_stoch", "iv_det"]
ARM_DIRS = {1: f"{R}/headroom_paired", 2: f"{R}/headroom_paired_s2", 3: f"{R}/headroom_paired_s3"}
HEAD_DIRS = {1: f"{R}/headroom", 2: f"{R}/headroom_paired_s2", 3: f"{R}/headroom_paired_s3"}
PAIRS = [("i_allk_N32", "ii_k0_N640"), ("i_allk_N32", "iii_stoch"), ("ii_k0_N640", "iii_stoch"), ("iii_stoch", "iv_det")]


def mcnemar_p(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n) if n else 1.0


out = {"arms": {}, "paired": {}, "headroom": {}}
succ = {}
for s, d in ARM_DIRS.items():
    rows = {r["arm"]: r for r in csv.DictReader(open(f"{d}/arms.csv"))}
    for a in ARMS:
        succ.setdefault(a, {})[s] = float(rows[a]["success"])
    out["paired"][s] = json.load(open(f"{d}/paired_tests.json"))
for a in ARMS:
    v = np.array(list(succ[a].values()))
    p = v.mean()
    out["arms"][a] = {"per_seed": succ[a], "pooled": float(p), "se_binom_600": float(np.sqrt(p * (1 - p) / 600)),
                      "seed_sd": float(v.std(ddof=1))}  # fmt: skip
for x, y in PAIRS:
    k = f"{x}_vs_{y}"
    b = sum(out["paired"][s][k]["x_only"] for s in ARM_DIRS)
    c = sum(out["paired"][s][k]["y_only"] for s in ARM_DIRS)
    diffs = [succ[x][s] - succ[y][s] for s in ARM_DIRS]
    out["paired"][f"pooled_{k}"] = {"x_only": b, "y_only": c, "p_exact_mcnemar": mcnemar_p(b, c),
                                    "diff_pooled": float(np.mean(diffs)), "diff_per_seed": diffs}  # fmt: skip

for s, d in HEAD_DIRS.items():
    hs = json.load(open(f"{d}/headroom_summary.json"))
    out["headroom"][s] = {k: hs[k] for k in ["argmax_k_hist", "frac_argmax_k0", "frac_argmax_k_le3", "argmax_k_mean",
                                             "split_half_argmax_agreement", "split_half_chance_agreement",
                                             "gap_best_minus_k0", "gap_cv_halves", "gap_over_std0",
                                             "perm_test_argmaxk_vs_phase", "perm_test_argmaxk_vs_tau_decile"]}  # fmt: skip
    out["headroom"][s]["by_phase"] = {p: {k: v[k] for k in ["n", "argmax_k_median", "frac_argmax_k_ge10", "gap_cv_median"]}
                                      for p, v in hs["by_phase"].items()}  # fmt: skip
H = out["headroom"]
out["headroom"]["pooled"] = {
    "argmax_k_hist": np.sum([H[s]["argmax_k_hist"] for s in HEAD_DIRS], 0).tolist(),
    "split_half_agreement_mean": float(np.mean([H[s]["split_half_argmax_agreement"] for s in HEAD_DIRS])),
    "split_half_chance_mean": float(np.mean([H[s]["split_half_chance_agreement"] for s in HEAD_DIRS])),
    "gap_median_mean": float(np.mean([H[s]["gap_best_minus_k0"]["median"] for s in HEAD_DIRS])),
    "gap_cv_median_mean": float(np.mean([H[s]["gap_cv_halves"]["median"] for s in HEAD_DIRS])),
    "gap_over_std0_median_mean": float(np.mean([H[s]["gap_over_std0"]["median"] for s in HEAD_DIRS])),
}
json.dump(out, open(f"{R}/pooled_seeds.json", "w"), indent=2)
for a in ARMS:
    print(a, out["arms"][a])
for x, y in PAIRS:
    print(x, y, out["paired"][f"pooled_{x}_vs_{y}"])
for s in list(HEAD_DIRS) + ["pooled"]:
    print("headroom", s, json.dumps({k: v for k, v in H[s].items() if k != "by_phase"}))
    if s != "pooled":
        print("   by_phase", json.dumps(H[s]["by_phase"]))
