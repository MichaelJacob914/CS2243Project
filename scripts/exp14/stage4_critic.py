"""exp14 Stage 4 critic check, per level (post-hoc descriptive; not a preregistered rule).

For each Stage 4 level k (inside / around the measured value window) and state:
  - within-state AUC of Q^A(s, a) for episode success (does the critic rank regenerated chunks by success?),
  - success rate, and the between-point share of success variance (R_success) vs that of Q^A (R_value),
  - mean success of the chunks the critic ranks in its top quarter vs the rest (steering-by-critic gain).
Excluded (diverged) states are skipped and listed, as in analyze.py.

    python scripts/exp14/stage4_critic.py --analysis results/exp14/analysis/analysis.json --sim results/exp14/sim \
        --act_dir /n/netscratch/.../exp14/actions --out results/exp14/analysis/stage4_critic.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402,F401
import windows as W  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", required=True)
    p.add_argument("--sim", required=True)
    p.add_argument("--act_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_boot", type=int, default=2000)
    args = p.parse_args()
    from sklearn.metrics import roc_auc_score

    a = json.load(open(args.analysis))
    excluded = {d["i"] for d in a.get("diverged_states", [])} if not a.get("keep_diverged") else set()
    per_level, skipped, states = {}, [], []
    for f in sorted(glob.glob(os.path.join(args.act_dir, "s4_*.npz"))):
        i = int(os.path.basename(f)[3:6])
        if i in excluded:
            skipped.append(i)
            continue
        d = np.load(f)
        q, ks = d["q"], d["ks"]
        L, P4, M4 = q.shape
        succ = np.zeros((L, P4, M4))
        for lv in range(L):
            for pt in range(P4):
                succ[lv, pt] = np.load(os.path.join(args.sim, "stage4", f"s4_{i:03d}_{lv}_{pt:02d}.npz"))["success"]
        states.append(i)
        for lv, k in enumerate(ks):
            s, qq = succ[lv].ravel(), q[lv].ravel()
            e = per_level.setdefault(int(k), {"auc": [], "rate": [], "top_q_gain": [], "R_success": [], "R_value": []})
            e["rate"].append(s.mean())
            e["auc"].append(roc_auc_score(s, qq) if 0 < s.mean() < 1 else np.nan)
            top = qq >= np.quantile(qq, 0.75)
            e["top_q_gain"].append(s[top].mean() - s[~top].mean())
            e["R_success"].append(float(W.resolved_fraction(succ[lv][..., None])))
            e["R_value"].append(float(W.resolved_fraction(q[lv][..., None].astype(np.float64))))
    rng = np.random.default_rng(0)
    out = {"states": states, "skipped_states": skipped, "per_level": {}}
    for k, e in sorted(per_level.items()):
        row = {"n_states": len(e["rate"])}
        for name, v in e.items():
            v = np.asarray(v, float)
            ok = v[np.isfinite(v)]
            boots = [np.mean(rng.choice(ok, len(ok))) for _ in range(args.n_boot)] if len(ok) > 1 else [np.nan]
            row[name] = {"mean": float(np.mean(ok)) if len(ok) else np.nan, "ci95": np.percentile(boots, [2.5, 97.5]).tolist(), "n": int(len(ok))}
        out["per_level"][k] = row
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    for k, r in out["per_level"].items():
        print(f"k={k}: success {r['rate']['mean']:.3f} | AUC {r['auc']['mean']:.3f} {np.round(r['auc']['ci95'], 3)} (n={r['auc']['n']}) | "
              f"top-quarter-Q gain {r['top_q_gain']['mean']:+.3f} {np.round(r['top_q_gain']['ci95'], 3)} | "
              f"R_success {r['R_success']['mean']:.3f} R_value {r['R_value']['mean']:.3f}")  # fmt: skip
    print("skipped", skipped)


if __name__ == "__main__":
    main()
