"""exp13: exact McNemar tests between paired rollout arms (same initial states), run on a headroom out dir.

    python scripts/exp13_paired_tests.py results/exp13/headroom_paired
"""
import glob, json, sys, numpy as np
D = sys.argv[1] if len(sys.argv) > 1 else "results/exp13/headroom"
from math import comb
arms = ["iii_stoch", "iv_det", "i_allk_N32", "ii_k0_N640"]
S, O0 = {}, {}
for a in arms:
    d = np.load(glob.glob(f"{D}/arm_{a}/eval_rollouts/*.npz")[0], allow_pickle=True)
    ep, ch = d["episode"], d["chunk"]
    m = ch == 0
    order = np.argsort(ep[m])
    O0[a] = d["obs"][m][order]
    S[a] = d["episode_success"][m][order].astype(bool)
for a in arms[1:]:
    print(a, "same initial obs as iii:", np.allclose(O0[a][:, :18], O0["iii_stoch"][:, :18]), "max diff", float(np.abs(O0[a] - O0["iii_stoch"]).max()))
def mcnemar(x, y):
    b, c = int((x & ~y).sum()), int((~x & y).sum()); n = b + c
    p = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n * 2 if n else 1.0
    return b, c, min(p, 1.0)
out = {}
for x, y in [("i_allk_N32", "ii_k0_N640"), ("i_allk_N32", "iii_stoch"), ("ii_k0_N640", "iii_stoch"), ("iii_stoch", "iv_det")]:
    b, c, p = mcnemar(S[x], S[y]); out[f"{x}_vs_{y}"] = {"x_only": b, "y_only": c, "p_exact_mcnemar": p}
    print(x, "vs", y, "x-only", b, "y-only", c, "p", round(p, 3))
json.dump(out, open(f"{D}/paired_tests.json", "w"), indent=2)
