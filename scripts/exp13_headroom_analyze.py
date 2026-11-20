"""exp13 Phase 0B analysis (CPU): phase labels and headroom summaries from scripts/exp13_headroom.py outputs.

Transport obs layout (normalized; obs_raw is unnormalized): robot0 eef_pos 0:3, eef_quat 3:7, gripper_qpos 7:9,
robot1 eef_pos 9:12, eef_quat 12:16, gripper_qpos 16:18, object 18:59 with (robosuite TwoArmTransport order)
payload_pos +0:3, payload_quat +3:7, trash_pos +7:10, trash_quat +10:14, lid_handle_pos +14:17,
lid_handle_quat +17:21, target_bin_pos +21:24, trash_bin_pos +24:27, gripper0_to_payload +27:30,
gripper0_to_lid_handle +30:33, gripper1_to_payload +33:36, gripper1_to_trash +36:39,
payload_in_target_bin +39, trash_in_trash_bin +40.

Coarse phase of the payload (hammer) sub-task, using the episode's first obs as reference:
  0 lid_on         lid handle moved < LID_MOVED from its start (reach / grasp the lid), payload not held
  1 pre_grasp      lid moved, payload unmoved and not held (lid removal, trash, reaching the payload)
  2 arm0_holds     gripper0 holds the payload, gripper1 does not (lift / carry)
  3 handover       both grippers hold the payload
  4 arm1_holds     gripper1 holds it alone (carry to / place in target bin)
  5 payload_in_bin payload_in_target_bin (remaining: trash / settling)
  6 loose          payload moved but not held, not in bin (dropped, regrasp)

    python scripts/exp13_headroom_analyze.py --dir results/exp13/headroom
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

O = 18
PHASES = ["lid_on", "pre_grasp", "arm0_holds", "handover", "arm1_holds", "payload_in_bin", "loose"]
LID_MOVED, PAY_MOVED, HOLD0, HOLD1, CLOSED = 0.03, 0.03, 0.10, 0.06, 0.06


def features(raw):
    pay, lid = raw[:, O : O + 3], raw[:, O + 14 : O + 17]
    return {
        "payload": pay, "lid": lid, "target_bin": raw[:, O + 21 : O + 24],
        "g0_pay": np.linalg.norm(raw[:, O + 27 : O + 30], axis=1), "g1_pay": np.linalg.norm(raw[:, O + 33 : O + 36], axis=1),
        "pay_in_bin": raw[:, O + 39] > 0.5, "trash_in_bin": raw[:, O + 40] > 0.5,
        "g0_open": raw[:, 7] - raw[:, 8], "g1_open": raw[:, 16] - raw[:, 17],
        "eef0": raw[:, 0:3], "eef1": raw[:, 9:12],
    }  # fmt: skip


def phase_of(raw, raw0):
    """Holding = gripper closed (finger gap < CLOSED; open is ~0.08, closed on an object 0.02-0.045)
    and within HOLD of the payload center (arm0 grasps ~0.07 from it, arm1 ~0.03; see exp13 doc)."""
    f, f0 = features(raw), features(raw0)
    lid_moved = np.linalg.norm(f["lid"] - f0["lid"], axis=1) > LID_MOVED
    pay_moved = np.linalg.norm(f["payload"] - f0["payload"], axis=1) > PAY_MOVED
    h0 = (f["g0_pay"] < HOLD0) & (f["g0_open"] < CLOSED)
    h1 = (f["g1_pay"] < HOLD1) & (f["g1_open"] < CLOSED)
    ph = np.full(len(raw), 6)  # payload moved, not held: dropped / regrasp
    ph[h0 & ~h1] = 2
    ph[h0 & h1] = 3
    ph[~h0 & h1] = 4
    ph[~pay_moved & ~h0 & ~h1] = 1
    ph[~lid_moved & ~pay_moved & ~h0 & ~h1] = 0
    ph[f["pay_in_bin"]] = 5
    return ph


def dump_phases(dump):
    """Phase for every row of an eval_rollouts dump (reference = the episode's chunk-0 obs)."""
    raw, ep, ch = dump["obs_raw"], dump["episode"], dump["chunk"]
    first = {e: i for i, (e, c) in enumerate(zip(ep, ch)) if c == 0}
    raw0 = raw[[first[e] for e in ep]]
    return phase_of(raw, raw0)


def check_layout(raw):
    """gripper0_to_payload should equal payload_pos - eef0_pos (and same for arm1) if the layout is right."""
    f = features(raw)
    e0 = np.abs(raw[:, O + 27 : O + 30] - (f["payload"] - f["eef0"])).max()
    e1 = np.abs(raw[:, O + 33 : O + 36] - (f["payload"] - f["eef1"])).max()
    return float(e0), float(e1)


def summarize_headroom(h, phases, out_dir):
    q = h["q"]  # [S, K, N]
    S, K, N = q.shape
    qmax = q.max(2)  # [S, K]
    best_k = qmax.argmax(1)
    gap = qmax.max(1) - qmax[:, 0]
    spread_std0 = q[:, 0].std(1)
    spread_rng0 = q[:, 0].max(1) - q[:, 0].min(1)
    # split-half reliability of argmax-k
    h1, h2 = q[:, :, : N // 2].max(2).argmax(1), q[:, :, N // 2 :].max(2).argmax(1)
    agree = float((h1 == h2).mean())
    p1 = np.bincount(h1, minlength=K) / S
    p2 = np.bincount(h2, minlength=K) / S
    chance = float((p1 * p2).sum())
    # gap under split halves: choose k on half 1, evaluate max on half 2 (removes max-over-k selection bias)
    q2 = q[:, :, N // 2 :].max(2)
    gap_cv = q2[np.arange(S), h1] - q2[:, 0]
    rank0 = (qmax > qmax[:, :1]).sum(1)  # how many k beat k=0 per state
    out = {
        "n_states": S, "K": K, "n_w": N,
        "argmax_k_hist": np.bincount(best_k, minlength=K).tolist(),
        "frac_argmax_k0": float((best_k == 0).mean()),
        "frac_argmax_k_le3": float((best_k <= 3).mean()),
        "argmax_k_mean": float(best_k.mean()), "argmax_k_median": float(np.median(best_k)),
        "split_half_argmax_agreement": agree, "split_half_chance_agreement": chance,
        "gap_best_minus_k0": {"mean": float(gap.mean()), "median": float(np.median(gap)),
                              "p90": float(np.percentile(gap, 90))},
        "gap_cv_halves": {"mean": float(gap_cv.mean()), "median": float(np.median(gap_cv)),
                          "frac_pos": float((gap_cv > 0).mean())},
        "q_spread_k0_std_over_w": {"mean": float(spread_std0.mean()), "median": float(np.median(spread_std0))},
        "q_spread_k0_range_over_w": {"mean": float(spread_rng0.mean()), "median": float(np.median(spread_rng0))},
        "gap_over_std0": {"median": float(np.median(gap / spread_std0)), "mean": float(np.mean(gap / spread_std0))},
        "gap_over_range0": {"median": float(np.median(gap / spread_rng0)), "mean": float(np.mean(gap / spread_rng0))},
        "q_level": {"mean_qmax_k0": float(qmax[:, 0].mean()), "mean_q_base": float(h["q_base"].mean()),
                    "mean_q_all": float(q.mean())},
        "per_k_mean_qmax_minus_k0": (qmax - qmax[:, :1]).mean(0).tolist(),
        "per_k_mean_q_minus_k0": (q.mean(2) - q[:, 0].mean(1)[:, None]).mean(0).tolist(),
        "per_k_mean_l2_a_minus_base": h["l2"].mean((0, 2)).tolist(),
        "per_k_mean_abs_a_minus_base": h["mabs"].mean((0, 2)).tolist(),
        "n_k_beating_k0_mean": float(rank0.mean()),
    }  # fmt: skip
    # by phase and by time decile
    tau_bin = np.minimum((h["tau"] * 10).astype(int), 9)
    for name, lab, labels in [("by_phase", phases, PHASES), ("by_tau_decile", tau_bin, [f"d{i}" for i in range(10)])]:
        d = {}
        for j, lbl in enumerate(labels):
            m = lab == j
            if m.sum() == 0:
                continue
            d[lbl] = {"n": int(m.sum()), "argmax_k_mean": float(best_k[m].mean()),
                      "argmax_k_median": float(np.median(best_k[m])),
                      "frac_argmax_k0": float((best_k[m] == 0).mean()),
                      "frac_argmax_k_le3": float((best_k[m] <= 3).mean()),
                      "frac_argmax_k_ge10": float((best_k[m] >= 10).mean()),
                      "gap_median": float(np.median(gap[m])), "gap_over_std0_median": float(np.median(gap[m] / spread_std0[m])),
                      "gap_cv_median": float(np.median(gap_cv[m])),
                      "argmax_k_hist": np.bincount(best_k[m], minlength=K).tolist()}  # fmt: skip
        out[name] = d
    # chi-square-ish test of argmax-k (binned 0-3,4-9,10-19) vs phase via permutation of labels
    kb = np.digitize(best_k, [4, 10])

    def stat(lab):
        tab = np.zeros((lab.max() + 1, 3))
        np.add.at(tab, (lab, kb), 1)
        tab = tab[tab.sum(1) > 0]
        e = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
        return float((((tab - e) ** 2) / np.maximum(e, 1e-9)).sum())

    rng = np.random.default_rng(0)
    for name, lab in [("phase", phases), ("tau_decile", tau_bin)]:
        s0 = stat(lab)
        null = np.array([stat(rng.permutation(lab)) for _ in range(2000)])
        out[f"perm_test_argmaxk_vs_{name}"] = {"chi2": s0, "p": float((null >= s0).mean())}
    with open(os.path.join(out_dir, "headroom_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def summarize_arms(d, out_dir):
    rows = list(csv.DictReader(open(os.path.join(d, "arms.csv"))))
    out = []
    for r in rows:
        arm_dir = os.path.join(d, f"arm_{r['arm']}")
        dumps = sorted(glob.glob(os.path.join(arm_dir, "eval_rollouts", "*.npz")))
        dd = dict(np.load(dumps[-1], allow_pickle=True))
        diff = dd["action"] - dd["base_action"]
        succ_ep = dd["episode_success"].astype(bool)
        ph = dump_phases(dd)
        # pre-success rows only (post-success chunks are irrelevant to the outcome)
        ep, ch, ss = dd["episode"], dd["chunk"], dd["success_step"].astype(bool)
        first = {}
        for e, c, s in zip(ep, ch, ss):
            if s and e not in first:
                first[e] = c
        pre = np.array([c <= first.get(e, 10**9) for e, c in zip(ep, ch)])
        row = {**r, "mean_l2_pre_success": float(np.linalg.norm(diff[pre], axis=1).mean()),
               "mean_abs_a_minus_base": float(np.abs(diff).mean()),
               "mean_abs_pre_success": float(np.abs(diff[pre]).mean()),
               "k_hist_pre_success": np.bincount(dd["k"][pre], minlength=20).tolist(),
               "phase_hist_pre_success": np.bincount(ph[pre], minlength=len(PHASES)).tolist(),
               "n_success_eps": int(len({e for e, s_ in zip(ep, succ_ep) if s_}))}  # fmt: skip
        ch_path = os.path.join(arm_dir, "chosen.npz")
        if os.path.exists(ch_path):
            c = np.load(ch_path)
            row["mean_q_chosen"] = float(c["q_chosen"].mean())
            row["mean_q_candidates"] = float(c["q_mean"].mean())
            # k chosen by phase (pre-success)
            kpre, phpre = dd["k"][pre], ph[pre]
            row["mean_k_chosen_by_phase"] = {PHASES[j]: float(kpre[phpre == j].mean()) for j in range(len(PHASES))
                                             if (phpre == j).any()}  # fmt: skip
        out.append(row)
        print(json.dumps({k: v for k, v in row.items() if "hist" not in k}))
    with open(os.path.join(out_dir, "arms_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results/exp13/headroom")
    args = p.parse_args()
    d = args.dir
    hp = os.path.join(d, "headroom_states.npz")
    if os.path.exists(hp):
        h = dict(np.load(hp))
        dump = dict(np.load(glob.glob(os.path.join(d, "arm_iii_stoch", "eval_rollouts", "*.npz"))[0], allow_pickle=True))
        print("layout check (max |g_to_payload - (payload - eef)|):", check_layout(dump["obs_raw"]))
        ph_all = dump_phases(dump)
        phases = ph_all[h["dump_row"]]
        # phase vs time sanity: mean tau per phase over all pre-success rows
        print("phase counts (300 states):", {PHASES[j]: int((phases == j).sum()) for j in range(len(PHASES))})
        for j in range(len(PHASES)):
            m = phases == j
            if m.any():
                print(f"  {PHASES[j]:15s} n={m.sum():3d} tau mean={h['tau'][m].mean():.2f} chunk mean={h['chunk'][m].mean():.1f}")
        np.save(os.path.join(d, "headroom_phases.npy"), phases)
        s = summarize_headroom(h, phases, d)
        print(json.dumps({k: v for k, v in s.items() if not k.startswith("by_")}, indent=1))
        print(json.dumps(s["by_phase"], indent=1))
        print(json.dumps(s["by_tau_decile"], indent=1))
    if os.path.exists(os.path.join(d, "arms.csv")):
        summarize_arms(d, d)


if __name__ == "__main__":
    main()
