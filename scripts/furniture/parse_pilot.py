"""Summarize exp_6 throughput-pilot logs (train_adsrl.py with log_train_stdout=True).

    python scripts/furniture/parse_pilot.py logs/exp6/<job dir> [...]

Per run log (<name>.log): base-rollout env steps/s ([init]), deterministic eval time ([eval det]),
and over the [train] lines after the first one: env steps/s and gradient updates/s during training,
train success, peak torch GPU memory. Per job dir: max per-process GPU memory from gpu_mem.csv.
"""

import glob
import os
import re
import sys


def parse_log(path):
    txt = open(path, errors="replace").read()
    out = {"name": os.path.basename(path)[:-4]}
    m = re.search(r"\[init\] (\d+) base-policy env steps in ([\d.]+)s", txt)
    if m:
        out["init_sps"] = int(m.group(1)) / float(m.group(2))
    m = re.findall(r"\[eval det\] (\d+) episodes in ([\d.]+)s", txt)
    if m:
        out["eval_s"] = float(m[0][1])
        out["eval_eps"] = int(m[0][0])
    rows = [tuple(float(x) for x in r) for r in re.findall(
        r"\[train\] env_steps=(\d+) wall_s=([\d.]+) n_updates=(\d+) success=([\d.na]+) gpu_mem_gb=([\d.]+)",
        txt.replace("nan", "-1"))]  # fmt: skip
    if len(rows) >= 2:
        (s0, w0, u0, *_), (s1, w1, u1, succ, mem) = rows[0], rows[-1]
        out.update(train_sps=(s1 - s0) / (w1 - w0), ups=(u1 - u0) / (w1 - w0), n_train_lines=len(rows),
                   last_success=succ, torch_mem_gb=mem, sec_per_episode=(w1 - w0) / (len(rows) - 1))  # fmt: skip
    out["train_success"] = [r[3] for r in rows]
    if "Traceback" in txt or "Error" in txt.split("\n")[-5:]:
        out["error"] = txt.strip().split("\n")[-1][:200]
    return out


def gpu_mem(job_dir):
    p = os.path.join(job_dir, "gpu_mem.csv")
    if not os.path.exists(p):
        return None
    best = {}
    for line in open(p):
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 4 and parts[3].endswith("MiB"):
            best[parts[1]] = max(best.get(parts[1], 0), int(parts[3].split()[0]))
    return best


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(f"== {d}  gpu_mem_MiB per pid: {gpu_mem(d)}")
        for f in sorted(glob.glob(os.path.join(d, "*.log"))):
            r = parse_log(f)
            succ = r.pop("train_success")
            print("  " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
            if succ:
                print(f"    train success per episode: {[round(s, 3) for s in succ]}")
