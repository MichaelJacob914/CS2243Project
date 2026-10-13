#!/usr/bin/env python3
"""Merge the results_shard*.jsonl files of an eval folder into a per-task success table.

    python scripts/aggregate.py $RUNS_ROOT/eval/pi0_libero_spatial
    python scripts/aggregate.py $RUNS_ROOT/eval/*_fixed*          # several folders, one line each

Writes summary.json in each folder and prints a markdown table.
"""

import collections
import json
import math
import pathlib
import sys


def summarize(folder: pathlib.Path) -> dict:
    rows = [json.loads(line) for f in sorted(folder.glob("results_shard*.jsonl")) for line in f.read_text().splitlines()]
    per_task = collections.defaultdict(list)  # key: task, or task/noise<seed> for fixed-noise sweeps
    for r in rows:
        seed = r.get("noise_seed")
        per_task[str(r["task"]) if seed is None else f"{r['task']}/noise{seed}"].append(r)
    tasks = {
        t: {
            "prompt": rs[0]["prompt"],
            "episodes": len(rs),
            "success": sum(r["success"] for r in rs) / len(rs),
            "errors": sum(r["error"] is not None for r in rs),
        }
        for t, rs in sorted(per_task.items(), key=lambda kv: (len(kv[0]), kv[0]))
    }
    n = len(rows)
    p = sum(r["success"] for r in rows) / n if n else float("nan")
    out = {
        "episodes": n,
        "success": p,
        "stderr": math.sqrt(p * (1 - p) / n) if n else float("nan"),
        "errors": sum(r["error"] is not None for r in rows),
        "tasks": tasks,
    }
    rates = sorted(v["success"] for k, v in tasks.items() if "/noise" in k)
    if rates:  # spread of success across fixed noises = how much the noise alone can steer
        out["noise_spread"] = {"seeds": len(rates), "min": rates[0], "median": rates[len(rates) // 2], "max": rates[-1]}
    (folder / "summary.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    folders = [pathlib.Path(a) for a in sys.argv[1:]]
    if not folders:
        sys.exit(__doc__)
    for folder in folders:
        s = summarize(folder)
        print(f"\n## {folder.name}: {100 * s['success']:.1f} % ± {100 * s['stderr']:.1f} ({s['episodes']} episodes, {s['errors']} errors)\n")
        if "noise_spread" in s:
            ns = s["noise_spread"]
            print(f"fixed-noise spread over {ns['seeds']} seeds: min {100 * ns['min']:.0f} % / "
                  f"median {100 * ns['median']:.0f} % / max {100 * ns['max']:.0f} %\n")  # fmt: skip
        if len(s["tasks"]) > 1 or len(folders) == 1:
            print("| task | success | episodes | prompt |\n|---|---|---|---|")
            for t, v in s["tasks"].items():
                print(f"| {t} | {100 * v['success']:.0f} % | {v['episodes']} | {v['prompt']} |")


if __name__ == "__main__":
    main()
