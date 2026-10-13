#!/usr/bin/env python3
"""Evaluate a policy in simulation from a config file.

    python scripts/eval.py configs/eval/pi0_libero90_t57.yaml
    python scripts/eval.py configs/eval/pi0_libero_spatial.yaml episodes=2 tasks=[0] --tag quick
    python scripts/eval.py configs/eval/pi0_libero90_t57.yaml noise.mode=fixed noise.seed=3 --tag fixed3
    python scripts/eval.py configs/eval/pi0_libero_spatial.yaml --shard 3/10     # tasks 3, 13, ...

Overrides are key=value (dotted keys reach into nested dicts; values parsed as YAML).
Output: $RUNS_ROOT/eval/<config name>[_<tag>]/ -- rerun the same command to resume.
Merge shards with scripts/aggregate.py <output dir>.
"""

import argparse
import os
import pathlib

import yaml


def load_config(path: str, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(pathlib.Path(path).read_text())
    for item in overrides:
        key, _, value = item.partition("=")
        if not _:
            raise SystemExit(f"override {item!r} is not key=value")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*", help="key=value")
    ap.add_argument("--tag", default="", help="suffix for the output folder (separate repeats/variants)")
    ap.add_argument("--shard", default=None, help="i/n: run every n-th task starting at i (default: Slurm array)")
    ap.add_argument("--out", default=None, help="output folder (default: $RUNS_ROOT/eval/<name>)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    name = pathlib.Path(args.config).stem + (f"_{args.tag}" if args.tag else "")
    out = pathlib.Path(args.out) if args.out else pathlib.Path(os.environ["RUNS_ROOT"]) / "eval" / name

    if args.shard:
        i, n = map(int, args.shard.split("/"))
    elif "SLURM_ARRAY_TASK_ID" in os.environ:
        i, n = int(os.environ["SLURM_ARRAY_TASK_ID"]), int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    else:
        i, n = 0, 1

    from steer.eval.runner import run

    print(f"config {args.config} -> {out}  shard {i}/{n}", flush=True)
    run(cfg, out, shard=(i, n))


if __name__ == "__main__":
    main()
