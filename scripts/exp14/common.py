"""exp14 paths. adsrl / train_adsrl are imported read-only from the frozen snapshot (another agent edits adsrl/)."""

from __future__ import annotations

import os
import subprocess
import sys

PROJECT = "/n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project"
SCRATCH = "/n/netscratch/olveczky_lab/Lab/charleszhang/exp14"
SNAP = os.environ.get("EXP14_CODE_ROOT", os.path.join(SCRATCH, "code_snapshot"))
PYLIB = os.path.join(SCRATCH, "pylib")
DSRL = os.path.join(SNAP, "dsrl")
DATASET = "/n/netscratch/olveczky_lab/Lab/charleszhang/data/robomimic/transport_mh/low_dim_v141.hdf5"
CRITIC_CKPT = os.path.join(SCRATCH, "uniformk_s1", "checkpoint", "final.pt")
NORM = os.path.join(DSRL, "dppo", "log", "robomimic", "transport", "normalization.npz")

for p in (SNAP, os.path.join(SNAP, "scripts"), os.path.join(DSRL, "dppo"), DSRL):
    if p not in sys.path:
        sys.path.insert(0, p)
if PYLIB not in sys.path:
    sys.path.append(PYLIB)  # after the venv, so it never shadows numpy/scipy


def env_info():
    try:
        commit = subprocess.check_output(["git", "-C", PROJECT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    snap_time = open(os.path.join(SNAP, "SNAPSHOT_TIME")).read().strip() if os.path.exists(os.path.join(SNAP, "SNAPSHOT_TIME")) else None
    return {"commit": commit, "snapshot": SNAP, "snapshot_time": snap_time, "host": os.uname().nodename,
            "slurm_job": os.environ.get("SLURM_JOB_ID")}
