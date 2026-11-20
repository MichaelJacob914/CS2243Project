"""ChunkReplayBuffer save/load, including loading into a smaller buffer (resume with a smaller budget)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adsrl_pi0.buffer import ChunkReplayBuffer  # noqa: E402

SHAPES = {"pixels": (4, 4, 3, 1), "state": (8, 1), "x_T": (32,)}
DTYPES = {"pixels": np.uint8, "state": np.float32, "x_T": np.float32}


def _fill(buf, n, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(n):
        o = {k: rng.integers(0, 255, s).astype(DTYPES[k]) for k, s in SHAPES.items()}
        buf.insert(o, o, rng.standard_normal(32), i % 10, -1.0, 1.0, 0.98)


def test_roundtrip_and_grow(tmp_path):
    a = ChunkReplayBuffer(SHAPES, DTYPES, 32, capacity=20, seed=0)
    _fill(a, 12)
    path = str(tmp_path / "buf.pkl")
    a.save(path)
    b = ChunkReplayBuffer(SHAPES, DTYPES, 32, capacity=5, seed=1)
    b.load(path)
    assert len(b) == 12 and b.capacity == 12
    np.testing.assert_array_equal(b.data["observations"]["pixels"][:12], a.data["observations"]["pixels"][:12])
    np.testing.assert_array_equal(b.data["k"][:12], a.data["k"][:12])
    batch = b.sample(7, with_k=True)
    assert batch["actions"].shape == (7, 1, 32) and batch["k"].shape == (7,)
    _fill(b, 3, seed=2)  # keeps working after growth (wraps)
    assert len(b) == 12
