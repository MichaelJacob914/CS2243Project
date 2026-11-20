"""Preallocated ring replay buffer of chunk transitions (DSRL-pi0's ReplayBuffer contents plus the steering k)."""

from __future__ import annotations

import os
import pickle

import jax
import numpy as np
from flax.core import frozen_dict


class ChunkReplayBuffer:
    def __init__(self, obs_shapes: dict, obs_dtypes: dict, noise_dim: int, capacity: int, seed: int = 0):
        self.capacity = int(capacity)
        mk = lambda: {k: np.zeros((self.capacity, *s), obs_dtypes[k]) for k, s in obs_shapes.items()}  # noqa: E731
        self.data = {
            "observations": mk(),
            "next_observations": mk(),
            "actions": np.zeros((self.capacity, 1, noise_dim), np.float32),
            "k": np.zeros((self.capacity,), np.int32),
            "rewards": np.zeros((self.capacity,), np.float32),
            "masks": np.zeros((self.capacity,), np.float32),
            "discount": np.zeros((self.capacity,), np.float32),
        }
        self.size = 0
        self.ptr = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def insert(self, obs, next_obs, action, k, reward, mask, discount):
        i = self.ptr
        for key in self.data["observations"]:
            self.data["observations"][key][i] = obs[key]
            self.data["next_observations"][key][i] = next_obs[key]
        self.data["actions"][i] = np.asarray(action, np.float32).reshape(1, -1)
        self.data["k"][i] = k
        self.data["rewards"][i] = reward
        self.data["masks"][i] = mask
        self.data["discount"][i] = discount
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, with_k: bool):
        idx = self.rng.integers(0, self.size, batch_size)
        out = {}
        for key, v in self.data.items():
            if key == "k" and not with_k:
                continue
            out[key] = {kk: vv[idx] for kk, vv in v.items()} if isinstance(v, dict) else v[idx]
        return frozen_dict.freeze(jax.device_put(out))

    def save(self, path: str):
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"data": {k: (({kk: vv[: self.size] for kk, vv in v.items()}) if isinstance(v, dict)
                                      else v[: self.size]) for k, v in self.data.items()},
                         "size": self.size, "ptr": self.ptr, "rng": self.rng.bit_generator.state}, f, protocol=4)
        os.replace(tmp, path)

    def load(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        n = d["size"]
        if n > self.capacity:  # resumed with a smaller --total_env_steps: grow to fit the saved data
            grow = lambda a: np.concatenate([a, np.zeros((n - self.capacity, *a.shape[1:]), a.dtype)])  # noqa: E731
            self.data = {k: ({kk: grow(vv) for kk, vv in v.items()} if isinstance(v, dict) else grow(v))
                         for k, v in self.data.items()}
            self.capacity = n
        for k, v in d["data"].items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    self.data[k][kk][:n] = vv
            else:
                self.data[k][:n] = v
        self.size, self.ptr = n, d["ptr"] % self.capacity
        self.rng.bit_generator.state = d["rng"]
