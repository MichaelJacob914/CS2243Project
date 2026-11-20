"""Adaptive-step DSRL for pi0 on LIBERO (JAX, built on dsrl_pi0's jaxrl2 + openpi).

Import `adsrl_pi0.setup_env()` before jax/libero/mujoco to set the env defaults used everywhere.
"""

import os
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def setup_env():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(PROJECT_ROOT / ".libero"))
    os.environ.setdefault("OPENPI_DATA_HOME", "/n/netscratch/olveczky_lab/Lab/charleszhang/openpi_cache")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/n/netscratch/olveczky_lab/Lab/charleszhang/jax_cache_pi0")
