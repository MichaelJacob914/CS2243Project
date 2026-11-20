"""Minimal stand-in for mujoco_py.

robomimic 0.3.0 (env_robosuite.py) imports mujoco_py at module level only to name
`mujoco_py.builder.MujocoException` in `rollout_exceptions`. robosuite 1.4.1 runs on the
official `mujoco` bindings, so the real mujoco-py (and its MuJoCo 2.1 / compile toolchain)
is not needed. Added to the venv's sys.path by .venv/lib/python3.9/site-packages/adsrl_stubs.pth.
"""
import types


class MujocoException(Exception):
    pass


builder = types.SimpleNamespace(MujocoException=MujocoException)
