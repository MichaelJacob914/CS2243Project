"""Eval-protocol check (GPU node, robosuite/EGL): do eval initial states depend on eval_env.seed(), and are
they distinct across the parallel envs? Builds the eval vec env exactly as scripts/base_eval.py does.

    MUJOCO_GL=egl python -m pytest tests/test_eval_seeding.py -s
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

pytest.importorskip("robosuite")


def _make_eval_env(n):
    import base_eval  # noqa: F401  (sets sys.path)
    from base_eval import load_cfg
    from env_utils import ActionChunkWrapper, ObservationWrapperRobomimic, make_robomimic_env
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from adsrl.env import XTAugmentWrapper

    cfg = load_cfg("can", None, n, n, os.path.join(ROOT, "logs", "test_eval_seeding"), 0)

    def make_env():
        env = make_robomimic_env(env=cfg.env_name, normalization_path=cfg.normalization_path,
                                 low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys, dppo_path=cfg.dppo_path)
        env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
        return ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)

    return XTAugmentWrapper(make_vec_env(make_env, n_envs=n, vec_env_cls=SubprocVecEnv), cfg.act_steps, cfg.action_dim), cfg


def _initial_states(seed, n=4, rounds=2):
    env, cfg = _make_eval_env(n)
    env.seed(seed)
    out = [env.reset()[:, : cfg.obs_dim].copy() for _ in range(rounds)]
    env.close()
    return np.stack(out)  # (rounds, n, obs_dim)


def test_initial_states_distinct_across_envs():
    s = _initial_states(1000)
    for r in range(s.shape[0]):
        d = np.abs(s[r][:, None] - s[r][None]).max(-1)
        np.fill_diagonal(d, np.inf)
        print("round", r, "min pairwise max|diff| between envs:", d.min())
        assert d.min() > 1e-6, "parallel eval envs start from identical states"


@pytest.mark.xfail(strict=True, reason="known upstream bug, not fixed: ObservationWrapperRobomimic.reset ignores "
                   "reset(seed=...), so eval initial states are unseeded (see exp_2_wave0_can_2026-11-03.md)")
def test_eval_env_seed_controls_initial_states():
    a, b = _initial_states(1000), _initial_states(1000)
    print("same-seed max|diff|:", np.abs(a - b).max())
    assert np.allclose(a, b), "eval_env.seed() does not make eval initial states reproducible"
