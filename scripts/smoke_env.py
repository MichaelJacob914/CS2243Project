"""Smoke test: imports + robomimic can/square/transport env reset/step + base policy forward on GPU. Run with cwd = dsrl/."""
import os, sys, time
sys.path.insert(0, os.getcwd()); sys.path.append('./dppo')
import numpy as np, torch
from env_utils import make_robomimic_env, ObservationWrapperRobomimic
keys = {'can': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'square': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'transport': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot1_eef_pos', 'robot1_eef_quat', 'robot1_gripper_qpos', 'object']}
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
for task in ['can', 'square', 'transport']:
    env = make_robomimic_env(env=task, normalization_path=f'./dppo/log/robomimic/{task}/normalization.npz', low_dim_keys=keys[task], dppo_path='./dppo')
    env = ObservationWrapperRobomimic(env, reward_offset=1)
    obs = env.reset(); a = env.action_space.sample()
    t = time.perf_counter(); n = 200
    for _ in range(n): obs, r, d, info = env.step(np.zeros_like(a))
    print(task, 'obs', obs.shape, 'act', a.shape, f'{n/(time.perf_counter()-t):.0f} raw env steps/s (1 proc)')
