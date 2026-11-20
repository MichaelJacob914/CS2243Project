"""Time DSRL-NA gradient updates (fixed k=0, one_leg sizes, K from config) vs batch size, no simulator.

    python scripts/furniture/bench_updates.py --batches 256 1024 2048 --ags -1 8

Builds DSRLFixedK exactly as train_adsrl.py does (3x2048 nets, layer norm, noise critic) on a dummy
VecEnv with the furniture obs/action spaces, fills the replay buffer with random transitions, and
times model.train(gradient_steps=64, ...) with noise_critic_grad_steps=32 (the sweep's per-vec-step
work). Prints ms per gradient step (incl. its share of distillation) per (batch, actor_gradient_steps).
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import train_adsrl  # noqa: E402

import gymnasium as gym  # noqa: E402
import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from adsrl.agent import DSRLFixedK, SteerSACPolicy  # noqa: E402
from adsrl.base_policy import SteeredBasePolicy  # noqa: E402
from adsrl.env import XTAugmentWrapper  # noqa: E402


class Dummy(gym.Env):
    def __init__(self, obs_dim, act_dim):
        self.observation_space = gym.spaces.Box(-5, 5, (obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (act_dim,), np.float32)

    def reset(self, seed=None, options=None):
        return np.zeros(self.observation_space.shape, np.float32), {}

    def step(self, a):
        return np.zeros(self.observation_space.shape, np.float32), 0.0, False, False, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="one_leg_low")
    ap.add_argument("--batches", type=int, nargs="+", default=[256, 1024, 2048])
    ap.add_argument("--ags", type=int, nargs="+", default=[-1, 8])
    ap.add_argument("--utd", type=int, default=64)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    cfg = OmegaConf.load(os.path.join(ROOT, "cfg", "adsrl", f"{args.task}.yaml"))
    for key in ["base_policy_path", "dppo_path", "normalization_path"]:
        cfg[key] = train_adsrl.abs_path(cfg[key])
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    base = hydra.utils.instantiate(OmegaConf.create(model_cfg)).eval()
    steered = SteeredBasePolicy(base, cfg.obs_dim, cfg.act_steps, cfg.action_dim)
    H, A = cfg.act_steps, cfg.action_dim
    env = XTAugmentWrapper(DummyVecEnv([lambda: Dummy(cfg.obs_dim, H * A)]), H, A)
    net = [cfg.train.layer_size] * cfg.train.num_layers
    print(f"GPU {torch.cuda.get_device_name(0)}, K={steered.K}, utd={args.utd}, nc={args.utd // 2}", flush=True)
    for ags in args.ags:
        for B in args.batches:
            model = DSRLFixedK(
                SteerSACPolicy, env, k=0, steered_policy=steered, horizon=H, act_dim=A,
                learning_rate=3e-4, buffer_size=200_000, learning_starts=1, batch_size=B, tau=0.005, gamma=0.99,
                train_freq=1, gradient_steps=args.utd, ent_coef="auto", target_entropy=0.0, verbose=0,
                noise_critic_grad_steps=args.utd // 2, critic_backup_combine_type="min", device="cuda:0",
                actor_gradient_steps=ags,
                policy_kwargs=dict(net_arch=dict(pi=net, qf=net), activation_fn=torch.nn.Tanh, log_std_init=0.0,
                                   post_linear_modules=[torch.nn.LayerNorm], n_critics=2,
                                   xt_inputs="none", state_dim=cfg.obs_dim),
            )  # fmt: skip
            model._setup_learn(1, None)
            rb = model.replay_buffer
            n = 200_000
            rb.observations[:n] = np.random.randn(n, 1, rb.observations.shape[-1]).astype(np.float32)
            rb.next_observations[:n] = np.random.randn(n, 1, rb.observations.shape[-1]).astype(np.float32)
            rb.actions[:n] = np.random.uniform(-1, 1, (n, 1, rb.actions.shape[-1])).astype(np.float32)
            rb.rewards[:n] = -8.0
            rb.full, rb.pos = True, 0
            model.train(gradient_steps=args.utd, batch_size=B)  # warm-up
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(args.reps):
                model.train(gradient_steps=args.utd, batch_size=B)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / args.reps
            print(f"ags={ags} batch={B}: {1000 * dt:.0f} ms per train() = {1000 * dt / args.utd:.1f} ms per gradient step "
                  f"(incl. 0.5 distillation steps)", flush=True)  # fmt: skip
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
