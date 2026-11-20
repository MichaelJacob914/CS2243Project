"""Base-policy success on a Furniture-Bench task (w = x_T ~ N(0, I), no RL) for several DDIM K.

    python scripts/furniture/base_eval_furniture.py --task one_leg_low --n_envs 1024 --rounds 1 --ks 100 20 10

Uses the same env adapter (FurnitureVecEnv + XTAugmentWrapper) and sampler (SteeredBasePolicy.base_action)
as train_adsrl.py. Reports two success definitions per K:
  success      all parts assembled at any sim step of the episode (DPPO's best_reward >= threshold)
  success_chunk  some chunk with all parts assembled during the whole chunk (AdsrlCallback's test,
               reward > -reward_offset)
and the rollout throughput (sim + diffusion) and diffusion-only time per chunk.
Writes JSON to --out (default results/exp6/base_eval_<task>.json).
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import train_adsrl  # noqa: E402  (imports isaacgym before torch; registers OmegaConf resolvers)

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from adsrl.base_policy import SteeredBasePolicy  # noqa: E402
from adsrl.env import XTAugmentWrapper  # noqa: E402
from adsrl.furniture_env import make_furniture_vec_env  # noqa: E402


def load_cfg(task, overrides):
    cfg = OmegaConf.load(os.path.join(ROOT, "cfg", "adsrl", f"{task}.yaml"))
    cfg.merge_with_dotlist(list(overrides))
    for key in ["base_policy_path", "dppo_path", "normalization_path"]:
        cfg[key] = train_adsrl.abs_path(OmegaConf.select(cfg, key))
    cfg.log_dir = train_adsrl.abs_path(cfg.log_dir, ROOT)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--n_envs", type=int, default=1024)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--ks", type=int, nargs="+", default=[100, 20, 10], help="DDIM K; 0 = stochastic DDPM (100 steps, DPPO's own sampler)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="extra config overrides, e.g. +env.furniture.physx.max_gpu_contact_pairs=8388608")
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, "results", "exp6", f"base_eval_{args.task}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    torch.manual_seed(args.seed)
    cfg = load_cfg(args.task, [f"env.n_envs={args.n_envs}", f"seed={args.seed}"] + [o.lstrip("+") for o in args.set])
    t0 = time.time()
    venv = make_furniture_vec_env(cfg)
    print(f"sim created in {time.time() - t0:.1f}s: {args.task} n_envs={venv.num_envs} "
          f"chunks/episode={venv.chunks_per_episode}", flush=True)  # fmt: skip
    env = XTAugmentWrapper(venv, cfg.act_steps, cfg.action_dim, seed=args.seed)
    H, N = venv.chunks_per_episode, venv.num_envs

    results = {"task": args.task, "n_envs": N, "rounds": args.rounds, "chunks_per_episode": H, "set": args.set,
               "act_steps": cfg.act_steps, "gpu": torch.cuda.get_device_name(0), "per_k": {}}  # fmt: skip
    for K in args.ks:
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        ddpm = K == 0
        if ddpm:  # sanity check against the sampler DPPO itself uses for its pre-trained policies
            model_cfg["use_ddim"] = False
        else:
            model_cfg["ddim_steps"] = int(K)
        base = hydra.utils.instantiate(OmegaConf.create(model_cfg)).eval()
        if ddpm:
            def act(obs_aug):
                o = torch.as_tensor(obs_aug, device=cfg.device, dtype=torch.float32)
                x_T = o[:, cfg.obs_dim :].reshape(-1, cfg.act_steps, cfg.action_dim)
                with torch.no_grad():
                    return base({"state": o[:, : cfg.obs_dim], "noise_action": x_T}, deterministic=False).trajectories
        else:
            steered = SteeredBasePolicy(base, cfg.obs_dim, cfg.act_steps, cfg.action_dim)
            assert steered.K == K
            act = steered.base_action
        succ_any, succ_chunk, t_diff, t_total = [], [], 0.0, 0.0
        for r in range(args.rounds):
            obs = env.reset()
            first = np.full(N, -1)
            t_r = time.time()
            for t in range(H):
                torch.cuda.synchronize()
                t1 = time.time()
                a = act(obs).reshape(N, -1)
                torch.cuda.synchronize()
                t_diff += time.time() - t1
                obs, rew, done, infos = env.step(a.cpu().numpy())
                first[(first < 0) & np.array([i["chunk_success"] for i in infos])] = t
            assert done.all()
            t_total += time.time() - t_r
            succ_any.append(np.array([info["is_success"] for info in infos]))
            succ_chunk.append(first >= 0)
            print(f"K={K} round {r}: success={succ_any[-1].mean():.3f} success_chunk={succ_chunk[-1].mean():.3f} "
                  f"({time.time() - t_r:.1f}s)", flush=True)  # fmt: skip
        sa, sc = np.concatenate(succ_any), np.concatenate(succ_chunk)
        n = sa.size
        chunks = args.rounds * H
        res = {
            "success": float(sa.mean()), "success_ci95": float(1.96 * np.sqrt(sa.mean() * (1 - sa.mean()) / n)),
            "success_chunk": float(sc.mean()), "n_episodes": int(n),
            "env_steps_per_s": float(chunks * N * cfg.act_steps / t_total),
            "diffusion_ms_per_chunk": float(1000 * t_diff / chunks),
            "sim_ms_per_chunk": float(1000 * (t_total - t_diff) / chunks),
        }  # fmt: skip
        results["per_k"][str(K)] = res
        print(f"K={K}: {json.dumps(res)}", flush=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=1)
        del base, act
        torch.cuda.empty_cache()
    print(f"wrote {out}; peak torch mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    venv.close()


if __name__ == "__main__":
    main()
