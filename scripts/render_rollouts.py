"""Render robomimic rollouts to annotated MP4 clips for the report page.

    python scripts/render_rollouts.py --task can --policy base
    python scripts/render_rollouts.py --task transport --policy base --ddim_steps 20
    python scripts/render_rollouts.py --task can --policy fixed_k --k 3 --ckpt logs/adsrl/can/<run>/checkpoint/final.pt
    python scripts/render_rollouts.py --task can --policy adaptive --ckpt logs/adsrl/can/<run>/checkpoint/final.pt
    python scripts/render_rollouts.py --task can --policy residual --ckpt logs/adsrl/can/residual_b0.1/seed1/checkpoint/final.pt

Each (task, policy) rolls out --n_episodes episodes on the fixed seed list seed0..seed0+N-1. Seed s
fixes the initial state (np.random.seed(s) right before env.reset) and the per-chunk x_T stream
(np.random.default_rng(s + XT_SEED_OFFSET)), so different methods start from the same states and see
the same x_T. The episode protocol matches AdsrlCallback.evaluate: a chunk counts as a success when
sum_i (r_i - reward_offset) > -reward_offset over its act_steps env steps, and the episode succeeds
at its first successful chunk. Clips stop --tail_steps env steps after success.

Up to --max_success successes and --max_fail failures are saved as
<out_dir>/<task>__<method>__<success|fail>__<seed>.mp4 (H.264, yuv420p, 20 fps) and listed in
<out_dir>/manifest.json (one entry per clip; reruns replace entries with the same file). The success
rate over all N episodes goes to <out_dir>/rollout_stats.json. Needs a GPU node (robosuite renders
with EGL): MUJOCO_GL=egl.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import train_adsrl  # noqa: E402, F401  (registers OmegaConf resolvers, puts dsrl/ and dppo/ on sys.path)
from omegaconf import OmegaConf  # noqa: E402

from adsrl.base_policy import SteeredBasePolicy  # noqa: E402

XT_SEED_OFFSET = 100_000
# agentview plus one context camera: the wrist view for the single-arm tasks (grasp/insertion detail);
# for transport, agentview only shows part of the scene, so frontview (both arms) is added.
# Both-camera clips stay well under 2 MB (see report/README.md).
DEFAULT_CAMERAS = {"can": "agentview,robot0_eye_in_hand", "square": "agentview,robot0_eye_in_hand",
                   "transport": "agentview,frontview"}  # fmt: skip
HEADER_H = 48  # px; frame height = size + HEADER_H (multiple of 16 for H.264 macroblocks at size=256)

# Overlay colors (text in neutral ink; status green carries an icon + label, never color alone).
INK = (240, 239, 236)
INK_MUTED = (170, 169, 163)
HEADER_BG = (26, 26, 25)
ACCENT = (235, 104, 52)  # adaptive accent, same as METHOD_COLORS["adaptive"] in scripts/plots.py
BAR_IDLE = (110, 109, 104)
SUCCESS_BG = (0, 131, 0)


# ----------------------------------------------------------------------------------------
# config / policies
# ----------------------------------------------------------------------------------------
def base_cfg(task: str, ddim_steps: int | None, device: str):
    import base_eval  # scripts/base_eval.py

    cfg = base_eval.load_cfg(task, ddim_steps, n_envs=1, n_episodes=1, logdir=os.path.join(ROOT, "logs", "render"), seed=0)
    cfg.device = device
    cfg.model.device = device
    return cfg


def run_cfg(ckpt: str, device: str):
    """Resolved training config saved by train_adsrl.py in <logdir>/run_info.json."""
    logdir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
    with open(os.path.join(logdir, "run_info.json")) as f:
        cfg = OmegaConf.create(json.load(f)["cfg"])
    cfg.device = device
    cfg.model.device = device
    return cfg


def build_steered(cfg) -> SteeredBasePolicy:
    import hydra

    model = hydra.utils.instantiate(OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))).eval()
    w_scale = OmegaConf.select(cfg, "steer.w_scale", default=1.0)  # base-policy cfgs have no steer
    param = OmegaConf.select(cfg, "steer.param", default="absolute")
    steer = OmegaConf.select(cfg, "steer", default=None)
    dims = None if steer is None else train_adsrl.steer_w_scale_dims(steer, cfg.act_steps, cfg.action_dim)
    return SteeredBasePolicy(model, cfg.obs_dim, cfg.act_steps, cfg.action_dim, w_scale=w_scale, param=param,
                             w_scale_dims=dims)


class _SpacesOnlyEnv:
    """Minimal env for the SB3 constructor (it only reads the spaces); never stepped."""

    def __init__(self, obs_dim, act_dim):
        import gymnasium
        from gymnasium import spaces

        class _E(gymnasium.Env):
            observation_space = spaces.Box(-1.0, 1.0, (obs_dim,), np.float32)
            action_space = spaces.Box(-1.0, 1.0, (act_dim,), np.float32)

            def reset(self, seed=None, options=None):
                return np.zeros(obs_dim, np.float32), {}

            def step(self, action):
                raise RuntimeError("spaces-only env")

        self.cls = _E


def build_agent(cfg, steered, ckpt: str):
    """Rebuild DSRLFixedK / DSRLAdaptiveK / ResidualSAC exactly as train_adsrl.main does and load
    ``ckpt`` (network/optimizer/temperature state; the replay buffer is not loaded)."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    from adsrl.agent import DSRLAdaptiveK, DSRLFixedK, SteerSACPolicy
    from adsrl.env import XTAugmentWrapper

    E = _SpacesOnlyEnv(cfg.obs_dim, cfg.act_steps * cfg.action_dim).cls
    env = XTAugmentWrapper(DummyVecEnv([E]), cfg.act_steps, cfg.action_dim)
    net_arch = [cfg.train.layer_size] * cfg.train.num_layers
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, qf=net_arch),
        activation_fn=torch.nn.Tanh,
        log_std_init=0.0,
        post_linear_modules=[torch.nn.LayerNorm] if cfg.train.use_layer_norm else None,
        n_critics=cfg.train.n_critics,
        # same x_T routing as training (otherwise input widths mismatch the checkpoint)
        xt_inputs=train_adsrl.resolve_xt_inputs(cfg.steer),
        state_dim=cfg.obs_dim,
    )
    common = dict(
        learning_rate=cfg.train.actor_lr, buffer_size=1000, learning_starts=1, batch_size=cfg.train.batch_size,
        tau=cfg.train.tau, gamma=cfg.train.discount, train_freq=cfg.train.train_freq, gradient_steps=cfg.train.utd,
        action_noise=None, optimize_memory_usage=False,
        ent_coef=train_adsrl.resolve_ent_coef(cfg.train.ent_coef), target_update_interval=1,
        target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent, verbose=0,
        policy_kwargs=policy_kwargs, device=cfg.device, noise_critic_grad_steps=cfg.train.noise_critic_grad_steps,
        critic_backup_combine_type=cfg.train.critic_backup_combine_type, steered_policy=steered,
        horizon=cfg.act_steps, act_dim=cfg.action_dim, use_tf32=False, compile_networks=False,
    )  # fmt: skip
    tr = cfg.train.get("trust_region")
    if tr is not None and tr.enabled and cfg.steer.mode in ("fixed", "adaptive"):
        common["trust_region"] = dict(bound=tr.bound, lr=tr.lr, init=tr.init)  # so trust_region.lam loads
    if cfg.steer.mode == "fixed":
        model = DSRLFixedK(SteerSACPolicy, env, k=cfg.steer.k, **common)
    elif cfg.steer.mode == "residual":
        from adsrl.residual import ResidualSAC

        # the spaces-only env has the right shapes: obs [s, a_base] has the same length as [s, x_T]
        for key in ("noise_critic_grad_steps", "critic_backup_combine_type", "horizon", "act_dim"):
            common.pop(key)
        model = ResidualSAC(SteerSACPolicy, env, beta=cfg.steer.residual_beta, state_dim=cfg.obs_dim, **common)
    else:
        model = DSRLAdaptiveK(
            SteerSACPolicy, env, num_k=steered.K, rho=cfg.steer.rho, k_ent_coef=cfg.steer.k_ent_coef,
            freeze_k_uniform=cfg.steer.freeze_k_uniform,
            k_subset=None if cfg.steer.k_subset is None else list(cfg.steer.k_subset), **common,
        )  # fmt: skip
    model.load_checkpoint(ckpt, load_replay_buffer=False)
    return model


def method_name(cfg, policy: str) -> str:
    """Method slug used in file names; matches the method keys in scripts/plots.py."""
    if policy == "base":
        return "base"
    if cfg.steer.mode == "residual":
        return "residual_sac"
    if cfg.steer.mode == "fixed":
        return "dsrl" if int(cfg.steer.k) == 0 else f"k{int(cfg.steer.k)}"
    if cfg.steer.freeze_k_uniform:
        return "uniform_k"
    if cfg.steer.k_subset is not None:
        return "adaptive_subset"
    return "adaptive"


METHOD_LABEL = {"base": "base policy", "dsrl": "DSRL (k=0)", "uniform_k": "uniform-random k",
                "adaptive": "adaptive k", "adaptive_subset": "adaptive k (subset)",
                "residual_sac": "residual SAC"}  # fmt: skip


# ----------------------------------------------------------------------------------------
# env
# ----------------------------------------------------------------------------------------
def make_env(cfg, size: int):
    from env_utils import make_robomimic_env

    env = make_robomimic_env(
        render=True, env=cfg.env_name, normalization_path=cfg.normalization_path,
        low_dim_keys=list(cfg.env.wrappers.robomimic_lowdim.low_dim_keys), dppo_path=cfg.dppo_path,
    )  # fmt: skip
    env.render_hw = (size, size)
    return env  # RobomimicLowdimWrapper: reset() -> {"state"}, step(a_normalized) -> (obs, r, done, info)


def render_cams(env, cams, size):
    return np.concatenate(
        [env.env.render(mode="rgb_array", height=size, width=size, camera_name=c) for c in cams], axis=1
    )


# ----------------------------------------------------------------------------------------
# overlay
# ----------------------------------------------------------------------------------------
def _font(size):
    from matplotlib import font_manager
    from PIL import ImageFont

    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)


class Overlay:
    def __init__(self, task, method_label, n_chunks, num_k):
        self.task, self.method, self.n_chunks, self.num_k = task, method_label, n_chunks, num_k
        self.f_big, self.f_small = _font(13), _font(11)

    def __call__(self, img, chunk, k, probs, success_step):
        from PIL import Image, ImageDraw

        h, w, _ = img.shape
        canvas = Image.new("RGB", (w, h + HEADER_H), HEADER_BG)
        canvas.paste(Image.fromarray(img), (0, HEADER_H))
        d = ImageDraw.Draw(canvas)
        d.text((8, 6), f"{self.task}  |  {self.method}", font=self.f_big, fill=INK)
        k_txt = "k = -" if k is None else f"k = {k}"
        d.text((8, 27), f"chunk {chunk + 1}/{self.n_chunks}   {k_txt}", font=self.f_small, fill=INK_MUTED)
        if probs is not None:  # pi(k|s): one bar per k, chosen k in the accent color
            bw = max(2, min(8, 96 // self.num_k))
            x0, y0, bh = w - 8 - bw * self.num_k, 8, 30
            d.text((x0 - 30, 22), "π(k)", font=self.f_small, fill=INK_MUTED)
            for i, p in enumerate(probs):
                hgt = max(1, round(bh * float(p)))
                col = ACCENT if (k is not None and i == k) else BAR_IDLE
                d.rectangle([x0 + i * bw, y0 + bh - hgt, x0 + (i + 1) * bw - 2, y0 + bh], fill=col)
            d.line([x0, y0 + bh + 1, x0 + bw * self.num_k - 2, y0 + bh + 1], fill=INK_MUTED, width=1)
        if success_step is not None:
            label = f"✓ success @ step {success_step}"
            tw = d.textlength(label, font=self.f_small)
            d.rounded_rectangle([6, HEADER_H + 6, 6 + tw + 12, HEADER_H + 26], radius=4, fill=SUCCESS_BG)
            d.text((12, HEADER_H + 9), label, font=self.f_small, fill=(255, 255, 255))
        return np.asarray(canvas)


# ----------------------------------------------------------------------------------------
# rollout
# ----------------------------------------------------------------------------------------
def rollout(env, cfg, act, seed, overlay, writer, cams, size, tail_steps):
    """One episode under the eval protocol. ``act(obs_aug) -> (action (A*H,), k, probs)``.
    Frames go to ``writer`` (None = no rendering). Returns the episode record."""
    H, A, off = cfg.act_steps, cfg.action_dim, cfg.env.reward_offset
    n_chunks = int(cfg.env.max_episode_steps / H)
    np.random.seed(seed)
    torch.manual_seed(seed)
    obs = env.reset()["state"].flatten().astype(np.float32)
    init_hash = hashlib.sha1(np.round(obs, 5).tobytes()).hexdigest()[:10]
    xt_rng = np.random.default_rng(seed + XT_SEED_OFFSET)
    success_step, ks, steps, stop_at = None, [], 0, None
    if writer is not None:
        writer.append_data(overlay(render_cams(env, cams, size), 0, None, None, None))
    for t in range(n_chunks):
        x_T = xt_rng.standard_normal(H * A).astype(np.float32)
        obs_aug = np.concatenate([obs, x_T])[None]
        action, k, probs = act(obs_aug)
        ks.append(None if k is None else int(k))
        chunk_rew = 0.0
        for a in action.reshape(H, A):
            o, r, _, _ = env.step(a)
            steps += 1
            chunk_rew += float(r) - off
            if writer is not None:
                frame = render_cams(env, cams, size)
                writer.append_data(overlay(frame, t, k, probs, success_step))
            if stop_at is not None and steps >= stop_at:
                break
        obs = o["state"].flatten().astype(np.float32)
        if success_step is None and chunk_rew > -off:
            success_step = (t + 1) * H  # same definition as eval_summary.csv mean_ep_len_success
            stop_at = steps + tail_steps
        if stop_at is not None and steps >= stop_at:
            break
    return {"seed": seed, "success": success_step is not None, "steps_to_success": success_step,
            "episode_length": steps, "n_chunks": len(ks), "k_sequence": ks, "init_state_hash": init_hash}  # fmt: skip


def make_actor(policy, steered, agent, deterministic):
    def act_base(obs_aug):
        return steered.base_action(obs_aug).reshape(-1).cpu().numpy(), None, None

    def act_residual(obs_aug):  # residual SAC sees [s, a_base(s, x_T)] instead of [s, x_T]
        a_base = steered.base_action(obs_aug).reshape(len(obs_aug), -1).cpu().numpy()
        obs_res = np.concatenate([obs_aug[:, : steered.obs_dim], a_base], axis=-1)
        action, _ = agent.predict_diffused(obs_res, deterministic=deterministic)
        return np.asarray(action).reshape(-1), None, None

    def act_agent(obs_aug):
        action, _ = agent.predict_diffused(obs_aug, deterministic=deterministic)
        k = int(np.asarray(agent.last_k).reshape(-1)[0])
        probs = None if agent.last_probs is None else np.asarray(agent.last_probs)[0]
        return np.asarray(action).reshape(-1), k, probs

    return {"base": act_base, "residual": act_residual}.get(policy, act_agent)


def ffmpeg_info(path):
    """(codec, pix_fmt, duration_s, n_frames) via the imageio-ffmpeg binary (no ffprobe on the cluster)."""
    import re
    import subprocess

    import imageio_ffmpeg

    out = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True, text=True, check=False).stderr
    m = re.search(r"Video: (\w+)[^,]*, (\w+)", out)
    n_frames, secs = imageio_ffmpeg.count_frames_and_secs(path)
    return (m.group(1) if m else None), (m.group(2) if m else None), float(secs), int(n_frames)


def rel_ckpt(ckpt):
    if ckpt is None:
        return None
    path = os.path.abspath(ckpt)
    return os.path.relpath(path, ROOT) if path.startswith(ROOT + os.sep) else path


@contextlib.contextmanager
def locked(path):
    with open(path + ".lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def update_json_list(path, entries, key):
    """Replace entries with the same ``key`` value and append the rest (process-safe)."""
    with locked(path):
        old = []
        if os.path.exists(path):
            with open(path) as f:
                old = json.load(f)
        new_keys = {e[key] for e in entries}
        merged = [e for e in old if e.get(key) not in new_keys] + entries
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(merged, f, indent=1)
        os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=["can", "square", "transport"])
    p.add_argument("--policy", required=True, choices=["base", "fixed_k", "adaptive", "residual"])
    p.add_argument("--ckpt", help="checkpoint/latest.pt or final.pt of a train_adsrl.py run (fixed_k/adaptive/residual)")
    p.add_argument("--k", type=int, help="fixed_k: steering step (checked against the run config)")
    p.add_argument("--ddim_steps", type=int, help="base: DDIM K (default: config; use 20 for transport)")
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--seed0", type=int, default=0, help="seeds are seed0..seed0+n_episodes-1")
    p.add_argument("--max_success", type=int, default=2)
    p.add_argument("--max_fail", type=int, default=2)
    p.add_argument("--cameras", default=None,
                   help="comma-separated robosuite cameras, side by side (default: DEFAULT_CAMERAS[task])")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--crf", type=int, default=26)
    p.add_argument("--tail_steps", type=int, default=30, help="env steps kept after success")
    p.add_argument("--stochastic", action="store_true", help="sample (k, w) instead of argmax k / mean w")
    p.add_argument("--method", help="override the method slug in file names")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_dir", default=os.path.join(ROOT, "report", "assets", "videos"))
    args = p.parse_args()

    import imageio

    t0 = time.time()
    if args.policy == "base":
        cfg = base_cfg(args.task, args.ddim_steps, args.device)
        steered, agent = build_steered(cfg), None
    else:
        assert args.ckpt, "--ckpt is required for fixed_k/adaptive/residual"
        cfg = run_cfg(args.ckpt, args.device)
        assert cfg.env_name == args.task, (cfg.env_name, args.task)
        want = {"fixed_k": "fixed"}.get(args.policy, args.policy)
        assert cfg.steer.mode == want, f"checkpoint is steer.mode={cfg.steer.mode}, not {want}"
        if args.policy == "fixed_k" and args.k is not None:
            assert int(cfg.steer.k) == args.k, f"checkpoint is k={cfg.steer.k}, not {args.k}"
        steered = build_steered(cfg)
        agent = build_agent(cfg, steered, args.ckpt)
    method = args.method or method_name(cfg, args.policy)
    label = METHOD_LABEL.get(method, f"fixed k={method[1:]}" if method.startswith("k") else method)
    print(f"task={args.task} method={method} K={steered.K} ckpt={args.ckpt}", flush=True)

    cams = (args.cameras or DEFAULT_CAMERAS[args.task]).split(",")
    env = make_env(cfg, args.size)
    act = make_actor(args.policy, steered, agent, deterministic=not args.stochastic)
    n_chunks = int(cfg.env.max_episode_steps / cfg.act_steps)
    overlay = Overlay(args.task, label, n_chunks, steered.K)
    os.makedirs(args.out_dir, exist_ok=True)

    records, clips, saved = [], [], {"success": 0, "fail": 0}
    for seed in range(args.seed0, args.seed0 + args.n_episodes):
        need = saved["success"] < args.max_success or saved["fail"] < args.max_fail
        tmp = os.path.join(args.out_dir, f".tmp_{args.task}__{method}__{seed}.mp4")
        writer = None
        if need:
            writer = imageio.get_writer(
                tmp, fps=args.fps, codec="libx264", pixelformat="yuv420p", macro_block_size=16, quality=None,
                ffmpeg_params=["-crf", str(args.crf), "-preset", "slow", "-movflags", "+faststart"],
            )  # fmt: skip
        te = time.time()
        rec = rollout(env, cfg, act, seed, overlay, writer, cams, args.size, args.tail_steps)
        records.append(rec)
        outcome = "success" if rec["success"] else "fail"
        print(f"seed={seed} {outcome} steps_to_success={rec['steps_to_success']} len={rec['episode_length']} "
              f"({time.time() - te:.0f}s)", flush=True)  # fmt: skip
        if writer is None:
            continue
        writer.close()
        quota = args.max_success if rec["success"] else args.max_fail
        if saved[outcome] >= quota:
            os.remove(tmp)
            continue
        saved[outcome] += 1
        name = f"{args.task}__{method}__{outcome}__{seed}.mp4"
        path = os.path.join(args.out_dir, name)
        os.replace(tmp, path)
        codec, pix_fmt, dur, n_frames = ffmpeg_info(path)
        clips.append({
            "file": name, "task": args.task, "method": method, "method_label": label, "seed": seed,
            "success": rec["success"], "episode_length": rec["episode_length"],
            "steps_to_success": rec["steps_to_success"], "k_sequence": rec["k_sequence"],
            "init_state_hash": rec["init_state_hash"], "size_bytes": os.path.getsize(path),
            "duration_s": round(dur, 2), "n_frames": n_frames, "fps": args.fps, "codec": codec,
            "pix_fmt": pix_fmt, "cameras": cams, "ddim_steps": steered.K, "deterministic": not args.stochastic,
            "ckpt": rel_ckpt(args.ckpt),
        })  # fmt: skip
        if os.path.getsize(path) > 2e6:
            print(f"  WARNING: {name} is over 2 MB; consider --cameras agentview or a higher --crf", flush=True)
        print(f"  saved {name} {os.path.getsize(path) / 1e6:.2f} MB {codec}/{pix_fmt} {dur:.1f}s", flush=True)

    n_succ = sum(r["success"] for r in records)
    stats = {
        "key": f"{args.task}__{method}", "task": args.task, "method": method, "ddim_steps": steered.K,
        "n_episodes": len(records), "n_success": n_succ, "success_rate": n_succ / len(records),
        "seeds": [r["seed"] for r in records], "episodes": records, "deterministic": not args.stochastic,
        "ckpt": rel_ckpt(args.ckpt), "wall_s": round(time.time() - t0, 1),
    }  # fmt: skip
    update_json_list(os.path.join(args.out_dir, "manifest.json"), clips, key="file")
    update_json_list(os.path.join(args.out_dir, "rollout_stats.json"), [stats], key="key")
    print("RENDER " + json.dumps({k: v for k, v in stats.items() if k != "episodes"}), flush=True)


if __name__ == "__main__":
    main()
