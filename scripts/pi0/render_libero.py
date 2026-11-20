"""Render LIBERO rollouts of steered pi0 / pi0.5 (or the plain base policy) to annotated MP4 clips.

    python scripts/pi0/render_libero.py --run <logdir of a train_adsrl_pi0.py run> --method pi05_k3 --label "..."
    python scripts/pi0/render_libero.py --base pi05 --task_id 8 --query_freq 10 --method pi05_base --label "..."

Episodes run on benchmark init states 0..n_episodes-1 through adsrl_pi0.rollout.run_episodes, exactly as
train_adsrl_pi0.run_eval does: trained runs restore <run>/checkpoint with build_agent + restore_agent and act
deterministically (argmax k, mean w) with run_eval's RNG setup (eval_index 10**6, as --eval_only). --base runs
plain sampling as scripts/pi0/base_eval.py --noise full (k=0, w ~ N(0, I) of shape (H, 32)).

Frames: each LIBERO worker records agentview and wrist images (256x256, rotated 180 degrees as in
adsrl_pi0.env.process_obs) after every env step and returns them with its reply. The clip shows
agentview | wrist under a header (method, episode = init state, step, chunk and its k) at 20 fps; the last frame
is held for --hold_s seconds (episodes end at success). Up to --max_success successes and --max_fail failures are
encoded with the imageio-ffmpeg binary of .venv-pi0 (H.264, yuv420p) as
<out_dir>/libero<task_id>__<method>__<success|fail>__<init state>.mp4 and added to <out_dir>/manifest.json;
per-episode results go to <out_dir>/rollout_stats.json.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import adsrl_pi0  # noqa: E402

adsrl_pi0.setup_env()

import numpy as np  # noqa: E402

from adsrl_pi0 import env as envmod  # noqa: E402  (numpy only at import)

_ORIG_WORKER = envmod._worker
HEADER_H = 48
INK, INK_MUTED, HEADER_BG, SUCCESS_BG, FAIL_BG = (240, 239, 236), (170, 169, 163), (26, 26, 25), (0, 131, 0), (176, 32, 32)
FONT = glob.glob(os.path.join(ROOT, ".venv-pi05/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"))
FFMPEG = glob.glob(os.path.join(ROOT, ".venv-pi0/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux*"))


def _frame(obs):
    a = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    w = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return np.concatenate([a, w], axis=1)


class _RecRemote:
    """Pipe proxy for a worker: replies to 'reset' and 'chunk' go out as (reply, frames recorded since the last
    command); other replies ('info', read directly by LiberoVecEnv.__init__) are sent unchanged."""

    def __init__(self, remote, frames):
        self.remote, self.frames, self.cmd = remote, frames, None

    def recv(self):
        msg = self.remote.recv()
        self.cmd = msg[0]
        self.frames.clear()
        return msg

    def send(self, x):
        self.remote.send((x, list(self.frames)) if self.cmd in ("reset", "chunk") else x)
        self.frames.clear()

    def close(self):
        self.remote.close()


def _rec_worker(remote, *args):
    """adsrl_pi0.env._worker with an OffScreenRenderEnv that records a frame after every step."""
    import libero.libero.envs as E

    frames = []
    base = E.OffScreenRenderEnv

    class Rec(base):
        def step(self, action):
            out = super().step(action)
            frames.append(_frame(out[0]))
            return out

    E.OffScreenRenderEnv = Rec
    _ORIG_WORKER(_RecRemote(remote, frames), *args)


class RecVecEnv(envmod.LiberoVecEnv):
    """LiberoVecEnv whose workers record frames; recordings[init_idx] = [frames after reset, chunk 1, ...]."""

    def __init__(self, *a, **kw):
        self._last = {}
        envmod._worker = _rec_worker  # LiberoVecEnv.__init__ looks the target up in the module
        try:
            super().__init__(*a, **kw)
        finally:
            envmod._worker = _ORIG_WORKER
        self.slot_idx, self.recordings = {}, {}

    def _recv(self, i):
        msg, frames = super()._recv(i)
        self._last[i] = frames
        return msg

    def reset(self, ids, init_idx):
        out = super().reset(ids, init_idx)
        for s, k in zip(ids, init_idx, strict=True):
            self.slot_idx[s] = int(k)
            self.recordings[int(k)] = [self._last[s][-1:]]  # the obs after the settle steps
        return out

    def step_chunk(self, ids, actions):
        out = super().step_chunk(ids, actions)
        for s in ids:
            self.recordings[self.slot_idx[s]].append(self._last[s])
        return out


class Overlay:
    def __init__(self, label, num_k, max_steps, show_probs):
        from PIL import ImageFont

        self.label, self.num_k, self.max_steps, self.show_probs = label, num_k, max_steps, show_probs
        self.f_big, self.f_small = ImageFont.truetype(FONT[0], 13), ImageFont.truetype(FONT[0], 11)

    def __call__(self, img, ep, step, chunk, k, probs, done):
        from PIL import Image, ImageDraw

        h, w, _ = img.shape
        c = Image.new("RGB", (w, h + HEADER_H), HEADER_BG)
        c.paste(Image.fromarray(img), (0, HEADER_H))
        d = ImageDraw.Draw(c)
        d.text((8, 6), f"LIBERO-10 #8  |  {self.label}", font=self.f_big, fill=INK)
        k_txt = "k = -" if k is None else f"k = {k}"
        if self.show_probs and probs is not None and k is not None:
            k_txt += f"  (π(k) = {float(probs[k]):.2f})"
        d.text((8, 27), f"episode {ep}   step {step:3d}/{self.max_steps}   chunk {chunk}   {k_txt}", font=self.f_small,
               fill=INK_MUTED)
        d.text((w - 118, 27), "agentview | wrist", font=self.f_small, fill=INK_MUTED)
        if done is not None:
            label, bg = (f"✓ success @ step {step}", SUCCESS_BG) if done else (f"✗ timed out @ step {step}", FAIL_BG)
            tw = d.textlength(label, font=self.f_small)
            d.rounded_rectangle([6, HEADER_H + 6, 6 + tw + 12, HEADER_H + 26], radius=4, fill=bg)
            d.text((12, HEADER_H + 9), label, font=self.f_small, fill=(255, 255, 255))
        return np.asarray(c)


def encode(path, frames, fps, crf):
    h, w, _ = frames[0].shape
    cmd = [FFMPEG[0], "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "slow",
           "-movflags", "+faststart", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f, np.uint8).tobytes())
    p.stdin.close()
    assert p.wait() == 0, f"ffmpeg failed on {path}"


def clip_frames(rec, e, overlay, query_freq, fps, hold_s):
    """Annotated frames of one episode; rec = recordings[e.init_idx] (reset frame + one list per chunk)."""
    assert len(rec) == len(e.k) + 1, (len(rec), len(e.k))
    out, step = [], 0
    k0 = e.k[0]
    p0 = e.k_probs[0] if e.k_probs else None
    out.append(overlay(rec[0][0], e.init_idx, 0, 1, k0, p0, None))
    for c, fr in enumerate(rec[1:]):
        probs = e.k_probs[c] if e.k_probs else None
        for f in fr:
            step += 1
            out.append(overlay(f, e.init_idx, step, c + 1, e.k[c], probs, None))
    assert step == e.steps, (step, e.steps)
    last = rec[-1][-1]
    out += [overlay(last, e.init_idx, step, len(e.k), e.k[-1], e.k_probs[-1] if e.k_probs else None, e.success)] \
        * int(hold_s * fps)
    return out


def update_json_list(path, entries, key):
    """Replace entries with the same `key` value and append the rest (same as scripts/render_rollouts.py, which
    imports torch and so cannot be imported from the pi0 venvs)."""
    import fcntl

    with open(path + ".lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        old = json.load(open(path)) if os.path.exists(path) else []
        new_keys = {e[key] for e in entries}
        merged = [e for e in old if e.get(key) not in new_keys] + entries
        with open(path + ".tmp", "w") as f:
            json.dump(merged, f, indent=1)
        os.replace(path + ".tmp", path)
        fcntl.flock(lf, fcntl.LOCK_UN)


def main():
    import jax

    import train_adsrl_pi0 as T
    from adsrl_pi0.agent import restore_agent
    from adsrl_pi0.rollout import run_episodes
    from adsrl_pi0.sampler import SteeredPi0

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", nargs="*", default=[], help="train_adsrl_pi0.py logdirs (same model)")
    p.add_argument("--method", nargs="+", required=True, help="method slug per run (then the base one, if --base)")
    p.add_argument("--label", nargs="+", required=True, help="overlay label per method")
    p.add_argument("--base", choices=["pi0", "pi05"], default=None, help="also render the plain base policy")
    p.add_argument("--task_id", type=int, default=8)
    p.add_argument("--query_freq", type=int, default=10, help="--base only")
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--base_episodes", type=int, default=20)
    p.add_argument("--n_envs", type=int, default=10)
    p.add_argument("--max_success", type=int, default=2)
    p.add_argument("--max_fail", type=int, default=2)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--crf", type=int, default=28)
    p.add_argument("--hold_s", type=float, default=1.5)
    p.add_argument("--out_dir", default=os.path.join(ROOT, "report", "assets", "videos"))
    a = p.parse_args()
    assert len(a.method) == len(a.label) == len(a.run) + (a.base is not None)
    assert FONT and FFMPEG, (FONT, FFMPEG)

    infos = [json.load(open(os.path.join(r, "run_info.json"))) for r in a.run]
    models = {i["args"].get("model", "pi0") for i in infos} | ({a.base} if a.base else set())
    assert len(models) == 1, models
    model = models.pop()
    pi0 = SteeredPi0.load_model(model)
    venv = RecVecEnv(a.n_envs, a.task_id, suite="libero_10", seed=0)
    print(f"task: {venv.prompt}", flush=True)
    entries, stats = [], []

    jobs = [(r, i) for r, i in zip(a.run, infos, strict=True)] + ([(None, None)] if a.base else [])
    for (run, info), method, label in zip(jobs, a.method, a.label, strict=True):
        t0 = time.time()
        if run is not None:
            args = T.parse(["--task_id", str(a.task_id)])
            vars(args).update(info["args"])
            assert args.task_id == a.task_id and args.suite == "libero_10"
            agent = T.build_agent(args)
            step = restore_agent(agent, os.path.join(run, "checkpoint"))
            assert step is not None, run
            keys = list(T.obs_spec(args)[0])
            assert keys == info["obs_keys"], (keys, info["obs_keys"])
            seed, eval_index, det = args.seed, 10**6, True  # run_eval's RNGs, as --eval_only
            rng = np.random.default_rng([seed, 456, eval_index, int(det)])
            agent.set_rng(jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(seed + 456), eval_index), int(det)))

            def act_fn(batch, agent=agent, keys=keys):
                return agent.act(jax.device_put(T.agent_input(batch, keys)), deterministic=True)

            qf, kw, idx = args.query_freq, T.rollout_kw(args), list(range(a.n_episodes))
            ckpt = f"{run}/checkpoint/agent_{step}"
        else:
            rng = np.random.default_rng(a.task_id)  # base_eval.py: default_rng(seed * 1000 + task_id), seed 0

            def act_fn(batch, rng=rng):
                B = batch["state"].shape[0]
                return np.zeros(B, np.int32), rng.standard_normal((B, pi0.horizon, 32)).astype(np.float32), None

            qf, kw, idx, ckpt = a.query_freq, {}, list(range(a.base_episodes)), f"{model} base (openpi)"
        venv.recordings = {}
        eps, _ = run_episodes(venv, pi0, idx, act_fn, qf, rng, compare_base=run is not None, **kw)
        eps.sort(key=lambda e: e.init_idx)
        n_s = sum(e.success for e in eps)
        print(f"[{method}] {n_s}/{len(eps)} success; lengths "
              f"{[(e.init_idx, e.steps, int(e.success)) for e in eps]} ({time.time() - t0:.0f}s)", flush=True)
        stats.append({"key": f"libero{a.task_id}__{method}", "task": f"libero{a.task_id}", "method": method,
                      "query_freq": qf, "ckpt": ckpt, "n_episodes": len(eps), "n_success": int(n_s),
                      "success_rate": n_s / len(eps),
                      "episodes": [{"seed": e.init_idx, "success": e.success, "episode_length": e.steps,
                                    "k_sequence": [int(k) for k in e.k]} for e in eps]})
        overlay = Overlay(label, T.NUM_STEPS, venv.max_steps, show_probs=run is not None and args.mode == "adaptive")
        chosen = [e for e in eps if e.success][: a.max_success] + [e for e in eps if not e.success][: a.max_fail]
        for e in chosen:
            tag = "success" if e.success else "fail"
            fname = f"libero{a.task_id}__{method}__{tag}__{e.init_idx}.mp4"
            path = os.path.join(a.out_dir, fname)
            frames = clip_frames(venv.recordings[e.init_idx], e, overlay, qf, a.fps, a.hold_s)
            encode(path, frames, a.fps, a.crf)
            entries.append({"file": fname, "task": f"libero{a.task_id}", "method": method, "method_label": label,
                            "seed": e.init_idx, "success": e.success, "episode_length": e.steps,
                            "steps_to_success": e.steps if e.success else None, "k_sequence": [int(k) for k in e.k],
                            "size_bytes": os.path.getsize(path), "n_frames": len(frames), "fps": a.fps,
                            "codec": "h264", "pix_fmt": "yuv420p", "cameras": ["agentview", "robot0_eye_in_hand"],
                            "query_freq": qf, "deterministic": run is not None, "ckpt": ckpt})
            print(f"  wrote {fname} ({os.path.getsize(path) / 1e6:.2f} MB, {len(frames)} frames)", flush=True)
        del venv.recordings
    venv.close()

    update_json_list(os.path.join(a.out_dir, "manifest.json"), entries, "file")
    update_json_list(os.path.join(a.out_dir, "rollout_stats.json"), stats, "key")


if __name__ == "__main__":
    main()
