"""Train DSRL-NA steered at a fixed or adaptive denoising step (trimmed copy of dsrl/train_dsrl.py).

    python train_adsrl.py --config-name=can steer.mode=fixed steer.k=0
    python train_adsrl.py --config-name=can steer.mode=adaptive steer.rho=0.5
    python train_adsrl.py --config-name=can steer.mode=residual steer.residual_beta=0.1
    python train_adsrl.py --config-name=can name=<same as before> resume=true

Units: ``train.total_timesteps`` and every ``env_steps`` value are raw env steps (chunks x
act_steps, summed over the training envs), including the warm-start base-policy rollouts.
SB3's internal ``num_timesteps`` counts chunks.

Outputs in ``logdir``: checkpoint/latest.pt (+ replay buffer), checkpoint/final.pt,
eval_summary.csv, eval_rollouts/step_<env_steps>[_det].npz, run_info.json, wandb_id.txt.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DSRL_DIR = os.path.join(PROJECT_ROOT, "dsrl")
for _p in (PROJECT_ROOT, DSRL_DIR, os.path.join(DSRL_DIR, "dppo")):
    if _p not in sys.path:
        sys.path.append(_p)

try:  # env.env_type=furniture: isaacgym must be imported before torch (only installed in .venv-furniture)
    import isaacgym  # noqa: F401, E402
except ImportError:
    pass

import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: E402

from adsrl.agent import DSRLAdaptiveK, DSRLFixedK, SteerSACPolicy  # noqa: E402
from adsrl.base_policy import SteeredBasePolicy, load_w_scale_dims  # noqa: E402
from adsrl.env import XTAugmentWrapper  # noqa: E402
from adsrl.residual import BaseActionObsWrapper, ResidualSAC  # noqa: E402

for _name, _fn in [("eval", eval), ("round_up", math.ceil), ("round_down", math.floor)]:
    OmegaConf.register_new_resolver(_name, _fn, replace=True)


def abs_path(p, root=DSRL_DIR):
    """Config paths are relative to dsrl/ (as for upstream train_dsrl.py, which runs from dsrl/)."""
    return p if p is None or os.path.isabs(p) else os.path.normpath(os.path.join(root, p))


def steer_w_scale(steer) -> float:
    """steer.w_scale (default 1): the sampler decodes w_scale * w. Residual SAC has no w, so only 1 is allowed."""
    c = float(steer.get("w_scale", 1.0))
    if steer.mode == "residual" and c != 1.0:
        raise ValueError(f"steer.w_scale={c} is not supported in residual mode (no steering noise)")
    return c


def steer_w_scale_dims(steer, horizon: int, act_dim: int):
    """steer.w_scale_dims (default null): path (relative to the project root) to a per-dim noise scale, applied on
    top of steer.w_scale; see adsrl.base_policy.load_w_scale_dims. Residual SAC has no noise, so only null."""
    path = steer.get("w_scale_dims")
    if path is None:
        return None
    if steer.mode == "residual":
        raise ValueError("steer.w_scale_dims is not supported in residual mode (no steering noise)")
    return load_w_scale_dims(abs_path(path, PROJECT_ROOT), horizon, act_dim)


def steer_param(steer) -> str:
    """steer.param (default absolute): absolute = the actor's w is the step-k noise; kick = the noise is
    eps_hat_k + w_scale * w (w = 0 decodes the base action). Residual SAC has no noise, so only absolute."""
    p = steer.get("param", "absolute")
    if steer.mode == "residual" and p != "absolute":
        raise ValueError(f"steer.param={p} is not supported in residual mode (no steering noise)")
    return p


def steer_actor_zero_init(steer) -> bool:
    """steer.actor_zero_init (default False): zero the actor's mean output layer (fixed/adaptive only)."""
    z = bool(steer.get("actor_zero_init", False))
    if steer.mode == "residual" and z:
        raise ValueError("steer.actor_zero_init is only implemented for fixed/adaptive")
    return z


def resolve_ent_coef(v):
    """train.ent_coef: -1 -> "auto" (learned, init 1.0); "auto" / "auto_<init>" -> learned with that initial
    value (SB3 parses it); any number -> fixed."""
    if isinstance(v, str):
        if v != "auto" and not (v.startswith("auto_") and float(v[len("auto_"):]) > 0):
            raise ValueError(f"train.ent_coef={v!r}: use -1, 'auto', 'auto_<init>' or a number")
        return v
    return "auto" if v == -1 else float(v)


def early_eval_schedule(train) -> tuple:
    """(every, until, episodes) of the extra deterministic evals, or None when train.early_eval_every is unset."""
    every = train.get("early_eval_every")
    if every is None:
        return None
    until = train.get("early_eval_until")
    assert until is not None and int(every) > 0, "train.early_eval_every needs train.early_eval_until"
    return int(every), int(until), int(train.get("early_eval_episodes", 50))


def early_eval_due(env_steps: int, step_size: int, every: int, until: int) -> bool:
    """True on the step that crosses a multiple m of ``every`` with m <= until (env steps). Stateless in
    env_steps, so it survives resume (env_steps is restored from the checkpoint)."""
    m = (env_steps // every) * every
    return env_steps - step_size < m <= until and m > 0


def env_type(cfg) -> str:
    """cfg.env.env_type: robomimic (default; SubprocVecEnv + separate eval envs) | furniture (one
    IsaacGym FurnitureVecEnv, also used for evals)."""
    t = cfg.env.get("env_type", "robomimic")
    assert t in ("robomimic", "furniture"), t
    return t


def run_info(cfg):
    def git(*args, cwd=PROJECT_ROOT):
        try:
            return subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "git_commit_dsrl": git("rev-parse", "HEAD", cwd=DSRL_DIR),
        "git_commit_dppo": git("rev-parse", "HEAD", cwd=os.path.join(DSRL_DIR, "dppo")),
        "git_commit_sb3": git("rev-parse", "HEAD", cwd=os.path.join(DSRL_DIR, "stable-baselines3")),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "runs_per_gpu": float(os.environ.get("ADSRL_RUNS_PER_GPU", 1)),
        "argv": sys.argv,
        "w_scale": steer_w_scale(cfg.steer),
        "steer_param": steer_param(cfg.steer),
        "w_scale_dims": cfg.steer.get("w_scale_dims"),
        "sb3_path": os.path.dirname(sys.modules["stable_baselines3"].__file__),
    }


def collect_base_rollouts(model, env, num_steps, steered, residual=False):
    """Warm-start the buffer with the unsteered base policy a_base(s, x_T) (upstream
    ``collect_rollouts``, NA branch). With x_T ~ N(0, I) in the obs this is the base policy.
    Residual mode: obs is [s, a_base], the env executes a_base and the buffer stores da = 0."""
    obs = env.reset()
    for _ in range(num_steps):
        if residual:
            action = model.a_base(obs)
            buffer_action = np.zeros_like(action)  # scaled residual of a zero da
        else:
            action = steered.base_action(obs).reshape(obs.shape[0], -1).cpu().numpy()
            buffer_action = action
        next_obs, reward, done, info = env.step(action)
        real_next = next_obs.copy()
        for i, d in enumerate(done):
            if d and info[i].get("terminal_observation") is not None:
                real_next[i] = info[i]["terminal_observation"]
        model.replay_buffer.add(obs=obs, next_obs=real_next, action=buffer_action, reward=reward, done=done, infos=info)
        obs = next_obs
    model.replay_buffer.final_offline_step()


def resolve_xt_inputs(steer):
    """steer.xT_inputs: auto | none | actor_noise_critic | all. auto = none at fixed k=0 with absolute noise
    (x_T never affects the action there), else actor_noise_critic (Q(s,a) never needs x_T). Kick at k=0
    decodes DDIM(x_T + Delta), so x_T matters and none is rejected."""
    mode = steer.get("xT_inputs", "auto")
    if steer.mode == "residual":
        return "all"  # residual obs is [s, a_base]; there is no x_T to drop
    kick = steer.get("param", "absolute") == "kick"
    if mode == "auto":
        mode = "none" if steer.mode == "fixed" and steer.k == 0 and not kick else "actor_noise_critic"
    assert mode in ("none", "actor_noise_critic", "all"), mode
    assert mode != "none" or (steer.mode == "fixed" and steer.k == 0), "xT_inputs=none requires fixed k=0"
    assert mode != "none" or not kick, "xT_inputs=none is invalid with steer.param=kick (the action depends on x_T)"
    return mode


def chunk_success(rewards, infos, rew_offset, use_flag=False):
    """Per-env success of the last chunk. robomimic: reward > -reward_offset (unchanged). furniture
    (use_flag): infos[i]["chunk_success"] = all parts assembled, so partial-credit rewards (lamp,
    round_table) are never counted as success."""
    if use_flag:
        return np.array([bool(info["chunk_success"]) for info in infos])
    return np.asarray(rewards) > -rew_offset


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0] / max(p.sum(), 1e-12)
    return float(-(p * np.log(p)).sum()) if p.size else 0.0


class AdsrlCallback(BaseCallback):
    """Train/eval logging, eval rollout dumps, eval_summary.csv and periodic checkpoints."""

    CSV_FIELDS = [
        "env_steps", "mode", "wall_s", "gpu_s", "success", "reward", "mean_ep_len_success",
        "mean_a_minus_base", "k_entropy", "pi_k_entropy", "episodes",
    ]  # fmt: skip

    def __init__(self, cfg, eval_env, steered, obs_min, obs_max, wandb_run, state=None):
        super().__init__(0)
        self.cfg = cfg
        self.eval_env = eval_env
        self.steered = steered
        self.obs_min, self.obs_max = obs_min, obs_max
        self.wandb = wandb_run
        self.act_steps = cfg.act_steps
        self.n_train = cfg.env.n_envs
        self.n_eval = cfg.env.n_eval_envs
        self.rew_offset = cfg.env.reward_offset
        # furniture: eval_env is the training env (one IsaacGym sim per process); evals run whole
        # lockstep episodes of ceil(max_episode_steps / act_steps) chunks, at episode boundaries.
        self.shared_env = env_type(cfg) == "furniture"
        if self.shared_env:
            self.max_steps = math.ceil(cfg.env.max_episode_steps / cfg.act_steps)
            assert cfg.eval_interval % self.max_steps == 0, (cfg.eval_interval, self.max_steps)
        else:
            self.max_steps = int(cfg.env.max_episode_steps / cfg.act_steps)
        self.log_freq = self.max_steps
        self.log_stdout = bool(cfg.get("log_train_stdout", False))  # also print train throughput lines
        self.eval_freq = cfg.eval_interval
        self.eval_rounds = max(int(cfg.num_evals / self.n_eval), 1)
        # train.early_eval_*: extra det-only evals (no dump/checkpoint, n_evals unchanged) up to `until`
        self.early = early_eval_schedule(cfg.get("train") or {})
        if self.early is not None:
            assert not self.shared_env, "early evals need separate eval envs (robomimic)"
        self.runs_per_gpu = float(os.environ.get("ADSRL_RUNS_PER_GPU", 1))
        self.logdir = cfg.logdir
        # residual SAC: eval obs is [s, a_base] (BaseActionObsWrapper), x_T lives in eval_env.x_T
        self.residual = cfg.steer.mode == "residual"
        os.makedirs(os.path.join(self.logdir, "eval_rollouts"), exist_ok=True)
        os.makedirs(os.path.join(self.logdir, "checkpoint"), exist_ok=True)
        state = state or {}
        self.env_steps = state.get("env_steps", 0)
        self.n_evals = state.get("n_evals", 0)
        self.wall_offset = state.get("wall_s", 0.0)
        self.gpu_offset = state.get("gpu_s", 0.0)
        self.last_eval_steps = state.get("last_eval_steps", -1)  # env_steps of the last eval
        self.last_eval_dumped = state.get("last_eval_dumped", False)
        self.t_start = time.time()
        self.last_ckpt_t = self.t_start
        self._reset_train_stats()

    # ---- bookkeeping ----
    def wall_s(self):
        return self.wall_offset + (time.time() - self.t_start)

    def gpu_s(self):
        return self.gpu_offset + (time.time() - self.t_start) / self.runs_per_gpu

    def state(self):
        return {"env_steps": self.env_steps, "n_evals": self.n_evals, "wall_s": self.wall_s(), "gpu_s": self.gpu_s(),
                "last_eval_steps": self.last_eval_steps, "last_eval_dumped": self.last_eval_dumped}  # fmt: skip

    def log(self, d):
        d = {**d, "env_steps": self.env_steps, "time/wall_s": self.wall_s(), "time/gpu_s": self.gpu_s()}
        if self.wandb is not None:
            self.wandb.log(d)

    def _reset_train_stats(self):
        self.ep_rewards, self.ep_lengths = [], []
        self.ep_success = np.zeros(self.n_train)
        self.ep_completed = np.zeros(self.n_train)

    # ---- training ----
    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
                self.ep_lengths.append(info["episode"]["l"])
        rew = self.locals["rewards"]
        self.ep_success[chunk_success(rew, self.locals["infos"], self.rew_offset, self.shared_env)] = 1
        self.ep_completed[self.locals["dones"]] = 1
        self.env_steps += self.act_steps * self.model.n_envs

        if self.n_calls % self.log_freq == 0:
            self.log_train()
        if self.n_calls % self.eval_freq == 0:
            self.run_evals()
            if self.shared_env:
                # The eval episodes moved the shared env. This step ended a training episode in every
                # env (eval_interval is a multiple of the episode length), so the stored transitions
                # use infos' terminal_observation and only the next obs is replaced, in place, because
                # SB3 sets _last_obs = new_obs after this callback returns.
                assert np.all(self.locals["dones"]), "furniture evals must run at episode boundaries"
                self.locals["new_obs"][...] = self.eval_env.reset()
            if self.n_evals % self.cfg.checkpoint_every_evals == 0:
                self.save_checkpoint()
        elif self.early is not None and early_eval_due(self.env_steps, self.act_steps * self.model.n_envs, *self.early[:2]):
            self.evaluate(deterministic=True, dump=False, rounds=max(self.early[2] // self.n_eval, 1))
        ckpt_min = self.cfg.get("checkpoint_every_min") or 0
        if ckpt_min > 0 and time.time() - self.last_ckpt_t > 60 * ckpt_min:
            self.save_checkpoint()  # bounds the work lost when a multi-day run hits its walltime
        return True

    def log_train(self):
        stats = dict(getattr(self.model, "last_train_stats", {}) or {})
        d = {k if k.startswith("train/") else f"train/{k}": v for k, v in stats.items()}
        if self.ep_completed.sum() > 0:
            d["train/success_rate"] = float(self.ep_success.sum() / self.ep_completed.sum())
        if self.ep_rewards:
            d["train/ep_rew_mean"] = float(np.mean(self.ep_rewards))
        counts = self.model.pop_k_counts()
        if counts is not None and counts.sum() > 0:
            frac = counts / counts.sum()
            d["train/k_hist_entropy"] = entropy(counts)
            d["train/k_mean"] = float((frac * np.arange(len(frac))).sum())
            if self.wandb is not None:
                import wandb

                d["train/k_hist"] = wandb.Histogram(np_histogram=(counts, np.arange(len(counts) + 1)))
            if len(frac) <= 16:
                d.update({f"train/k_frac/k{i}": float(f) for i, f in enumerate(frac)})
        self.log(d)
        if self.log_stdout:
            print(f"[train] env_steps={self.env_steps} wall_s={self.wall_s():.1f} n_updates={self.model._n_updates} "
                  f"success={d.get('train/success_rate', float('nan')):.3f} "
                  f"gpu_mem_gb={torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0:.2f}",
                  flush=True)  # fmt: skip
        self._reset_train_stats()

    # ---- eval ----
    def raw_obs(self, obs_aug):
        obs = obs_aug[:, : self.cfg.obs_dim]
        return (obs / 2 + 0.5) * (self.obs_max - self.obs_min + 1e-6) + self.obs_min

    def run_evals(self, final=False):
        dump = final or (self.n_evals % self.cfg.eval_dump_every == 0)
        if final and self.env_steps == self.last_eval_steps and self.last_eval_dumped:
            return  # the last periodic eval already covered this step (avoids duplicate rows / overwritten dumps)
        if self.cfg.get("stochastic_eval", True):
            self.evaluate(deterministic=False, dump=dump)
        if self.cfg.deterministic_eval:
            self.evaluate(deterministic=True, dump=dump)
        self.n_evals += 1
        self.last_eval_steps, self.last_eval_dumped = self.env_steps, dump

    @torch.no_grad()
    def evaluate(self, deterministic=False, dump=True, rounds=None):
        agent, env = self.model, self.eval_env
        n, H, K = self.n_eval, self.max_steps, getattr(self.steered, "K", 1)
        rounds = self.eval_rounds if rounds is None else rounds
        rows = {k: [] for k in ["episode", "chunk", "env_t", "obs", "obs_raw", "x_T", "action", "base_action",
                                "k", "w", "k_probs", "reward", "success_step"]}  # fmt: skip
        ep_success, ep_len_success, rews, dists, ks, pik_ent = [], [], [], [], [], []
        t_eval = time.time()
        for r in range(rounds):
            obs = env.reset()
            first_success = np.full(n, -1)
            for t in range(H):
                action, _ = agent.predict_diffused(obs, deterministic=deterministic)
                if self.residual:
                    base, x_T = obs[:, self.cfg.obs_dim :], env.x_T.reshape(n, -1).copy()
                else:
                    base = self.steered.base_action(obs).reshape(n, -1).cpu().numpy()
                    x_T = obs[:, self.cfg.obs_dim :]
                k = np.broadcast_to(np.asarray(agent.last_k), (n,)).astype(np.int64)
                w = agent.last_w if agent.last_w is not None else np.full_like(action, np.nan)
                probs = getattr(agent, "last_probs", None)
                next_obs, reward, done, info = env.step(action)
                succ = chunk_success(reward, info, self.rew_offset, self.shared_env)
                first_success[(first_success < 0) & succ] = t
                dists.append(np.linalg.norm(action - base, axis=-1))
                ks.append(k)
                rews.append(reward)
                if probs is not None:
                    pik_ent.append(-(probs * np.log(np.clip(probs, 1e-12, None))).sum(-1))
                if dump:
                    rows["episode"].append(r * n + np.arange(n))
                    rows["chunk"].append(np.full(n, t))
                    rows["env_t"].append(np.full(n, t * self.act_steps))
                    rows["obs"].append(obs[:, : self.cfg.obs_dim])
                    rows["obs_raw"].append(self.raw_obs(obs))
                    rows["x_T"].append(x_T)
                    rows["action"].append(action)
                    rows["base_action"].append(base)
                    rows["k"].append(k)
                    rows["w"].append(w)
                    if probs is not None:
                        rows["k_probs"].append(probs)
                    rows["reward"].append(reward)
                    rows["success_step"].append(succ)
                obs = next_obs
            ep_success.append(first_success >= 0)
            ep_len_success.extend(((first_success[first_success >= 0] + 1) * self.act_steps).tolist())

        ep_success = np.concatenate(ep_success)
        ks = np.concatenate(ks)
        k_counts = np.bincount(ks[ks >= 0], minlength=K)  # residual SAC has no k (last_k = -1)
        res = {
            "success": float(ep_success.mean()),
            "reward": float(np.mean(rews)),
            "mean_ep_len_success": float(np.mean(ep_len_success)) if ep_len_success else float("nan"),
            "mean_a_minus_base": float(np.mean(np.concatenate(dists))),
            "k_entropy": entropy(k_counts),
            "pi_k_entropy": float(np.mean(np.concatenate(pik_ent))) if pik_ent else float("nan"),
            "episodes": int(ep_success.size),
        }
        mode = "det" if deterministic else "stoch"
        sfx = "_deterministic" if deterministic else ""
        d = {f"eval/{k}{sfx}": v for k, v in res.items()}
        d[f"eval/success_rate{sfx}"] = res["success"]
        d["eval/timesteps"] = self.env_steps
        if self.wandb is not None:
            import wandb

            d[f"eval/k_hist{sfx}"] = wandb.Histogram(np_histogram=(k_counts, np.arange(K + 1)))
        self.log(d)
        print(f"[eval {mode}] env_steps={self.env_steps} success={res['success']:.3f} "
              f"k_entropy={res['k_entropy']:.3f} |a-a_base|={res['mean_a_minus_base']:.3f}", flush=True)  # fmt: skip
        if self.log_stdout:
            print(f"[eval {mode}] {n * rounds} episodes in {time.time() - t_eval:.1f}s", flush=True)

        self.write_csv_row({"env_steps": self.env_steps, "mode": mode, "wall_s": round(self.wall_s(), 1),
                            "gpu_s": round(self.gpu_s(), 1), **res})  # fmt: skip

        if dump:
            ep_succ_rows = np.repeat(ep_success.reshape(rounds, 1, n), H, axis=1).reshape(-1)
            arrays = {k: np.concatenate(v) for k, v in rows.items() if v}
            arrays["episode_success"] = ep_succ_rows
            arrays["meta"] = np.array(json.dumps({
                "env_steps": self.env_steps, "deterministic": deterministic, "mode": self.cfg.steer.mode,
                "fixed_k": self.cfg.steer.k, "num_k": K, "act_steps": self.act_steps,
                "action_space": "normalized model space [-1,1]; env unnormalizes",
                "w": "residual da (residual mode)" if self.residual else
                     "kick Delta (noise = eps_hat_k + w_scale * Delta)" if steer_param(self.cfg.steer) == "kick" else
                     "steering noise w",
            }))  # fmt: skip
            name = f"step_{self.env_steps}{'_det' if deterministic else ''}.npz"
            np.savez_compressed(os.path.join(self.logdir, "eval_rollouts", name), **arrays)
        return res

    def write_csv_row(self, row):
        """Appends to eval_summary.csv. An existing file keeps its header, so CSVs started before the
        "episodes" column stay aligned (the new column is dropped there)."""
        csv_path = os.path.join(self.logdir, "eval_summary.csv")
        new = not os.path.exists(csv_path)
        fields = self.CSV_FIELDS
        if not new:
            with open(csv_path, newline="") as f:
                fields = next(csv.reader(f), None) or fields
        with open(csv_path, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new:
                wr.writeheader()
            wr.writerow(row)

    def save_checkpoint(self, name="latest"):
        path = os.path.join(self.logdir, "checkpoint", f"{name}.pt")
        self.model.save_checkpoint(path, extra={"callback": self.state()}, save_replay_buffer=name == "latest")
        self.last_ckpt_t = time.time()
        print(f"saved checkpoint {path} at env_steps={self.env_steps}", flush=True)


@hydra.main(config_path=os.path.join(PROJECT_ROOT, "cfg/adsrl"), config_name="can", version_base=None)
def main(cfg):
    for key in ["base_policy_path", "offline_data_path", "dppo_path", "normalization_path"]:
        if cfg.get(key) is not None:
            cfg[key] = abs_path(cfg[key])
    cfg.log_dir = abs_path(cfg.log_dir, PROJECT_ROOT)
    OmegaConf.resolve(cfg)
    assert cfg.algorithm == "dsrl_na", "train_adsrl.py only supports DSRL-NA"

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    os.makedirs(cfg.logdir, exist_ok=True)
    ckpt_path = os.path.join(cfg.logdir, "checkpoint", "latest.pt")
    resuming = bool(cfg.resume) and os.path.exists(ckpt_path)
    if os.path.exists(ckpt_path) and not cfg.resume:
        raise FileExistsError(f"{ckpt_path} exists; pass resume=true or a different name=")

    n_env = cfg.env.n_envs
    furniture = env_type(cfg) == "furniture"
    if furniture:
        # IsaacGym: create the sim before torch initializes CUDA in this process; evals share it
        from adsrl.furniture_env import make_furniture_vec_env

        furniture_venv = make_furniture_vec_env(cfg)
        cfg.env.n_eval_envs = n_env
        chunks = furniture_venv.chunks_per_episode
        assert cfg.train.init_rollout_steps % chunks == 0, f"init_rollout_steps must be a multiple of {chunks}"

    info = run_info(cfg)
    print(json.dumps(info, indent=1), flush=True)
    with open(os.path.join(cfg.logdir, "run_info.json" if not resuming else "run_info_resume.json"), "w") as f:
        json.dump({"run": info, "cfg": OmegaConf.to_container(cfg, resolve=True)}, f, indent=1)

    wandb_run = None
    if cfg.use_wandb:
        import wandb

        id_file = os.path.join(cfg.logdir, "wandb_id.txt")
        run_id = open(id_file).read().strip() if os.path.exists(id_file) else wandb.util.generate_id()
        with open(id_file, "w") as f:
            f.write(run_id)
        wandb_run = wandb.init(
            project=cfg.wandb.project, name=cfg.name, group=cfg.wandb.group, mode=cfg.wandb.mode,
            id=run_id, resume="allow", dir=cfg.logdir, save_code=True,
            config={**OmegaConf.to_container(cfg, resolve=True), "run_info": info},
        )  # fmt: skip
        wandb_run.define_metric("env_steps")
        wandb_run.define_metric("*", step_metric="env_steps")

    # ---- base policy and envs ----
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    base_model = hydra.utils.instantiate(OmegaConf.create(model_cfg)).eval()
    steered = SteeredBasePolicy(base_model, cfg.obs_dim, cfg.act_steps, cfg.action_dim,
                                w_scale=steer_w_scale(cfg.steer), param=steer_param(cfg.steer),
                                w_scale_dims=steer_w_scale_dims(cfg.steer, cfg.act_steps, cfg.action_dim))
    print(f"DDIM steps K={steered.K}, w_scale={steered.w_scale}, param={steered.param}, "
          f"w_scale_dims={cfg.steer.get('w_scale_dims')}", flush=True)
    if steered.w_scale_dims is not None:
        d = steered.w_scale_dims
        print(f"w_scale_dims shape={tuple(d.shape)} min={float(d.min()):.4f} max={float(d.max()):.4f} "
              f"sum={float(d.double().sum()):.6f}", flush=True)

    residual = cfg.steer.mode == "residual"
    if furniture:
        env = XTAugmentWrapper(furniture_venv, cfg.act_steps, cfg.action_dim)
        env.seed(cfg.seed + 1)
        if residual:
            env = BaseActionObsWrapper(env, steered)
        eval_env = env
        norm = {"obs_min": furniture_venv.obs_min, "obs_max": furniture_venv.obs_max}
    else:
        from env_utils import ActionChunkWrapper, ObservationWrapperRobomimic, make_robomimic_env

        def make_env():
            env = make_robomimic_env(
                env=cfg.env_name, normalization_path=cfg.normalization_path,
                low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys, dppo_path=cfg.dppo_path,
            )  # fmt: skip
            env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
            return ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)

        env = XTAugmentWrapper(make_vec_env(make_env, n_envs=n_env, vec_env_cls=SubprocVecEnv), cfg.act_steps, cfg.action_dim)
        env.seed(cfg.seed + 1)
        eval_env = XTAugmentWrapper(
            make_vec_env(make_env, n_envs=cfg.env.n_eval_envs, vec_env_cls=SubprocVecEnv), cfg.act_steps, cfg.action_dim
        )
        eval_env.seed(cfg.seed + n_env + 1)
        if residual:
            env, eval_env = BaseActionObsWrapper(env, steered), BaseActionObsWrapper(eval_env, steered)
        norm = np.load(cfg.normalization_path)

    # ---- agent ----
    net_arch = [cfg.train.layer_size] * cfg.train.num_layers
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, qf=net_arch),
        activation_fn=torch.nn.Tanh,
        log_std_init=0.0,
        post_linear_modules=[torch.nn.LayerNorm] if cfg.train.use_layer_norm else None,
        n_critics=cfg.train.n_critics,
    )
    policy_kwargs.update(xt_inputs=resolve_xt_inputs(cfg.steer), state_dim=cfg.obs_dim)
    print(f"xT_inputs={policy_kwargs['xt_inputs']}", flush=True)
    common = dict(
        learning_rate=cfg.train.actor_lr,
        buffer_size=int(cfg.train.buffer_size),
        learning_starts=1,
        batch_size=cfg.train.batch_size,
        tau=cfg.train.tau,
        gamma=cfg.train.discount,
        train_freq=cfg.train.train_freq,
        gradient_steps=cfg.train.utd,
        action_noise=None,
        optimize_memory_usage=False,
        ent_coef=resolve_ent_coef(cfg.train.ent_coef),
        target_update_interval=1,
        target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,
        verbose=1,
        policy_kwargs=policy_kwargs,
        device=cfg.device,
        noise_critic_grad_steps=cfg.train.noise_critic_grad_steps,
        critic_backup_combine_type=cfg.train.critic_backup_combine_type,
        steered_policy=steered,
        horizon=cfg.act_steps,
        act_dim=cfg.action_dim,
        use_tf32=cfg.train.tf32,
        compile_networks=cfg.train.compile,
        compile_mode=cfg.train.compile_mode,
    )
    if cfg.train.get("actor_gradient_steps") is not None:  # optional: actor steps per train() call (upstream default -1 = every step)
        common["actor_gradient_steps"] = int(cfg.train.actor_gradient_steps)
    tr = cfg.train.get("trust_region")
    trust_region = dict(bound=tr.bound, lr=tr.lr, init=tr.init) if tr is not None and tr.enabled else None
    if trust_region is not None:
        assert cfg.steer.mode in ("fixed", "adaptive"), "the noise trust region applies to noise-space steering"
        common["trust_region"] = trust_region
    if steer_actor_zero_init(cfg.steer):
        common["actor_zero_init"] = True
    if cfg.steer.get("actor_log_std_init") is not None:
        assert cfg.steer.mode in ("fixed", "adaptive"), "steer.actor_log_std_init is fixed/adaptive only"
        common["actor_log_std_init"] = float(cfg.steer.actor_log_std_init)
    if cfg.steer.mode == "fixed":
        assert 0 <= cfg.steer.k < steered.K, (cfg.steer.k, steered.K)
        model = DSRLFixedK(SteerSACPolicy, env, k=cfg.steer.k, **common)
    elif cfg.steer.mode == "residual":
        # same SAC hyperparameters/networks as DSRL; no noise critic, no sampler in the updates
        for key in ("noise_critic_grad_steps", "critic_backup_combine_type", "horizon", "act_dim"):
            common.pop(key)
        common.pop("actor_gradient_steps", None)
        model = ResidualSAC(SteerSACPolicy, env, beta=cfg.steer.residual_beta, state_dim=cfg.obs_dim, **common)
    elif cfg.steer.mode == "adaptive":
        model = DSRLAdaptiveK(
            SteerSACPolicy, env, num_k=steered.K, rho=cfg.steer.rho, k_ent_coef=cfg.steer.k_ent_coef,
            freeze_k_uniform=cfg.steer.freeze_k_uniform, k_entropy_bias=cfg.steer.k_entropy_bias,
            k_subset=None if cfg.steer.k_subset is None else list(cfg.steer.k_subset), **common,
        )  # fmt: skip
    else:
        raise ValueError(cfg.steer.mode)

    # ---- resume / warm start ----
    cb_state = None
    if resuming:
        extra = model.load_checkpoint(ckpt_path)
        cb_state = extra.get("callback")
        print(f"resumed from {ckpt_path}: num_timesteps={model.num_timesteps} {cb_state}", flush=True)
    callback = AdsrlCallback(cfg, eval_env, steered, norm["obs_min"], norm["obs_max"], wandb_run, state=cb_state)
    init_chunks = cfg.train.init_rollout_steps * n_env
    if not resuming:
        callback.init_callback(model)
        callback.run_evals()
        if cfg.load_offline_data:
            raise NotImplementedError("offline data has no x_T; not supported in train_adsrl.py")
        if cfg.train.init_rollout_steps > 0:
            t_init = time.time()
            collect_base_rollouts(model, env, cfg.train.init_rollout_steps, steered, residual=residual)
            callback.env_steps = init_chunks * cfg.act_steps
            if callback.log_stdout:
                print(f"[init] {callback.env_steps} base-policy env steps in {time.time() - t_init:.1f}s", flush=True)

    total_chunks = int(cfg.train.total_timesteps) // cfg.act_steps - init_chunks
    remaining = total_chunks - model.num_timesteps
    if remaining > 0:
        # furniture: all n_envs episodes end on the same step, so SB3's per-4-episodes console dump would
        # print n_envs/4 tables at every episode boundary; AdsrlCallback logs train stats instead
        learn_kwargs = {"log_interval": 10**9} if furniture else {}
        model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=not resuming, **learn_kwargs)
    else:
        callback.init_callback(model)

    callback.run_evals(final=True)
    callback.save_checkpoint("final")
    callback.save_checkpoint("latest")
    env.close()
    eval_env.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
