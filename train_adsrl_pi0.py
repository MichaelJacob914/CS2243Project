"""DSRL-pi0 on LIBERO with steering at a fixed or adaptive flow step k (JAX; DSRL-pi0's alternating loop).

    python train_adsrl_pi0.py --task_id 3 --steer.mode fixed --steer.k 0 --total_env_steps 300000
    python train_adsrl_pi0.py --task_id 3 --steer.mode adaptive --steer.rho 0.5 --total_env_steps 300000
    (same command again resumes from <logdir>/checkpoint if it exists)

Loop (examples/train_utils_sim.py:trajwise_alternating_training_loop, batched over --n_envs):
collect n_envs episodes (pure Gaussian noise until the first update, as DSRL-pi0), add their chunk transitions,
then take (#chunks * multi_grad_step) SAC updates once the buffer holds > start_online_updates transitions.
Units: env_steps are raw LIBERO steps (excluding the 10 settle steps). Eval every --eval_every env steps on
the benchmark init states 0..eval_episodes-1.

Outputs in logdir: eval_summary.csv, train_log.csv, run_info.json, checkpoint/{agent_*,buffer.pkl,state.pkl},
wandb/ (offline).
"""

import argparse
import csv
import json
import os
import pickle
import platform
import socket
import subprocess
import sys
import time
import uuid

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import adsrl_pi0  # noqa: E402

adsrl_pi0.setup_env()

import numpy as np  # noqa: E402

from adsrl_pi0.rollout import N_REAL  # noqa: E402  (numpy only)

# jax/flax/openpi are imported inside main(): the LIBERO workers are spawned processes that re-import this module,
# and importing those from the lab filesystem in 12 processes at once made startup take ~10 min.
NUM_STEPS = 10  # pi0 flow steps (adsrl_pi0.sampler.NUM_STEPS)
NOISE_DIM = 32  # pi0 action dim; the agent's w is 32-d (decoded over the 50-step horizon by --steer.w_layout)
HORIZON = {"pi0": 50, "pi05": 10}  # action_horizon of pi0_libero / pi05_libero (checked against the loaded model)
DEFAULT_QUERY_FREQ = {"pi0": 20, "pi05": 5}  # DSRL-pi0's 20 of 50; openpi's LIBERO eval replans pi0.5 every 5 of 10
CSV_FIELDS = ["env_steps", "grad_steps", "mode", "wall_s", "success", "n_episodes", "mean_ep_len_success",
              "mean_a_minus_base", "k_entropy", "pi_k_entropy", "k_hist", "train_k_hist", "train_success"]


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", default=None)
    p.add_argument("--model", choices=["pi0", "pi05"], default="pi0",
                   help="base policy (adsrl_pi0.sampler.MODELS): pi0 needs .venv-pi0, pi05 needs .venv-pi05")
    p.add_argument("--logdir_root", default="/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs/exp7")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--task_id", type=int, required=True)
    p.add_argument("--total_env_steps", type=int, default=300_000)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--steer.mode", dest="mode", choices=["fixed", "adaptive"], default="fixed")
    p.add_argument("--steer.k", dest="k", type=int, default=0)
    p.add_argument("--steer.k_subset", dest="k_subset", type=int, nargs="+", default=None)
    p.add_argument("--steer.rho", dest="rho", type=float, default=0.5)
    p.add_argument("--steer.freeze_k_uniform", dest="freeze_k_uniform", type=int, default=0)
    p.add_argument("--steer.xt_layout", dest="xt_layout", choices=["tiled", "full"], default="tiled",
                   help="x_T for steps < k: one N(0, I_32) row tiled over the horizon (32-d obs), or full "
                        "N(0, I) of shape (50, 32) (1600-d obs).")
    p.add_argument("--steer.w_layout", dest="w_layout", choices=["tiled", "structured", "lowfreq"],
                   default="tiled",
                   help="how the agent's w becomes pi0's (50, 32) noise (adsrl_pi0.rollout.decode_noise): tiled: a "
                        "32-d w repeated over the horizon (DSRL-pi0); structured: the observed x_T with each channel's "
                        "time-mean replaced by w/sqrt(50) (32-d w); lowfreq: the first --steer.n_modes DCT modes of "
                        "each of the 7 real channels of x_T replaced by w (n_modes*7-d w). structured and lowfreq "
                        "need --steer.xt_layout full and use the same formula at every k")
    p.add_argument("--steer.n_modes", dest="n_modes", type=int, default=3, help="DCT modes per channel (lowfreq)")
    p.add_argument("--steer.obs_extra", dest="obs_extra", choices=["xt", "abase"], default="xt",
                   help="xt: x_T in the obs as before (see uses_xt_obs); abase: no x_T, instead pi0's plan for this "
                        "chunk's x_T (fixed k=0 and adaptive: the plain sample; fixed k>0: a_hat_{k-1}), 7 real channels x 50 steps "
                        "in normalized model space (350-d); one extra pi0 call per chunk")
    p.add_argument("--steer.force_xt_obs", dest="force_xt_obs", type=int, choices=[0, 1], default=0,
                   help="1: put x_T in the agent obs even where it cannot affect the action (fixed k=0, tiled w); a "
                        "control for the 1600-d obs input. Needs --steer.xt_layout full.")
    p.add_argument("--hybrid.k_entropy_bias", dest="k_entropy_bias", choices=["keep", "detach"], default="keep",
                   help="adaptive only. keep: pi(k) multiplies a_w log pi(w_k|k) in the actor loss (gives pi(k) an "
                        "entropy bonus toward k with high-entropy w heads); detach: stop-gradient on that pi(k), so "
                        "the categorical is trained only by Q and a_k H(pi_k); heads unchanged")
    # DSRL-pi0 hyperparameters (examples/scripts/run_libero.sh + launch_train_sim.py defaults)
    p.add_argument("--query_freq", type=int, default=None, help="env steps per chunk (default: 20 pi0, 5 pi05)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--discount", type=float, default=0.999)
    p.add_argument("--multi_grad_step", type=int, default=20)
    p.add_argument("--start_online_updates", type=int, default=500)
    p.add_argument("--resize_image", type=int, default=64)
    p.add_argument("--action_magnitude", type=float, default=1.0, help="tanh bound b of the agent's w in [-b, b]")
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[128])
    p.add_argument("--actor_lr", type=float, default=1e-4)
    p.add_argument("--critic_lr", type=float, default=3e-4)
    p.add_argument("--temp_lr", type=float, default=3e-4)
    p.add_argument("--num_qs", type=int, default=10)
    p.add_argument("--target_entropy", default="auto")
    # eval / logging / checkpointing
    p.add_argument("--eval_every", type=int, default=50_000, help="env steps between evals")
    p.add_argument("--eval_episodes", type=int, default=50, help="episodes of the stochastic eval (main metric)")
    p.add_argument("--eval_det", type=int, default=1,
                   help="also run a deterministic eval (argmax k, mean w) with one episode per init state")
    p.add_argument("--eval_at_start", type=int, default=0)
    p.add_argument("--eval_only", type=int, default=0,
                   help="restore the run's checkpoint, run a stoch and a det eval into eval_final.csv, exit")
    p.add_argument("--log_every_updates", type=int, default=1000)
    p.add_argument("--max_wall_hours", type=float, default=1e9, help="checkpoint and exit after this long")
    p.add_argument("--use_wandb", type=int, default=1)
    p.add_argument("--wandb_project", default="adsrl")
    p.add_argument("--wandb_group", default="exp7_pi0")
    a = p.parse_args(argv)
    if a.query_freq is None:
        a.query_freq = DEFAULT_QUERY_FREQ[a.model]
    if a.query_freq > HORIZON[a.model]:
        p.error(f"--query_freq {a.query_freq} > {a.model} action horizon {HORIZON[a.model]}")
    if a.w_layout == "structured" and a.xt_layout != "full":
        p.error("--steer.w_layout structured needs --steer.xt_layout full")
    if a.w_layout == "lowfreq" and a.xt_layout != "full":
        p.error("--steer.w_layout lowfreq needs --steer.xt_layout full")
    if a.force_xt_obs and a.xt_layout != "full":
        p.error("--steer.force_xt_obs 1 needs --steer.xt_layout full")
    if a.force_xt_obs and a.obs_extra == "abase":
        p.error("--steer.force_xt_obs 1 and --steer.obs_extra abase contradict (abase has no x_T in the obs)")
    if a.name is None:
        tag = f"k{a.k}" if a.mode == "fixed" else f"adaptive_rho{a.rho}"
        tag += "_xtfull" if a.xt_layout == "full" else ""
        tag += {"tiled": "", "structured": "_wstruct", "lowfreq": f"_lf{a.n_modes}"}[a.w_layout]
        tag += "_xtobs" if a.force_xt_obs else ""
        tag += "_abase" if a.obs_extra == "abase" else ""
        tag += "_kdetach" if a.k_entropy_bias == "detach" else ""
        tag += f"_b{a.action_magnitude:g}" if a.action_magnitude != 1.0 else ""
        a.name = f"{a.model}_t{a.task_id}_{tag}_s{a.seed}"
    a.logdir = os.path.join(a.logdir_root, a.name)
    return a


def uses_xt_obs(a):
    """With --steer.obs_extra xt, x_T is in the agent's obs unless it cannot affect the action: fixed k=0 with tiled
    w (exactly DSRL-pi0). Structured/lowfreq w is built from x_T, so the critic needs Q(s, x_T, w) to stay Markov.
    --steer.force_xt_obs 1 keeps x_T in the obs in every case. With abase, never (the plan feature replaces it)."""
    if a.obs_extra == "abase":
        return False
    return bool(a.force_xt_obs) or not (a.mode == "fixed" and a.k == 0 and a.w_layout == "tiled")


def action_dim(a):
    """Dimension of the agent's w: n_modes * 7 real channels for lowfreq, else one 32-d row."""
    return a.n_modes * N_REAL if a.w_layout == "lowfreq" else NOISE_DIM


def rollout_kw(a):
    """run_episodes keyword arguments for this run's noise layout and obs (training and eval)."""
    return dict(xt_full=a.xt_layout == "full", w_layout=a.w_layout, n_modes=a.n_modes, obs_extra=a.obs_extra,
                feature_k=a.k if a.mode == "fixed" else 0)  # adaptive: k is chosen after observing


def agent_input(obs, keys):
    """The agent's view of a rollout obs dict (which always carries x_T): only the obs_spec keys."""
    return {k: obs[k] for k in keys}


def add_episodes(buffer, eps, discount_chunk, keys):
    """Insert every chunk transition of `eps` (obs/next_obs restricted to `keys`); returns the count."""
    n = 0
    for e in eps:
        for o, no, w, k, r, m, disc in e.transitions(discount_chunk):
            buffer.insert(agent_input(o, keys), agent_input(no, keys), w, k, r, m, disc)
            n += 1
    return n


def sac_kwargs(a):
    """train_args_dict of dsrl_pi0/examples/launch_train_sim.py with the run_libero.sh overrides."""
    return dict(actor_lr=a.actor_lr, critic_lr=a.critic_lr, temp_lr=a.temp_lr, hidden_dims=tuple(a.hidden_dims),
                cnn_features=(32, 32, 32, 32), cnn_strides=(2, 1, 1, 1), cnn_padding="VALID", latent_dim=50,
                discount=a.discount, tau=0.005, critic_reduction="mean", dropout_rate=0.0, aug_next=1,
                use_bottleneck=True, encoder_type="small", encoder_norm="group", use_spatial_softmax=True,
                softmax_temperature=-1, target_entropy=a.target_entropy, num_qs=a.num_qs,
                action_magnitude=a.action_magnitude, num_cameras=1)


def obs_spec(a):
    r = a.resize_image
    shapes = {"pixels": (r, r, 3, 1), "state": (8, 1)}
    dtypes = {"pixels": np.uint8, "state": np.float32}
    if uses_xt_obs(a):
        shapes["x_T"], dtypes["x_T"] = (NOISE_DIM * (HORIZON[a.model] if a.xt_layout == "full" else 1),), np.float32
    if a.obs_extra == "abase":
        shapes["a_base"], dtypes["a_base"] = (HORIZON[a.model] * N_REAL,), np.float32
    return shapes, dtypes


def build_agent(a):
    from adsrl_pi0.agent import FixedKAgent, HybridPixelSACLearner

    shapes, dtypes = obs_spec(a)
    sample_obs = {k: np.zeros((1, *s), dtypes[k]) for k, s in shapes.items()}
    kw = sac_kwargs(a)
    if a.mode == "fixed":
        return FixedKAgent(a.k, NUM_STEPS, a.seed, sample_obs, np.zeros((1, 1, action_dim(a)), np.float32), **kw)
    allowed = a.k_subset if a.k_subset is not None else list(range(NUM_STEPS))
    return HybridPixelSACLearner(a.seed, sample_obs, action_dim(a), NUM_STEPS, allowed, rho=a.rho,
                                 freeze_k_uniform=bool(a.freeze_k_uniform), k_entropy_bias=a.k_entropy_bias, **kw)


def head_log_std_eval(agent, buffer, *, seed, eval_index, n=256):
    """Per-k w head mean log_std (adaptive) on n buffer observations drawn with an RNG of their own (the buffer's
    sampling RNG is untouched); None if the buffer is empty."""
    if len(buffer) == 0:
        return None
    import jax

    idx = np.random.default_rng([seed, 789, eval_index]).integers(0, len(buffer), n)
    obs = {k: v[idx] for k, v in buffer.data["observations"].items()}
    return agent.head_log_std(jax.device_put(obs))


def eval_init_indices(n_init_states, eval_episodes, deterministic):
    """Benchmark init-state index of each eval episode: deterministic evals run exactly one episode per init
    state; stochastic evals run eval_episodes episodes cycling over the init states."""
    if deterministic:
        return list(range(n_init_states))
    return [i % n_init_states for i in range(eval_episodes)]


def run_eval(venv, pi0, agent, act_fn, deterministic, *, seed, eval_index, eval_episodes, query_freq, rollout_kw,
             train_rng):
    """One evaluation (episodes only; logging is done by the caller).

    Uses its own RNGs, fixed by (seed, eval_index, mode) like DSRL-pi0's PRNGKey(seed + 456): numpy for the x_T
    noise and a JAX key for the agent's (k, w) sampling. The training streams (train_rng, agent rng) are left
    exactly as they were, so evals do not change training. train_rng is only passed in to make that explicit.
    """
    import jax

    from adsrl_pi0.rollout import run_episodes

    del train_rng  # never used during eval
    rng = np.random.default_rng([seed, 456, eval_index, int(deterministic)])
    saved = agent.get_rng()
    agent.set_rng(jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(seed + 456), eval_index),
                                     int(deterministic)))
    try:
        eps, _ = run_episodes(venv, pi0, eval_init_indices(venv.n_init_states, eval_episodes, deterministic),
                              act_fn, query_freq, rng, compare_base=True, **rollout_kw)
    finally:
        agent.set_rng(saved)
    return eps


def resume_agent(agent, ckpt_dir, expected_step):
    """Restore the agent checkpoint written together with state.pkl; raise instead of continuing with a fresh
    agent if it is missing, fails to load, or is from a different update step than state.pkl."""
    from adsrl_pi0.agent import restore_agent

    step = restore_agent(agent, ckpt_dir)  # raises if the checkpoint exists but cannot be restored
    if step is None:
        raise FileNotFoundError(f"state.pkl exists but there is no agent checkpoint in {ckpt_dir}")
    if step != expected_step:
        raise RuntimeError(f"agent checkpoint is at grad step {step} but state.pkl at {expected_step} ({ckpt_dir})")
    return step


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    import jax

    from adsrl_pi0.agent import save_agent
    from adsrl_pi0.buffer import ChunkReplayBuffer
    from adsrl_pi0.env import LiberoVecEnv
    from adsrl_pi0.rollout import run_episodes, summarize
    from adsrl_pi0.sampler import NUM_STEPS as SAMPLER_STEPS
    from adsrl_pi0.sampler import SteeredPi0

    assert SAMPLER_STEPS == NUM_STEPS
    a = parse()
    os.makedirs(os.path.join(a.logdir, "checkpoint"), exist_ok=True)
    t_start = time.time()
    use_xt = uses_xt_obs(a)
    ckpt_dir = os.path.join(a.logdir, "checkpoint")
    state_path = os.path.join(ckpt_dir, "state.pkl")
    resume = os.path.exists(state_path)

    gpu = os.popen("nvidia-smi --query-gpu=name --format=csv,noheader -i 0").read().strip()
    info = {"args": vars(a), "git": git_hash(), "host": socket.gethostname(), "gpu": gpu,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "python": platform.python_version(),
            "jax": jax.__version__, "devices": [str(d) for d in jax.devices()], "use_xt_obs": use_xt,
            "model": a.model, "w_layout": a.w_layout, "xt_layout": a.xt_layout, "action_bound": a.action_magnitude,
            "force_xt_obs": a.force_xt_obs, "n_modes": a.n_modes, "obs_extra": a.obs_extra,
            "k_entropy_bias": a.k_entropy_bias,
            "action_dim": action_dim(a), "obs_keys": list(obs_spec(a)[0]),
            "resume": resume}
    with open(os.path.join(a.logdir, "run_info.json" if not resume else f"run_info_resume_{int(t_start)}.json"),
              "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1), flush=True)

    wandb_run = None
    if a.use_wandb:
        import wandb

        id_file = os.path.join(a.logdir, "wandb_id.txt")
        run_id = open(id_file).read().strip() if os.path.exists(id_file) else uuid.uuid4().hex[:8]
        open(id_file, "w").write(run_id)
        wandb_run = wandb.init(project=a.wandb_project, group=a.wandb_group, name=a.name, id=run_id,
                               resume="allow", mode="offline", dir=a.logdir, config=info)
        wandb_run.define_metric("env_steps")
        wandb_run.define_metric("*", step_metric="env_steps")

    pi0 = SteeredPi0.load_model(a.model)
    assert pi0.horizon == HORIZON[a.model] and pi0.action_dim == NOISE_DIM, (pi0.horizon, pi0.action_dim)
    print(f"[startup] pi0 loaded at {time.time() - t_start:.0f}s", flush=True)
    venv = LiberoVecEnv(a.n_envs, a.task_id, suite=a.suite, seed=a.seed, resize_image=a.resize_image)
    print(f"[startup] {a.n_envs} LIBERO workers up at {time.time() - t_start:.0f}s", flush=True)
    agent = build_agent(a)
    shapes, dtypes = obs_spec(a)
    obs_keys = list(shapes)
    capacity = int(1.2 * a.total_env_steps / a.query_freq) + 1000
    buffer = ChunkReplayBuffer(shapes, dtypes, action_dim(a), capacity, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    allowed = np.array(a.k_subset if a.k_subset is not None else range(NUM_STEPS)) if a.mode == "adaptive" \
        else np.array([a.k])

    st = {"env_steps": 0, "grad_steps": 0, "episodes": 0, "next_eval": a.eval_every, "wall_s": 0.0, "n_evals": 0,
          "train_k_counts": np.zeros(NUM_STEPS, np.int64), "train_succ": []}
    if resume:
        with open(state_path, "rb") as f:
            st = pickle.load(f)
        rng.bit_generator.state = st.pop("np_rng")
        if not a.eval_only:
            buffer.load(os.path.join(ckpt_dir, "buffer.pkl"))
        step = resume_agent(agent, ckpt_dir, st["grad_steps"])
        print(f"resumed at env_steps={st['env_steps']} grad_steps={st['grad_steps']} (agent ckpt {step})", flush=True)
        csv_path = os.path.join(a.logdir, "eval_summary.csv")
        if os.path.exists(csv_path):  # drop eval rows written after the checkpoint
            rows = list(csv.DictReader(open(csv_path)))
            keep = [r for r in rows if int(r["env_steps"]) <= st["env_steps"]]
            with open(csv_path, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                wr.writeheader()
                wr.writerows(keep)
    wall0 = time.time() - st["wall_s"]

    def wall():
        return time.time() - wall0

    def log(d):
        if wandb_run is not None:
            wandb_run.log({**d, "env_steps": st["env_steps"]})

    def act_agent(deterministic):
        def f(batch):
            return agent.act(jax.device_put(agent_input(batch, obs_keys)), deterministic=deterministic)
        return f

    def act_random(batch):
        B = batch["state"].shape[0]
        w = rng.standard_normal((B, action_dim(a))).astype(np.float32)
        return allowed[rng.integers(0, len(allowed), B)], w, None

    def evaluate_mode(deterministic, eval_index, csv_name):
        t0 = time.time()
        mode = "det" if deterministic else "stoch"
        eps = run_eval(venv, pi0, agent, act_agent(deterministic), deterministic, seed=a.seed, eval_index=eval_index,
                       eval_episodes=a.eval_episodes, query_freq=a.query_freq, rollout_kw=rollout_kw(a),
                       train_rng=rng)
        s = summarize(eps, NUM_STEPS, a.query_freq)
        tc = st["train_k_counts"]
        row = {"env_steps": st["env_steps"], "grad_steps": st["grad_steps"], "mode": mode,
               "wall_s": round(wall(), 1), **{k: s[k] for k in ["success", "n_episodes", "mean_ep_len_success",
                                                                "mean_a_minus_base", "k_entropy", "pi_k_entropy",
                                                                "k_hist"]},
               "train_k_hist": " ".join(str(int(c)) for c in tc),
               "train_success": round(float(np.mean(st["train_succ"])), 4) if st["train_succ"] else float("nan")}
        csv_path = os.path.join(a.logdir, csv_name)
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new:
                wr.writeheader()
            wr.writerow(row)
        print(f"[eval {mode}] env_steps={st['env_steps']} success={s['success']:.3f} k_hist={s['k_hist']} "
              f"|a-a_base|={s['mean_a_minus_base']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        pre = "eval" if mode == "stoch" else "eval_det"
        d = {f"{pre}/{k}": v for k, v in s.items() if k != "k_hist"}
        if wandb_run is not None:
            import wandb

            ks = np.concatenate([np.asarray(e.k) for e in eps])
            d[f"{pre}/k_hist"] = wandb.Histogram(np_histogram=(np.bincount(ks, minlength=NUM_STEPS),
                                                               np.arange(NUM_STEPS + 1)))
            if mode == "stoch":
                d["train/k_hist"] = wandb.Histogram(np_histogram=(tc, np.arange(NUM_STEPS + 1)))
        log(d)

    def evaluate(csv_name="eval_summary.csv", eval_index=None):
        """Stochastic eval (main metric, as DSRL-pi0) and, with --eval_det, a deterministic one; one CSV row each."""
        idx = st["n_evals"] if eval_index is None else eval_index
        evaluate_mode(False, idx, csv_name)
        if a.eval_det:
            evaluate_mode(True, idx, csv_name)
        if a.mode == "adaptive":
            ls = head_log_std_eval(agent, buffer, seed=a.seed, eval_index=idx)
            if ls is not None:
                ks = [int(x) for x in np.asarray(agent.allowed_k)]
                path = os.path.join(a.logdir, "head_log_std.csv")
                new = not os.path.exists(path)
                with open(path, "a") as f:
                    if new:
                        f.write("env_steps,grad_steps," + ",".join(f"k{kk}" for kk in ks) + "\n")
                    f.write(f"{st['env_steps']},{st['grad_steps']}," + ",".join(f"{v:.4f}" for v in ls) + "\n")
                print("[eval] head mean log_std " + " ".join(f"k{kk}={v:.2f}" for kk, v in zip(ks, ls)), flush=True)
                log({f"eval/head_log_std_k{kk}": float(v) for kk, v in zip(ks, ls)})
        if eval_index is None:
            st["n_evals"] += 1
            st["train_k_counts"] = np.zeros(NUM_STEPS, np.int64)
            st["train_succ"] = []

    def checkpoint():
        t0 = time.time()
        save_agent(agent, ckpt_dir, st["grad_steps"])
        buffer.save(os.path.join(ckpt_dir, "buffer.pkl"))
        st["wall_s"] = wall()
        tmp = state_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({**st, "np_rng": rng.bit_generator.state}, f)
        os.replace(tmp, state_path)
        print(f"checkpoint at env_steps={st['env_steps']} ({time.time() - t0:.0f}s)", flush=True)

    if a.eval_only:
        assert resume, f"--eval_only needs a checkpoint in {ckpt_dir}"
        evaluate(csv_name="eval_final.csv", eval_index=10**6)  # fixed index: final evals are reproducible
        venv.close()
        return

    if a.eval_at_start and st["env_steps"] == 0:
        evaluate()

    train_log = open(os.path.join(a.logdir, "train_log.csv"), "a")
    if os.path.getsize(train_log.name) == 0:
        train_log.write("env_steps,grad_steps,wall_s,round_s,collect_s,pi0_s_per_call,mean_batch,env_s,update_s,"
                        "updates,env_steps_per_hour\n")
    while st["env_steps"] < a.total_env_steps:
        t_round = time.time()
        acting = act_random if st["grad_steps"] == 0 else act_agent(False)
        init = rng.integers(0, venv.n_init_states, a.n_envs)
        eps, tm = run_episodes(venv, pi0, init, acting, a.query_freq, rng, **rollout_kw(a))
        t_collect = time.time() - t_round
        n_chunks = add_episodes(buffer, eps, a.discount ** a.query_freq, obs_keys)
        for e in eps:
            st["train_k_counts"] += np.bincount(e.k, minlength=NUM_STEPS)[:NUM_STEPS]
            st["train_succ"].append(e.success)
        st["env_steps"] += tm["env_steps"]
        st["episodes"] += len(eps)

        t_upd = time.time()
        n_upd = 0
        info_u = None
        if len(buffer) > a.start_online_updates:
            for _ in range(n_chunks * a.multi_grad_step):
                info_u = agent.update(buffer.sample(a.batch_size, with_k=a.mode == "adaptive"))
                st["grad_steps"] += 1
                n_upd += 1
                if st["grad_steps"] % a.log_every_updates == 0:
                    info_u = jax.device_get(info_u)
                    log({f"training/{k}": float(v) for k, v in info_u.items() if np.ndim(v) == 0})
                    if "k_probs" in info_u:
                        log({f"training/pi_k_{int(kk)}": float(info_u["k_probs"][j])
                             for j, kk in enumerate(np.asarray(agent.allowed_k))})
            jax.block_until_ready(agent._save_dict["critic"].params)
        t_upd = time.time() - t_upd
        round_s = time.time() - t_round
        line = [st["env_steps"], st["grad_steps"], round(wall(), 1), round(round_s, 2), round(t_collect, 2),
                round(tm["pi0_s"] / max(tm["pi0_calls"], 1), 4), round(tm["chunks"] / max(tm["pi0_calls"], 1), 2),
                round(tm["env_s"], 2), round(t_upd, 2), n_upd, round(tm["env_steps"] / round_s * 3600)]
        train_log.write(",".join(map(str, line)) + "\n")
        train_log.flush()
        succ = np.mean([e.success for e in eps])
        sac = ""
        if info_u is not None:
            iu = jax.device_get(info_u)
            sac = " " + " ".join(f"{k}={float(iu[k]):.3g}" for k in
                                 ["q", "critic_loss", "temperature", "entropy", "k_temperature", "k_entropy"] if k in iu)
        print(f"env_steps={st['env_steps']} grad_steps={st['grad_steps']} buffer={len(buffer)} train_succ={succ:.2f} "
              f"collect={t_collect:.1f}s (pi0 {tm['pi0_s']:.1f}s feat {tm['feat_s']:.1f}s env {tm['env_s']:.1f}s) "
              f"update={t_upd:.1f}s ({n_upd} upd) wall={wall() / 3600:.2f}h{sac}", flush=True)
        log({"train/success": succ, "train/episodes": st["episodes"], "train/grad_steps": st["grad_steps"],
             "time/env_steps_per_hour": tm["env_steps"] / round_s * 3600, "time/wall_h": wall() / 3600})

        if st["env_steps"] >= st["next_eval"] or st["env_steps"] >= a.total_env_steps:
            evaluate()
            st["next_eval"] += a.eval_every
            checkpoint()
        if time.time() - t_start > a.max_wall_hours * 3600:
            print("wall limit reached; checkpointing and exiting", flush=True)
            checkpoint()
            break
    else:
        open(os.path.join(a.logdir, "DONE"), "w").write(str(st["env_steps"]))
    train_log.close()
    venv.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
