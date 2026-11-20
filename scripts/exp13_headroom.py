"""exp13 Phase 0B: headroom for learned k under a trained action critic Q^A (eval only).

Agent: a uniform-random-k adaptive checkpoint (hybrid actor: per-k Gaussian heads pi(w|s,k)).

Steps (one process, in this order):
  1. Arm (iii): the plain stochastic policy (k ~ uniform, w ~ pi(w|s,k)), --n_episodes episodes,
     with rollout dumps. Its own rollouts are the state pool for step 2.
  2. Headroom: pick --n_states states from (iii)'s pre-success chunks, stratified over deciles of
     normalized episode time. For each state (with its x_T) and each k in 0..K-1, draw --n_w w from
     the k-th head, decode, and score with Q^A = min over the twin critics. Saves Q[S, K, n_w] etc.
  3. Arms (iv) deterministic policy, (i) critic-chosen (k, w) over all K heads x --n_cand samples,
     (ii) best-of-(K * --n_cand) at k=0 (k=0 head). Same eval-env seed for every arm.

Phase labels and summaries are computed afterwards by scripts/exp13_headroom_analyze.py.

    python scripts/exp13_headroom.py --ckpt <copy of checkpoint>.pt --out_dir results/exp13/headroom
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.environ.get("EXP13_CODE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import train_adsrl  # noqa: E402  (registers resolvers, puts dsrl/ and dppo/ on sys.path)
from best_of_n_eval import make_eval_env  # noqa: E402
from render_rollouts import build_agent, build_steered, run_cfg  # noqa: E402


# ----------------------------------------------------------------------------------------
# paired eval env: seeded initial states (works around the exp_2 eval-seeding bug without
# touching the library: eval_env.seed() never reaches ObservationWrapperRobomimic, whose reset
# only reads options["seed"], so initial states come from each worker's unseeded np RNG)
# ----------------------------------------------------------------------------------------
def make_paired_eval_env(cfg, n_envs):
    from env_utils import ActionChunkWrapper, ObservationWrapperRobomimic, make_robomimic_env
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnvWrapper

    from adsrl.env import XTAugmentWrapper

    class SeedableChunkEnv(ActionChunkWrapper):
        def seed_np(self, seed):
            np.random.seed(int(seed))  # robosuite's placement sampler draws from the global np RNG
            return True

    class SeededResetVecEnv(VecEnvWrapper):
        """Before every explicit reset(), seed worker i with base + 1000 * n_resets + i."""

        def __init__(self, venv):
            super().__init__(venv)
            self.base, self.n_resets = 0, 0

        def seed(self, seed=None):
            self.base, self.n_resets = int(seed or 0), 0
            return self.venv.seed(seed)

        def reset(self):
            for i in range(self.num_envs):
                self.venv.env_method("seed_np", self.base + 1000 * self.n_resets + i, indices=[i])
            self.n_resets += 1
            return self.venv.reset()

        def step_wait(self):
            return self.venv.step_wait()

    def make_env():
        env = make_robomimic_env(
            env=cfg.env_name, normalization_path=cfg.normalization_path,
            low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys, dppo_path=cfg.dppo_path,
        )  # fmt: skip
        env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
        return SeedableChunkEnv(env, cfg, max_episode_steps=cfg.env.max_episode_steps)

    venv = SeededResetVecEnv(make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv))
    return XTAugmentWrapper(venv, cfg.act_steps, cfg.action_dim)


# ----------------------------------------------------------------------------------------
# candidate generation / scoring
# ----------------------------------------------------------------------------------------
def q_min(model, obs, action):
    """Q^A(s, a) = min over the critics, as in the target backup. obs: obs_aug rows."""
    return model._combine_q(torch.cat(model.critic(obs, action), dim=1)).reshape(-1)


@torch.no_grad()
def candidates(model, obs, head_idx, n, sub_batch=16384, base=None):
    """For each row of obs_aug [B, ...]: n squashed w per head in head_idx (list of head indices),
    decoded and scored. Returns dict with q [B, len(heads), n], and (flattened) action / w_sq / k
    [B, len(heads), n, ...]; if base [B, D] is given also l2 / mabs distance to it [B, L, n]."""
    ha = model.hybrid_actor
    B, L = obs.shape[0], len(head_idx)
    _, mean, log_std = ha(obs)  # [B, K', D]
    hi = torch.as_tensor(head_idx, device=obs.device, dtype=torch.long)
    mean, std = mean[:, hi], log_std[:, hi].exp()  # [B, L, D]
    D = mean.shape[-1]
    eps = torch.randn(B, L, n, D, device=obs.device)
    w_sq = torch.tanh(mean[:, :, None] + std[:, :, None] * eps)  # [B, L, n, D]
    k = ha.allowed_k[hi].view(1, L, 1).expand(B, L, n)
    obs_rep = obs[:, None, None].expand(B, L, n, obs.shape[-1]).reshape(-1, obs.shape[-1])
    w_flat, k_flat = w_sq.reshape(-1, D), k.reshape(-1)
    acts, qs = [], []
    for i in range(0, obs_rep.shape[0], sub_batch):
        a = model._decode(obs_rep[i : i + sub_batch], k_flat[i : i + sub_batch], w_flat[i : i + sub_batch])
        acts.append(a)
        qs.append(q_min(model, obs_rep[i : i + sub_batch], a))
    act = torch.cat(acts).view(B, L, n, -1)
    out = {"q": torch.cat(qs).view(B, L, n), "action": act, "w_sq": w_sq, "k": k}
    if base is not None:
        d = act - base[:, None, None]
        out["l2"] = d.norm(dim=-1)
        out["mabs"] = d.abs().mean(-1)
    return out


class CandidateAgent:
    """Agent stand-in for AdsrlCallback.evaluate. mode="all_k": argmax Q^A over every head x n samples;
    mode="k0": argmax over K * n samples from the k=0 head (equal decode count)."""

    def __init__(self, model, mode, n):
        self.model, self.mode, self.n = model, mode, int(n)
        ha = model.hybrid_actor
        allowed = ha.allowed_k.cpu().tolist()
        self.heads = list(range(len(allowed))) if mode == "all_k" else [allowed.index(0)]
        self.n_per_head = self.n if mode == "all_k" else self.n * len(allowed)
        self.last_k = self.last_w = self.last_probs = None
        self.log = {"k": [], "q_chosen": [], "q_mean": [], "q_max_k0": []}

    @torch.no_grad()
    def predict_diffused(self, obs, deterministic=False):
        m = self.model
        obs_t = torch.as_tensor(np.asarray(obs), device=m.device, dtype=torch.float32)
        c = candidates(m, obs_t, self.heads, self.n_per_head)
        B = obs_t.shape[0]
        q = c["q"].reshape(B, -1)
        idx = q.argmax(1)
        rows = torch.arange(B, device=obs_t.device)
        act = c["action"].reshape(B, -1, c["action"].shape[-1])[rows, idx]
        w_sq = c["w_sq"].reshape(B, -1, c["w_sq"].shape[-1])[rows, idx]
        k = c["k"].reshape(B, -1)[rows, idx]
        self.last_k = k.cpu().numpy()
        self.last_w = m._unscale_t(w_sq).cpu().numpy()
        self.log["k"].append(self.last_k)
        self.log["q_chosen"].append(q[rows, idx].cpu().numpy())
        self.log["q_mean"].append(q.mean(1).cpu().numpy())
        return act.cpu().numpy(), None


# ----------------------------------------------------------------------------------------
# state selection + headroom
# ----------------------------------------------------------------------------------------
def select_states(dump, n_states, rng, n_bins=10):
    """Pre-success chunks of the (iii) rollouts, stratified over deciles of normalized episode time
    t / T_end (T_end = first success chunk + 1 for successes, else the episode length)."""
    ep, ch, succ = dump["episode"], dump["chunk"], dump["success_step"].astype(bool)
    n_chunks = int(ch.max()) + 1
    t_end = {}
    for e in np.unique(ep):
        m = ep == e
        s = np.where(succ[m])[0]
        t_end[e] = (int(ch[m][s[0]]) + 1) if len(s) else n_chunks
    t_end_row = np.array([t_end[e] for e in ep])
    pool = np.where(ch < t_end_row)[0]
    tau = ch / t_end_row
    bins = np.minimum((tau * n_bins).astype(int), n_bins - 1)
    per_bin = n_states // n_bins
    chosen = []
    for b in range(n_bins):
        cand = pool[bins[pool] == b]
        chosen.extend(rng.choice(cand, size=min(per_bin, len(cand)), replace=False).tolist())
    chosen = np.array(sorted(chosen))
    return chosen, tau, t_end_row


@torch.no_grad()
def headroom(model, steered, obs_aug, n_w, batch_states=12):
    K = len(model.hybrid_actor.allowed_k)
    heads = list(range(K))
    keys = ["q", "l2", "mabs"]
    res = {k: [] for k in keys + ["q_base", "q_det", "l2_det"]}
    for i in range(0, len(obs_aug), batch_states):
        o = obs_aug[i : i + batch_states]
        base = steered.base_action(o).reshape(len(o), -1).to(model.device, torch.float32)
        c = candidates(model, o, heads, n_w, base=base)
        for k in keys:
            res[k].append(c[k].cpu().numpy())
        res["q_base"].append(q_min(model, o, base).cpu().numpy())
        # deterministic w (tanh of the head mean) per k
        _, mean, _ = model.hybrid_actor(o)
        w_det = torch.tanh(mean)  # [b, K, D]
        o_rep = o[:, None].expand(len(o), K, o.shape[-1]).reshape(-1, o.shape[-1])
        k_rep = model.hybrid_actor.allowed_k.repeat(len(o))
        a_det = model._decode(o_rep, k_rep, w_det.reshape(-1, w_det.shape[-1]))
        res["q_det"].append(q_min(model, o_rep, a_det).view(len(o), K).cpu().numpy())
        res["l2_det"].append((a_det.view(len(o), K, -1) - base[:, None]).norm(dim=-1).cpu().numpy())
    return {k: np.concatenate(v) for k, v in res.items()}


# ----------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_episodes", type=int, default=200)
    p.add_argument("--n_envs", type=int, default=25)
    p.add_argument("--n_states", type=int, default=300)
    p.add_argument("--n_w", type=int, default=32, help="samples per k for the state-level headroom")
    p.add_argument("--n_cand", type=int, default=32, help="samples per k for arm (i); (ii) uses K * n_cand at k=0")
    p.add_argument("--arms", default="iii,headroom,iv,i,ii")
    p.add_argument("--seed", type=int, default=0, help="eval-env seed offset (same for every arm)")
    p.add_argument("--paired", action="store_true", help="seed every worker before each reset (same initial states per arm)")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    assert args.n_episodes % args.n_envs == 0
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = run_cfg(args.ckpt, args.device)
    assert cfg.steer.mode == "adaptive", cfg.steer.mode
    steered = build_steered(cfg)
    model = build_agent(cfg, steered, args.ckpt)
    model.policy.set_training_mode(False)
    model.hybrid_actor.train(False)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ckpt_steps = int(ck["extra"]["callback"]["env_steps"])
    del ck
    K = steered.K
    print(f"loaded {args.ckpt}: env_steps={ckpt_steps} K={K} freeze_k_uniform={cfg.steer.freeze_k_uniform} "
          f"w_scale={steered.w_scale} code_root={ROOT}", flush=True)  # fmt: skip

    cfg.env.n_eval_envs = args.n_envs
    cfg.num_evals = args.n_episodes
    cfg.eval_dump_every = 1
    env = make_paired_eval_env(cfg, args.n_envs) if args.paired else make_eval_env(cfg, args.n_envs)
    norm = np.load(cfg.normalization_path)
    summary_path = os.path.join(args.out_dir, "arms.csv")
    meta = {"ckpt": args.ckpt, "ckpt_env_steps": ckpt_steps, "K": K, "code_root": ROOT, **vars(args)}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    def run_arm(name, agent, deterministic):
        cfg.logdir = os.path.join(args.out_dir, f"arm_{name}")
        env.seed(cfg.seed + 10_000 + args.seed)
        torch.manual_seed(1234)
        np.random.seed(1234)
        cb = train_adsrl.AdsrlCallback(cfg, env, steered, norm["obs_min"], norm["obs_max"], None,
                                       state={"env_steps": ckpt_steps})  # fmt: skip
        cb.model = agent
        torch.cuda.synchronize()
        t0 = time.time()
        res = cb.evaluate(deterministic=deterministic, dump=True)
        torch.cuda.synchronize()
        wall = time.time() - t0
        se = float(np.sqrt(res["success"] * (1 - res["success"]) / args.n_episodes))
        row = {"arm": name, "n_episodes": args.n_episodes, "success": res["success"], "se": se,
               "mean_l2_a_minus_base": res["mean_a_minus_base"], "mean_ep_len_success": res["mean_ep_len_success"],
               "wall_s": round(wall, 1), "ckpt_env_steps": ckpt_steps,
               "decodes_per_chunk": getattr(agent, "n_per_head", 1) * len(getattr(agent, "heads", [0]))}  # fmt: skip
        new = not os.path.exists(summary_path)
        with open(summary_path, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)
        if isinstance(agent, CandidateAgent):
            np.savez_compressed(os.path.join(cfg.logdir, "chosen.npz"),
                                **{k: np.concatenate(v) for k, v in agent.log.items() if v})  # fmt: skip
        print("ARM " + json.dumps(row), flush=True)
        return res

    for arm in args.arms.split(","):
        if arm == "iii":
            run_arm("iii_stoch", model, deterministic=False)
        elif arm == "iv":
            run_arm("iv_det", model, deterministic=True)
        elif arm == "i":
            run_arm(f"i_allk_N{args.n_cand}", CandidateAgent(model, "all_k", args.n_cand), deterministic=False)
        elif arm == "ii":
            run_arm(f"ii_k0_N{K * args.n_cand}", CandidateAgent(model, "k0", args.n_cand), deterministic=False)
        elif arm == "headroom":
            t0 = time.time()
            dump_path = os.path.join(args.out_dir, "arm_iii_stoch", "eval_rollouts", f"step_{ckpt_steps}.npz")
            dump = dict(np.load(dump_path, allow_pickle=True))
            rng = np.random.default_rng(0)
            idx, tau, t_end = select_states(dump, args.n_states, rng)
            obs_aug = np.concatenate([dump["obs"][idx], dump["x_T"][idx]], axis=1).astype(np.float32)
            torch.manual_seed(4321)
            res = headroom(model, steered, torch.as_tensor(obs_aug, device=model.device), args.n_w)
            np.savez_compressed(
                os.path.join(args.out_dir, "headroom_states.npz"), dump_row=idx, episode=dump["episode"][idx],
                chunk=dump["chunk"][idx], tau=tau[idx], t_end=t_end[idx], obs=dump["obs"][idx],
                obs_raw=dump["obs_raw"][idx], x_T=dump["x_T"][idx],
                episode_success=dump["episode_success"][idx], **res,
            )  # fmt: skip
            print(f"HEADROOM {len(idx)} states x {K} k x {args.n_w} w in {time.time() - t0:.1f}s", flush=True)
    env.close()


if __name__ == "__main__":
    main()
