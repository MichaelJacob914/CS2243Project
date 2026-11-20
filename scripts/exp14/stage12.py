"""exp14 transport GPU driver: Stage 1 (256 stochastic samples, clustering), Stage 2a (resolved fraction,
policy + Gaussian null + independent null replicate, 64 points x 16 regenerations, 20 DDIM levels + t=99 check),
Stage 2b (open fraction, plan form + steering form + null plan form, 8 bases x 32 copies), value via the critic.
Also generates the Stage 4 actions (mode stage4).

One pickle per state in out_dir/state_XXX.pkl (resumable; --shard i/n splits states across processes / GPUs).
Actions needed by the simulator (Stage 1 samples for all states; 2a/2b actions for the Stage 3 subset) go to
--act_dir as npz.

    python scripts/exp14/stage12.py --states results/exp14/states/states.npz --out_dir results/exp14/stage12 \
        --act_dir /n/netscratch/.../exp14/actions --shard 0/2
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402
import measure  # noqa: E402
import windows as W  # noqa: E402

H, A = 8, 14
P, M = 128, 16  # Stage 2a (Design change 4: 128 x 16 on transport, null matched)
B2, C2 = 8, 32  # Stage 2b
N1 = 256  # Stage 1
NBOOT = 1000
STAGE3_PER_PHASE = 20


def load_policy(device):
    from adsrl.model_utils import build_model, load_task_cfg

    cfg = load_task_cfg("transport", os.path.join(common.PROJECT, "cfg", "adsrl", "transport.yaml"))
    return build_model(cfg, device=device, ddim_steps=20)


def load_critic(device):
    from render_rollouts import build_agent, build_steered, run_cfg

    cfg = run_cfg(common.CRITIC_CKPT, device)
    steered = build_steered(cfg)
    agent = build_agent(cfg, steered, common.CRITIC_CKPT)
    agent.policy.set_training_mode(False)
    ck = torch.load(common.CRITIC_CKPT, map_location="cpu", weights_only=False)
    steps = int(ck["extra"]["callback"]["env_steps"])
    del ck
    assert steps == 6_000_000, steps
    return agent, cfg


class Critic:
    """Q^A(s, a) = min over the twin critics (as in the target backup). The critic drops x_T (actor_noise_critic)."""

    def __init__(self, device):
        self.agent, cfg = load_critic(device)
        self.obs_dim = cfg.obs_dim
        self.xt_dim = cfg.act_steps * cfg.action_dim
        self.device = device

    @torch.no_grad()
    def q(self, obs, actions, xt=None, bs=8192):
        """obs (obs_dim,) np/tensor; actions (N, H, A) np -> (N,) np."""
        o = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        out = []
        for i in range(0, len(actions), bs):
            a = torch.as_tensor(actions[i : i + bs].reshape(-1, self.xt_dim), dtype=torch.float32, device=self.device)
            n = len(a)
            x = torch.zeros(n, self.xt_dim, device=self.device) if xt is None else xt[i : i + bs].to(self.device)
            oa = torch.cat([o[None].expand(n, -1), x], 1)
            qs = torch.cat(self.agent.critic(oa, a), dim=1)
            out.append(self.agent._combine_q(qs).reshape(-1).cpu().numpy())
        return np.concatenate(out)


def make_feat_fn(critic, obs_np, st1):
    pcs, mu, labeler = st1["pcs"], st1["mu"], st1["labeler"]

    def f(a):
        a = np.asarray(a, np.float32)
        X = a.reshape(len(a), -1).astype(np.float64)
        Z = (X - mu) @ pcs.T  # (N, 5)
        out = {
            "grip0": np.sign(a[:, :, 6]).mean(1, keepdims=True),
            "grip1": np.sign(a[:, :, 13]).mean(1, keepdims=True),
            "grip0_mean": a[:, :, 6].mean(1, keepdims=True).astype(np.float64),  # Design change 5 (exploratory)
            "grip1_mean": a[:, :, 13].mean(1, keepdims=True).astype(np.float64),
            # Design change 5: value on the clipped chunk (what the env executes; the critic was trained in range)
            "value": critic.q(obs_np, np.clip(a, -1.0, 1.0))[:, None].astype(np.float64),
            "arm0": a[:, :, 0:7].reshape(len(a), -1),
            "arm1": a[:, :, 7:14].reshape(len(a), -1),
            "slow": a.mean(1),
            "fast": (a - a.mean(1, keepdims=True)).reshape(len(a), -1),
        }
        for i in range(Z.shape[1]):
            out[f"pc{i + 1}"] = Z[:, i : i + 1]
        if labeler is not None:
            out["cluster_label"] = W.cluster_label_onehot(labeler, a)
        return out

    return f


def curve_pack(summ, x_ls, x_k, rng, nboot=NBOOT):
    """name -> dict(R (L,), boot stats (n_boot,) of width / mid in log-SNR and mid in step index, fbar for d=1)."""
    out = {}
    for name, (fbar, wvar) in summ.items():
        R = W.resolved_from_summary(fbar, wvar, M)
        Rb = W.bootstrap_resolved_gram(fbar[:20], wvar[:20], M, nboot, rng)  # bootstrap on the 20 DDIM levels
        sb = W.curve_stats(x_ls, Rb)
        sk = W.curve_stats(x_k, Rb)
        out[name] = {"R": R, "boot_width": sb["width"], "boot_mid": sb["mid"], "boot_mid_k": sk["mid"],
                     "R_lo": np.nanpercentile(Rb, 2.5, 0), "R_hi": np.nanpercentile(Rb, 97.5, 0),
                     "fbar": fbar if fbar.shape[-1] == 1 else None, "wvar": wvar if fbar.shape[-1] == 1 else None}  # fmt: skip
    return out


def stage3_subset(states, rng_seed=314):
    rng = np.random.default_rng(rng_seed)
    sel = []
    for p in range(5):
        for src in (0, 1):
            idx = np.where((states["phase"] == p) & (states["source"] == src))[0]
            sel += rng.choice(idx, size=min(STAGE3_PER_PHASE // 2, len(idx)), replace=False).tolist()
    return sorted(sel)


def run_state(i, states, model, critic, dev, act_dir, in_s3, log):
    t0 = time.time()
    gen = torch.Generator(device=dev).manual_seed(1_000_000 + i)
    rng = np.random.default_rng(2_000_000 + i)
    obs_np = states["obs"][i].astype(np.float32)
    o = torch.as_tensor(obs_np, device=dev)
    levels = W.ddim_levels(model)
    lv = list(levels) + [model.denoising_steps - 1]
    x_ls = W.logsnr(model, levels)
    x_k = np.arange(len(levels), dtype=float)
    # ---- Stage 1
    samp = W.ddpm_sample(model, {"state": o[None]}, N1, gen)[0].cpu().numpy()
    ct = W.cluster_tests(samp, rng_seed=0)
    null = W.GaussNull(samp)
    gm = W.GaussEpsModel(model, null)
    nsamp = W.ddpm_sample(gm, {"state": torch.zeros(1, 1, device=dev)}, N1, gen)[0].cpu().numpy()
    ct_null = W.cluster_tests(nsamp, rng_seed=0)
    st1 = {"pcs": ct["pcs"], "mu": samp.reshape(N1, -1).mean(0), "labeler": ct["labeler"] if ct["clustered"] else None}
    feat = make_feat_fn(critic, obs_np, st1)
    q1 = critic.q(obs_np, np.clip(samp, -1.0, 1.0))
    # ---- Stage 2a (policy clean points drawn separately from the Stage 1 samples)
    clean = W.ddpm_sample(model, {"state": o[None]}, P, gen)[0]
    eps = torch.randn(clean.shape, device=dev, generator=gen)
    s_pol, a_pol = measure.stage2a_policy(model, o, clean, eps, lv, M, gen, feat, keep_actions=in_s3)
    top = a_pol[-1].reshape(-1, H, A) if in_s3 else None
    s_n1, a_n1 = measure.stage2a_null(gm, P, torch.randn(clean.shape, device=dev, generator=gen), lv, M, gen, feat, keep_actions=in_s3)
    s_n2, _ = measure.stage2a_null(gm, P, torch.randn(clean.shape, device=dev, generator=gen), lv, M, gen, feat)
    c_pol, c_n1, c_n2 = curve_pack(s_pol, x_ls, x_k, rng), curve_pack(s_n1, x_ls, x_k, rng), curve_pack(s_n2, x_ls, x_k, rng)
    # top-of-chain check on the policy: regeneration from t = 99 vs fresh samples (energy test on subsample)
    if top is None:
        _, a_top = measure.stage2a_policy(model, o, clean[:16], eps[:16], [model.denoising_steps - 1], M, gen, lambda a: {"x": a[:, 0, :1]}, keep_actions=True)
        top = a_top[0].reshape(-1, H, A)
    e_stat, e_p = measure.energy_test(top[:256], samp[:256], n_perm=200, rng=rng)
    # ---- Stage 2b
    ks = list(range(len(levels)))
    xT = torch.randn((B2, H, A), device=dev, generator=gen)
    e2 = torch.randn((B2, C2, H, A), device=dev, generator=gen)
    w2 = torch.randn((B2, C2, H, A), device=dev, generator=gen)
    r2b, base2b, acts2b = measure.stage2b_policy(model, o, xT, e2, ks, feat, steer_w=w2, keep_actions=in_s3)
    fresh = W.ddim_sample(model, {"state": o[None].expand(256, -1)}, torch.randn((256, H, A), device=dev, generator=gen)).cpu().numpy()
    ff = feat(fresh)
    tv = {n: W.total_var(v) for n, v in ff.items()}
    r2b_null = measure.stage2b_null(gm, torch.randn((B2, H, A), device=dev, generator=gen), torch.randn((B2, C2, H, A), device=dev, generator=gen), ks, feat)
    nfresh = W.ddim_sample(gm, {"state": torch.zeros(256, 1, device=dev)}, torch.randn((256, H, A), device=dev, generator=gen)).cpu().numpy()
    nff = feat(nfresh)
    open_ = {}
    for form, d in list(r2b.items()) + [("null_plan", r2b_null)]:
        tvx = {n: W.total_var(v) for n, v in nff.items()} if form == "null_plan" else tv
        open_[form] = {n: {"O": W.open_fraction(v, tvx[n]), "per_base": v.var(-2, ddof=1).sum(-1) / max(tvx[n], 1e-12)} for n, v in d.items()}
    # ---- save actions for the simulator
    np.savez(os.path.join(act_dir, f"s1_{i:03d}.npz"), samp=samp.astype(np.float32), q=q1)
    if in_s3:
        np.savez(os.path.join(act_dir, f"s3_{i:03d}.npz"), a2a=a_pol[:20].astype(np.float32), a2a_null=a_n1[:20].astype(np.float32),
                 a2b_plan=acts2b["plan"].astype(np.float32), a2b_steer=acts2b["steer"].astype(np.float32))  # fmt: skip
    res = {
        "i": i, "phase": int(states["phase"][i]), "source": int(states["source"][i]),
        "cluster": {k: v for k, v in ct.items() if k not in ("labeler", "pcs")}, "cluster_null": {k: ct_null[k] for k in ("dbic", "dip_p", "clustered", "n_comp")},
        "lam": null.lam[::-1][:20].copy(),
        # Design change 5: everything needed to recompute features later (PCs, PCA mean, labeler)
        "feat_basis": {"pcs": ct["pcs"], "mu": st1["mu"], "pc_ev": ct["pc_ev"],
                       "labeler": None if st1["labeler"] is None else {
                           "mu": st1["labeler"]["mu"], "V": st1["labeler"]["V"], "gmm": st1["labeler"]["gmm"],
                           "weights": st1["labeler"]["gmm"].weights_, "means": st1["labeler"]["gmm"].means_,
                           "covariances": st1["labeler"]["gmm"].covariances_}},
        "frac_outside_box_max": float(np.abs(samp).max()), "frac_outside_box": float((np.abs(samp) > 1).mean()),
        "q1_mean": float(q1.mean()), "q1_std": float(q1.std()),
        "pol": c_pol, "null": c_n1, "null2": c_n2, "open": open_, "tv": tv,
        "top_check": {"energy_stat": e_stat, "energy_p": e_p, "R_t99": {n: float(c["R"][-1]) for n, c in c_pol.items()}},
        "seconds": time.time() - t0,
    }  # fmt: skip
    log(f"state {i} phase {res['phase']} src {res['source']} clustered={ct['clustered']} (null {ct_null['clustered']}) "
        f"lam1={res['lam'][0]:.4f} q1={q1.mean():.1f}+-{q1.std():.2f} topR={max(abs(v) for v in res['top_check']['R_t99'].values()):.3f} "
        f"energy_p={e_p:.2f} t={res['seconds']:.1f}s")  # fmt: skip
    return res


@torch.no_grad()
def gen_stage4(states, model, critic, dev, out_dir, act_dir, sel, ks, log, P4=32, M4=8):
    """Stage 4 actions: per selected state and level, 32 points x 8 stochastic regenerations (2a design)."""
    for i in sel:
        gen = torch.Generator(device=dev).manual_seed(3_000_000 + i)
        o = torch.as_tensor(states["obs"][i].astype(np.float32), device=dev)
        clean = W.ddpm_sample(model, {"state": o[None]}, P4, gen)[0]
        eps = torch.randn(clean.shape, device=dev, generator=gen)
        levels = W.ddim_levels(model)
        lv = [int(levels[k]) for k in ks]
        feat = lambda a: {"value": critic.q(states["obs"][i], np.clip(a, -1.0, 1.0))[:, None].astype(np.float64)}  # noqa: E731
        summ, acts = measure.stage2a_policy(model, o, clean, eps, lv, M4, gen, feat, keep_actions=True)
        q = np.stack([summ["value"][0][:, :, 0]])  # point means (L, P)
        val = acts  # (L, P, M, H, A)
        qall = np.stack([critic.q(states["obs"][i], np.clip(a.reshape(-1, H, A), -1.0, 1.0)).reshape(P4, M4) for a in val])
        np.savez(os.path.join(act_dir, f"s4_{i:03d}.npz"), actions=val.astype(np.float32), q=qall, ks=np.array(ks))
        log(f"stage4 actions state {i}: {val.shape}, q mean {qall.mean():.1f}")
        del q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--states", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--act_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--limit", type=int, default=None, help="only the first N states of this shard (timing)")
    p.add_argument("--mode", default="stage12", choices=["stage12", "stage4"])
    p.add_argument("--stage4_states", default=None, help="comma list (mode stage4)")
    p.add_argument("--stage4_ks", default=None, help="comma list of DDIM steps (mode stage4)")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.act_dir, exist_ok=True)
    logf = open(os.path.join(args.out_dir, f"log_{args.mode}_{args.shard.replace('/', 'of')}.txt"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log(f"=== stage12 {time.ctime()} {vars(args)} {common.env_info()}")
    states = dict(np.load(args.states))
    model = load_policy(args.device)
    critic = Critic(args.device)
    # critic check: Q must not depend on the x_T slot
    a0 = np.random.default_rng(0).uniform(-1, 1, (16, H, A)).astype(np.float32)
    q0 = critic.q(states["obs"][0], a0)
    q1 = critic.q(states["obs"][0], a0, xt=torch.randn(16, H * A))
    log(f"critic x_T invariance: max |dQ| = {np.abs(q0 - q1).max():.2e}")
    assert np.abs(q0 - q1).max() < 1e-4
    if args.mode == "stage4":
        sel = [int(x) for x in args.stage4_states.split(",")]
        ks = [int(x) for x in args.stage4_ks.split(",")]
        gen_stage4(states, model, critic, args.device, args.out_dir, args.act_dir, sel, ks, log)
        return
    s3 = set(stage3_subset(states))
    with open(os.path.join(args.out_dir, "stage3_subset.json"), "w") as f:
        json.dump(sorted(int(x) for x in s3), f)
    si, sn = (int(x) for x in args.shard.split("/"))
    todo = [i for i in range(len(states["phase"])) if i % sn == si]
    if args.limit:
        todo = todo[: args.limit]
    t0 = time.time()
    for n, i in enumerate(todo):
        path = os.path.join(args.out_dir, f"state_{i:03d}.pkl")
        if os.path.exists(path):
            continue
        res = run_state(i, states, model, critic, args.device, args.act_dir, i in s3, log)
        with open(path + ".tmp", "wb") as f:
            pickle.dump(res, f)
        os.replace(path + ".tmp", path)
        if n % 10 == 0:
            log(f"progress {n + 1}/{len(todo)} elapsed {time.time() - t0:.0f}s")
    log(f"=== done {len(todo)} states in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
