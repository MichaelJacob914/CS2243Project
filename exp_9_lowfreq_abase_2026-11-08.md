# Experiment: exp9 low-frequency noise + a_base plan feature for DSRL on pi0 / LIBERO-10
Status: Completed

## Goal
Fix the two failure modes found in exp7/exp8: (1) tiling one 32-d row over 50 steps puts pi0 off-distribution;
(2) a raw 1600-d x_T in the agent obs breaks k=0 DSRL (exp8 control t4 0.14, t8 ~0.1 vs 0.82/0.94). Test (A) a
compact plan feature in place of x_T and (B) steering the first m DCT modes of each real action channel, for k=0 and
k=3.

## Setup
- **A. `--steer.obs_extra abase`** (default `xt` = previous behaviour). The agent obs has no x_T. Instead it gets
  `a_base` (350 floats): pi0's plan for this chunk's x_T, real LIBERO channels only (7, from openpi `LiberoOutputs`),
  all 50 steps, in pi0's normalized model space (before Unnormalize).
  - k=0: the plain pi0 sample from x_T. k>0: â_{k−1} = x_{k−1} − t_{k−1} v_{k−1} on the plain trajectory from x_T.
  - `adsrl_pi0.sampler.plain_features` is the prefix half. The suffix is the existing one-shot
    `steered_sample_actions`, which recomputes the k plain steps instead of resuming from a cached state, so the
    actions are exactly those of the one-shot sampler by construction. Tested: the k=0 feature equals openpi's plain
    sample, and the k>0 feature equals the â_{k−1} that the one-shot sampler re-noises at k.
  - Cost: one extra pi0 call per chunk batch (k steps for k>0, a full sample for k=0), plus one per batch of finished
    episodes for their final next_obs.
  - Each stored obs and next_obs carries the feature computed for its own x_T (tested), stored in the buffer like
    x_T was. The noise decode still uses x_T, which stays in the rollout obs but not in the agent's view.
  - Q(s, a_base, z) is not strictly Markov, because the x_T residual also affects the action. This is accepted.
- **B. `--steer.w_layout lowfreq --steer.n_modes m`**: B = first m orthonormal DCT-II vectors over the 50 steps
  (mode 0 = time-mean). For real channels c<7: w[:,c] = B z_c + (I − B Bᵀ) x_T[:,c]. Padding channels: w = x_T.
  - z has m·7 dims, mode-major (z.reshape(m,7)[j,c]), tanh bound b.
  - Tested: z ~ N(0,I) gives w iid N(0,I); z = Bᵀx_T gives w = x_T; m=1 matches the structured time-mean on the real
    channels; batched matches per-sample; the jaxrl2 actor has action_dim 21 and target entropy 'auto' = −10.5.
- **Scope**: fixed k (any k) only. Adaptive with abase or lowfreq raises NotImplementedError.
- **Runs** (1M env steps, seed 0, 12 envs, stoch (50 eps) + det (50 init states) eval every 50k; other
  hyperparameters as exp7 k0: batch 256, discount 0.999, multi_grad_step 20, hidden [128], lr 1e-4/3e-4/3e-4,
  num_qs 10, target_entropy auto):
  - main: tasks 8,4,6 × k∈{0,3} × lowfreq m=3 × abase × b=2.5 (6 runs) → `pi0_t{t}_k{k}_xtfull_lf3_abase_b2.5_s0`
  - ablation (obs change alone): tasks 8,4,6 × k=0 × structured × abase × b=1.0 (3 runs) →
    `pi0_t{t}_k0_xtfull_wstruct_abase_s0`; compare to exp8 structured b=1 (x_T obs): det 0.70 / 0.76 / 0.78 (t8/t4/t6).
- Hardware: 1 H100 each (kempner_h100, kempner_pehlevan_lab), 16 CPUs, 64G, 8h walltime (`--max_wall_hours 7.6`).
- Scripts: `bash scripts/pi0/sweep_exp9.sh` → `scripts/pi0/generated/exp9/run_*.sbatch`; run dirs
  `/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl_runs/exp9/`; logs `logs/exp9/`; wandb group exp9_pi0.
- Baselines: exp7 k0 tiled det 0.94 / 0.82 / 0.96 (t8/t4/t6); exp7 k3 (tiled, x_T obs) det 0.08 / 0.20 / 0.52 (t8/t4/t6); plain pi0 0.655 / 0.76 / 0.78.
- Code: untracked in git, so the run_info `git` field does not identify it. sha256 at prep time:
  - adsrl_pi0/rollout.py 0225b3a6afac9020d455dcbc3ae3d8151869a68d240a541605bd80c5f7c1dfdc
  - adsrl_pi0/sampler.py f9033ee8181bbeed8f6e003f379aafffc79b2ab1808dae415f7d174cc2d5c908
  - train_adsrl_pi0.py 856c522b68e24a8149e6fd6b1ba496754550beb05599e3aece25c2a92b009e6e
  - unchanged: agent.py 9a2184b3…, buffer.py 88329f79…, env.py 9a326b32…, \_\_init\_\_.py bbd33b57… (as exp8)
  - tests/test_pi0_exp9.py acfd4d04…, tests/test_pi0_sampler.py 7fcae3e9…, scripts/pi0/sweep_exp9.sh 29ec1f8d…
  - train_adsrl_pi0.py API changes (callers updated): `obs_spec(a)`, `build_agent(a)`,
    `agent_input/add_episodes(..., keys)`, `run_eval(..., rollout_kw=...)`; new `action_dim(a)`, `rollout_kw(a)`.
    Defaults keep the previous behaviour (exp8 args give the same obs keys, action dim and decode).

## Progress Log
- [2026-11-09 00:20] All 12 exp9 runs COMPLETED (incl. adaptive detach 59024905/6/7). Results and Analysis written below.
- [2026-11-08 19:25] Submitted the adaptive runs with the k-entropy fix only (`--hybrid.k_entropy_bias detach`, lowfreq m=3, abase obs, b=2.5, eval_at_start), on kempner_h100: t8 59024905, t4 59024906, t6 59024907. The keep (old loss) variant was not launched, by user choice. Pending (Priority); 12 kempner GPUs in total once running.
- [2026-11-08 19:30] **Adaptive variant + k-entropy-bias flag prepared (not submitted).**
  - The 9 fixed-k exp9 runs (59007654/55/56/59/60/79/81/83, 59013803) were already running when the code changed.
    Fixed-k and default paths behave the same under the new code, so a resubmission after a crash gives the same
    behaviour.
  - **Adaptive + abase + lowfreq** (parse no longer raises):
    - The feature is always the k=0 one (the plain π0 sample for the chunk's x_T), because k is chosen after
      observing (`rollout_kw` feature_k=0 for adaptive).
    - The hybrid learner's noise dim is `action_dim(a)` (21), with per-head target entropy 'auto' = −10.5.
    - Per-row k with per-row lowfreq decode goes through the same `steered_sample_actions` path as exp7 adaptive.
    - k grid: all 10 steps, ρ=0.5 (as exp7 `pi0_t8_adaptive_rho0.5_xtfull_s0`), b=2.5, `--eval_at_start 1`.
  - **`--hybrid.k_entropy_bias {keep,detach}`** (default keep = old loss):
    - detach: stop_gradient on the π(k) that multiplies α_w log π(w_k|k), so the categorical is trained only by Q
      and α_k H(π_k).
    - The heads keep their π(k)-weighted α_w term. Their gradients are identical to keep (tested), and the forward
      loss value is the same.
    - Uniform weighting was not used because it would change the heads' updates too, and the test asked for those
      to be unchanged.
    - Run-name tag `_kdetach`.
  - Adaptive runs now write `head_log_std.csv` at each eval: per-k mean log_std on 256 buffer obs drawn with their
    own RNG (the buffer RNG is untouched, tested). It is also printed and sent to wandb `eval/head_log_std_k*`.
    Fixed-k CSVs are unchanged.
  - **LIBERO EOFError evidence**: `faulthandler.enable()` in `_worker`, and `LiberoVecEnv._recv` prints the worker
    index, pid, alive flag and exitcode, then re-raises (tested with a worker that exits with code 3).
  - CPU tests: 59015622 had 96 passed, 3 skipped before the detach flag. 59016583 (full suites incl. detach) had 99
    passed, 3 skipped, 1 failure; the failure was a run-name order mismatch in the test (`_kdetach` precedes
    `_b2.5`). After fixing the test, 59018551 ran test_pi0_exp9.py: 36 passed.
  - GPU smoke 59015676, adaptive keep, t8, 22 min:
    - eval at start: stoch 0.67, det 0.72.
    - 10.4k: 0.50 / 0.50. 22k: 0.25 / 0.46. Then resumed from checkpoint agent_17040 and trained on to 33.7k:
      0.33 / 0.30.
    - Det k_hist shifts toward k=8 at 22k, then spreads out by 33.7k.
    - Head log_std about −0.8 at 33.7k. The 3 checkpoint tests and the mixed-k lowfreq sampler test pass on GPU.
  - GPU check 59016545, adaptive detach, t8, 20k steps, 11 min:
    - 10.7k: 0.42 / 0.74. 21.8k: 0.08 / 0.04.
    - Det k_hist at 21.8k has 714 of 1296 on k=9, so mass still moves to late k early with detach. Far too early to
      judge; the head log_std is equal across k at this stage (−0.35).
    - run_info `k_entropy_bias`: detach.
  - Throughput (adaptive): steady 380–405k env steps/h, against exp7 t8 adaptive at the same stage at 410–430k/h
    (exp7 t8 adaptive took 3.7 h in total). About 5–10% slower; estimate ~4.5 h per 1M run, so 8h is ample.
  - Scripts: `scripts/pi0/sweep_exp9_ad.sh` (writes only `run_t{8,4,6}_ad_lf3_abase_b2.5.sbatch`, keep variant).
    `KBIAS=detach bash scripts/pi0/sweep_exp9_ad.sh` writes `run_t{8,4,6}_ad_lf3_abase_kdetach_b2.5.sbatch`: the same
    command plus `--hybrid.k_entropy_bias detach` (run name gets `_kdetach`). Both sets are generated.
  - sha256 now:
    - agent.py 9c3b3709…, env.py a0d5a454…, train_adsrl_pi0.py 98bd5a6d…
    - unchanged since the fixed-k exp9 launch: rollout.py 0225b3a6…, sampler.py f9033ee8…, buffer.py 88329f79…
    - tests/test_pi0_exp9.py 85f66ceb…, tests/test_pi0_sampler.py dcd93e08…
    - The 9 running fixed-k runs used agent.py 9a2184b3…, env.py 9a326b32…, train_adsrl_pi0.py 856c522b….
- [2026-11-08 17:50] t4_k3_lf3 59007658 FAILED at 13 min with the LIBERO worker EOFError (4th occurrence overall; empty checkpoint) → resubmitted fresh as 59013803. Others at 100–150k: t8_k0_lf3 det 0.78 (start) → 0.02 (52k) → 0.76 (102k); t8_k3_lf3 0.84 → 0.00 → 0.88; t6_k3_lf3 0.92 → 0.86 → 0.94 → 0.90; t6_k0_lf3 0.82 → 0.56 → 0.94; t4_k0_lf3 0.86 → 0.72 → 0.78; wstruct_abase b=1: t8 0.62 → 0.64 → 0.84, t4 0.80 → 0.84 → 0.76, t6 0.86 → 0.76 → 0.94.
- [2026-11-08 ~17:30] Fable review: fine to launch, no result-breaking issues. Smoke collapse (det 0.92 → 0.02 by 21k, α 0.26 → 0.0025) matches the parametrization's expected entropy/temperature path; exp7 t8 runs were also ~0 at 55k. Notable: det eval at 10.6k (z ≈ 0, i.e. x_T with its 3 low-frequency modes zeroed per real channel) scored 0.92 on t8 vs plain π0 0.655. Added `--eval_at_start 1` to all exp9 sbatch (and sweep_exp9.sh) to record that z=0 baseline per run. Submitted on kempner_h100: t8_k0_lf3 59007654, t8_k3_lf3 59007655, t4_k0_lf3 59007656, t4_k3_lf3 59007658, t6_k0_lf3 59007659, t6_k3_lf3 59007660, t8_k0_wstruct_abase_b1.0 59007679, t4 59007681, t6 59007683 (~4.5 h each). Watch first 100k steps for collapse.
- [2026-11-08 17:10] GPU smoke 59000504 (kempner_h100, 33 min, COMPLETED): t8, k=3, lowfreq m=3, abase, b=2.5.
  - run_info: action_dim 21, obs_keys [pixels, state, a_base], use_xt_obs false.
  - Trained to 21.5k env steps, then resumed from the checkpoint (agent_16560) and trained on to 33k.
  - det eval: 0.92 at 10.6k, 0.02 at 21.6k, 0.02 at 33k. Stoch eval (12 episodes): 0.67, 0.17, 0.00.
  - Entropy fell 32.7 → −2.6 (target −10.5); temperature 0.26 → 0.0025.
  - The early collapse matches exp7: t8 k0 and k3 were both at 0.00 det at 55k and recovered later. So no conclusion
    yet, but watch the first 100k.
  - The 3 GPU-only checkpoint plan-feature tests pass, including the k=0 feature matching the plain sample on the real model.
  - Throughput: steady rounds take 52 s for 5.45k env steps (375–381k env steps/h) against exp7 t8 k3/k0 at the
    same stage, 420–460k/h, so about 13% slower. The feature call costs about the same as the steered call (3.5–7.7 s
    vs 4–6 s per round). 1M-step estimate: about 4.5 h including evals (exp7 t8 k3 took 3.7 h). The 8h walltime is ample.
- [2026-11-08 16:55] CPU tests 59000449 (test partition): tests/test_pi0_{exp9,structured_noise,train_fixes,agent,
  buffer,sampler}.py, 89 passed, 3 skipped (GPU-only), 16.6 min.
- [2026-11-08 16:30] Implemented A and B (tests first).

## Results
All 12 runs completed 1M env steps (3.3–4.3 h each). Metric: deterministic eval success, 50 episodes per eval, mean of the last 3 evals (each eval has about ±0.06 SE).

| Run | t4 | t6 | t8 |
|---|---|---|---|
| lowfreq+abase k=0 | 0.93 | 0.99 | 0.99 |
| lowfreq+abase k=3 | 0.93 | 0.98 | 0.95 |
| lowfreq+abase adaptive (detach) | 0.91 | 0.97 | 0.95 |
| structured+abase k=0, b=1 | 0.85 | 0.86 | 0.87 |
| exp7 tiled k=0 (mean of last 3) | 0.83 | 0.97 | 0.89 |
| exp7 tiled k=3 (final) | failed (0.08–0.72 range across k>0) | | |
| plain π0 | 0.76 | 0.78 | 0.655 |
| eval_at_start (z=0) of lowfreq runs | 0.82–0.86 | 0.82–0.92 | 0.72–0.84 |

Adaptive step choice (det eval k histogram, final eval): t8 mostly k=1–2 (k0..9: 115 187 169 63 77 70 29 46 37 6); t6 spread over k=0–4, few late; t4 k=0 and k=3 most common, k=9 rare. No drift toward late steps.

## Analysis
- The lowfreq noise layout plus the abase observation fixes later-step steering: k=3 went from failing in exp7 to 0.93–0.98, matching k=0.
- k=0, k=3 and adaptive are all within eval noise of each other, and all are at 0.91–0.99, near the ceiling. These three LIBERO tasks can no longer separate the methods; a harder task or more seeds is needed.
- With the k-entropy fix, adaptive moved toward early/middle steps instead of late ones, so the late-step drift seen before the fix did not occur here.
- The structured (rank-1 per channel) layout lags lowfreq by about 0.1 on every task.
- 1 seed per cell.

## Next Steps
