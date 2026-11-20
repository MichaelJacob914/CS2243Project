# Experiment: Kick parameterization, headroom for learned k, equal control strength, and the early dip (transport)
Status: Running

## Goal
Four follow-ups from the user's notes (2026-11-10), all on robomimic transport:
1. **Kick parameterization.** The actor outputs a perturbation Δ added to a reference noise, so Δ=0 reproduces the base policy exactly at every steering step k (LP-DS, arXiv 2606.01151, does this at k=0).
2. **Is k more than an authority dial?** Give k=0, 3, 5, 8 equal control strength with the per-k noise scale (`steer.w_scale`, exp4), then train.
3. **Headroom for learned k.** Before more work on adaptive k: for a few hundred states, find the best k by brute force under a trained critic, and check whether it changes with task phase. Compare critic-chosen k against best-of-N sampling at k=0 with equal compute.
4. **Fix the early dip.** Every method except k=0 falls from base-level success (0.17–0.36) to 0.01–0.10 in the first ~256k learning steps, then recovers. Test a noise trust region, a zero-initialized actor (with the kick), and a lower starting entropy.

## Decisions (user, 2026-11-10)
- Kick reference = **implied noise**: the noise that puts the sampler exactly on the base trajectory given x_T, which is already in the obs. noise = ε̂_k(s, x_T) + Δ. At k=0 this is x_T + Δ (the LP-DS form). Δ=0 gives the deterministic base action, so the problem stays Markov.
- Dip ablation: k=0 and k=3, 3 seeds, 3M env steps.
- Equal control strength: k ∈ {0, 3, 5, 8}, **1 seed**, 9M env steps.
- Headroom critic: Q^A from the uniform-random-k run, seed 1 (at 6M, 0.79 success; its data covers every k).

## Setup
### Phase 0: analyses, no training (~10 GPU-h, ~0.5 day)
- **Authority calibration at K=20.** `scripts/authority.py` on transport, DDIM K=20, dataset states, k ∈ {0,3,5,8}, w_scale grid. Choose c_k so per-dim action std over w matches k=0 at c=1. Note: exp4 on can found that matched authority closes most but not all of the late-k gap (k7 c3 0.85 vs k0 c1 0.93).
- **Headroom.** Load the uniform-k s1 agent at 6M. Collect ~300 states from its own rollouts, labelled with task phase and time in episode. Per state and per k in 0..19, draw N actor samples, decode, and score with Q^A (min of the twin critics). Report: the distribution of argmax-k, how it varies with phase, and the Q gap between best-k and k=0. Then env rollouts, 200 episodes each:
  - (i) per chunk, the critic-chosen (k, w) over all k with N samples per k;
  - (ii) best-of-(20·N) at k=0, equal decode compute;
  - (iii) the plain policy (N=1).
  - If (i) does not beat (ii), learned k has little headroom.

### Phase 1: code (subagent, tests, Fable review)
- `steer.param: absolute | kick` (default absolute). Kick: noise = ε̂_k + b·tanh(Δ_raw), for fixed and adaptive, in every decode path (acting, eval, target backup, distillation, best-of-N, render).
- `steer.actor_zero_init`: zero-init the actor's mean output layer.
- Starting entropy: `train.ent_coef` accepts `auto_<init>`; use init 0.05 against the default 1.0.
- Dense early evals: eval every ~64k env steps up to 1.6M (50 episodes), then the usual cadence. This resolves the dip, which falls between evals today.

### Phase 2: training on transport, kempner_h100, 3 runs per H100 (MPS)
- **Dip ablation**, 3M, seeds 1–3, k ∈ {0, 3}. Arms:
  - (a) trust region (existing, bound 0.3);
  - (b) kick + zero-init actor;
  - (c) starting entropy 0.05.
  - Baseline = exp10 fixed k0/k3 seeds 1–3 at 3M (same config).
  - 18 runs = 6 packs, ~25 h each ≈ 150 GPU-h.
- **Equal strength**, 9M, seed 1, k ∈ {0,3,5,8} at the calibrated c_k. 1 pack of 4, ~100k env steps/h per run → ~90 h with 2 chained follow-ups ≈ 90 GPU-h.
- **Equal strength, adaptive k** (added 2026-11-12, user request "run just adaptive k, single seed"): `adaptivedetach_rho0.5_eq_s1` = exp10 `adaptivedetach_rho0.5_s*` settings (steer.mode=adaptive, rho=0.5, k_entropy_bias=detach) plus steer.w_scale=1.0, steer.w_scale_dims=`results/exp13/authority/w_scale_dims_allk_transport_K20.npy` (per-k (20, 14) scales; each chunk uses its own k's row), and the same early evals as the fixed eq runs. Seed 1, 9M, 1 H100 (kempner_h100), 2-day packs chained with afterany; ~4 days expected. Compare with `fixed_k{3,5,8}_eq_s1` and exp10 `adaptivedetach_rho0.5_s{1,2,3}`. Launch via `scripts/slurm/exp13_dip.sbatch`. **Held for a user decision on the large late-k scales (see Progress Log 2026-11-12).**
- Total ≈ 250 GPU-h on top of the 9M extension (exp10). Peak ≈ 13 kempner GPUs (6 exp10 transport + 7 here), so it fits under the 15 cap.

### Expected results
- Kick + zero-init removes the dip, since the start is the exact base policy. A lower starting entropy should shrink it. A trust region limits how far noise strays, but may slow the late climb.
- If k only acts as an authority dial, matched strength makes k=0/3/5/8 tie. If late k still differs, the step matters through which features of the action it can change.

## Progress Log
- [2026-11-20 12:30] After the filesystem outage: follow-up 60087686 FAILED on Nov 16 17:02 after 55 min with an empty run log (no traceback; consistent with the outage). The run is at 7.04M (last det 0.905/0.90/0.93 at 6.53–7.04M); checkpoint latest.pt + replay buffer from Nov 16 16:03 are intact. Kempner was at its 16-GPU cap (other projects), so resubmitted the same command on gpu_h200 / olveczky_lab: 62382400 (pending, resume=true). About 2M steps left, ~26 h of training. Report v29: all figures regenerated.
- [2026-11-16 11:02] Adaptive eq (60087684) at 6.78M: last 3 det 0.905/0.90/0.905 (standard-noise adaptive seed 1 at the same steps 0.87/0.92/0.895). 60087684 hits its 2-day limit in about 5 h; 60087686 (afterany) resumes from checkpoint. ETA 9M around Nov 17 evening.
- [2026-11-14 06:19] Equal-strength runs finished 9M (59633900 → 59633907 COMPLETED 05:21). Mean of last 3 det evals at ≤9M, seed 1: k3_eq 0.883, k5_eq 0.908, k8_eq 0.845; standard noise seed 1: k0 0.832, k3 0.908, k5 0.903, k8 0.823 (at its 6M budget). k8_eq peaked near 0.95 around 6M and settled at 0.845. Adaptive eq (60087683) 0.17 at 2.94M.
- [2026-11-13 16:30] Adaptive-eq (60087683) at 1.92M: det 0.065 (standard adaptive s1 0.16, k0 s1 0.105 at the same step). Throughput 1.65x slower per env step than fixed-k eq (76.8k s vs 46.4k s wall to 1.92M); GPU kernel-busy 35% of the time (dmon), so the run is bound by per-process overhead, not the GPU. ETA 9M about Nov 17-18. Added panel d (adaptive k) to exp13_equal_strength; report v26.
- [2026-11-12 16:05] User approved launching adaptive k at equal strength with the uncapped all-k scales (up to 26.6 at k=19, gripper). Submitted adaptivedetach_rho0.5_eq_s1 to 9M on kempner_h100: 60087683 (running), follow-ups 60087684, 60087686 (afterany, resume from latest.pt). Watch the k histogram: if it settles on k=18–19, read as additive residual control, not diffusion steering.
- [2026-11-12 15:45] Second Fable review (test fix, calibration, late-k safety): no bugs. The test fix is right, and the per-dim test means what it did before. The calibration converged at every k (in-sample error ≤ 0.001 for k ≤ 18 and 0.003 at k=19). Row 0 is ones; rows 3/5/8 match the old files. The replicate puts calibration noise at about 3–7%. On safety, the injected noise is s_k·v·w, where s_k = sqrt(1-ᾱ_k) is the noise std the denoiser was trained on; for |w| = 1 its size is never more than about 1.4 and is 0.67 (gripper) and 0.15 (other dims) at k=19. So actions cannot blow up: the denoised clamp keeps them in [-1, 1]. Two concerns:
  - k=19 is nearly direct additive gripper control: a ≈ clamp(a_base + 0.67·w).
  - k=14–18 feeds the denoiser 3.5–9× its trained noise level, so the w → action map there is less smooth and harder for Q^W to fit.
  - If π(k) settles on k=18–19, read it as "additive residual control wins", not as diffusion steering. Per (h, a) chunk position the spread still differs from k=0 (0.67–1.96 at k=14–16); a (K, H, A) file would fix that, and the loader accepts one, but the script does not produce it.
  - Distillation draws w ~ N(0, I) without clipping, so at late k Q^W also trains on w the actor never produces.
- [2026-11-12 15:30] All-k calibration and tests done. **Run not launched: the late-k scales are large, so the launch waits for the user's decision.**
  - Tests. Red, on the old base_policy.py (via a scratch copy of the tree; job 60033508): 7 old tests pass, all 6 new ones fail. Green, first try (job 60033622) failed 2 new tests because of a test bug: the equal-rows comparison started from weights already changed by an earlier training step. Fixed so every variant runs from one saved state. Rerun (job 60061769): test_w_scale_dims.py 13 passed; adsrl suite 164 passed, 1 failed (the known test_eval_env_seed_controls_initial_states).
  - Calibration (job 60061683, H100; 1000 dataset states × 128 w, seed 0, verify seed 1, the settings the k=3/5/8 vectors used per their JSON). The first try at damping 0.5 (job 60052107) oscillated on the gripper dims at k=11–17 (fresh-draw ratio down to 0.45), because gripper spread grows about as v^3.8 there. It is kept as `w_scale_dims_allk_damp0.5_*`. Rerun with `--damp 0.2 --iters 25`: `results/exp13/authority/w_scale_dims_allk_transport_K20.npy` (20 × 14, row 0 = ones; sha256 32b73634…, sum 456.213; file made read-only) plus `w_scale_dims_allk_calib_transport_K20.json`.
  - Fresh-draw check, per action dim std ratio to k=0: within 0.975–1.038 for every k. Per (h, a) chunk position it is 0.67–1.96 at k=13–16, since one vector per k can't fix the spread across the chunk.
  - k=3/5/8 rows vs the existing files: max relative difference 3.4e-4 (same seed, so this checks convergence, not noise). Replicate with fresh states and w (seeds 2/3, `/n/netscratch/olveczky_lab/Lab/charleszhang/exp13_wsd_allk_rep_60061683/`): max relative difference ≤ 3.2% for k ≤ 18 and 5.2% at k=19.
  - Scale range per k (min–max; the max is always a gripper dim, 6 or 13): k1 1.00–1.10, k2 1.00–1.14, k3 1.00–1.20, k4 1.01–1.27, k5 1.01–1.37, k6 1.01–1.48, k7 1.01–1.62, k8 1.01–1.78, k9 1.01–1.95, k10 1.01–2.15, k11 1.01–2.38, k12 1.01–2.66, k13 1.01–3.00, k14 1.01–3.46, k15 1.02–4.10, k16 1.02–5.03, k17 1.04–6.54, k18 1.12–9.29, k19 1.96–26.6. The largest non-gripper value per k is 1.57 at k16, 1.78 at k17, 2.28 at k18 and 6.06 at k19.
  - Launched 2026-11-12 16:05 with the scales as they are (user decision); jobs 60087683 → 60087684 → 60087686. Why it was held: at step k the noise enters the denoiser input as sqrt(1-ᾱ_k)·v·w, so v is roughly how many times its trained noise level the network sees. Up to about k=12 (v ≤ 2.7) this stays near N(0,1) tails. At k=16–19 the gripper noise is 5–27× and at k=19 every dim is ≥2×. Options for the user: launch as is, cap v (a value to choose), or launch anyway and watch how much the agent uses late k.
- [2026-11-12 14:25] Adaptive equal-strength arm (`adaptivedetach_rho0.5_eq_s1`), work in progress, nothing launched yet.
  - Code: `steer.w_scale_dims` now also takes a per-k file, shape (K, A) (tiled over the horizon) or (K, H, A). Each sample's w is scaled by the row of its own k, so mixed-k batches work in every decode path. 1-D files keep their old meaning. Change is in `load_w_scale_dims` and `SteeredBasePolicy` (adsrl/base_policy.py); `train_adsrl.py` now also logs the loaded scale's shape, min, max and sum. `scripts/exp13_wscale_dims_calib.py --allk` calibrates k = 1..19 into one (20, 14) file. `scripts/slurm/exp13_dip.sbatch` accepts `adaptivedetach_rho<rho>_eq_s<seed>`.
  - Tests: 4 new tests in `tests/test_w_scale_dims.py` (loader, real-sampler mixed-k batch, every agent path for fixed and adaptive, config plumbing). They include bitwise checks that a per-k file with equal rows gives the same result as the single-vector file.
  - Fable review of code, tests, calibration script and sbatch dry run: no bugs. Risks it noted: (1) the run records only the file path, so overwriting the all-k file would silently change a resumed run (mitigated: the train log now prints a fingerprint, and I will make the file read-only after calibration); (2) a 2-D (H, A) file is now read as per-k (fails loudly unless a task has H == K; no such files exist); (3) kick + per-k scales is mechanically fine but was not calibrated.
  - Blocker: the lab filesystem (holylfs06) reads at 0.7–1.7 MB/s since about 12:40, so jobs stall for an hour or more while importing torch/CUDA from `.venv`. Calibration job 60027369 timed out at 1:30 while still importing (no output). Resubmitted as 60052107 (6 h limit, same node). Test jobs 60033508 (red, run against the old base_policy.py through a scratch shadow tree) and 60033622 (green, adsrl suite) are still importing. First red run 60024340 was cancelled and 60027329 timed out, both stuck in imports.
- [2026-11-11 09:40] Dip ablation CANCELLED by user request once concluded (59633793, 59633827, 59633833, 59633861, 59633866, 59633884; all 18 runs at 1.66M, past the early-eval window that ends at 1.6M; checkpoints kept, resumable). Conclusion: none of the three fixes prevents the dip. With early evals every 64k, det success falls within the first 64k learning steps (640k warm start → 704k) in every arm. Step 0 vs 704k: kickzi 0.17–0.21 → 0.00 (then ≤0.08 through 1.66M, the worst arm); trust region k3 0.14–0.23 → 0.00–0.12, k0 stays 0–0.3; ent 0.05 k3 0.18–0.26 → 0.00–0.16, k0 ≈0–0.18. At 1.66M: tr k3 0.16–0.22, k0 0.01–0.11; ent005 k3 0.11–0.16, k0 0.03–0.05; kickzi k3 0.02–0.055, k0 0.015–0.11. Starting exactly at the base action (kickzi) does not help and slows recovery, so the dip is not caused by where the actor starts or how widely it explores. The next suspect is early critic error (untested; proposed critic-only warm-up arm). Equal-strength runs (59633888 chain) continue.
- [2026-11-10 22:20] User approved submission. Dip ablation (scripts/slurm/exp13_dip.sbatch, 3 seeds per pack, 3M, early evals every 64k to 1.6M at 50 eps): k0_tr 59633793, k0_kickzi 59633827, k0_ent005 59633833, k3_tr 59633861, k3_kickzi 59633866, k3_ent005 59633884. Equal control strength (new `eq` arm: steer.w_scale=1.0 + per-dim vectors results/exp13/authority/w_scale_dims_k{3,5,8}_transport_K20.npy, Fable-reviewed OK; ADSRL_EXTRA total 9M; seed 1; k=0 reference = exp10 fixed_k0_s1 at 9M): 59633888 → 59633900 → 59633907 (afterany chain). All PENDING on kempner_h100; would bring kempner to 13 GPUs.
- [2026-11-10 21:00] Equal-strength code done (not launched). New key `steer.w_scale_dims` (default null): a path to a .npy/.json with H*A values, or A values tiled over the horizon. It multiplies w per dimension on top of `steer.w_scale`, so noise = w_scale * dims * w (kick: eps_hat_k + that). It is applied in SteeredBasePolicy, so every decode path gets it.
  - Tests in `tests/test_w_scale_dims.py`, written first; they failed before the code existed (job 59623485). Now 7 passed; adsrl suite 158 passed, 1 known failure (job 59624659).
  - Calibration: `scripts/exp13_wscale_dims_calib.py` (job 59624201, H100, 75 s; 1000 dataset states x 128 w, w = clip(N(0,1)) * v). Damped fixed-point iteration, starting from authority_per_dim_ck. The undamped update oscillated on the gripper dims at k=8 (first try, job 59623638).
  - Vectors: `results/exp13/authority/w_scale_dims_k{3,5,8}_transport_K20.npy`, summary in `w_scale_dims_calib_transport_K20.json`. Ranges: k3 1.004–1.195, k5 1.006–1.371, k8 1.010–1.779; the max is gripper dim 6 each time, dim 13 is next.
  - Verified on fresh w draws, per-action-dim std ratio to k=0: k3 0.999–1.003, k5 0.998–1.009, k8 0.997–1.017, against unscaled 0.78–0.98 / 0.62–0.97 / 0.31–0.94. Per (h, a) chunk position: 0.98–1.02 / 0.96–1.05 / 0.89–1.16; a vector tiled over the horizon can't fix the per-position spread.
- [2026-11-10 20:15] Dip ablation (18 runs, 6 packs) prepared but NOT submitted: the auto-mode permission classifier denied the sbatch calls. Waiting for the user to submit or allow it.
  - Script: `scripts/slurm/exp13_dip.sbatch` (copy of exp3_wave1 plus arm cases). Run names `fixed_k{0,3}_{tr,kickzi,ent005}_s{1,2,3}`; none of these logdirs exist yet.
  - Per-arm overrides on top of the exp3/exp10 base (`--config-name=transport env.n_eval_envs=10 wandb.mode=offline log_dir=.../adsrl_runs resume=true checkpoint_every_min=120 model.ddim_steps=20 steer.mode=fixed steer.k=K seed=S train.early_eval_every=64000 train.early_eval_until=1600000 train.early_eval_episodes=50`):
    - tr: `train.trust_region.enabled=true train.trust_region.bound=0.3`
    - kickzi: `steer.param=kick steer.actor_zero_init=true steer.actor_log_std_init=-1.4 train.ent_coef=auto_0.05`
    - ent005: `train.ent_coef=auto_0.05`
  - Dry run (`DRY_RUN=1`) printed all 18 commands. A Hydra compose check (job 59622808) resolved xT_inputs: k0 tr/ent005 = none, k0 kickzi = actor_noise_critic, all k3 = actor_noise_critic; total 3M, K=20.
- [2026-11-10 19:50] Seed replicate 59591203 COMPLETED (60 min, ≈1 GPU-h; Phase 0 total ≈ 2.3 GPU-h). s2: (i) 0.915, (ii) 0.945, (iii) 0.87, (iv) 0.88. s3: 0.865 / 0.87 / 0.795 / 0.775. Pooled over 3 seeds (600 episodes): (i) 0.878, (ii) 0.873, (iii) 0.825, (iv) 0.828. Paired (i) vs (ii): p = 0.83, so **no headroom for learned k**. Both best-of-640 arms beat the policy (p = 0.005 and 0.011). The s1 phase dependence does not replicate. Results and Analysis updated; pooled numbers in `results/exp13/pooled_seeds.json` (`scripts/exp13_pool_seeds.py`).
- [2026-11-10 19:40] Review fixes (not committed).
  - (1) Bug: `resolve_xt_inputs` mapped auto to `none` at fixed k=0 even with kick, although DDIM(x_T + Δ) depends on x_T. Now auto picks `actor_noise_critic` under kick. `xT_inputs=none` with kick is rejected both in `resolve_xt_inputs` and in `DSRLFixedK`. Repro tests failed on the old code (job 59594937) and pass now.
  - (2) New option `steer.actor_log_std_init` (default null, no change). It zeroes the log_std head weight and sets the bias to the value, on the fixed actor and on all HybridActor heads. It is applied before the checkpoint load, so a resume overwrites it. Planned value for kick runs: -1.4 (σ≈0.24) with ent_coef auto_0.05.
  - Tests: test_kick + test_w_scale + test_sampler 100 passed; adsrl suite 151 passed, 1 failed (the known test_eval_env_seed_controls_initial_states). Both in job 59595345.
- [2026-11-10 18:55] Phase 1 code done (not committed; Fable review pending). New keys: `steer.param` (absolute|kick), `steer.actor_zero_init`, `train.ent_coef=auto_<init>`, `train.early_eval_{every,until,episodes}`. Kick is done once in `steered_ddim_sample(kick=True)` (noise = eps_hat_k + w_scale*Delta; x_T + Delta at k=0), so every decode path that goes through SteeredBasePolicy gets it. Distillation in kick mode samples Delta ~ U[-1,1]^D. Tests: new `tests/test_kick.py` (Delta=0 is the exact base action at every k, fixed and adaptive, mixed k; zero-init; ent init; early-eval schedule incl. resume; checkpoint round trip). Adsrl suite 143 passed, 1 known failure (test_eval_env_seed_controls_initial_states); jobs 59581912, 59582983. Old-vs-new default path bitwise identical (trained params + actions, fixed/fixed+TR/adaptive), job 59582404. Smoke pack (transport, 10k learning steps, 5 runs/1 H100) job 59582842, logs /n/netscratch/olveczky_lab/Lab/charleszhang/exp13_smoke/59582842. Step-0 det: k3 kick+zero-init 0.165 with |a-a_base|=0.000 (200 ep); adaptive kick+zero-init 0.26, |a-a_base|=0.000 (50 ep); k3 absolute+zero-init (w=0) 0.195 (200 ep), so on transport w=0 at k=3 is not catastrophic the way it is at k=0. Kick stoch eval at step 0 is low (0.09 vs 0.39 absolute): tanh(N(0,~1)) exploration is added on top of eps_hat. ent auto_0.05: alpha starts at 0.05 (0.032 after ~2k updates, vs 0.55 for the default). Early evals fired on schedule (det only, 50 ep, `episodes` column in the CSV), and resume continued the schedule.
- [2026-11-10 18:55] Seed replicate submitted as job **59591203** (`scripts/slurm/exp13_phase0_seeds.sbatch`, 1 GPU, ~1.5 h expected). It runs uniformk_s2 and uniformk_s3 from `final.pt` (the job asserts env_steps = 6M), each copied to scratch. Same settings as s1: paired arms (iii, iv, i, ii), 200 episodes, 640 candidates, plus the 300-state headroom with the split-half check. Outputs: `results/exp13/headroom_paired_s{2,3}/`. Paired tests: `scripts/exp13_paired_tests.py`.
- [2026-11-10 18:45] **Phase 0 done.** Paired job 59586146 COMPLETED (43 min, ≈0.7 GPU-h). Paired success: (i) 0.855, (ii) 0.805, (iii) 0.81, (iv) 0.83. Pooled with the unpaired run (400 episodes per arm): 0.865 / 0.823 / 0.825 / 0.82. Paired McNemar (i) vs (ii): p = 0.14. Phase 0 total ≈ 1.3 GPU-h (smoke 59582001, 59582989, 59586146). Results and Analysis updated. Calibrated c_k for the Phase 2 equal-strength runs: k3 1.06, k5 1.12, k8 1.26 (gripper dims would need 1.19 / 1.35 / 1.74).
- [2026-11-10 18:00] Phase 0B arms rerun with paired initial states: job **59586146** (`scripts/slurm/exp13_phase0_paired.sbatch`, `exp13_headroom.py --paired`, out `results/exp13/headroom_paired/`). The script seeds each worker's global np RNG before every reset through a new env subclass in the script; no library change. Its built-in check passed: initial obs are identical across arms (max diff 0.0) and differ across episodes. The unpaired run is kept as a second 200-episode replicate.
- [2026-11-10 17:55] Phase 0 job 59582989 COMPLETED (27 min, ≈0.45 GPU-h). A done (Results). B, first pass (unpaired): (iii) stoch 0.84, (iv) det 0.81, (i) critic-chosen (k,w) 0.875, (ii) best-of-640 at k=0 0.84. Wall time per arm 334 / 339 / 434 / 438 s. **Caveat found:** "same eval seed" did not give the same initial states. The known exp_2 eval-seeding bug makes `env.seed()` a no-op for robomimic initial states, so these four arms are unpaired: a McNemar test on episode index is meaningless, and the initial obs differ. Headroom states (300) come from this run's arm (iii). Phase labels were revised after looking at the gripper/payload features (arm0 grasps ~0.07 m from the payload center, arm1 ~0.03 m; finger gap is ~0.08 open and 0.02–0.045 closed on an object). The revised phases follow the expected order in time.
- [2026-11-10 17:27] Phase 0: smoke 59582001 passed (5-episode arms, 20-state headroom; checkpoint confirmed env_steps=6,000,000; obs layout check passes: gripper-to-payload = payload − eef to 1e-7). Arm cost at 640 decodes/chunk is only ~10% above the plain policy, so N=32 is affordable for (i)/(ii). Submitted full Phase 0 job **59582989** (kempner_h100, 1 GPU, 23 CPUs, 4 h limit; expected ~1 h ≈ 1 GPU-h): A (500 dataset states, 64 clipped w, k∈{0,3,5,8}, c∈{1,1.25,1.5,1.75,2,2.5,3,4,5}) then B (arm iii → 300-state headroom with N=32 per k → arms iv, i, ii; 200 episodes each on 20 envs, same eval seed). Outputs: `results/exp13/authority/`, `results/exp13/headroom/`.
- [2026-11-10 17:20] Phase 0: smoke job 59582001 (kempner_h100, 1 GPU) for the new scripts `scripts/exp13_authority_calib.py`, `scripts/exp13_headroom.py`, `scripts/exp13_headroom_analyze.py` (sbatch: `scripts/slurm/exp13_phase0.sbatch`). Phase 0 runs from a frozen code snapshot (`/n/netscratch/olveczky_lab/Lab/charleszhang/exp13_phase0/code_snapshot`, taken 17:17) because adsrl/ is being edited for Phase 1; the default absolute path is unchanged there. Checkpoint: uniformk_s1 `final.pt` (written 11-07 at the end of the 6M run; latest.pt is being overwritten by the 9M extension) copied to `.../exp13_phase0/uniformk_s1/checkpoint/final.pt`.
- [2026-11-10 15:40] Doc created. User answered the 4 design questions (above). Starting Phase 0 analyses and Phase 1 code in parallel.

## Results
### Phase 0A: authority calibration (job 59582989, `results/exp13/authority/`)
Transport, DDIM K=20, 500 dataset states, 64 w per state, w = clip(N(0,1), −1, 1)·c (the actor's tanh box scaled by `steer.w_scale`); same w draws for every (k, c). Reference = k=0, c=1: mean per-dim action std 0.0526, mean |a − a_base| 0.069 (L2 over the 112-d chunk 1.55).

| k | std ratio at c=1 | **c_k** (match mean std) | c_k (log-LSQ over dims) | per-dim std ratio at c_k (min–max) | mean \|a − a_base\| at c_k (k0 c1: 0.069) | L2 at c_k (k0 c1: 1.55) | gripper-dim c (dims 6, 13) |
|---|---|---|---|---|---|---|---|
| 0 | 1.00 | 1.00 | 1.00 | 1.00 | 0.069 | 1.55 | 1.00, 1.00 |
| 3 | 0.92 | **1.06** | 1.04 | 0.87–1.07 | 0.066 | 1.46 | 1.19, 1.16 |
| 5 | 0.85 | **1.12** | 1.09 | 0.73–1.14 | 0.064 | 1.37 | 1.37, 1.32 |
| 8 | 0.73 | **1.26** | 1.21 | 0.46–1.30 | 0.060 | 1.22 | 1.76, 1.72 |

- The calibration is small at K=20: c_8 = 1.26. The scaled noise stays in [−1.26, 1.26] with RMS 0.90 (N(0,1): RMS 1, 4.6% beyond ±2). **No c_k leaves the N(0,1) range.** Even the gripper-matched c ≈ 1.75 stays inside ±2 (RMS ≈ 1.26).
- k is not a pure authority dial. Later k loses authority mostly on the two gripper dims (6, 13). At c_8 the 6 arm-pose dims per arm are 1.2–1.3× over-matched while the grippers sit at 0.46–0.50×. A single scalar c cannot match both. Matching the grippers needs c ≈ 1.2 / 1.35 / 1.75 for k = 3 / 5 / 8, and that over-drives the pose dims.
- At matched std, mean |a − a_base| is still lower for late k (0.060 vs 0.069 at k=8). Late-k samples spread around a point nearer the base action.
- Action std grows faster than linearly in c, even at k=0 (c=2 gives 2.9× the std). Large c moves the network out of its input distribution.
- Files: `authority_grid_transport_K20.{csv,npz}`, `authority_calib_transport_K20.json`, `authority_per_dim_ck_transport_K20.json`.

### Phase 0B: headroom for learned k (uniformk_s1, `final.pt`, env_steps = 6,000,000; job 59582989; `results/exp13/headroom/`)
**State-level headroom.** 300 states from the agent's own stochastic rollouts (arm iii): pre-success chunks, 30 per decile of normalized episode time τ = chunk / (first-success chunk + 1). For each state and its x_T, k = 0..19, N = 32 w from that k's head; scored by Q^A = min of the twin critics.
- **argmax-k distribution** (k = 0..19): [48, 23, 28, 24, 26, 22, 10, 17, 16, 14, 12, 10, 9, 7, 5, 5, 8, 8, 7, 1]. k=0 wins on 16% of states, k ≤ 3 on 41%; mean 5.9, median 5.
- **argmax-k is mostly noise.** Splitting the 32 w into two halves of 16 and taking argmax-k on each, the halves agree on 17% of states, against 7% by chance. On average k=0 has the highest max-Q, and both max-Q and mean Q fall steadily with k (k=19 is −10 Q below k=0).
- **Q gap (best k − k=0, per state):** median 1.02, mean 2.38, p90 5.0 Q units (Q ≈ −170). Relative to the Q spread across w at k=0: median gap / std = 0.77, gap / range = 0.18. **Cross-validated gap** (choose k on one half, score on the other): median 0.16, mean 1.30, positive on only 53% of states. Most of the raw gap is the selection bias of a max over 20 k.
- **Phase dependence** (phases from the obs, below): the permutation test of argmax-k bin (0–3 / 4–9 / 10–19) vs phase gives p = 0.013. vs time decile, p = 0.91. The effect comes from one phase, **loose** (payload moved but not held, i.e. dropped or regrasping; n = 39): median argmax-k 9, 49% at k ≥ 10, cross-validated gap 2.9. The other phases (lid_on 46, pre_grasp 103, arm0_holds 63, handover 8, arm1_holds 37, payload_in_bin 4) have median argmax-k 3–6 and cross-validated gap ≤ 0.35.
- Phase labels: lid_on (lid not yet moved) → pre_grasp (lid off, payload untouched; also covers the trash sub-task) → arm0_holds → handover → arm1_holds → payload_in_bin, plus loose. Their mean τ is 0.10 / 0.38 / 0.58 / 0.77 / 0.89 / 0.97, the expected order (loose 0.68). "Holds" means the gripper is closed (finger gap < 0.06) and within 0.10 m (arm0) or 0.06 m (arm1) of the payload center. Layout check: gripper-to-payload equals payload − eef to 2e-7.
- Files: `headroom_states.npz` (Q[300, 20, 32], distances, obs, x_T), `headroom_summary.json`, `headroom_phases.npy`, `distance_check.json`, `analyze.log`.

**Env rollouts**, 200 episodes per arm per run, N = 32 per k. (i) and (ii) both decode 640 candidates per chunk. Run P (job 59586146, `results/exp13/headroom_paired/`) is paired: all arms start from identical initial states, checked to max diff 0.0. Run U (job 59582989) is unpaired, because of the eval-seeding bug. Distances are over pre-success chunks.

| arm | success U | success P | pooled (400 ep) ± SE | mean success ep len (U / P) | wall s per arm (U / P) | L2 \|a − a_base\| (P) | mean \|a − a_base\| per dim (P) | mean k chosen |
|---|---|---|---|---|---|---|---|---|
| (i) critic-chosen (k, w), 20 × 32 | 0.875 ± 0.023 | 0.855 ± 0.025 | **0.865 ± 0.017** | 295 / 295 | 434 / 621 | 1.88 | 0.091 | ~6.5; k-hist falls steadily from k=0 (937) to k=19 (147) |
| (ii) best-of-640 at k=0 | 0.84 ± 0.026 | 0.805 ± 0.028 | **0.823 ± 0.019** | 285 / 286 | 438 / 623 | 2.05 | 0.100 | 0 |
| (iii) stochastic policy (N=1, k uniform) | 0.84 ± 0.026 | 0.81 ± 0.028 | **0.825 ± 0.019** | 345 / 345 | 334 / 510 | 1.40 | 0.071 | 9.5 |
| (iv) deterministic (k still uniform under the frozen π(k), w = tanh(mean)) | 0.81 ± 0.028 | 0.83 ± 0.027 | **0.82 ± 0.019** | 334 / 334 | 339 / 509 | 1.47 | 0.075 | 9.5 |

- Paired exact McNemar (run P, episodes where only one arm succeeds): (i) vs (ii) 24 vs 14, p = 0.14. (i) vs (iii) 30 vs 21, p = 0.26. (ii) vs (iii) 24 vs 25, p = 1.0. (iii) vs (iv) 22 vs 26, p = 0.67 (`headroom_paired/paired_tests.json`).
- Both selection arms finish successful episodes ~50 steps faster than the policy (≈290 vs 345).
- Wall time: the 640-candidate arms cost +22–30% per arm over N=1. Env stepping dominates. The two runs ran on different nodes, hence the different absolute times.
- **Q^A overestimation check (step 4).** Chosen actions sit further from a_base than the policy's own: L2 1.88 for (i) and 2.05 for (ii), vs 1.40 for (iii). This is mostly a k effect, not the critic seeking far actions. At fixed k, the Q-argmax w is only ~5% further than an average candidate (k=0: 2.35 vs 2.23), and the within-(s, k) Spearman correlation between Q and distance is 0.08. Across k, Q^A prefers early k, and those decode further from base. Arm (ii) gains +4.1 Q over its candidate mean but no success over the policy, so at k=0 the critic's ranking among the policy's own samples is not useful.

### Phase 0B seed replicate: 3 seeds (uniformk_s1/s2/s3, each `final.pt` at 6M; s2/s3 from job 59591203, `results/exp13/headroom_paired_s{2,3}/`, pooled in `results/exp13/pooled_seeds.json`)
All rollouts here are paired: every arm starts from identical initial states (checked, max diff 0.0). s1 uses the paired run 59586146. 200 episodes per arm per seed.

| arm | s1 | s2 | s3 | **pooled (600 ep)** ± binom SE | seed SD |
|---|---|---|---|---|---|
| (i) critic-chosen (k, w), 20 × 32 | 0.855 | 0.915 | 0.865 | **0.878 ± 0.013** | 0.032 |
| (ii) best-of-640 at k=0 | 0.805 | 0.945 | 0.87 | **0.873 ± 0.014** | 0.070 |
| (iii) stochastic policy | 0.81 | 0.87 | 0.795 | **0.825 ± 0.016** | 0.040 |
| (iv) deterministic (k still uniform) | 0.83 | 0.88 | 0.775 | **0.828 ± 0.015** | 0.053 |

Paired exact McNemar (discordant episodes: x-only vs y-only, then p):

| comparison | s1 | s2 | s3 | **pooled** (summed discordant) | mean diff |
|---|---|---|---|---|---|
| (i) vs (ii) | 24 / 14, p 0.14 | 7 / 13, p 0.26 | 16 / 17, p 1.0 | **47 / 44, p 0.83** | +0.005 (+0.05, −0.03, −0.005) |
| (i) vs (iii) | 30 / 21, p 0.26 | 20 / 11, p 0.15 | 27 / 13, p 0.038 | **77 / 45, p 0.005** | +0.053 |
| (ii) vs (iii) | 24 / 25, p 1.0 | 22 / 7, p 0.008 | 29 / 14, p 0.032 | **75 / 46, p 0.011** | +0.048 |
| (iii) vs (iv) | 22 / 26, p 0.67 | 16 / 18, p 0.86 | 25 / 21, p 0.66 | **63 / 65, p 0.93** | −0.003 |

Per-state headroom by seed (300 states each, N = 32 w per k):

| seed | frac argmax k=0 / k≤3 | mean argmax-k | split-half agreement (chance) | gap median (raw) | gap median (cross-validated), frac > 0 | gap / std₀ median | argmax-k vs phase p / vs time p | loose phase: n, median argmax-k, CV gap |
|---|---|---|---|---|---|---|---|---|
| s1 | 0.16 / 0.41 | 5.9 | 0.17 (0.07) | 1.02 | 0.16, 53% | 0.77 | 0.013 / 0.91 | 39, 9, 2.9 |
| s2 | 0.12 / 0.41 | 6.0 | 0.16 (0.07) | 0.95 | 0.17, 59% | 0.74 | 0.41 / 0.16 | 35, 7, 1.2 |
| s3 | 0.14 / 0.45 | 5.9 | 0.22 (0.07) | 1.37 | 0.24, 56% | 0.79 | 0.17 / 0.27 | 41, 4, 1.1 |

- Pooled argmax-k histogram (900 states, k = 0..19): [127, 81, 91, 83, 78, 57, 47, 48, 37, 33, 27, 29, 25, 27, 18, 21, 20, 22, 23, 6].
- Wall time per arm (s2 / s3): (iii) 357 / 358 s, (iv) 368 / 357 s, (i) 479 / 498 s, (ii) 445 / 502 s.
- Pre-success distance to a_base, as L2 (per-dim mean): (iii) 1.45 (0.073) / 1.42 (0.072), (i) 1.98 (0.095) / 1.86 (0.089), (ii) 2.21 (0.104) / 2.11 (0.100). Same ordering as s1.
- Successful episodes are shorter under (i)/(ii) in every seed (283–296 vs 336–351 steps for (iii)).

## Analysis
### Phase 0
- **Equal control strength is cheap at K=20.** c_k = 1.06 / 1.12 / 1.26 for k = 3 / 5 / 8 (mean-std match), and the noise stays well inside N(0,1). The catch is that late k loses gripper authority much faster than arm-pose authority. A scalar w_scale cannot equalize both, so the Phase 2 "equal strength" runs at these c_k still give late k about half the k=0 gripper authority. If the question is whether k matters beyond authority, a gripper-matched variant (c ≈ 1.2 / 1.35 / 1.75) is the other end of the bracket.
- **Learned k has no detectable headroom (3 seeds).** In rollouts, critic-chosen (k, w) and equal-compute best-of-640 at k=0 tie: 0.878 vs 0.873 pooled, paired p = 0.83. The s1 margin (+0.05) reversed in s2 (−0.03) and vanished in s3. **Best-of-N under Q^A does help**, by about +0.05 over the plain policy for both arms (pooled p = 0.005 and 0.011), and it shortens successful episodes by ~50 steps. So the critic's ranking is useful, but adding k to the search adds nothing over more samples at k=0. The s1-only reading that "best-of-N at k=0 adds nothing" was seed noise.
- Per-state, the picture repeats in every seed. Best-k is spread out (k=0 best on 12–16% of states), but split-half agreement is only 0.16–0.22 against 0.07 by chance. The cross-validated Q gain from choosing k is a median 0.16–0.24 Q (positive on 53–59% of states), against a raw gap of ~1 Q and a Q std over w of ~1.5. Phase dependence (p = 0.013) appears only in s1; s2 and s3 give p = 0.41 and 0.17, and time dependence is null in all seeds. The "loose" (dropped / regrasp) phase keeps a positive cross-validated gap in every seed (2.9 / 1.2 / 1.1), but its preferred k is not consistent (median 9 / 7 / 4). The only recurring signal is "recovery states have more spread to exploit", not "recovery states want a particular k".
- Read: do not invest in adaptive-k selection on transport on headroom grounds. If eval-time compute is acceptable, plain best-of-N under Q^A at k=0 gets the same +0.05.
- Caveats: one 6M checkpoint per seed; Q^A comes from a policy whose data are uniform over k; the deterministic arm still draws k uniformly (frozen π(k)), so it is not a "mode" policy.

## Next Steps
- (done 2026-11-10, job 59591203) Seed replicate on uniformk_s2/s3: the (i)-vs-(ii) margin did not replicate (pooled p = 0.83). Both best-of-N arms beat the policy by ~+0.05.
