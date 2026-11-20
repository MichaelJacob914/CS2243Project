# Experiment: Noise-bound sweep on can (does a wider w bound give late steps back their control?)
Status: Completed (figures and report update pending)

## Goal
Test whether scaling the steering noise by c (sampler decodes c·w, actor still outputs w in [-1, 1]) lets late steering steps (k=4, 6, 7) catch up with early ones on can, and whether a wider bound helps or hurts plain DSRL (k=0).

## Setup
- **Implementation (new config key `steer.w_scale`, default 1.0).** `SteeredBasePolicy.__call__` multiplies w by `w_scale` just before `steered_ddim_sample` (skipped at 1.0, so the default path is bit-identical). Every steered decode goes through that call: acting (policy and warm-up uniform w), deterministic and stochastic eval (`predict_diffused`), the target backup, noise-critic distillation (w ~ N(0, I) → sampler gets c·w, Q^W is fed the unscaled w), best-of-N (`adsrl/best_of_n.py`), and render (`scripts/render_rollouts.build_steered` reads `steer.w_scale` from run_info.json, default 1 for old runs). `base_action` (unsteered x_T, used for the 1501-step warm start and residual SAC) is unaffected.
  - The actor, noise critic, entropy target and trust region (if enabled) all stay in the unscaled [-1, 1] space. Eval dumps (`eval_rollouts/*.npz` key `w`) store the unscaled w; the sampler saw c·w.
  - Fixed-k and adaptive both support it. Residual raises ValueError if w_scale ≠ 1 (`train_adsrl.steer_w_scale`).
  - Recorded in run_info.json as `run.w_scale` and `cfg.steer.w_scale`.
  - Files: `adsrl/base_policy.py`, `train_adsrl.py`, `scripts/render_rollouts.py`, `cfg/adsrl/{can,square,transport}.yaml`, tests in `tests/test_w_scale.py`.
- **Tests** (`tests/test_w_scale.py`, 18 tests): w_scale=1 bit-identical to the current sampler (real random DPPO model); c·w reaches the sampler on acting (policy + warm-up), det/stoch eval, target backup + distillation (fixed, fixed + trust region, adaptive), distillation critic input stays unscaled, best-of-N; checkpoint round-trip at c=2.5 (fixed + adaptive); config plumbing (residual rejection, yaml defaults, render/best-of-N rebuild from run_info.json). Full suite on a test node (srun job 57759209): 76 passed, 1 failed = `test_eval_seeding.py::test_eval_env_seed_controls_initial_states`, the known eval-seeding bug documented in exp_2 (expected to fail, unrelated).
- **Runs (12):** fixed k ∈ {0, 4, 6, 7} × w_scale ∈ {1.5, 2, 3}, can, seed 1, 1M env steps. Everything else identical to exp_2 Wave 0 (`--config-name=can seed=1 env.n_eval_envs=10 wandb.mode=offline resume=true`, `xT_inputs=auto`, eval every 3000 vec-steps, det + stoch eval, 200 eps). c=1 baselines are the Wave 0 runs `logs/adsrl/can/fixed_k{0,4,6,7}_s1`.
- **Packs** (CUDA MPS, 6 runs per H100, kempner_h100, kempner_pehlevan_lab, -c 24 --mem 96G, exclude holygpu8a12204, 16 h walltime):
  - Pack A: fixed_k0_c{1.5,2,3}_s1, fixed_k4_c{1.5,2,3}_s1
  - Pack B: fixed_k6_c{1.5,2,3}_s1, fixed_k7_c{1.5,2,3}_s1
- Script: `scripts/slurm/exp4_bound_sweep_can.sbatch <A|B>` (waits only on training PIDs). Resubmitting a pack resumes from checkpoint/latest.pt.
- Logdirs: `logs/adsrl/can/fixed_k<k>_c<c>_s1` (scratch via the symlink). stdout: `logs/exp4/bound_<pack>_<jobid>/<name>.log`; slurm logs `logs/slurm/e4_bound_<jobid>.*`.
- Commit: a6bd481 (+ uncommitted working tree).
- GPU budget: 9 kempner GPUs in use by Wave 1 at submission, +2 here = 11 (< 16 cap).

### Hypothesis and expected result
No-training authority on can (`results/authority_wscale/authority_can_K8_dataset_clip1_c*.csv`, mean per-dim action std over w ~ clip(N(0,I), ±1)·c; step 0 at c=1 is 0.084):

| k | c=1 | c=1.5 | c=2 | c=3 |
|---|---|---|---|---|
| 0 | 0.084 | 0.141 | 0.232 | 0.463 |
| 4 | 0.045 | 0.075 | 0.123 | 0.287 |
| 6 | 0.034 | 0.055 | 0.081 | 0.154 |
| 7 | 0.025 | 0.039 | 0.053 | 0.083 |

Hypothesis: the late-k deficit at c=1 (final det success k0 .93, k4 .90, k6 .84, k7 .77) is caused by too little action authority, so raising c until a late step's spread matches step 0 at c=1 should close the gap.
Expected: k4 reaches k0-level success at c=1.5–2, k6 at c=2–3, k7 only at c=3 (spread 0.083 ≈ 0.084). If authority is the whole story, k7 at c=3 should reach ≈ .90+. For k=0, c>1 pushes x_T outside the range the base was trained on (|x_T| up to 3 vs N(0,1)), so success may fall, most at c=3. That would show that more authority is not free. If late k stays below k0 even at matched spread, the deficit is about *which* directions the late step can move (low-noise steps only edit details), not how far.

## Progress Log
- [2026-11-04 23:40] Both packs COMPLETED (57759602 10:52 elapsed, 57759605 10:46, exit 0). Final det success (mean of last 3 evals, 200 eps each, 1 seed) by k at c=1.5/2/3 vs c=1 from Wave 0: k0 .927/.932/.000 (c=1 .932); k4 .902/.965/.933 (c=1 .902); k6 .858/.868/.923 (c=1 .840); k7 .765/.813/.852 (c=1 .773). k0 c3 stayed at ~0 for the entire run.
- [2026-11-04 11:45] All 12 runs started cleanly (w_scale printed and in run_info.json, no tracebacks) and logged the step-0 stochastic eval (untrained actor, 200 eps). Success by k/c: k0 c1.5 .565, c2 .28, c3 .005; k4 .575/.455/.155; k6 .445/.525/.385; k7 .465/.435/.33. Wave 0 c=1 step-0 stoch is not in this table; base is ~0.42. At c=3, k0 collapses to 0.005 before any training (mean |a - a_base| 2.45). This matches the off-manifold concern: x_T scaled up to |3| is far outside what the base was trained on. Late steps lose less (k7 c3 .33). Det evals and throughput are not in yet. Projection uses the Wave 0 pack-A rate (6 fixed-k runs per GPU, 24.3–24.5 env steps/s with eval time included): ~11.4 h per run, so both packs should finish around 23:00–23:30 Nov 4, well inside 16 h. Confirm against the 72k→120k evals.
- [2026-11-04 11:40] Submitted pack A = 57759602 (holygpu8a13301), pack B = 57759605 (holygpu8a15303); both started immediately.
- [2026-11-04 11:35] Implemented `steer.w_scale`, tests pass (srun 57759209: 76 passed, 1 known failure). Created this doc; submitting packs A and B.

## Results
(pending)

## Analysis
(pending)

## Next Steps
- If late k catches up at matched spread: repeat on square/transport and try adaptive with per-k scaling (c_k chosen to equalize spread).
- If k0 degrades at c>1, report the success vs c curve per k as the authority/off-manifold trade-off.
