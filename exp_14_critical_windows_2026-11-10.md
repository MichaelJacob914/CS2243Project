# Experiment: Critical windows in the transport diffusion policy
Status: Completed (all stages; final analysis 2026-11-12; fix and numbers checked by a Fable review)

## Goal
Test whether the frozen transport diffusion policy decides parts of its action chunk inside narrow ranges of noise level ("windows"), whether those windows move across task phases, and whether the step at which the outcome is decided predicts where RL steering helped. The result decides whether choosing a steering step per chunk is worth pursuing. Plan: Claude Docs "Critical windows in a diffusion policy: experiment plan for robomimic transport" (read-only).

## Setup

### Fixed facts (checked in code)
- Policy: DPPO diffusion MLP (3×2048, residual), `cfg/adsrl/transport.yaml`; checkpoint `dsrl/dppo/log/robomimic-pretrain/transport/..._ta8_td100_ddim-100steps_size2048/.../state_3000.pt`. Low-dim obs, obs_dim 59. Action chunk 8 × 14, normalized to [-1, 1] by `dsrl/dppo/log/robomimic/transport/normalization.npz`.
- Action layout (to verify in sim): dims 0–6 robot0 (OSC delta pos 0:3, delta rot 3:6, gripper 6), dims 7–13 robot1 (same; gripper 13). Gripper dims have action_min = −1, action_max = 1, so their normalized and raw values coincide.
- Trained with 100 DDPM steps, cosine schedule. Steering uses DDIM K = 20 over that schedule: step k takes input at DDPM index t = 95 − 5k with ᾱ from 0.0039 (k=0, log-SNR −5.55) to 0.99937 (k=19, log-SNR +7.37). The true top of the chain is t = 99 (ᾱ = 2.4e-7).
- Sampler clipping in the config: denoised_clip_value 1.0 (x̂0 clamp, also re-derives ε̂ under DDIM), randn_clip_value 3 (DDPM noise clamp), DDPM std floor 1e-3. All three are off in the stochastic sampler used here. The deterministic sampler keeps the evaluation settings (DDIM, x̂0 clamp at 1.0).
- Critic for the value feature: Q^A(s, a) = min of the twin critics of the uniform-random-k seed-1 agent, `logs/adsrl/transport/uniformk_s1/checkpoint/final.pt` (env_steps 6,000,000; md5 460527cc...; copied to `/n/netscratch/olveczky_lab/Lab/charleszhang/exp14/uniformk_s1/`). It takes the final action. Caveat: it values the trained agent's continuation (0.79 success at 6M), not the base policy's.
- Max episode length 800 env steps (100 chunks). Initial-state seeding: each worker's global np RNG is seeded before reset (the exp13 workaround).

### Code
- New code in `scripts/exp14/`. adsrl/ and train_adsrl.py are imported read-only from a frozen snapshot at `/n/netscratch/olveczky_lab/Lab/charleszhang/exp14/code_snapshot` (copy of the exp13 Phase 0 snapshot, 2026-11-10 17:17; `adsrl/sampler.py` identical to the working tree).
- Extra packages (scikit-learn 1.5.2, diptest 0.8.1) are installed to `/n/netscratch/olveczky_lab/Lab/charleszhang/exp14/pylib`, not into .venv.
- Repo commit a6bd481 (all project code is untracked; the snapshot is the record).
- Data: robomimic transport multi-human low-dim `low_dim_v141.hdf5` (300 demos, md5 bf868a1d...), `/n/netscratch/olveczky_lab/Lab/charleszhang/data/robomimic/transport_mh/`.

### Estimators (fixed before results)
- **Stochastic sampler** = DDPM ancestral sampling with the 100-step training schedule, posterior variance β̃_t, no x̂0 clamp, no noise clamp, no std floor; the last step (t = 0) returns the mean.
- **Resolved fraction R_f(k)** (Stage 2a). At each state, P clean actions a_i from the stochastic sampler (P = 128 on transport, Design change 4; P = 64 in the toy gate runs, Design change 1), noised once to level k with one unit noise ε_i that is reused at every level, then 16 stochastic regenerations (M = 16) from DDPM index t = 95 − 5k down to 0. With per-point means f̄_i and within-point variance W = mean_i Var_j(f_ij) and S_b = Var_i(f̄_i) (both ddof 1), R = (S_b − W/16) / (S_b − W/16 + W), the one-way ANOVA intraclass correlation. Vector features use traces. A check level at t = 99 must give R ≈ 0 and samples indistinguishable from fresh ones.
- **Open fraction O_f(k)** (Stage 2b, deterministic DDIM-20 as evaluated). 8 base actions from DDIM with fresh x_T; each noised to level k with 32 unit noises reused at every level, x = √ᾱ_k a + √(1−ᾱ_k) ε; then DDIM steps k..19. O = mean_b Var_c(f) / Var(f), with Var(f) over 256 DDIM samples from fresh x_T at that state. The existing steering form (as in `scripts/authority.py`: keep x_T of base b, inject fresh w at step k via `steered_ddim_sample`) is run on the same 8 × 32 design.
- **Gaussian null.** N(μ, Σ) fitted to the 256 Stage 1 actions of the state. The same P × 16 procedure (128 × 16 on transport, Design change 4) is run through the identical sampler code with the null's closed-form ε-predictor (Design change 2); the exact posterior, for reference: in Σ's eigenbasis each coordinate has posterior mean μ_i + √ᾱ λ_i (y_i − √ᾱ μ_i)/(ᾱ λ_i + 1 − ᾱ) and variance λ_i(1 − ᾱ)/(ᾱ λ_i + 1 − ᾱ). The same feature functions are applied (the cluster label uses the policy's GMM; the value uses the same critic).
- **Midpoint, width.** R is made monotone (non-decreasing) in log-SNR over the 20 levels by averaging its upper envelope (running maximum from the noisy end) and its lower envelope (running minimum from the clean end); this is not isotonic regression, but it is vectorized over bootstrap curves and coincides with it for a monotone curve; crossings at 0.1, 0.5, 0.9 are linearly interpolated in log-SNR (and, for rule 2, in step index k). If R(k=0) ≥ 0.1 or R(k=19) < 0.9 after smoothing, the width is censored and that state is left out of the width test for that feature (counts reported).
- **Bootstrap.** 1000 resamples of the P noisy points (128 on transport) (the same indices at every level), for the policy and the null separately; 95% percentile intervals.
- **Two-rise fit.** Least-squares fits of one logistic in log-SNR (2 parameters) and a weighted sum of two logistics (5 parameters) to the raw 20-level R curve; BIC = n ln(RSS/n) + p ln n. "Two-rise preferred" = ΔBIC ≥ 10, both weights ≥ 0.2, and the two midpoints ≥ 2 log-SNR units apart.
- **False-positive calibration.** Every state-level test is also run null-vs-null (a second, independent null replicate in place of the policy). Its rate is reported next to the policy's.
- **Stage 1 clustering.** Project the 256 actions onto the principal directions holding 90% of variance; fit GMMs with 1, 2, 3 components (full covariance, 5 inits); dip test on PC1. "Two or more clusters" = min(BIC2, BIC3) < BIC1 − 10 **and** dip p < 0.05. The same tests on 256 samples from the fitted Gaussian give the false-positive rate. The cluster label feature = argmax component of the preferred GMM (2 or 3), applied in the same PCA space.

### Design change (2026-11-10 22:45, before any transport data)
Stage 2a uses **P = 64 noisy points × M = 16 regenerations**, for both the policy and the Gaussian null (plan and first preregistration: 16 × 16). [Superseded for transport by Design change 4: P = 128.] Stage 4 scales the same way (32 instead of 8 points), and so does the Stage 3 short-horizon effect on the 2a regenerations. Reason: the 16-point design failed the toy power check. Criterion (a) passed in only 2 of 5 measurement seeds, and the two-rise test fired falsely in 2 of 5. The 64-point design passed (a) in 4 of 4 with no two-rise false positives (Progress Log, 22:30). No transport data had been collected when this change was made. The decision rules and the other estimator settings are unchanged. Stage 0 is rerun once at P = 64 on a fresh measurement seed (6) as the gate. The gate = the plan's three criteria (a), (b), (c); the top-of-chain check is reported as a sanity check but does not gate.

### Design change 2 (2026-11-10 23:00, after a code review, before any transport data)
A code review found that the null and the policy did not run the same procedure, which biased the width test toward finding windows. Changes:
1. **Null through the identical sampler.**
   - The problem:
     - The policy's stochastic regeneration (`ddpm_regenerate`) runs only the t = 0 mean step from the last level (k = 19), so its within-point variance is 0 and R(k=19) ≡ 1 for every policy feature. The exact-posterior null gave 0.64–0.9998 there.
     - At k ≤ 18 the discretized ancestral chain is under-dispersed against the exact posterior (≈ 0.59× the variance at k = 18 and 0.69–0.76× at k ≤ 15 for λ ≈ 0.005–0.01).
     - So for low-variance features, the transport regime, the policy looked narrower and earlier than the null with no window at all. The toy's past-obstacle PC2 showed it: policy R at k = 17–19 was 0.09 / 0.24 / 1.0 against the null's 0.05 / 0.11 / 0.64.
   - The fix (reviewer's option a; it matches the plan's "identical procedure"):
     - The Stage 2a null now runs through the same `ddpm_regenerate`, with the null's closed-form ε-predictor ε(x,t) = √(1−ᾱ)(ᾱΣ + (1−ᾱ)I)⁻¹(x − √ᾱμ), evaluated in Σ's eigenbasis (`windows.GaussEpsModel`).
     - The Stage 2b null uses the same DDIM-20 `ddim_step` path with the same clamps, not the continuous-time ODE map.
     - The null's clean points stay exact Gaussian draws.
   - Test written first (`scripts/exp14/test_null_path.py`). It plugs the closed-form ε into both arms and requires the two R / open-fraction curves to agree within 4.5 combined bootstrap SEs, at λ = 0.003–2.
     - Pre-fix code (srun 59641510): fails, max |ΔR|/SE = 12.7 at k = 19 and max |ΔO|/SE = 11.0.
     - Fixed code (srun 59641183 and the rerun after it): passes, max 2.25 (2a) and 2.28 (2b).
     - Logs: `results/exp14/test_null_path_{PRE,POST}_FIX.log`.
2. **Censored bootstraps.** `width_test` no longer drops censored bootstrap replicates silently, and no longer accepts as few as 100 of 1000. Each arm needs ≥ 95% complete replicates; otherwise that state × feature is marked censored and excluded, the same rule as for point estimates and the same for both arms. For criterion (c), a censored state does not count as "matching".
3. **Coarse late levels (no change, as the user chose).** The 20 DDIM levels are kept. The k = 18 → 19 gap is 2.87 log-SNR units, so widths and midpoints of late-resolved features are coarse. This is now symmetric between the arms (both have R(k=19) = 1 by construction).
4. **Separate draws.** The policy's clean points in Stage 2a are drawn separately from the 256 Stage 1 samples used for the labeler, the PCs and the null fit. This is done in the toy now and is required of the transport driver (not yet written).
5. **Known limitations, noted and not fixed:**
   - `two_rise_test` can absorb a null plateau below 1 (review item 4).
   - The Stage 2b null open fraction is not used by any decision rule.
   - Stage 1 GMM-by-BIC has limited power in 112-D (PCA to 90% variance, 256 samples).
- **Gate.** Stage 0 is rerun once on a fresh measurement seed (7) with the plan's three criteria plus the null-vs-null check: no null-vs-null "narrower" on the gate features (start-state label, past-obstacle PC1). The earlier gate run (seed 6, 22:29) used the biased null and is superseded.

### Design change 3 (2026-11-10 23:15, user decision after review, before any transport data)
1. **Rule 1 = width test only.** The user dropped the two-rise branch: a window exists at a state when a primary feature's policy width interval lies entirely below its null's. The two-rise fit stays as a reported descriptive, with its null-vs-null rate (a second independent null replicate run through the same fit). Reason: in the toy gate (seed 7), two-rise fired falsely on a late-resolved unimodal feature. Its fit rests on the coarse k = 18 → 19 jump to R = 1, so it would create spurious windows on transport, where low-variance features resolve late. Toy criterion (c) is likewise intervals overlap, without the two-rise clause.
2. **Strict procedural identity of the null (reviewer's optional fix).**
   - The null's Stage 2a clean points now come from the same stochastic chain (`ddpm_sample` with the GaussEpsModel), and its 2b base actions from the same DDIM sampler from fresh x_T. Before, they were exact N(μ, Σ) draws.
   - Reason: at small λ the discretized chain under-disperses its own marginal, so exact draws are not the identical procedure.
   - `test_null_path.py` now draws both arms' clean / base points this way.
- Operationalization item 1 above ("or two-rise is preferred") is superseded by this entry. The summary-table mapping is unchanged.

### Design change 4 (2026-11-11, user decision, before any transport data)
1. **Gate accepted** on the 9-seed evidence: (a) passed 8/9 (seed 8 failed by an interval overlap of 0.47); (b), (c) and null-vs-null passed 9/9.
2. **Transport Stage 2a uses P = 128 noisy points × M = 16 regenerations**, the null matched (128 × 16, same procedure). This doubles Stage 2a over the toy-validated 64 to raise power on transport (estimate ≈ 2× the Stage 2a GPU cost, ≈ 700M sample-calls).
3. **Stage 3 stays matched to Stage 2a** (all 128 × 16 regenerations per level) only if its timed CPU cost stays under ≈ 100 core-h. Otherwise it is subsampled (a fixed random subset of points, the same for the policy and the null), and the doc says so.
- Stage 4 is unchanged by this entry (20 states × 5 levels × 32 points × 8 regenerations).

### Design change 5 (2026-11-11, after review of stage12.py, before any transport data)
1. **Clipping for value and execution only.** The policy's stochastic samples leave [−1, 1]: 10.5% of entries in the smoke run, max 1.77. The env executes the clipped chunk, and the critic was trained on in-range actions. So, for both arms:
   - The value feature Q^A is computed on np.clip(a, −1, 1).
   - Every simulator execution (Stage 1 executions, Stages 3–4, including the base-policy continuation) uses the clipped chunk.
   - All other features (gripper sign, PCs, arm split, slow/fast, cluster label) stay on the unclipped normalized actions.
2. **Gripper primaries and the null.** The fitted Gaussian almost never flips a gripper's sign, so the null's sign-averaged gripper feature is often constant. Its curve is then undefined and the width test is censored.
   - The sign-averaged gripper features stay primary as specified.
   - The analysis reports "null degenerate" (null feature constant) separately from "policy censored" (policy bootstrap < 95% complete).
   - New exploratory features (FDR like the others): the continuous chunk-mean gripper command, dims 6 and 13 (`grip0_mean`, `grip1_mean`).
   - Consequence: rule 1 may rest mainly on the cluster label and the value.
3. **Audit trail.** Each state's pickle stores what is needed to recompute features: the top-5 PCs, the PCA mean, and the labeler (the GMM object, its weights / means / covariances, and the labeler's mu and V).
4. Stale P = 64 mentions in Estimators and Stage 3 now point to Design change 4 (P = 128).
5. `bootstrap_resolved_gram` centers f̄ per level before forming G and uses ((C G) ∘ C) summed over points. Its output is mathematically unchanged, and the equality test still passes.

### Design change 6 (2026-11-11, after timing, before any Stage 3 data)
Stage 3 fully matched to Stage 2a would be ≈ 9.2M chunk executions ≈ 740 core-h (0.29 s per restore + chunk). That breaks the user's ≈ 100 core-h cap, so **Stage 3 is subsampled**:
- 100 states (unchanged).
- 2a policy and null: the first 16 of the 128 points (i.i.d., so a random subset, the same rule for both arms) × all 16 regenerations, i.e. 256 chunks per level per arm.
- 2b plan form: all 8 bases × the first 8 of 32 copies.
- 2b steering form: not executed.
- Total ≈ 100 × 20 × (2 × 256 + 64) = 1.15M chunks ≈ 93 core-h.
- Consequence: Stage 3 width tests have the power of the 16-point design, which the toy showed is low, so Stage 3 is descriptive / exploratory only (as preregistered, with FDR). No null-vs-null replicate is executed for the sim features.
- Stage 4 is unchanged (25,600 episodes). Timed estimate ≈ 140 core-h (≈ 20 s per episode), above the earlier 40–80 core-h estimate.

### Design change 7 (2026-11-11, after the simulator-executor review, before any Stage 3 / 4 data; Stage 1 executions are rerun)
- **B1, stale OSC controller cache (bug, fixed).** After env construction or reset, `controller.new_update` is False, so the first step after a restore used the reset pose's eef pose / Jacobian as the base of the delta goal. The first chunk per worker was off by 26–63 mm (eef) and up to 46 mm (objects), corrupting 64 Stage 1 chunks. `TransportSim.restore` now forces `controller.update(force=True)` for both arms.
- **C1, OSC nullspace target (fixed).** `controller.initial_joint` is hidden per-reset state with random init noise, and it differed per worker (up to 59–91 mm difference after 10 chunks). `restore` now sets it to the episode's own t = 0 arm qpos: rollouts use the episode's chunk-0 sim state, demos `states[0]`. It is stored as `init_joint` in `states.npz`, with a backup of the previous file kept.
- **Tests first** (`scripts/exp14/test_restore.py`).
  - Pre-fix code fails all three (`results/exp14/test_restore_PRE_FIX.log`):
    - fresh-env vs repeat first chunk differs by 27–48 mm (after a reset, 2.6–16 mm);
    - `initial_joint` is not set;
    - a state with t = 824 is accepted by `run_episode`.
  - Post-fix: `results/exp14/test_restore_POST_FIX.log`.
- **B2, Stage 4 budget (fixed).** Budget = 800 − t is negative or tiny for late demo states (6 states have t ≥ 800). Stage 4 states are now chosen from rollout states only (all with t < 800; additionally 800 − t ≥ 80 is required), and `run_episode` asserts a budget ≥ 8 env steps.
- **Stage 4 design rule (fixed now, before Stage 4 data; `scripts/exp14/select_stage4.py`).**
  - States: 4 rollout states per phase with the largest spread of Q^A over their Stage 1 samples, excluding diverged states.
  - Levels: the 5 DDIM steps where those states' median value R is closest to 0.1 / 0.3 / 0.5 / 0.7 / 0.9.
- **C2, reproducibility.** Repeats are deterministic only up to BLAS-level noise, which contacts amplify over long horizons (up to 140 mm after 10 chunks at one state). Stage 4 per-episode success is therefore reproducible only in distribution, which is what the estimator needs.
- **C3, safe defaults.** The Design change 6 subsample is now the default of `sim_exec.py` (`--s3_points 16 --s3_copies 8 --s3_keys a2a,a2a_null,a2b_plan`), so a plain call cannot launch the ≈ 740 core-h version.
- **C4, observable lag.** The one-chunk features use the raw obs returned by the last env step, which lags the state by one substep through robosuite's observable sampling. The lag is the same for both arms (policy and null).
- **Consequences.**
  - Stage 1 simulator executions (job 59742199) are rerun with the fixed restore; the Stage 1 gripper-target summary is the only analysis output that uses them.
  - Rules 1–3 use only GPU-side Stage 2 features and are unaffected; they are re-checked when the analysis reruns.

### Features
| Feature | Definition | Role |
|---|---|---|
| Cluster label | GMM component (clustered states only), one-hot, trace R | primary |
| Gripper commands | per arm: mean over the 8 steps of sign(raw gripper command) (dims 6, 13) | primary (2 features) |
| Outcome: value | Q^A(s, a) of the regenerated chunk | primary |
| Outcome: success | full rollout with the base policy (Stage 4 subset) | primary on subset, critic check |
| Principal directions | projection on each of the top 5 PCs of the 256 Stage 1 actions | exploratory |
| Arm split | trace R of each arm's 56 dims (7 × 8) | exploratory |
| Slow / fast | chunk mean (14-d); deviation from it (112-d) | exploratory |
| Short-horizon effect | execute the chunk from the saved sim state: displacement of payload, lid handle, trash (9-d); which object each gripper approached most (one-hot) | exploratory |
Exploratory tests use a bootstrap p-value (share of bootstrap pairs with policy width ≥ null width) and Benjamini–Hochberg at q = 0.05 across states × features.

### Decision rules (copied verbatim from the plan; fixed before results)
> The rules below are fixed before looking at the transport results.
>
> For each state and feature, compute:
> - The midpoint, where the curve crosses 0.5, interpolated in log signal-to-noise.
> - The width, from 0.1 to 0.9.
> - The same two numbers for the Gaussian null. Get them by running the identical procedure on samples from the fitted Gaussian, using its exact posterior.
> - A fit with one rise and a fit with two, compared by BIC, to detect a plateau.
> - Bootstrap intervals from resampling the noisy points.
>
> Decision rules:
> - **A window exists at a state** when a primary feature's width is below its null's width and the 95% intervals do not overlap, or when the two-rise fit is preferred.
> - **Windows differ across phases** when the median midpoints of two phases differ by more than one denoising step, with a permutation test below 0.01.
> - **The measurement is useful for steering** when the open fraction of the outcome ranks the steps in the same order as your per-step RL gains.
>
> The primary features are the cluster label, the gripper commands, and the outcome. Declaring them in advance limits the number of tests. The other features are exploratory, with a false-discovery correction across states and features.

Operationalization (fixed now, before any transport result):
1. Window at a state: for some primary feature, the upper end of the policy width's 95% interval is below the lower end of the null width's interval. [Design change 3: the original "or two-rise is preferred" clause was dropped.] Reported per feature as the share of states, next to the null-vs-null rate. "Windows exist" at the population level if, for some primary feature, the policy share exceeds the null-vs-null share by at least 10 points and a one-sided Fisher exact test gives p < 0.01.
2. Phase differences: midpoints in fractional step index k for the 300 phase states (ambiguous states excluded). Statistic = the largest |difference of phase medians| over the 10 phase pairs. p from 10,000 permutations of the phase labels (max statistic, so family-wise over pairs). Holds for a feature if the statistic > 1.0 step and p < 0.01.
3. Usefulness for steering: steps k ∈ {0, 3, 5, 8, 11, 14, 16, 19}. RL gain g_k = mean deterministic success of `logs/adsrl/transport/fixed_k{k}_s1/eval_summary.csv` over the evals at 5.56M, 5.82M and 6.0M env steps, minus the no-learning success at k (w ~ N(0, I), `results/base_eval_wk/transport.csv`, gauss, 600 episodes). Outcome open fraction = median over all 330 states of the value's open fraction (plan form, 2b) at level k. [As run: over the 319 non-diverged states (11 diverged excluded); the 330-state sensitivity run gives the same ρ.] Holds if Spearman ρ ≥ 0.7 and the exact one-sided permutation p (all 8! orders) < 0.05. Secondary readouts: the steering form of the open fraction, gains at 4.79M and at 6.0M alone, stochastic eval.
- Summary-table row: "move across phases" if rules 1 and 2 hold; "one place" if rule 1 holds and rule 2 fails; "smooth drift" if rule 1 fails.

### Stages
- **Stage 0 (toy control, gates the rest).** 2D point mass, obstacle disc at the origin (radius 0.5), start (−1, 0), goal (1, 0). Demos go above or below (50/50) via waypoints (0, ±0.8). Obs = position; action chunk 8 × 2 velocities; same dppo DiffusionModel class, 100 DDPM steps, cosine schedule, DDIM 20. Same pipeline code as transport. Pass = all three:
  (a) the cluster label's width at the start state is below its null's, with non-overlapping 95% intervals;
  (b) its midpoint is within 1.0 log-SNR unit of −ln λ_max (λ_max = top eigenvalue of the action covariance at that state; Biroli et al. speciation time);
  (c) at a state past the obstacle, the PC1 curve matches its null (width intervals overlap and two-rise not preferred).
  Plus the sampler check: regeneration from t = 99 gives R ≈ 0 and fresh-looking samples.
- **States (330).** 165 rollout + 165 demo. Five phases × 60 (30 rollout + 30 demo; shortfalls filled from the other source and logged): (1) before contact, (2) opening the container / grasping the hammer, (3) clearing trash, (4) handover, (5) placing the hammer. Plus 30 ambiguous states = first step of an episode (15 rollout, 15 demo). Phase labels from the raw low-dim obs (extends the exp13 labels with the trash sub-task). Each state stores its full MuJoCo state. Demo obs are checked against the env after restoring the state.
- **Stage 1.** 256 stochastic samples per state (84,480), clustering tests, and each chunk executed once from the saved state.
- **Stage 2a / 2b.** As above, all 330 states, 20 levels (+ t = 99 check in 2a).
- **Stage 3.** Short-horizon effect: 100 states (20 per phase) × 20 levels, on the 2a regenerations (128 × 16 = 2,048 actions per level per arm since Design change 4, policy and null; ≈ 8.2M chunks if not subsampled) and the 2b copies (8 × 32 = 256 per level, ≈ 0.51M chunks); ≈ 2.6M chunk executions.
- **Stage 4.** Success: 20 states (4 per phase, chosen where the value varies most across the Stage 1 samples) × 5 levels × 32 points × 8 regenerations (2a design; 4× the plan's 8 points, scaled with the Stage 2a change) = 25,600 episodes, continued with the base policy (DDIM-20, fresh x_T each chunk) to 800 total env steps. The 5 levels are chosen after Stage 2a to span the value's rise. Critic check: per-action AUC of Q^A for success, and value-vs-success R curves on the same states.

### Hardware and compute estimate (to be updated after timing)
- GPU (Stages 0, 1 sampling, 2a, 2b, critic): kempner_h100 / kempner_pehlevan_lab, 1 GPU. Network calls: Stage 2a ≈ 330 × 21 × 1,024 × ~50 DDPM calls ≈ 355M sample-calls of the 3×2048 MLP (≈ 18 MFLOP each, ≈ 6.4 PFLOP), plus the null (CPU, closed form) and the critic on every regenerated action; 2b ≈ 17M; Stage 1 ≈ 8.4M. Estimate ≈ 1–2 GPU-h total (to be timed on one batch first).
- CPU (rollouts for states, Stage 1 execution, Stage 3, Stage 4): sapphire / olveczky_lab, packed multi-process jobs. ≈ 2.6M chunk executions (8 env steps each) + 25,600 episodes of up to 800 steps. Untimed; at an assumed 3–5 ms per env step plus CPU policy inference, ≈ 20–30 core-h for Stage 3 and ≈ 40–80 core-h for Stage 4 (≈ 60–110 core-h in all). To be timed on one batch first.

### Expected results
- Stage 0 passes: the toy label curve is narrower than the null (null width for a thresholded Gaussian is ≈ 6 log-SNR units vs ≈ 3–4 for a real two-way choice).
- Transport: action features with small per-state variance (λ ≈ 0.01–0.05) will be decided late (midpoint near −ln λ ≈ 3–4.5 log-SNR, k ≈ 16–18), as the null predicts. A window would show as a narrower-than-null gripper or cluster-label curve. The prior from exp13 (no headroom for learned k; phase effect did not replicate) favors "one place" or "smooth drift".

## Progress Log
- [2026-11-12 14:20] Animated toy figure and explainer video revised after two rounds of blind clarity review by fresh subagents, which saw only stills and captions. Final renders: jobs 60046158, 60047447 and 60048878 (test partition). Outputs: `report/assets/anim/exp14_toy_window.{mp4,webm}` plus a poster PNG, `report/assets/video/critical_windows_explainer.mp4` (5 min 33 s, 720p), and the narration script `scripts/exp14/video/narration.md`. While making the video, I found that the report's finding-3 table of undecided shares (0.96, 0.91, ...) matches the analysis that keeps the 11 diverged states (330 states), whereas the caption next to it says 319 states. The 319-state values differ by at most 0.01. The video uses the 319-state values.
- [2026-11-12 14:10] Animated toy figure and explainer video, first full render. Code in `scripts/exp14/video/`; Python envs on netscratch (`videnv` for manim, matplotlib and ffmpeg; `toyenv` for CPU torch), because the project `.venv` on holylfs06 took over 2 minutes just to import numpy on compute nodes today.
  - Data job 60036820 (test, 8 CPUs, 13 min): toy redraws from the trained toy model through `windows.ddpm_regenerate` (4 chunks × 32 redraws per step, policy and Gaussian control, start and past-obstacle states) and per-state widths and midpoints for the video. An earlier attempt with the project `.venv` (srun 60030736) hung at import and was cancelled.
  - Render jobs 60038018, 60039145, 60040296, 60040861, 60042170, 60043120, 60043910 (test, 2–12 CPUs, 2–8 min each): toy animation plus 10 manim scenes, rendered in parallel. Earlier ones failed on LaTeX tick labels, missing imports and a text-cache race between parallel scenes; all fixed.
- [2026-11-12 01:45] **Fable review: the fix is OK and every number reproduces exactly.** Script and log: `/n/netscratch/olveczky_lab/Lab/charleszhang/exp14/review/verify.{py,log}`.
  - Re-derived from raw files: rule 1 counts and p-values; RL gains and ρ = 0.952 (p = 23/40320); Stage 4 pooled success, AUCs and per-level R; the list of 11 diverged states. No path in `main()` can reach an excluded state; the `--keep_diverged` run includes all 330.
  - Interpretation notes folded into Results / Analysis: the pooled AUC is between-state (centered-Q AUC 0.51); the within-state AUC is 0.53 without two near-saturated states; the top-quarter gain is ≤ 4 points over non-degenerate states.
  - Minor, not fixed: `stage4()` silently skips a state with missing sim files (not triggered: all 20 complete). The test covers stage3 / stage4 but not `memorization()`.
- [2026-11-12 01:10] **Final analysis done.**
  - Jobs: 59901402 (main, 319 states) and 59901419 (`--keep_diverged`, 330 states), both test partition, 13 min.
  - Stage 4 critic check: `scripts/exp14/stage4_critic.py` → `results/exp14/analysis/stage4_critic.json` (srun 59904617).
  - Results and Analysis sections written. Fable review of the fix and numbers requested.
- [2026-11-12 00:25] **Analysis exclusion bug fixed, test first.** Job 59860365 (coordinator-submitted final analysis) failed with `KeyError: 72` in `stage3()`. Cause: `main()` drops the 11 diverged states from `res_by_i`, but `stage3()` and `stage4()` globbed every sim file, and `memorization()` every action file.
  - Test `scripts/exp14/test_analyze_exclusion.py`: a synthetic sim dir with files for an included and an excluded state. Pre-fix: both tests fail with `KeyError: 72` (`results/exp14/test_analyze_exclusion_PRE_FIX.log`). Post-fix: pass (`..._POST_FIX.log`).
  - Fix: the three functions skip states not in `res_by_i` and list them (`stage3_skipped_states` = [72, 75, 122] in the main run; `stage4.skipped_states` = [], since Stage 4 used only non-diverged rollout states). Pre-fix copy: `/n/netscratch/olveczky_lab/Lab/charleszhang/exp14/analyze_prefix_exclusion.py.bak`.
- [2026-11-12 00:00] Stage 4 COMPLETED (coordinator-submitted job 59859920, serial_requeue, 48 CPUs, 3.2 h ≈ 155 core-h): 3,200 tasks = 25,600 episodes.
- [2026-11-12 00:19] Stage 4 59859920 COMPLETED (3.2 h, 48 cores). Analysis 59860365 FAILED: KeyError 72 in analyze.py stage3 (a diverged state, excluded from res_by_i, still has a Stage 3 file). Sent to the exp14 agent: reproduce with a test, fix, rerun analysis.
- [2026-11-11 22:19] Stage 4 59859920 at 1280/3200 episode tasks after 1.8 h (~64 tasks per 110 s); done ~23:15, then analysis 59860365.
- [2026-11-11 20:22] The serial_requeue copy 59859920 (48 cores) started first; 59838559 and 59859823 cancelled. Analysis resubmitted as 59860365 (afterok 59859920). At 48 cores Stage 4 (~140 core-h) should take ~3 h.
- [2026-11-11 20:25] Filesystem recovered at 18:20 (probe: numpy 1 s, torch 3 s); the watcher submitted Stage 4 59838559 (sapphire, 112 cores) and analysis 59838568, then exited. Stage 4 sat 2 h pending on Priority (lab fairshare 0.26), so added copies 59859823 (test, 96 cores) and 59859920 (serial_requeue, 48 cores, requeue-safe since finished tasks are skipped). A background loop keeps the first copy to start, cancels the others, and resubmits the analysis with afterok on it.
- [2026-11-11 18:19] Filesystem improving but not yet past the threshold: probes at 17:33 numpy 13 s / torch 104 s, 17:59 numpy 56 s / torch 226 s. Watcher 59815027 still polling.
- [2026-11-11 16:20] The first probe sat 55 min in the sapphire queue (Priority), so the watcher was restarted with probes on the test partition: watcher 59815027 (log `logs/exp14/e14_iowatch_59815027.out`), first probe 59815050 started at once. Stage 4 itself still goes to sapphire when submitted.
- [2026-11-11 15:05] Lab filesystem still slow on compute nodes (test job 59775557: numpy import 254 s, torch 19 min). Submitted watcher 59801093 (`scripts/exp14/slurm/iowatch.sbatch`, 1 core on test, 12 h limit): every 20 min it submits a fresh 1-core probe job that times the imports; once numpy < 15 s and torch < 90 s it submits Stage 4 (112 cores) and the analysis (afterok). Its log: `logs/exp14/e14_iowatch_59801093.out`.
- [2026-11-11 13:10] Stage 4 job 59759945 cancelled after 80 min with no output: 112 cores idle in I/O wait, about 150 core-h of allocation spent on start-up. No Stage 4 outputs exist, so nothing is lost.
  - Resubmit unchanged when the filesystem recovers: `sbatch --time=06:00:00 --cpus-per-task=112 --mem=300G --job-name=e14_sim4 scripts/exp14/slurm/sim.sbatch stage4`. The executor skips finished tasks.
  - Then run the final analysis: `sbatch scripts/exp14/slurm/analyze.sbatch`.
- [2026-11-11 13:05] Filesystem still degraded. Test-node import timings (srun 59770628): `import numpy` 103 s (normally ≈ 1 s); `import torch` did not finish in the remaining ≈ 12 min. Stage 4 (job 59759945) has produced no output after 75 min; it is left running (6 h limit) and will start once imports go through. The Stage 3 preview analysis (job 59767237) was cancelled; the final analysis runs after Stage 4. Reported to the coordinator.
- [2026-11-11 12:45] **Stage 3 COMPLETED**, job 59751800: 300 tasks (100 states × policy 2a, null 2a, 2b plan), 66 min on 112 sapphire CPUs ≈ 123 core-h (estimate 93). Outputs: `results/exp14/sim/stage3/`.
  - Stage 4 (job 59759945) is stalled in worker start-up. The lab filesystem (`/n/holylfs06`) is degraded for compute nodes: 111 of 112 workers sit in uninterruptible I/O wait reading tiny package-metadata files (e.g. `pandas-2.3.3.dist-info/entry_points.txt`) during import.
  - On a test node, a lone `import numpy; import torch` did not finish in 10 min (srun killed at its limit).
  - Pre-warming the node's page cache with the shared libraries (1.5 GB, 3.5 min) did not help: the stall is on metadata / small-file reads.
  - The job is left to proceed as the filesystem recovers (6 h limit). A preview analysis including Stage 3 (job 59767237) is stalled the same way.
- [2026-11-11 11:50] Stage 4 submitted: job 59759945, sapphire, 112 CPUs, 3,200 tasks of 8 episodes. A stand-alone timing run (srun 59757628) was cancelled after 12 min stuck importing torch from the lab filesystem (slow I/O, process idle), so timing comes from the job's own progress log.
  - Stage 4 actions were generated on CPU (test partition, job 59755464, ≈ 8 min) because gpu_h200 was pending on priority (job 59752667, cancelled).
  - The Stage 1 rerun (job 59751796, 14.4 min) leaves every rule outcome unchanged (check run in `/n/netscratch/olveczky_lab/Lab/charleszhang/exp14/analysis_after_stage1_rerun/`), as expected since the rules use only Stage 2 features.
  - The phase split of gripper targets changed for the place / ambiguous rows, which the earlier analysis had read while the first Stage 1 job was still writing.
- [2026-11-11 11:30] Stage 4 design chosen by the preregistered rule (`results/exp14/stage4_design.json`): rollout states 3, 6, 8, 22 / 60, 74, 76, 85 / 123, 126, 135, 144 / 185, 190, 195, 204 / 256, 258, 261, 264 (4 per phase); DDIM steps k = 4, 9, 12, 15, 18, where the median value R = 0.09 / 0.32 / 0.49 / 0.70 / 0.88. Stage 4 action generation submitted: job 59752667 (gpu_h200, 1 GPU).
- [2026-11-11 11:25] Fixed restore tests pass (`results/exp14/test_restore_POST_FIX.log`): first chunk identical on a fresh env and after a reset (≤ 2.5e-7 mm), 5 chunks across a reset ≤ 6e-8 mm, `initial_joint` restored, late state rejected. `init_joint` added to `states.npz` (t = 0 self-consistency exact). Submitted: Stage 1 executions rerun, job 59751796 (sapphire 64 CPUs; old outputs kept in `results/exp14/sim/stage1_prefix_B1_C1/`); Stage 3 (Design change 6 subsample), job 59751800 (sapphire 112 CPUs, ≈ 93 core-h expected).
- [2026-11-11 11:20] **Preregistered analysis of Stages 1–2 done** (rules 1–3 depend only on Stage 2). Jobs: 59743031 (main, `results/exp14/analysis/`) and 59743054 (sensitivity incl. diverged states, `results/exp14/analysis_keep_diverged/`, same conclusions). Figures: `report/assets/figs/exp14_{width_vs_null,midpoints_by_phase,open_vs_rlgain,example_curves}.png`. **Summary-table row: "Windows exist at one place."**
  - **Diverged states (excluded).** With x̂0 clipping off, the stochastic chain diverges at 11 of 330 states (all rollout states, phases open_grasp / trash): Stage 1 samples reach ~1e30, and the policy or null curves at the noisiest levels are NaN. Their null fit and curves are invalid, so the main analysis uses 319 states; the list is in `analysis.json` → `diverged_states`. The deployed DDIM sampler clamps x̂0, so this does not occur in use.
  - **Rule 1 HOLDS.**
    - Cluster label: 44 of 77 clustered states are narrower than the null, vs 0 of 77 null-vs-null; Fisher p = 9e-18.
    - grip0: 60 / 319 vs 1 / 319 (p = 1e-18). grip1: 71 / 319 vs 0 (p = 5e-24). The null is degenerate (constant) at 223 / 205 states, and the policy censored at 240 / 220, so the gripper tests rest on the ~80–100 states where both arms are defined.
    - Value: 11 / 319 vs 0 (p = 4.5e-4, but the share difference 3.4 points < 10) → fails.
    - Widths: cluster label ≈ 3–4.5 log-SNR vs null ≈ 4–8; value ≈ its null.
  - **Rule 2 FAILS** (max pairwise difference of phase medians, in steps):
    - Cluster label 0.42 (p = 0.66); grip0 3.64 (p = 0.044, place phase n = 3); grip1 0.62 (p = 0.86); value 0.75 (p = 0.087).
    - Median midpoints: cluster label k ≈ 5.5–5.9 in every phase; grippers k ≈ 4.5–5.7; value k ≈ 16–17.
    - Exploratory, not a rule: PC1, arm0, arm1 and slow midpoints do vary by phase (max diff 4–7 steps, permutation p 0.001–0.02).
  - **Rule 3 HOLDS.** Spearman ρ = 0.95, exact one-sided p = 5.7e-4, between the median value open fraction (plan form) and the RL gain over k ∈ {0,3,5,8,11,14,16,19}. Same ρ for the steering form; ρ = 0.95 for gains at 4.79M and stochastic; ρ = 0.81 (p = 0.011) for the single 6.0M eval.
    - Open fraction: 0.955 / 0.906 / 0.859 / 0.781 / 0.690 / 0.567 / 0.445 / 0.044.
    - Gains (det, mean of 5.56M / 5.82M / 6.0M minus no-learning gauss): +0.61 / +0.64 / +0.60 / +0.59 / +0.52 / +0.22 / +0.30 / −0.01.
  - Stage 1:
    - Share of states clustered by phase: 0.27 / 0.22 / 0.33 / 0.20 / 0.27; ambiguous (t = 0) 0.00. The null's false-positive rate is 0 in every phase.
    - 7–11% of sample entries fall outside [−1, 1].
  - Checks:
    - Top-of-chain: |R(t = 99)| ≤ 0.034; 3.6% of energy tests p < 0.01.
    - Memorization: the median distance to the nearest training chunk is the same at k = 0 / 10 / 17 / 18 / 19 (0.57 / 0.58 / 0.57 / 0.56 / 0.59), so no snapping to training actions at low noise.
    - Two-rise (descriptive): policy vs null-replicate rate — cluster label 5% vs 30%, value 39% vs 31%, grippers ≤ 1% vs ≤ 2.5%.
- [2026-11-11 10:55] Stages 1–2 COMPLETED, job 59737117 (1 H200, 33 min, 330 states; `results/exp14/stage12/`, actions in `/n/netscratch/olveczky_lab/Lab/charleszhang/exp14/actions/`). Stage 1 simulator executions (84k chunks) submitted: job 59742199, sapphire 64 CPUs. Stages 3–4 wait for the executor review.
- [2026-11-11 10:35] Timing batch 59736989 done (H200, 1.3 min for 3 states): ≈ 5–7 s per state after warm-up, so Stages 1–2 for 330 states ≈ 35–40 GPU-min. Critic x_T invariance exact (max |ΔQ| = 0). Submitted the full Stages 1–2 run, 1 GPU, gpu_h200 (job id in the next entry).
- [2026-11-11 10:20] Submitted the Stage 1–2 timing batch (3 states), job 59736989, gpu_h200 / olveczky_lab (kempner is at the 15-GPU cap).
- [2026-11-11 10:15] **State collection done**, job 59736632 (sapphire, 48 CPUs, 4.3 min). Outputs: `results/exp14/states/` (`states.npz`, `checks.json`, `selection.json`, rollouts and phases).
  - Rollouts: 192 episodes, success 0.224 (base policy, DDIM-20; matches 0.195 ± 0.016), 18,138 chunk states. Demos: 24,604 rows (every 8th step) from 300 demos.
  - Phase counts (unlabeled / before_contact / open_grasp / trash / handover / place):
    - rollouts 638 / 2095 / 8616 / 4013 / 2488 / 288;
    - demos 349 / 3457 / 8242 / 3190 / 7745 / 1621.
  - 330 states selected: 30 rollout + 30 demo per phase, 15 + 15 ambiguous (t = 0), no shortfall.
  - Checks:
    - Restore + same chunk twice is now exact: obs 4e-13, sim state 2e-11 (the observable-timer reset removed the 7e-4 residual).
    - Without the gripper restore, obs diverge by 7.4 cm.
    - Gripper replay vs live env: 0.0.
    - Demo obs vs env after restore: p99 2.4e-7 (normalized). The only exception is raw dim 57 (`payload_in_target_bin`, a contact flag), which flipped on at least one of 200 demo states. The dataset records it from contacts computed during the step (pre-step configuration); restore recomputes contacts at the current configuration, so borderline contacts can flip.
  - The selected rollout states' restored obs differ from the obs recorded live by up to 8.9 mm (149 / 165 states > 1e-5). The live obs came from robosuite's observable sampling, which lags the state within a step. The restored obs (exact for the saved state) is what the measurements use.
- [2026-11-11 10:05] Submitted full state collection, job 59736632: `scripts/exp14/slurm/collect_states.sbatch`, sapphire, 48 CPUs; 192 rollouts, demo states, checks, 330-state selection.
- [2026-11-11 09:55] **State-collection validation (smoke srun jobs 59734327 and the reruns).** Found and fixed a restore bug.
  - Demo obs vs env after restoring the state: max |Δ| 3e-7 raw, 6e-7 normalized (200 states). The demos are the policy's training data: nearest dppo train.npz state is 3e-8 away.
  - **Bug found: restore was not deterministic.** The same chunk from the same restored state diverged by 4–7.5 cm in obs. Cause: robosuite's PandaGripper keeps `current_action` outside the MuJoCo state. `format_action` integrates ±0.01·sign(cmd) at every substep (25 per env step), and the finger actuators follow it.
  - Fix: save / restore both grippers' `current_action` with the sim state; zero `qacc_warmstart` on restore; reset the observables' sampling timers.
    - For demo states the gripper state is reconstructed by replaying the update rule over the demo's earlier actions. Replay vs the live env: max |Δ| = 0.0 over 300 steps.
    - After the fix, the sim state after a chunk matches to 2e-11. A residual 7e-4 obs difference remained, which motivated the observable-timer reset; it will be rechecked in the full run.
  - Action layout confirmed (`robot.action_dim` = 7, 7):
    - dim 6 = +1 closes gripper 0 only (gap −0.038 vs −0.0005); dim 13 = +1 closes gripper 1 only.
    - dim 0 moves eef 0 +7.8 cm in x; dim 7 moves eef 1 +5.9 cm in x.
  - Timing: restore + one 8-step chunk = 0.29 s (≈ 36 ms per env step); one DDIM-20 policy call on 1 CPU thread = 0.032 s; a full 800-step rollout ≈ 50–90 s.
  - **Updated CPU estimates:**
    - Stage 1 executions: 84k chunks ≈ 7 core-h.
    - Stage 3, fully matched to P = 128 (policy + null 2a, 2b plan + steer): ≈ 9.2M chunks ≈ 740 core-h, far above the 100 core-h cap. It will be subsampled (Design change 6, to be recorded before Stage 3).
    - Stage 4: 25,600 episodes ≈ 20 s each ≈ 140 core-h (previous estimate 40–80).
- [2026-11-11 09:30] Design change 5 recorded and implemented (clipping for value / execution, gripper null-degenerate reporting plus `grip{0,1}_mean`, feature basis saved, P = 128 references, centered Gram bootstrap). `test_windows.py` incl. the Gram equality test passes (srun).
- [2026-11-11 09:10] Written: `scripts/exp14/sim_exec.py` (Stage 1 / 3 / 4 executor) and `scripts/exp14/analyze.py` (rules, summaries, figures). Not yet run.
- [2026-11-10 23:35] Diagnostic, **not a gate**: pass rate of the final procedure over measurement seeds 9–16. Job 59651324: `scripts/exp14/slurm/stage0_power2.sbatch`, test partition, 3.7 min; outputs `results/exp14/stage0_power2/`.
  - All 8 seeds pass (a), (b), (c) and null-vs-null. Label width 2.88–3.77 vs null 4.98–6.43; midpoints −0.97 to −1.28 vs −ln λ_max ≈ −1.16.
  - Counting seed 8, criterion (a) passes in 8 of 9 runs, so its power is ≈ 0.9 at P = 64.
  - Two-rise (descriptive): fires on the policy in 5 of 90 state × feature curves, and in 0 of 90 null-replicate curves.
- [2026-11-10 23:25] **Stage 0 gate (seed 8, final procedure) FAILED criterion (a).** Job 59650757: `scripts/exp14/slurm/stage0_gate3.sbatch`, test partition, 1.0 min; outputs `results/exp14/stage0_gate3_P64_g8/`. Final procedure = Design changes 1–3: null clean points drawn from the same chain, rule 1 width-only.
  - (a) Fails. Start-state label width 4.42 [3.29, 5.05] vs null 7.10 [4.58, 8.52]. The point estimates differ as expected, but the intervals overlap by 0.47.
  - (b) Passes: midpoint −1.34 vs −1.19.
  - (c) Passes: past-obstacle PC1 4.12 [3.85, 4.67] vs null 4.16 [3.77, 4.68].
  - Null-vs-null "narrower": none.
  - Per instruction, no transport stage was started. Reported to the coordinator.
- [2026-11-10 23:20] `test_null_path.py` passes with chain-drawn clean / base points in both arms (2a max |ΔR|/SE 2.29, 2b 1.98); log `results/exp14/test_null_path_POST_FIX2.log`. `test_windows.py` passes. Added `bootstrap_resolved_gram`, a memory-light bootstrap that uses the same rng draws and should give identical output; its equality test was added to `test_windows.py` but has not been run yet.
- Code written, not run: `scripts/exp14/stage12.py` (transport GPU driver: Stages 1, 2a, 2b, Stage 4 action generation).
- [2026-11-10 22:55] **Stage 0 gate PASSED again after the review fixes** (Design change 2). Job 59641582: `scripts/exp14/slurm/stage0_gate2.sbatch`, test partition, 16 CPUs, 1.3 min. Settings: P = 64 × M = 16, fresh measurement seed 7, null through the identical sampler, policy clean points drawn separately from the Stage 1 samples. Outputs: `results/exp14/stage0_gate2_P64_g7/`; figure `report/assets/figs/exp14_stage0_control.png` (replaced).
  - (a) Passes. Start-state label width 3.41 [2.75, 3.98] vs null 6.46 [4.91, 7.95].
  - (b) Passes. Midpoint −0.88 vs −ln λ_max = −1.15.
  - (c) Passes. Past-obstacle PC1 width 4.42 [3.91, 4.78] vs null 4.13 [3.63, 4.70], two-rise not preferred.
  - Null-vs-null "narrower": none, on any state or feature. No censored state × feature.
  - Sanity check: |R(t = 99)| ≤ 0.016; energy p = 0.31 / 0.82 / 0.35.
  - **New caveat (exploratory feature, not a gate feature):** two-rise is "preferred" for PC2 at the past-obstacle state (ΔBIC 17.2), a false positive. The curve is ≈ 0 up to k = 16, then 0.14 / 0.26 / 1.0 at k = 17–19. The 5-parameter fit uses a tiny early rise plus the final jump, which comes from the coarse k = 18 → 19 gap and the forced R(k=19) = 1. The null's identical curve gives ΔBIC 5.1, so not preferred. Because transport's low-variance features resolve late, this can create spurious "windows" through the two-rise branch of rule 1. Not fixed (stopping as instructed). Options: also require two-rise to beat the null-vs-null rate, or require the two rises to be separated from the k = 19 point.
- [2026-11-10 22:45] Review fixes implemented and tested (Design change 2). `test_null_path.py` fails on the pre-fix code (srun 59641510; |ΔR|/SE up to 12.7) and passes on the fixed code (59641183 + rerun; max 2.3). `test_windows.py` still passes.
- [2026-11-10 22:29] (Superseded by the 22:55 run: this one used the biased exact-posterior null.) **Stage 0 gate PASSED** at P = 64 × M = 16, fresh measurement seed 6, same trained toy model. Job 59634280: `scripts/exp14/slurm/stage0_gate.sbatch`, test partition, 16 CPUs, 6.7 min. Outputs: `results/exp14/stage0_gate_P64_g6/`; figure `report/assets/figs/exp14_stage0_control.png`.
  - (a) Passes. Start-state route label: width 3.88 [3.20, 4.25] vs null 6.86 [4.85, 8.57], intervals disjoint. Clustered: ΔBIC 716, dip p ≈ 0.
  - (b) Passes. Midpoint −1.23 vs −ln λ_max = −1.14 (λ_max 3.12).
  - (c) Passes. Past the obstacle, PC1 width 4.56 [4.07, 5.12] vs null 4.75 [3.93, 5.10], two-rise not preferred (ΔBIC −0.7).
  - No two-rise false positive at the start state (label ΔBIC 8.4, below 10). Null-vs-null "narrower": none.
  - Sanity check (does not gate): |R(t = 99)| ≤ 0.017; energy tests vs fresh samples p = 0.32 / 0.72 / 0.64.
  - Next: the user has the measurement code reviewed before any transport stage.
- [2026-11-10 22:20] Design re-preregistered (Setup, "Design change"): P = 64 for Stage 2a and the null; Stage 4 now 32 points (25,600 episodes); Stage 3 scaled to match; compute estimate updated. In `toy.py`, the default is now P = 64, and PASS = the plan's three criteria (the top check no longer gates). Gate rerun submitted as job 59634280.
- [2026-11-10 22:30] **Stage 0 diagnostic done** (CPU, test partition, job 59625978, 1.5 min × 32 cores; the GPU attempts 59625539 / 59625594 and the sapphire / shared attempts 59625696 / 59625872 sat in the queue on priority and were cancelled). Same trained toy model, measurement seeds 2–5 (seed 1 = the original run). Outputs `results/exp14/stage0_power/P{16,64}_g{2..5}/`.

  | design | (a) label narrower than null | (b) midpoint within 1 of −ln λ_max | (c) past obstacle matches null | two-rise false positive on the start-state label | null-vs-null "narrower" |
  |---|---|---|---|---|---|
  | P = 16 (preregistered), 5 seeds | 2 / 5 | 5 / 5 | 5 / 5 | 2 / 5 | 0 / 5 |
  | P = 64, 4 seeds | 4 / 4 | 4 / 4 | 4 / 4 | 0 / 4 | 0 / 4 |

  Label width (policy vs null, log-SNR): P = 16: 3.06/5.52, 4.06/3.72, 2.14/5.45, 3.45/6.01, 3.80/8.69. P = 64: 3.19/6.13, 3.43/6.48, 3.79/6.77, 3.20/6.46, with null intervals ≈ ±1 instead of ±2.5. Midpoints −1.0 to −1.6 vs −ln λ_max = −1.16. Reading: the pipeline finds the known window and is calibrated (no null-vs-null false positives, (c) always passes), but 16 noisy points per level are too few. The null's width interval is then too wide for (a), and the two-rise test fires falsely. With 64 points both problems go away. The extra top-of-chain check (my addition, not one of the plan's three criteria) failed twice. Both are threshold artefacts: one energy-test p = 0.004 among 18 tests, and one |R(t=99)| = 0.056 against the 0.05 threshold at P = 16. R at t = 99 was ≤ 0.03 in every P = 64 run.
- [2026-11-10 22:05] Submitted Stage 0 power diagnostic, job 59625539 (`scripts/exp14/slurm/stage0_power.sbatch`): same trained toy model, measurement seeds 2–5, at P = 16 (as preregistered) and P = 64. Diagnostic only; criteria and the preregistered result stay as below. No transport stage runs until the user decides.
- [2026-11-10 21:46] **Stage 0 FAILED criterion (a)**, job 59624808 (`scripts/exp14/slurm/stage0.sbatch`, 1 H100, ~5 min; train 30k steps in 35 s, loss 0.046). Start state: clustered (ΔBIC 752, dip p ≈ 0), λ_max 3.18, so −ln λ_max = −1.16. Cluster label: midpoint −1.51 (criterion b passes, |Δ| = 0.35), width 3.06 [2.44, 3.39] vs null 5.52 [3.09, 8.10]. The point estimates differ as predicted, but the null's 16-point bootstrap interval overlaps the policy's, so (a) fails. (c) passes: past the obstacle, PC1 width 4.64 [3.11, 5.95] vs null 5.48 [3.65, 5.99], no two-rise. The top-of-chain check passes: R at t = 99 within ±0.03 for all features; energy test vs fresh samples p = 0.375 and 0.112. Side finding: at the start state the two-rise test fires on the label (ΔBIC 14.9, rises at −2.18 and −0.14). The truth there is one rise, so with 16 points this test can give false positives (the noisy points are shared across levels, so their noise is correlated along the curve). Outputs: `results/exp14/stage0/`.
- [2026-11-10 21:40] Estimator checks passed (`scripts/exp14/test_windows.py`, srun job 59624768): the null procedure recovers the closed-form R (max error ≤ 0.026 over λ = 0.01–2), the open fraction under the exact ODE map equals 1 − R (≤ 0.036), and the closed-form null width is 4.39.
- [2026-11-10 21:38] Submitted Stage 0, job 59624808.
- [2026-11-10 21:50] Doc created before any job. Dataset downloaded and md5 recorded. Code snapshot and critic copy in `/n/netscratch/olveczky_lab/Lab/charleszhang/exp14/`.

## Results
Main analysis: 319 states (11 diverged states excluded; the 330-state sensitivity run gives identical rule outcomes). Files: `results/exp14/analysis/analysis.json`, `rule1_rows.csv`, `stage4_critic.json`; sensitivity `results/exp14/analysis_keep_diverged/`.

**Summary-table row: "Windows exist at one place."**

| Rule (preregistered) | Outcome | Numbers |
|---|---|---|
| 1. A window exists | **holds** | Cluster label narrower than its null at 44 / 77 clustered states vs 0 / 77 null-vs-null (Fisher p = 9e-18). grip0 60 / 319 vs 1 (p = 1e-18), grip1 71 / 319 vs 0 (p = 5e-24). Value 11 / 319 vs 0: difference 3.4 points < 10, fails. |
| 2. Windows differ across phases | **fails** | Largest phase-median gap (steps): cluster label 0.42 (p = 0.66), grip0 3.64 (p = 0.044), grip1 0.62 (p = 0.86), value 0.75 (p = 0.087) |
| 3. Open fraction ranks steps like the RL gains | **holds** | Spearman ρ = 0.95, exact p = 5.7e-4 (steering form identical; 4.79M gains 0.95; 6.0M alone 0.81, p = 0.011) |

- **Where things resolve.** Midpoints (DDIM step k): cluster label ≈ 5.5–5.9 and grippers ≈ 4.5–5.7 in every phase; value ≈ 16–17.
  - Cluster-label widths ≈ 3–4.5 log-SNR vs a null of 4–8.
  - Value widths match their null.
- **Value open fraction (median)**, plan form, at k = 0 / 3 / 5 / 8 / 11 / 14 / 16 / 19: 0.955 / 0.906 / 0.859 / 0.781 / 0.690 / 0.567 / 0.445 / 0.044.
  - RL gains (det, mean of 5.56M / 5.82M / 6.0M minus no-learning): +0.61 / +0.64 / +0.60 / +0.59 / +0.52 / +0.22 / +0.30 / −0.01.
- **Stage 1.**
  - Share of states clustered: before_contact 0.27, open_grasp 0.22, trash 0.33, handover 0.20, place 0.27, ambiguous (t = 0) 0.00. The null's false-positive rate is 0 in every phase.
  - In simulation (one chunk per Stage 1 sample), the gripper-0 target is split (second most common target ≥ 20%) at 28 / 60 before-contact, 20 / 52 open_grasp, 20 / 57 trash, 22 / 60 handover, 16 / 60 place and 28 / 30 ambiguous states; the gripper-1 target at 9 / 22 / 20 / 15 / 23 / 8.
- **Stage 3** (97 non-diverged states; exploratory, 16-point subsample, BH across states × features). Short-horizon effects are almost never narrower than the null:
  - object displacement 6 / 97 (7 BH-significant);
  - eef displacement 0 / 97 (1 BH);
  - which object each gripper approaches 0 / 97 (63 / 61 states censored: that feature is constant for 16 points).
- **Stage 4** (20 rollout states, k = 4 / 9 / 12 / 15 / 18, 25,600 episodes):
  - Success 0.476 overall and flat across levels (0.485 / 0.479 / 0.474 / 0.471 / 0.472). Per state, success ranges from 0.0 to 1.0 (states 74 and 85 never succeed; state 256 always does).
  - Between-point share of success variance R_success ≈ 0 at every level (−0.001 / 0.009 / 0.005 / 0.013 / 0.001), while the value's R on the same chunks rises 0.17 / 0.41 / 0.54 / 0.63 / 0.87.
  - Critic vs success: pooled AUC 0.86, essentially all between-state. The AUC of each state's mean Q is 0.866, and the pooled AUC of within-state-centered Q is 0.510. Mean within-state AUC 0.57 over the 17 states with 0 < success < 1; it is 0.53 without states 258 and 261, which have only 2–3 failures each.
  - Per level, within-state AUC 0.57 / 0.52 / 0.55 / 0.54 / 0.54. Picking the critic's top-quarter chunks raises success by +0.032 [0.004, 0.071] / +0.027 [0.002, 0.060] / +0.020 [−0.005, 0.047] / +0.020 [−0.006, 0.048] / +0.005 [−0.010, 0.016]. These are averaged over all 20 states, including those where success is all 0 or all 1. Over the non-degenerate states only: +0.043 / +0.034 / +0.025 / +0.026 / +0.006.
- **Checks.**
  - Top-of-chain: |R(t = 99)| ≤ 0.034.
  - No memorization: the nearest-training-chunk distance is the same at k = 0 / 10 / 17 / 18 / 19 (0.58 / 0.58 / 0.59 / 0.57 / 0.59).
  - Two-rise (descriptive): policy vs null-replicate rate — cluster label 5% vs 30%, value 39% vs 31%.
- **Figures** (`report/assets/figs/`): `exp14_stage0_control.png`, `exp14_width_vs_null.png`, `exp14_midpoints_by_phase.png`, `exp14_example_curves.png`, `exp14_open_vs_rlgain.png`, `exp14_stage4_success_vs_value.png`.

## Analysis
- **Windows exist, but at one place.**
  - The discrete choices the policy makes (which action cluster; whether to open or close a gripper) are decided in a window narrower than a Gaussian would give. For the cluster label this holds at 57% of clustered states, with zero false positives in the null-vs-null control.
  - Those windows sit at the same step, k ≈ 5–6 (log-SNR ≈ −1.4), in every task phase. So one well-chosen step covers them; a per-chunk step choice has nothing phase-specific to exploit.
  - This agrees with exp13: critic-chosen k had no headroom over best-of-N at k = 0, and its phase effect did not replicate.
- **The value (outcome) does not have a window.** It resolves late (k ≈ 16–17) and as smoothly as its null.
- **Rule 3 holds, but its evidence is mostly the coarse shape.** The open fraction of Q^A falls monotonically with k, and the RL gains fall with k, so the ranks agree (ρ = 0.95).
  - The ranking is driven by the late steps (14–19). Among k ≤ 11, both are nearly flat (open fraction 0.96 → 0.69; gains 0.52–0.64).
  - The measurement explains why late steps fail: there is little left to change. It does not discriminate among early steps.
- **Stage 4: one chunk barely matters for success, and the critic barely tells chunks apart.**
  - Regenerating only the first chunk leaves success unchanged at every level (R_success ≈ 0). Success here is set by the state and by the stochastic continuation, not by one chunk.
  - The critic's value of that chunk resolves across k (0.17 → 0.87), but this variation carries little success signal: within-state AUC 0.52–0.57, and 0.53 on average once two near-saturated states are dropped.
  - **So the trained critic does not predict higher success for steering inside the measured window.** Choosing chunks by Q^A gains ≤ 3 points over all 20 states (≤ 4 over the non-degenerate ones), and the gain is largest at the earliest levels (k = 4–9, CIs just above 0), not inside the late value window. It is indistinguishable from 0 at k ≥ 12.
  - The pooled AUC of 0.86 reflects between-state differences in success, which the critic captures well.
- **Caveats.**
  - The gripper rule-1 results rest on the ~80–100 states where both arms are defined (null degenerate at ~210–220 states).
  - The 11 diverged states come from the unclamped sampler the plan requires.
  - Stage 3 runs at 16-point power.
  - The Stage 4 state selection by Q spread still picked several states with success near 0 or 1.
  - Stage 4 changes one chunk only, whereas RL steers every chunk.

## Next Steps
- The step choice per chunk is not supported. Keep a fixed early step (k ≈ 3–8, where the discrete choices resolve and gains are flat) and drop the learned or phase-dependent choice. This is the "one place" row: choose the step by measurement.
- To test steering inside the window directly, steer every chunk (not one) with the critic at fixed k ∈ {4, 9, 15}, and compare success. The Stage 4 single-chunk effect is too small to resolve this.
