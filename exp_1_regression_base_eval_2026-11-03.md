# Experiment: exp_1 k=0 regression check + Step 2 base evaluation
Status: Completed

## Goal
(A) Check that `train_adsrl.py steer.mode=fixed steer.k=0` reproduces upstream DSRL-NA on can over ~200k env steps, and smoke-test adaptive / fixed k=4 / resume on the real env. (B) Measure base-policy success (200 episodes) on can/square/transport, pick the transport DDIM K, and run the authority diagnostic (plan: notes/plan_adaptive_dsrl.md, Verification + Step 2).

## Setup
- Commit a6bd481 (all project code is still untracked; git_dirty in run_info.json). venv `.venv`.
- **Task A pack** (one H100, CUDA MPS, `ADSRL_RUNS_PER_GPU=6`, `env.n_eval_envs=10`, 200 eval episodes, `deterministic_eval=True`, `eval_interval=1500` vec steps = 24k env steps for both implementations so evals land at matched env steps 24k+24k*i):
  - upstream DSRL-NA (`scripts/pilot_train_dsrl.py`, `dsrl/cfg/robomimic/dsrl_can.yaml`), seeds 1, 2; `+total_timesteps=44000` chunks (= 200k env steps incl. the 24k warm start). Eval success written by a wandb.log tee to `eval.csv` (`+eval_csv=`), with env_steps corrected for upstream's warm-start counter (it counts warm-start vec steps x n_envs, missing the x act_steps factor).
  - train_adsrl fixed k=0, seeds 1, 2, `train.total_timesteps=200000`.
  - smoke: adaptive (50k env steps) and fixed k=4 (50k, killed after its 2nd checkpoint and resumed with `resume=true`), both `eval_interval=500`.
- **Task B** (one GPU job): `scripts/base_eval.py` reuses `AdsrlCallback.evaluate` with an agent that plays `SteeredBasePolicy.base_action` (deterministic DDIM from x_T ~ N(0, I)); 200 episodes, 10 envs. can, square (K=8), transport K in {100, 20, 15, 10}. Then `scripts/authority.py --state_source dataset --n_states 500 --n_w 64` per task. Outputs: `results/base_eval/`, `results/authority/`.
- Expected: k=0 curves within seed noise of upstream; base success roughly can ~0.5, square ~0.4-0.5, transport lower (DSRL paper figs); transport success roughly flat down to K=20.

## Progress Log
- [2026-11-03 19:05] k=4 pack 57562281 COMPLETED (1.95 h): Q(s,a) without x_T (`actor_noise_critic`) 0.551 vs `all` 0.272 (stoch, 72-192k mean). Wrote Results/Analysis. Status -> Completed.
- [2026-11-03 18:55] x_T ablation 57555939 COMPLETED (2.1 h): k=0 without x_T matches upstream (0.533 vs 0.493 stoch, 0.476 vs 0.461 det, 72-192k mean over 3 seeds); k=0 with x_T 0.180 / 0.144 (3 seeds). **Regression passes with xT_inputs=none** (the new default at fixed k=0).
- [2026-11-03 17:05] Per coordinator: replaced `steer.obs_xT` with `steer.xT_inputs: auto|none|actor_noise_critic|all` (`SteerSACPolicy` in adsrl/agent.py; train_adsrl now passes it instead of "MlpPolicy"). Q(s,a) and its target never see x_T under `actor_noise_critic`; the actor and noise critic Q^W do. `none` drops x_T everywhere (fixed k=0 only); `all` = old behaviour. `auto` (new default in all 3 configs) = `none` at fixed k=0, else `actor_noise_critic`. Ablation job 57555939's `obs_xT=false` runs equal `xT_inputs=none`. Tests: test_fixed_k_xt_inputs (4 cases), test_fixed_k_xt_inputs_none_rejected_at_k_gt_0, test_adaptive_xt_inputs_actor_noise_critic (incl. checkpoint round trip), test_adaptive_rejects_xt_inputs_none, test_resolve_xt_inputs; pytest 42/42 on srun 57561126. Submitted 57562281 (`scripts/slurm/exp1_k4_xt_inputs.sbatch`, 4-run MPS pack): fixed k=4 with xT_inputs=all vs actor_noise_critic, seeds 1-2, 200k env steps.
- [2026-11-03 17:00] Ablation 57555939 step-0 evals: k=0 without x_T reproduces upstream's per-seed initial actor (det success s1 0.000 vs upstream 0.000, s2 0.355 vs 0.395; stoch s3 0.285 vs 0.275), as expected since identical input dims give identical init RNG draws.
- [2026-11-03 16:45] Submitted x_T ablation pack 57555939 (`scripts/slurm/exp1_xt_ablation.sbatch`, 5-run MPS pack): k=0 with `steer.obs_xT=false` (new DropXTExtractor drops x_T from actor/critic/noise-critic inputs; only allowed for fixed k=0) at seeds 1-3, plus upstream s3 and k=0 (with x_T) s3. pytest 35/35 on compute node (srun 57554972; new `test_drop_xt_extractor_ignores_xt`).
- [2026-11-03 16:30] Code comparison: FixedK runs upstream's train/target/noise-critic code unchanged; FastReplayBuffer is uniform like upstream's sampler (no offline mixing in NA); the sampler matches upstream bit for bit (tests). The remaining material difference is the 28 i.i.d. N(0,1) x_T dims appended to every network input (23 real state dims), which is pure input noise at k=0 at UTD 20. The x_T ablation tests this.
- [2026-11-03 16:06] Task A pack 57503563 COMPLETED (2.0 h, MaxRSS 53 GB). **Regression check FAILS**: upstream climbs back to 0.53-0.59 stochastic success by 192k env steps; train_adsrl k=0 stays at 0.14-0.25 (both seeds, every eval after 72k). Both dip the same way at 48k (0.21-0.38). Smoke runs and resume OK (see Results).
- [2026-11-03 15:05] Transport-K job 57503346 COMPLETED (34 min). Pooled transport base success: K=100 0.210, K=50 0.203, K=25 0.183, K=20 0.195, K=10 0.168 (see Results). Task A pack 57503563 running; the smoke_k4 kill/resume worked (resumed at env_steps 40016 with n_evals=3, next eval at 48016).
- [2026-11-03 14:12] Pack 57494227: all train_adsrl runs died at wandb.init: cfg `wandb.mode: online` overrides the env `WANDB_MODE=offline`, and compute nodes can't reach wandb (the debug job passed `wandb.mode=offline` explicitly). Only the 2 upstream runs were alive (at 48k env steps). Cancelled it and resubmitted the full 6-run pack with `wandb.mode=offline` as 57503563, so both implementations share the same packing.
- [2026-11-03 14:10] Task B 57494228 COMPLETED in 16.6 min (base evals 4-17 min each with 10 envs; authority 8 s (can) to 180 s (transport K=100)). Transport K=20 and K=10 both 0.140 vs K=100 0.215 (diff 7.5 +- 3.8 pts, inconclusive). K=15 (0.085) has a truncated schedule (100//15=6, so t only reaches 84). Submitted 57503346: 400 more transport episodes (eval seed 1) for K in {100,50,25,20,10} (divisors of 100) + authority at K=25, 50.
- [2026-11-03 14:30] Submitted Task A pack 57494227 (`scripts/slurm/exp1_regression_pack.sbatch`, kempner_h100, 1 GPU, -c 24, 96G, 5 h) and Task B 57494228 (`scripts/slurm/exp1_base_eval.sbatch`, kempner_h100, 1 GPU, -c 24, 96G, 3 h; runs authority for all transport K candidates so no second job is needed).
- [2026-11-03 14:25] pytest on a compute node (srun 57493606, test partition): 34 passed (32 old + 2 new).
- [2026-11-03 14:20] Bug fixed in train_adsrl.py: when training ended exactly on a periodic eval (or a finished run was resumed), the final eval ran again at the same env_steps, appending duplicate rows to eval_summary.csv and overwriting that step's npz dump. AdsrlCallback now tracks `last_eval_steps`/`last_eval_dumped` (saved in the checkpoint state) and skips the redundant final eval. Regression tests: `test_final_eval_not_duplicated_at_eval_boundary`, `test_final_eval_runs_when_last_eval_not_dumped`.
- [2026-11-03 14:15] Debug job 57485874 (12 min) ran every path on the real env: fixed k=4 tiny run + resume to a larger total (env_steps/CSV continue correctly), adaptive tiny run (train/k_frac/*, k_hist logged; eval npz has episode, chunk, env_t, obs, obs_raw, x_T, action, base_action, k, w, k_probs, reward, success_step, episode_success, meta), upstream tiny run with eval CSV, base_eval on can and transport K=10. sacct says FAILED only because of a bad `tail` at the end of the script. Observation: upstream deterministic eval at init (actor mean w = 0, i.e. x_T = 0) gave 0/10 success on can vs 7/10 stochastic.
- [2026-11-03 14:05] Submitted debug job 57485874 (kempner_h100, 16 CPUs, 45 min): tiny fixed k=4 run + resume, tiny adaptive run, tiny upstream run with eval CSV tee, base_eval on can and transport K=10.
- [2026-11-03 14:00] Wrote `scripts/base_eval.py`; added `+eval_csv` (wandb.log tee) to `scripts/pilot_train_dsrl.py`.

## Results

### Verdict: k=0 regression (3 seeds per arm; mean success over evals at 72k-192k env steps; SE across seeds)
| arm | stoch | det | job |
|---|---|---|---|
| upstream DSRL-NA | 0.493 +- 0.032 (0.557, 0.456, 0.467) | 0.461 +- 0.030 | 57503563 (s1, s2), 57555939 (s3) |
| train_adsrl k=0, x_T in all inputs (old) | 0.180 +- 0.008 (0.174, 0.171, 0.195) | 0.144 +- 0.005 | 57503563, 57555939 |
| train_adsrl k=0, xT_inputs=none (new default at k=0) | 0.533 +- 0.014 (0.552, 0.505, 0.542) | 0.476 +- 0.031 | 57555939 |
Final eval at 200k (k=0 none): 0.555/0.525/0.570 stoch. Per-eval curves are in `results/exp1_regression/` and `results/exp1_xt_ablation/` (eval_summary.csv; upstream eval.csv).

### Fixed k=4, can, xT_inputs all vs actor_noise_critic (job 57562281, 2 seeds, 200k env steps)
| env steps | all s1 | all s2 | anc s1 | anc s2 |
|---|---|---|---|---|
| 48k | 0.090 | 0.045 | 0.370 | 0.590 |
| 96k | 0.215 | 0.280 | 0.565 | 0.485 |
| 144k | 0.210 | 0.345 | 0.635 | 0.600 |
| 192k | 0.290 | 0.420 | 0.700 | 0.545 |
| 200k | 0.275 | 0.355 | 0.680 | 0.545 |
72-192k mean (stoch): all 0.272 (0.250, 0.294); actor_noise_critic 0.551 (0.585, 0.518). Det: 0.262 vs 0.536. |a - a_base| ~0.67 for both (vs ~1.3-1.6 at k=0). 4-pack throughput: 200k env steps in ~115 min per run incl. evals.

### Task A: k=0 regression (job 57503563), can, eval success (200 episodes) at matched env steps
| env steps | upstream s1 | upstream s2 | k0 s1 | k0 s2 | (det) up s1 | up s2 | k0 s1 | k0 s2 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.675 | 0.495 | 0.665 | 0.705 | 0.000 | 0.395 | 0.565 | 0.485 |
| 48k | 0.290 | 0.375 | 0.210 | 0.255 | 0.250 | 0.240 | 0.130 | 0.145 |
| 72k | 0.460 | 0.435 | 0.160 | 0.250 | 0.405 | 0.365 | 0.180 | 0.195 |
| 96k | 0.465 | 0.490 | 0.105 | 0.195 | 0.345 | 0.435 | 0.040 | 0.220 |
| 120k | 0.595 | 0.415 | 0.170 | 0.105 | 0.545 | 0.435 | 0.180 | 0.090 |
| 144k | 0.635 | 0.345 | 0.155 | 0.165 | 0.550 | 0.280 | 0.145 | 0.145 |
| 168k | 0.610 | 0.515 | 0.205 | 0.175 | 0.680 | 0.525 | 0.115 | 0.145 |
| 192k | 0.575 | 0.535 | 0.250 | 0.135 | 0.590 | 0.485 | 0.165 | 0.130 |
| 200k (final) | | | 0.280 | 0.315 | | | 0.230 | 0.240 |
Mean over 72k-192k (stoch): upstream 0.505, k0 0.172. Gap ~0.33 with per-eval SE ~0.035, consistent across seeds and time.
Note: the step-0 eval (base policy via x_T ~ N(0, I)) is 0.50-0.71 stochastic, above the 0.375 of the 200-episode base eval (Task B, different eval seed). Both dip to ~0.2-0.3 at 48k once RL starts.

### Smoke tests (job 57503563, 50k env steps)
- adaptive: trains stably; train/k_frac/k*, k_hist, k_hist_entropy, k_mean logged; eval_summary.csv has k_entropy 1.3-1.9 (log 8 = 2.08) and pi_k_entropy 1.05-1.25 by 32k+, so pi(k) sharpens from uniform; |a - a_base| grows 0.61 -> 0.93-1.04. Success 0.675 -> 0.11-0.17 at 50k (same early dip as k=0). Eval npz keys: episode, chunk, env_t, obs, obs_raw, x_T, action, base_action, k, w, k_probs, reward, success_step, episode_success, meta.
- fixed k=4: killed after its 2nd checkpoint (40016 env steps), resumed with resume=true: restored num_timesteps=4000 chunks and callback state (env_steps 40016, n_evals 3, wall/gpu offsets), next eval at 48016, final at 50000; CSV has no duplicated rows. |a - a_base| ~0.6 (vs 1.4-1.7 for k=0 at the same steps). Success 0.62 -> 0.18-0.22.

### Throughput (job 57503563, H100, MPS, 6 runs, n_eval_envs=10)
Training-only env steps/s per run (from eval timestamps): 6 runs packed: upstream 30-31, k0 30, fixed k=4 27-33, adaptive 26. After the 2 smoke runs finished (4 runs packed): upstream 38-42, k0 38-39. One 200-episode eval took 87-106 s per mode (x2 modes, stoch and det). Overall incl. evals: 28 env steps/s per long run (200k in 119 min); adaptive 22.6, k4 24.1. Aggregate in the 6-pack ~175 env steps/s = 3.9x a single run (45/s), matching the pilot.

### Task B: base policy (deterministic DDIM from x_T ~ N(0, I), train_adsrl eval protocol)
Binomial SE. Seed 0 = 200 episodes (job 57494228); seed 1 = 400 extra episodes (job 57503346); pooled where both exist.

| task | K | success | SE | n |
|---|---|---|---|---|
| can | 8 | 0.375 | 0.034 | 200 |
| square | 8 | 0.505 | 0.035 | 200 |
| transport | 100 | 0.210 | 0.017 | 600 (0.215 / 0.2075) |
| transport | 50 | 0.203 | 0.020 | 400 |
| transport | 25 | 0.183 | 0.019 | 400 |
| transport | 20 | 0.195 | 0.016 | 600 (0.140 / 0.2225) |
| transport | 15 | 0.085 | 0.020 | 200 (schedule truncated at t=84, not comparable) |
| transport | 10 | 0.168 | 0.015 | 600 (0.140 / 0.1825) |

Wall time per 200 episodes with 10 envs (6 evals packed, MPS): can 3.2 min, square 4.4 min, transport 13-16 min.

Transport K: vs K=100, K=20 is -1.5 +- 2.3 pts and K=10 is -4.2 +- 2.2 pts. K=10 meets the ~5-pt rule on its point estimate, but its 95% CI reaches -8.7 pts; K=20 is the safe choice at 2x the sampler cost.

### Authority (`results/authority/`, dataset states, 500 states x 64 w)
Mean per-dim std of the final action over w / mean ||a - a_base||, per k:
- can K=8: 0.131/1.22, 0.096/0.98, 0.090/0.88, 0.080/0.77, 0.071/0.67, 0.061/0.55, 0.051/0.44, 0.036/0.28
- square K=8: 0.078/0.72, 0.067/0.63, 0.063/0.58, 0.056/0.51, 0.047/0.44, 0.041/0.38, 0.034/0.30, 0.013/0.11
- transport K=10: 0.081/1.78, 0.072/1.63, 0.068/1.51, 0.063/1.36, 0.056/1.19, 0.050/1.04, 0.044/0.89, 0.038/0.73, 0.029/0.54, 0.011/0.19
- transport K=20 (k=0,5,10,15,19): 0.083/1.81, 0.073/1.55, 0.057/1.15, 0.041/0.78, 0.011/0.18
- transport K=100 (k=0,25,50,75,99): 0.088/1.88, 0.081/1.66, 0.064/1.27, 0.048/0.91, 0.012/0.17
Authority falls monotonically with k on every task. Can has a large drop from k=0 to k=1 (std 0.131 -> 0.096). Square's last step is nearly powerless (0.013).

## Analysis
- **The k=0 regression failure was caused by x_T in the network inputs**, not by the sampler, buffer or learning code. Dropping x_T from every network makes train_adsrl k=0 match upstream within seed noise, and even reproduces upstream's per-seed initial actor. The 28 i.i.d. N(0,1) x_T dims outnumber the 23 state dims. At UTD 20 with 3x2048 critics, the action critic apparently fits that noise, and the policy then never recovers from the early dip (both arms dip to ~0.2-0.3 at 48k, but only upstream recovers).
- **Q(s,a) should never see x_T.** x_T is resampled each chunk and its effect is already in a. Removing it from Q(s,a) and its target (actor and Q^W still see it) doubles k=4 success at 200k (0.27 -> 0.55). Every run of the old Step 1 code (x_T everywhere) is invalid for comparing k values. The Step 3 sweep must use `xT_inputs=auto`.
- With that fix, k=4 (0.551) is on par with k=0 (0.533) on can at 200k (2 vs 3 seeds; not a claim). k=4 stays about half as far from a_base (|a - a_base| 0.67 vs ~1.4).
- Early dip: every RL arm drops from ~0.5-0.7 (step 0) to ~0.1-0.4 at 48k before climbing. The `actor_noise_critic` k=4 runs dip least (0.37-0.59).
- Base policies: can 0.375, square 0.505, transport 0.210 (K=100), so there is room to improve on all three. Transport K=20 loses 1.5 +- 2.3 pts and K=10 loses 4.2 +- 2.2 pts vs K=100 (600 episodes each).
- Authority falls monotonically with k on every task. The ‖a - a_base‖ reachable at the last step is ~15-20% of that at k=0 on can, square and transport.

## Next Steps
- Use `steer.xT_inputs=auto` (none at k=0, actor_noise_critic otherwise) for all Step 3/4 runs. Optionally test whether the actor or Q^W also benefit from a lower-dim summary instead of raw x_T (e.g. a_base(s, x_T) or x0_hat_{k-1}).
- Transport K: K=10 by the ~5-pt rule (point estimate), K=20 if a safety margin is wanted (2x sampler cost). Authority outputs exist for K in {10, 15, 20, 25, 50, 100}.
- The can budget: k=0 and k=4 are both still rising at 200k (0.55-0.65), so keep ~1M for the sweep.
- The base-eval npz dumps (~480 MB) are in results/base_eval*/; the training checkpoints and replay buffers stay on scratch (`/n/netscratch/olveczky_lab/Lab/charleszhang/adsrl/exp1*`).
