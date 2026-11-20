# Plan: Adaptive-step DSRL (steer the diffusion policy at a chosen denoising step)

## Context
DSRL steers a frozen diffusion policy by having an RL agent pick the latent noise. In `dsrl/` (DPPO base policies + SB3 fork), that noise only sets the initial x_T of a deterministic DDIM sampler (eta=0, `dppo/model/diffusion/diffusion.py:266-322`, `dsrl/utils.py:5-15`). We want to test whether steering at a later denoising step works better: less authority over the action, but more in-distribution and better conditioned on what the base policy already "decided". We also want to test whether letting the policy choose that step per chunk beats any fixed step.

Phase A sweeps a fixed steering step k per task. Phase B trains a hybrid policy that picks (k, w) jointly. Everything stays in the torch `dsrl` repo; `dsrl_pi0` is out of scope because it is about 10-100x more expensive per step.

## Decisions (agreed)
- **Steering semantics (re-noise):** DDIM schedule indices are j = 0..K-1. Steering at k means: run plain DDIM from x_T up to step k, then replace that step's epsilon with w:
  `x_{t_k} = sqrt(ab_{t_k}) * x0_hat_{k-1} + sqrt(1 - ab_{t_k}) * w`, then continue plain DDIM.
  - k = 0 means x_T = w, which is exactly current DSRL.
  - Setting w equal to the model's own eps_hat recovers the unsteered DDIM output. This is used as a unit test.
- **Observation:** each chunk samples x_T ~ N(0, I) and appends it to the observation (obs grows by horizon*act_dim). Given (s, x_T, k, w) the action is deterministic, so the problem stays Markov.
- **Algorithm:** DSRL-NA on all tasks, including transport. The action critic Q(s,a) is unchanged. The noise critic becomes Q^W(s, x_T, k, w), with k fed as a one-hot input.
- **Adaptive actor:** hybrid SAC. It has a categorical head π(k|s,x_T) and K Gaussian heads π(w|s,x_T,k) with tanh squashing and the existing bound. The actor loss is an exact sum over k:
  `L = Σ_k π(k)[α_w log π(w_k|k) − Q^W(s,x_T,k,w_k)] + α_k Σ_k π(k) log π(k)`
  - There are separate auto-tuned temperatures α_w and α_k. Target entropy for k is ρ·log K, with ρ ≈ 0.5 (tunable).
  - Noise-critic distillation samples k uniformly and w ~ N(0, I).
  - Evaluation uses argmax k and the mean w when `deterministic_eval` is set; otherwise it samples.
- **Tasks:** can (short, medium difficulty), square (precision), transport (long-horizon, bimanual, 112-dim noise).
- **Seeds:** a 1-seed scan over all k, then 3 seeds for the promising k values, DSRL (k=0), and the adaptive version.

## Step 0: Setup
1. `git submodule update --init --recursive` inside `dsrl/` for the `dppo-dsrl` and `stable-baselines3-dsrl` forks. Download checkpoints and normalization files from the README's Google Drive folder into `dsrl/dppo/log` with `gdown`.
2. Create a uv venv at the project root: Python 3.9, torch 2.4.0, `dppo[robomimic]` editable, and the SB3 fork editable. `train_dsrl.py:11` imports d4rl unconditionally; our training script will not, so we can skip mujoco-py.
3. Smoke test on an interactive GPU node: `train_dsrl.py` on `dsrl_can` for about 20k steps. Record throughput in env steps/hr to size the walltimes.

## Step 1: Code (new package at project root; submodules stay untouched)
- **`adsrl/sampler.py`:** `steered_ddim_sample(model, cond, x_T, k, w)`.
  - It reuses `model.p_mean_var`, the `ddim_t` / `ddim_alphas` buffers, and the `denoised_clip_value` / x0-clamp logic from `diffusion.py` / `diffusion_eval.py`.
  - `k` is a per-sample LongTensor. Injection is a masked `torch.where` at loop index j == k, so a batch can mix different k values.
  - It also returns `x0_hat` at every step for diagnostics.
- **`adsrl/base_policy.py`:** `SteeredBasePolicy(obs_aug, k, w)`. It splits x_T off the augmented obs, normalizes as `DPPOBasePolicyWrapper` does (`dsrl/utils.py`), and calls the sampler.
- **`adsrl/env.py`:** `XTAugmentWrapper`, a VecEnv wrapper over the `ActionChunkWrapper` stack from `dsrl/env_utils.py`. It appends fresh x_T to every returned obs, including at reset, and widens `observation_space`.
- **`adsrl/agent.py`:** subclasses of the SB3 `DSRL` class (`stable_baselines3/dsrl/dsrl.py`):
  - `DSRLFixedK`: a constant k is passed to every diffusion call: `_sample_action` (:446), `predict_diffused` (:471), the target backup (:277), and `update_noise_critic` (:352).
  - `DSRLAdaptiveK`: the hybrid actor, a k-conditioned noise critic, and the two temperatures. k is stored in the buffer only for logging; Q(s,a) doesn't need it.
  - Override as few methods as possible. If the fork's structure makes subclassing awkward, copy `dsrl.py` into `adsrl/` rather than editing the submodule.
- **`train_adsrl.py` + `cfg/adsrl/{can,square,transport}.yaml`:**
  - A trimmed copy of `train_dsrl.py`.
  - New keys: `steer.mode: {fixed, adaptive}`, `steer.k`, `steer.k_grid`, `train.total_timesteps` (replacing the hardcoded 20M), `seed`.
  - Logs hyperparameters, the git hash, and the env to wandb (project `adsrl`). Per-eval it also logs: the histogram of chosen k, ‖a − a_base(x_T)‖, and success.
- **`tests/test_sampler.py`** (pytest, CPU, using a loaded can checkpoint):
  1. k=0 matches the original `Diffusion.forward` with `noise_action=w`, bit for bit.
  2. w = eps_hat at every k reproduces unsteered DDIM.
  3. A mixed-k batch equals separate per-k calls.
  4. Hybrid actor: log-prob and entropy shapes are correct, probabilities sum to 1, and a gradient reaches both heads.
- **`scripts/authority.py`:** a cheap diagnostic with no training. On about 500 dataset states, for each k, compute the spread of the final action over w ~ N(0, I) with x_T fixed. This shows how much control each k gives.

## Step 2: Base evaluation (exp_1)
- Evaluate the base policy (w ~ N(0, I), no RL) with 200 episodes per task. This confirms the room for improvement; DSRL's baseline numbers are not reported in the README.
- Transport: compare 100 vs 20 vs 10 DDIM steps. Use the smallest K that doesn't drop success by more than about 5 pts, since NA runs the sampler inside every critic backup.
- Run the authority diagnostic for every task.
- Pack everything into one GPU job.

## Step 3: Fixed-k sweep (exp_2)
- **Grid:** for can and square, all k ∈ {0..7} (8 DDIM steps). For transport, about 8 evenly spaced indices under the chosen K. That gives 24 runs at 1 seed.
- **Budget:** can ~1M env steps, square ~2M, transport ~3M. Calibrate these from Step 0 throughput and the DSRL paper's curves.
- **SLURM:**
  - One sbatch per task, packing all k values as parallel processes on one GPU. Use `kempner_pehlevan_lab`, a single kempner partition, about 32-48 CPUs (each run uses 4 train + 25 eval subprocess envs), and `XLA`-irrelevant torch.
  - If CPU becomes the bottleneck, drop to `n_eval_envs=10`.
- **Choosing what gets 3 seeds:** the best 2-3 k per task, plus k=0, at 3 seeds.
- **Metrics:** eval success vs env steps, and steps until reaching 80%/90% of the final success.

## Step 4: Adaptive k (exp_3)
- Train `DSRLAdaptiveK` on all 3 tasks with 3 seeds.
- Pilot-tune ρ ∈ {0.25, 0.5} on can first.
- Compare against: the base policy, DSRL (k=0), the best fixed k (oracle from the sweep), and adaptive.
- Ablation: adaptive with k restricted to the top-3 fixed-k set.

## Step 5: Analysis
- Curves of success vs steps with 3-seed mean ± SE, plus a per-task bar chart of final and AUC success.
- How the k distribution evolves over training. How chosen k depends on task phase: gripper state and object distance for can/square, handover phase for transport.
- Authority vs. k compared with the sweep results: does the most useful k track authority, or how in-distribution the steered action stays?

## Additions from the literature review
The full table is in the subagent report. The DSRL citation list is at `/n/netscratch/olveczky_lab/Lab/charleszhang/tmp/claude-64196/-n-holylfs06-LABS-olveczky-lab-Users-charleszhang-CS2243Project/0599ecae-81d1-44f9-8be1-9f0963d3fda5/scratchpad/dsrl_cites.txt`. Copy it into the project once we're out of plan mode.

- **Novelty.** No paper re-noises x0_hat at a step k chosen by a learned policy. These need to be cited and contrasted:
  - TMRL (2605.12236): picks a per-chunk timestep, but for noise on the *observation*, and needs special pretraining.
  - Ma et al. 2501.09732 ("search over paths"): uses the same re-noise operation, but as a search with a fixed schedule.
  - SDEdit and restart sampling: earlier uses of the operation.
  - DLSRL (2609.11270): steers intermediate hidden states.
- **Extra baselines** for Steps 3–4, all cheap:
  1. A constant-noise search in the style of Golden Ticket (2603.15757): CEM over one fixed w per task. Its Can result was 42.8→95%.
  2. Best-of-N noise at eval scored by Q^A(s, decode(w)), applied to both DSRL (k=0) and adaptive.
- **Improvements applied the same way to every variant (k=0, fixed k, adaptive)** so the comparison stays fair:
  - LayerNorm critics. The flag already exists in `train_dsrl.py`; check that it's on.
  - Base-rollout buffer warm start. This already exists as `init_rollout_steps`.
- **Ablation:** a noise trust region (Lagrangian bound on E‖w‖², following LP-DS 2606.01151). It is about 20 lines. Test it on can for k=0 and adaptive. If it helps both, keep it on for everything.
- **Stretch goal (not in the core plan):** an LPS-style actor (2603.05296) that backpropagates Q^A through the remaining K−k sampler steps instead of distilling Q^W. It gets cheaper as the steering step moves later.

## Comparisons and metrics
**Methods.** All use the same base checkpoint, 3 seeds, and the same env-step budget per task.

| # | Method | What it tests | Source |
|---|---|---|---|
| 1 | Base policy (w ~ N(0, I)) | Room for improvement | Step 2 |
| 2 | DSRL-NA (k=0) | Standard DSRL, the main baseline | Step 3 |
| 3 | Best fixed k (oracle, chosen on the 1-seed scan) | Does adaptive beat the best single step? | Step 3 |
| 4 | Uniform-random k (hybrid actor with π(k) frozen uniform) | Does a *learned* choice of k matter, or does mixing steps suffice? | Step 4, +3 runs/task |
| 5 | **Adaptive k (ours)** | – | Step 4 |
| 6 | Constant-noise CEM (Golden Ticket–style; one fixed w per task, no RL) | Is a learned noise policy even needed? | eval only |
| 7 | Residual SAC in action space | Noise-space vs action-space steering | new runs |
| ~~8~~ | ~~DPPO full fine-tuning~~ | Dropped 2026-11-04: a different method family on a different eval protocol. Cited in the text only, not plotted. | – |

Add-on applied to methods 2, 3 and 5: best-of-N noise at eval, scored by Q^A. This is reported as a separate column, not a separate method.

**Residual SAC (method 7):**
- New `adsrl/residual.py`: SAC whose action is Δa ∈ [−β, β]^(chunk), with a = a_base(s, x_T) + Δa and a_base included in the observation.
- β ∈ {0.1, 0.3}, chosen on can.
- Same UTD and network sizes as DSRL, and it reuses `SteeredBasePolicy` for a_base.

**Primary metrics:**
- **Eval success vs env steps.** 200 episodes per eval, same eval protocol for every method (`deterministic_eval` on).
- **Final success:** mean of the last 3 evals.
- **Sample efficiency:**
  - Env steps to reach 80% of the best final success across methods on that task.
  - Normalized AUC of the success curve.

**Secondary metrics:**
- **Compute efficiency:** success vs GPU-hours. This matters for the CEM and DPPO comparisons.
- **Speed:** mean episode length of successful episodes, i.e. how fast the task is solved.
- **Staying close to the base policy:** ‖a − a_base‖ per chunk, and the base model's denoising loss on executed actions.
- **Stability:** across-seed SE, and the number of runs that drop more than 20 pts below their peak.
- **Adaptive only:** the k histogram and its entropy over training, and k vs task phase (from the Analysis step).

**Stats:** 3 seeds, reported as mean ± SE with per-seed points shown. Make paired comparisons against DSRL per task. Don't claim significance from 3 seeds; report effect sizes.

## Compute estimates
These are a priori estimates. Nothing has been measured yet, so treat them as accurate to about 2×. The Step 0 pilot measures real throughput, and I'll re-plan before submitting Step 3.

**Cost model (DSRL-NA, one A100/H100, single process):**
- **Gradient updates dominate.** Each update is a critic step, a target backup that runs the full K-step sampler on a batch of 256, 10 noise-critic distillation steps that each also run the sampler, and an actor step.
- **Per-update time:** about 20 ms at K=8 (can, square) and about 35 ms at K≈15 (transport after DDIM reduction).
- **Updates per env step:** UTD 20 per vec-env step, 4 envs, and 4 env steps per chunk (8 for transport) gives about 1.25 updates per env step for can/square and about 0.6 for transport.
- **Eval overhead:** about 25%. That is 200 episodes per eval, every 3–8k vec steps.
- **Fixed-k cost is essentially the same for every k**, because the sampler always runs all K steps.
- **Adaptive k costs about 1.3× a fixed-k run.** The actor sums over K Gaussian heads, which only adds noise-critic forward passes and no extra sampler calls.

| Task | Env-step budget | Run-hours per run |
|---|---|---|
| can | 1M | ~9 h |
| square | 2M | ~18 h |
| transport | 3M | ~25 h |

At 100 DDIM steps, transport would be about 8× more, which is why Step 2 reduces K.

**Packing:** pack about 4 runs per GPU. The networks are small MLPs, so the GPU is mostly idle and 4 runs should get roughly 3× throughput. Each pack needs about 32–48 CPUs. Set `buffer_size` to the run's total chunk count instead of the default 10M; SB3 preallocates the buffer, and 10M would take 6–18 GB of RAM per run. Request about 64 GB per pack.

| Phase | Runs | Run-hours | GPU-hours (packed) | Wall-clock |
|---|---|---|---|---|
| 0. Setup + pilot (throughput, k=0 regression on can, 200k steps) | 2 | ~4 | ~4 | ~0.5 day, mostly install |
| 2. Base eval + transport DDIM-K choice + authority diagnostic | eval only | ~2 | ~2 | ~2 h, 1 job |
| 3a. Fixed-k scan, 1 seed (8 k × 3 tasks) | 24 | ~420 | ~140 | ~1.5 days (6 packs in parallel; transport sets the pace) |
| 3b. 3-seed follow-up (top 2–3 k + k=0, 2 extra seeds each) | ~24 | ~420 | ~140 | ~1.5 days |
| 4. Adaptive k: ρ pilot on can (2 runs) + 3 tasks × 3 seeds | 11 | ~230 | ~80 | ~1.5 days |
| 4b. Ablation: adaptive restricted to top-3 k (can + square, 3 seeds) | 6 | ~105 | ~35 | ~1 day |
| Lit baselines: constant-noise CEM, best-of-N eval | eval only | ~10 | ~10 | <0.5 day |
| Uniform-random-k control (3 tasks × 3 seeds) | 9 | ~200 | ~70 | ~1.5 days (packed with Step 4) |
| Residual SAC (β pilot on can + 3 tasks × 3 seeds; ~0.8× DSRL cost, no sampler in backups) | 11 | ~140 | ~45 | ~1 day |
| Trust-region ablation (can; k=0 and adaptive; 3 seeds) | 6 | ~60 | ~20 | ~0.5 day |
| **Total** | **~93 runs** | **~1,590** | **~545 GPU-h** | ~2.5 weeks if run phase by phase; **~4–5 days** with the waves below |

### GPU efficiency and parallelism (priority)
**Resource limits:**
- The kempner partitions (`kempner`, `kempner_h100`, `kempner_h200`, `kempner_rtx`; account `kempner_pehlevan_lab`) allow at most **16 GPUs at once across all jobs**. They are the primary pool.
- Overflow goes to `gpu`, `gpu_h200` or `seas_gpu` under `olveczky_lab`. In Step 0, check with `sacctmgr`/`sinfo` that `seas_gpu` accepts this account. These queues can be long, so send them only work that isn't on the critical path: the 3b extra seeds and the ablations.
- Every submission lists one partition. Before each submission, check free GPUs with `/slurm-opt` and pick the partition from that. Resubmit elsewhere if a job hasn't started after a few minutes.

**Getting the most out of each GPU.** All of this is measured in the Step 0 pilot and recorded in the exp doc:
1. **Profile a single run first.** Record steps/s, `nvidia-smi dmon` SM utilization, and the split between update time and env/eval time.
2. **Choose the packing factor from measurements.** Measure aggregate env-steps/s at 1, 2, 4, 6 and 8 runs per GPU. Pick the point where aggregate throughput stops rising, aiming for SM utilization above 70%.
   - Also try a user-level **CUDA MPS** daemon inside the job (`nvidia-cuda-mps-control -d`). It lets the packed processes run kernels concurrently instead of time-slicing.
   - Request CPUs to match: about 6–8 per run. Cut `n_eval_envs` to 10–16 if CPUs become the bottleneck.
3. **Speed up each update without changing the algorithm:**
   - Wrap the full update step (critic, noise-critic distillation, actor) in `torch.compile(mode="reduce-overhead")` (CUDA graphs). Shapes are fixed, so this should cut kernel-launch overhead, which is the main cost for small MLPs.
   - Turn on TF32 matmuls.
   - Keep the replay buffer on GPU so sampling avoids host→device copies.
   - Each of these gets a quick A/B on can. Keep only what is faster and leaves the k=0 regression check unchanged.
4. **If packing still leaves the GPU underused:** vectorize seeds within one process. Use a seed ensemble via `torch.func.stack_module_state` + `vmap` over S independent agents, each with its own envs. Treat this as an optional step, since it is more engineering.

**Schedule.** Phases overlap; only 3b and 4b depend on the sweep results. The figures below assume about 4 runs per GPU.
- **Wave 0 (gate, can only):** every method on can at 1 seed: 8 fixed-k runs, adaptive, uniform-k, residual (2 β values), CEM. That is about 13 runs on 4 GPUs, about 12 h. It catches pipeline bugs and sets the can budget before the expensive tasks.
- **Wave 1:** square and transport for the fixed-k scan (16 runs), plus can/square/transport × 3 seeds for adaptive, uniform-k and residual (~27 runs). About 43 runs on ≈11 kempner GPUs, about 1.5–2 days; transport sets the pace.
- **Wave 2:** the 3b extra seeds (~24), 4b (6), and the trust-region ablation (6). About 36 runs on ≈9 GPUs. Kempner is used first; extra seeds overflow to `gpu`/`gpu_h200`/`seas_gpu`. About 1.5–2 days.
- **Total GPU wall-clock: about 4–5 days**, compared with about 2.5 weeks done one phase at a time. Implementation (Step 1) and analysis come on top of that.

**Walltime:**
- Packed transport runs may slow to about 1.5× a single run, which is about 38 h. Request `--time=3-00:00:00` for transport packs and 1–2 days for the others.
- Add checkpoint/resume support: save the SB3 model and replay buffer every eval, so a timed-out or preempted pack can restart.

**Biggest uncertainties:**
1. Whether DSRL-NA actually needs the 1–3M env-step budgets on these tasks. If can converges by 300k steps, the can rows shrink about 3×.
2. The per-update time at batch 256 with 10 distillation steps.
3. How much success transport loses at reduced DDIM K.

The pilot pins down (2) directly, and the first can sweep pack pins down (1) before the larger tasks are launched.

## Documentation
Each phase gets a living document at the project root, following the CLAUDE.md template: `exp_1_base_eval_<date>.md`, `exp_2_fixed_k_sweep_<date>.md`, `exp_3_adaptive_k_<date>.md`. Each is created before its jobs are submitted, and progress entries include SLURM job IDs. Add a project `CLAUDE.md` that lists the ignored directories (`wandb/`, `logs/`, `outputs/`, `dsrl/dppo/log/`).

## Verification
- `pytest tests/` passes, especially the k=0 equivalence and eps_hat reconstruction tests.
- `train_adsrl.py steer.mode=fixed steer.k=0` on can reproduces the `train_dsrl.py` learning curve within seed noise over the first ~200k steps. This is a regression check against upstream DSRL.
- Adaptive on can with α_k frozen high (near-uniform k) trains stably, and its k histogram is logged.

## Decisions log
- [2026-11-03] Transport actor/critic width: 3×2048 (same as other NA configs; upstream SAC transport used 128).
- [2026-11-03] Uniform-random-k control: deterministic eval samples k uniformly (argmax of uniform would collapse to k=0); w uses the mean.
- [2026-11-03] Replay buffer: FastReplayBuffer (uniform randint, wrap-safe) replaces the SB3 fork's BaseBuffer.sample, which is O(N) per call and crashes after wrap-around.
- [2026-11-03] Pilot results (exp_0): can = 45 env steps/s single run on H100 (15.7 ms/update, 89% of time in updates, eval ~3%). Packing only helps with CUDA MPS: 6 runs/GPU = 3.8x aggregate. Standard pack: kempner_h100, 6 runs/GPU + MPS, -c 24 --mem 96G, n_eval_envs=10. 16-core kempner partitions: 4 runs/GPU. Can ≈ 10.3 h wall per 1M-step run at 6/GPU ≈ 1.7 GPU-h (plan assumed ~3). Revised total ≈ 310 GPU-h (was ~545). Never request extra GPUs just to get CPUs; exclude node holygpu8a12204 (ECC errors). seas_gpu: olveczky_lab OK, but ~1 day queue.
- [2026-11-03] Regression (exp_1): x_T must stay out of Q(s,a). With it in every input, k=0 learned at 0.18 vs upstream 0.49. With it removed, 0.53. New key `steer.xT_inputs: auto|none|actor_noise_critic|all`, default `auto`: `none` at fixed k=0, `actor_noise_critic` otherwise (actor + Q^W see x_T, Q(s,a) never does). At k=4 this doubles success (0.55 vs 0.27). All pre-fix cross-k results are void.
- [2026-11-03] Transport DDIM K stays 20 (0.195 vs 0.210 at K=100). K=10 is −4.2 ± 2.2 pts, with its 95% CI reaching −8.7, so it is too close to the 5-pt rule. 
- [2026-11-03] Wave 0 budget: can 1M env steps. Measured ~28 env steps/s per run with evals at 6/GPU, so ≈10 h/run.
- [2026-11-04] User: launch nothing beyond Wave 1 (square + transport + can seeds 2–3) and the base re-check until the initial results report is out. Wave 2 (extra fixed-k seeds, top-3-k, trust region, ρ seeds), the residual β sweep, best-of-N, CEM on square/transport, and harder tasks are all deferred.
- [2026-11-04] User approved one exception to the no-new-runs rule: a noise-bound training sweep on can (fixed k ∈ {0,4,6,7} × w_scale ∈ {1.5,2,3}, seed 1, 1M steps). It tests whether a wider bound gives late steps back their control. Doc: exp_4_bound_sweep_can_2026-11-04.md.
- [2026-11-04] User: compare mainly against DSRL. DPPO is removed from all figures and baselines and mentioned in one sentence of context. Main comparisons are DSRL, fixed k, random k, adaptive, and the no-RL references. Residual RL and fixed-noise CEM are secondary baselines.
- [2026-11-04] Wave 1 (exp_3): added `checkpoint_every_min` (default off; Wave 1 uses 120) because transport only checkpoints every 256k env steps. Generic pack script: `scripts/slurm/exp3_wave1.sbatch <task> <run names…>`. Transport yaml default `ddim_steps` changed 100→20 to match the exp_1 decision; Wave 1 already passed 20 explicitly.
- [2026-11-04] Residual SAC β=0.3 starts near 0 success on square/transport (an untrained actor mean of β·tanh(μ) pushes actions off the base policy). Candidate fix, deferred until after the report: zero-init the actor's last layer.
