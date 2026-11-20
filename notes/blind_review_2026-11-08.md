# Blind review: adaptive-step DSRL (re-noise steering at a chosen denoising step)

Reviewer: independent, read-only pass over the plan, exp_0–exp_9 logs, notes/, report/, adsrl/, adsrl_pi0/, and the raw eval_summary.csv / eval_rollouts npz under adsrl_runs/. Numbers below are deterministic-eval success, mean of the last 3 evals (200 episodes each on robomimic; 50 init states on LIBERO), unless stated.

## 1. Bottom line

The core hypothesis has two parts. (a) "Steering at a later step k beats k=0": on the only robomimic task that is not at ceiling (transport), one seed says yes by a small margin (k3 0.885, k5 0.868, k8 0.823 vs k0 0.793 at 6M), but the seed spread of the adaptive method on the same task (0.713 / 0.850 / 0.847) is larger than that margin, so this is unresolved. On can and square every k in 0..6 lands at 0.92–0.97 and only k=7 is clearly worse, so those tasks cannot discriminate. On π0/LIBERO every k>0 run lost badly, but that sweep is confounded twice (tiled noise, raw 1600-d x_T in the obs; exp_8 control confirms the second), so it says nothing about k. (b) "Adaptive k beats any fixed k": refuted in its strong form on all three robomimic tasks. Adaptive ties uniform-random k everywhere (can 0.934 vs 0.924, square 0.955 vs 0.955, transport 0.803 vs 0.806, 3 seeds each) and never beats the best fixed k. The learned π(k) also looks wrong: on transport it puts ~50% of its mass on k=16–18, steps that score 0.21–0.54 as fixed k; on π0 it puts ~45% on k=9. I think I can name the mechanism (Section 3.1) and it is fixable.

The most robust finding so far is not about RL at all: restarting denoising from x0_hat with zero added noise (w=0 at k≥1) raises can from 0.42 to 0.83 (1500 episodes), and a bounded tanh(N(0,I)) x_T raises it to 0.71. Much of what looks like "k>0 helps" is a sampler-temperature effect that the RL comparison does not yet control for.

## 2. What the evidence supports, and how strongly

| Claim | Evidence | Strength |
|---|---|---|
| Re-noise sampler is implemented correctly | tests: k=0 bit-identical to upstream; w=eps_hat reproduces unsteered DDIM; π0 k=0 equals openpi infer (exp_7). Regression at k=0 matches upstream once x_T is dropped from Q(s,a) (exp_1: 0.533 vs 0.493, 3 seeds). | Strong |
| Authority falls monotonically with k | results/authority (500 states × 64 w): can std 0.131→0.036; square 0.078→0.013; transport 0.088→0.012. | Strong (no training needed) |
| Later k learns faster early on can | first det eval ≥0.8: k1/k4/adaptive 216k, k0 408k (exp_2). | Weak: 1 seed, and the no-RL start differs by k (w=0 at k1 already 0.83) |
| Later k helps on transport | k3 0.885, k5 0.868 vs k0 0.793 at 6M; k3/k5 led from ~2.4M onward. | Weak: 1 seed; adaptive's seed-to-seed spread on this task is 0.14 |
| Very late k is worse | can k7 0.77, square k7 0.81, transport k14–19 0.21–0.54. Partly recovered by a wider bound (exp_4: k7 0.77→0.85 at c=3). | Moderate: consistent across tasks, 1 seed each |
| Adaptive k ≥ best fixed k | can 0.934 vs k1 0.957; square 0.955 vs k1 0.965; transport 0.803 vs k3 0.885. | Refuted at this budget (3 seeds vs 1) |
| Adaptive k > uniform-random k | ≤0.01 difference on every task, 3 seeds each. | Refuted |
| k choice tracks task phase | npz at final eval: can mean-k by chunk quartile 1.65→2.69 (s1), 1.47→1.79 (s2); transport no monotone pattern; π(k) differs a lot across seeds (can s1 peaks at k2, s2 at k1, s3 flat). | Not supported; looks like seed noise |
| Noise-space steering > residual action-space RL | residual SAC: can 0.46–0.56, square ≤0.10, transport 0.00. | Not interpretable: residual is mis-initialized (Section 3.4) |
| Constant noise (CEM) fails on this base | 0.09 on can. | Fine as a sanity check, not a tuned baseline |

Episode counts are adequate (200/eval, SE ≈ 0.03; mean of 3 evals ≈ 0.02 if evals were independent, which they are not quite, since the policy barely changes between them). Seeds are the limiting factor: every fixed-k conclusion rests on 1 seed. The eval-seeding bug (exp_2: `eval_env.seed()` does nothing, initial states come from unseeded worker RNG) means evals are unbiased but unpaired, so per-eval differences carry the full binomial variance of both arms.

## 3. Most important weaknesses

### 3.1 The hybrid SAC entropy term biases π(k) toward low-authority steps (method bug, likely explains the adaptive result)
`adsrl/agent.py:693-714`: the actor loss is Σ_k π(k)[α_w log π(w_k|k) − Q^W(s,k,w_k)] + α_k Σ_k π(k) log π(k). Each head has its own log_std. At a late k the decoded action barely depends on w, so Q^W(s,k,·) is flat and that head's optimum is maximum entropy (log_std → LOG_STD_MAX), making α_w log π(w_k|k) strongly negative. At an early k the head must keep w sharp to realize its Q advantage, so it pays an entropy price. The categorical therefore receives a per-k bonus of roughly α_w·[H(w|k)] that is largest where steering is least useful. The auto-tuned α_w targets Σ_k π(k) log π(w_k|k) ≈ target_ent (0 here), so the cheapest way to satisfy the entropy constraint is to shift mass to high-k heads rather than to explore w at useful k. Observed signature: transport adaptive chosen-k mass at k∈{16,17,18} is 0.51 / 0.39 / 0.42 across seeds although those k score 0.21–0.54 as fixed k; π0 adaptive puts ~45% on k=9 (t=0.1), the step with essentially no authority. The same structure exists in `adsrl_pi0/agent.py` (HybridPixelSACLearner). This is the known cross-branch entropy-comparability problem in hybrid action spaces (Neunert et al. 2020 handle it with per-branch constraints).

### 3.2 The comparison does not control for the no-RL sampler change
w=0 at k=1 gives 0.83 on can with no training; w~N(0,I) at k≥1 gives 0.51–0.67 vs base 0.42 (exp_2 base_eval_wk). Re-noising x0_hat (the posterior mean) rather than a posterior sample is a mode-seeking, lower-temperature sampler, as exp_2 correctly notes. So "fixed k>0 learns faster" is at least partly "starts higher". The missing control is k=0 DSRL with a matched low-temperature start (x_T scaled by c<1, or tanh-squashed as the untrained actor already does: 0.71). Without it the k-effect and the temperature effect are entangled.

### 3.3 Tasks and budgets cannot separate methods
Can and square saturate at 0.93–0.97 for every k in 0..6, adaptive, and uniform-k, so ~60 of the robomimic runs produced ties. Transport is the one informative task and has 1 seed per fixed k. The 3-seed budget was spent on adaptive/uniform/residual (which tie) rather than on the k=0 vs best-k contrast that the hypothesis actually needs. Exp_6 (furniture) was abandoned after ~a day of pilots; exp_7 (15 π0 runs, ~50 GPU-h) was invalidated by two confounds that a 30-minute no-RL eval would have caught first (the exp_7 team did this eval, but after the sweep).

### 3.4 The residual-SAC baseline is broken, not weak
exp_3 notes the untrained residual actor starts near 0 on square/transport because β·tanh(μ_init) with a random last layer pushes actions off-distribution. Standard residual RL zero-initializes the actor's output layer so Δa=0 at init (Silver et al. 2018; Johannink et al. 2019). As run, it is not evidence that noise space beats action space.

### 3.5 Reproducibility and bookkeeping
All project code is untracked in git (exp_8 resorted to sha256 of source files); run_info.json's commit hash identifies nothing. Eval seeding is ineffective (3.1 above). Transport configs say ddim_steps 100 while runs used 20 via override. These are cheap to fix and matter for any writeup.

### 3.6 The "more in-distribution" half of the hypothesis is never measured
The plan's Fig 3b (base model's denoising loss on executed actions) does not appear in results. Only ‖a−a_base‖ is logged. Re-noising x0_hat with a tanh-bounded, mean-shifted w is not the forward kernel either, so "late k is in-distribution" is an assumption, not a measurement. On π0 the tiled-noise finding shows how badly "in-distribution" intuitions can fail.

### 3.7 Representation handed to the actor
The actor and Q^W see raw x_T (28-d, or 1600-d on π0) and must learn the diffusion map to know what the base will do. exp_1's Next Steps already suggests feeding a_base(s,x_T) or x0_hat_{k−1}; exp_9 is trying this on π0 (abase feature), with promising early numbers (t6 k3 0.95 at 200k). On robomimic this was never tried, although it is the cheapest way to deliver the "better conditioned on what the base decided" part of the hypothesis.

## 4. Suggestions, prioritized

1. **Fix the entropy bias in the hybrid actor, then rerun adaptive on transport (3 seeds).**
   What: either (i) drop α_w log π(w_k|k) from the categorical's objective (gradient of π(k) from Q^W and α_k only; heads still get their own SAC entropy term), or (ii) measure the w-entropy in a k-invariant space, e.g. per-head entropy normalized by that head's max (subtract the LOG_STD_MAX baseline), or (iii) replace it by an action-space proxy (entropy of the decoded action, approximated by log|det| or by ‖∂a/∂w‖ from the authority table). Also log per-head mean log_std by k every eval.
   Why: Section 3.1. Prediction: π(k) mass moves away from k≥14 on transport; if adaptive then exceeds uniform-k by >0.05 the learned choice matters; if it still ties, adaptivity has no value at this authority structure and the paper's claim 3 should be dropped.
   Cost: ~1 h code + tests, 3 transport runs (~75 GPU-h at 3M, or reuse 1M can for a 2-h smoke first).

2. **Add the temperature-matched k=0 control before any more fixed-k runs.**
   What: k=0 DSRL whose warm start and untrained actor produce the same no-RL success as the k being compared (e.g., x_T scaled by c∈{0.5,0.7}, or a tanh-squashed x_T; `steer.w_scale` already exists for c>1, add c<1). Report every fixed-k run as Δ over its own no-RL row (w=0 and w~N at that k, already in results/base_eval_wk).
   Why: Section 3.2. If k=0 at c=0.6 matches k=3/5 on transport, the k story collapses into a temperature story. Cost: 2–3 transport runs (~60 GPU-h) or 3 can runs (~6 GPU-h) for a first look.

3. **Spend seeds where the hypothesis lives: k=0 vs k=3 vs k=5 on transport, 3 seeds each**, with paired evaluation (fix eval seeding so all arms see the same initial states and x_T streams; `tests/test_eval_seeding.py` already fails on this). Stop at 4M: the 4.8M–6M curves are flat-ish and noisy. Cost: 6 new runs (~150 GPU-h). Confirm: mean gap ≥0.05 with non-overlapping seed ranges; refute: gap within seed SE.

4. **Feed the actor and Q^W a base-plan feature instead of raw x_T on robomimic** (a_base(s,x_T) for k=0, x0_hat_{k−1} for k>0; the sampler already returns x0 at every step). exp_9 is doing this for π0; do it for transport fixed k and adaptive.
   Why: Section 3.7; it directly implements "conditioned on what the base decided", and removes the 28/112-d noise input that already broke Q(s,a) once. Cost: small code change, 3 transport runs. Confirm: faster rise (steps to 0.5) and smaller seed spread.

5. **Measure in-distribution-ness directly.** Log the base model's denoising loss (or ‖eps_hat − w‖ at step k) on executed chunks per eval, and the fraction of w at the tanh bound. Cost: eval-only, hours. This turns Fig 3 from authority-only into the authority vs OOD tradeoff the plan promised, and would explain why k=7 benefits from c=3 while k=0 collapses at c=3.

6. **Repair the residual baseline** (zero-init last layer, β=0.3, same warm start) and rerun 1 seed per task before citing it. Cost: ~15 GPU-h. Also run the planned best-of-N (Q^A-scored) eval add-on on k=0 and best-k checkpoints; it is cheap and tests whether the actor is the bottleneck or the critic is.

7. **Cheaper adaptive design worth one pilot: choose k from a small set with a per-k w_scale c_k** chosen so each step's authority matches (exp_4's table gives c_k: k4≈2, k6≈3, k7≈3). The exp_4 result (k4 c2 0.965, k6 c3 0.923, k7 c3 0.852, all above c=1) suggests the late-k deficit is mostly authority. Prediction: with matched authority, fixed-k curves flatten further and adaptive loses its remaining rationale; if late k with matched authority still underperforms, the deficit is directional (late steps only edit details), which is a publishable mechanism.

8. **π0 line: finish exp_9 (lowfreq + abase) with k∈{0,3} on 3 tasks, 2 seeds, before any new adaptive runs there.** The structured/lowfreq layouts fix the actual bug; the adaptive actor there has the same entropy bias as (1). Add w=0-at-k and iid-N(0,I)-at-k rows to every table as the no-RL reference (already done in exp_7; keep it).

9. **Housekeeping (hours, not GPU):** commit the code; record the hash in run_info; fix eval seeding; set transport.yaml ddim_steps to 20; write the Results/Analysis sections of exp_2, exp_3, exp_5, which are still "Pending" although the data exist.

## 5. Literature and standard practice that seem missing

- **Hybrid discrete–continuous SAC references and their entropy handling:** Neunert et al. 2020 (continuous–discrete RL for hybrid control in robotics; separate KL/entropy constraints per branch), Delalleau et al. 2019 (discrete and continuous action representation, mixture entropies), PDQN / HPPO. They motivate fix (1).
- **DPPO's own ablation on fine-tuning only the last K' denoising steps** (ft_denoising_steps in the very checkpoints used here) is the closest precedent for "later steps are where adaptation should act", and it is about optimization conditioning rather than authority. Cite and contrast.
- **Low-temperature / reduced-noise sampling as a baseline** for diffusion policies (noise scaling, DDIM with scaled x_T, "mode-seeking" sampling). The w=0-at-k result is this effect; the RL comparison needs it as a control (suggestion 2).
- **Residual RL initialization** (Silver et al. 2018; Johannink et al. 2019; Policy Decorator / ResiP for diffusion bases) for the residual baseline.
- **Paired evaluation with common random numbers** for small-gap comparisons; the current unpaired evals waste roughly half the statistical power.
- The lit review in notes/ is thorough on noise-space parametrization and KL-vs-box (LP-DS, NSL, HyperNoise); the noise trust region exists in `adsrl/trust_region.py` but the planned k=0 vs adaptive ablation on can never ran. Given LP-DS's finding that the Lagrangian constraint, not the box, prevents drift, this ablation is worth the ~6 GPU-h.

## 6. What I would tell the team in one paragraph

The sampler and infrastructure are solid, and the no-RL diagnostics (authority, w=0 at k, tiled vs iid noise on π0) are the best work in the project. But the headline hypothesis is currently neither confirmed nor cleanly refuted: can and square are at ceiling, transport has one seed per k, and the adaptive actor has a structural entropy bias that pushes it toward useless late steps, which is consistent with it tying uniform-random k on every task. Before more sweeps, fix the hybrid-actor entropy term, add the temperature-matched k=0 control, and put 3 seeds on k∈{0,3,5} transport with paired evals. Those three things decide whether there is a paper about k at all.
