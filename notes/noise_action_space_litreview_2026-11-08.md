# Literature review: noise/latent action space for steering frozen diffusion/flow policies

Date: 2026-11-08. Scope: DSRL's π0 noise parametrization (32-d vector repeated over 50 steps, tanh box), and what other work says about (a) the shape and dimension of the noise action, (b) bounds versus KL/prior regularization, and (c) injecting noise at later denoising steps.

PDFs and extracted text are in `/n/netscratch/olveczky_lab/Lab/charleszhang/tmp/claude-64196/-n-holylfs06-LABS-olveczky-lab-Users-charleszhang-CS2243Project/0599ecae-81d1-44f9-8be1-9f0963d3fda5/scratchpad/litreview/` (`pdfs/`, `txt/`; scratch is purged after 90 days).

**How each paper was read.**
- **[FULL]**: I downloaded the PDF and read the relevant sections (method, the noise parametrization, ablations, appendix hyperparameters). I searched by keyword first, so most of these were not read cover to cover. DSRL is the exception: I read its method, Section 5.5/5.6, all of Appendices A–C, and the discussion.
- **[ABSTRACT]**: I only saw the abstract (via OpenAlex or arXiv).
- **[KNOWN]**: Background I know well. I only confirmed the arXiv ID.

Page numbers for DSRL are the PDF page numbers of arXiv v2 (25 Jun 2025).

---

## 1. DSRL itself: Wagenmaker et al. 2025, arXiv:2506.15799 (CoRL 2025) [FULL]

**Formal noise space.** §3, p.4: "we define π^W_dp : S×W → A the deterministic output of the denoising process at state s when input noise w initializes the denoising process, and denote by W := R^d the latent-noise space." In general, W has the same dimension as the full noise.

**π0 dimension and the repeat trick.** Appendix C.5, "Training details", p.25. This is the only place the repeat choice is described:
> "The π0 policy is trained in a noise and action space with a chunk size of 50, where each noise and action is 32-dimensional, resulting in a total dimensionality of 1,600. Naively training the actor π^W(s) and critic Q^W(s,w) is challenging due to this high dimensionality. Here, we propose a simple yet effective approach to train DSRL-SAC on large action chunks. To avoid training the actor and critic on the whole noise space W = (R^d)^C … we instead train a single-step actor π^W_single(s) and critic Q^W_single(s, w_single) where W_single = R^d. At inference time, we simply sample w_single ∼ π^W_single and repeat it across the action chunk axis w = {(w1,…,wC) | wi = w_single}, and use it to query the π0 policy. We find that this makes SAC training more efficient and stable, while the repeated-noise formulation remains expressive enough to steer π0 to produce effective actions. This shows that the noise space of pre-trained π0 policy is a well-structured latent space, highlighting another advantage of running RL in this latent-noise action space."

The main text (§5.5, p.10) only says: "Steering π0 is non-trivial due to its high-dimensional action space—π0 uses an action chunk size of 50 and each action is 32-dimensional, resulting in a total action space dimension of 1,600."

For real-world π0-DROID (C.5, p.25): "We adopt the same method to handle large action chunks as is described above, though note that this π0-DROID variant we used only has an action chunk of size 10."

**Bounds.** Appendix B, "Best Practices", pp.19–20:
> "Action magnitude: When transforming the MDP from M to M^W, one design choice is how large a magnitude the actor can play in W. If π_dp is trained by adding noise sampled from a standard Gaussian to the demonstration actions, as is typical, then only w reasonably likely under a standard Gaussian would be 'in-distribution' for π_dp(s,w)—π_dp would have been unlikely to have denoised an example corresponding to w outside this range during training, and therefore its behavior on such examples may be unreliable. To incorporate this, we simply constrain the actions to a rectangle of some width b_W, i.e. in practice we set W ← [−b_W, b_W]^d. We found that choosing b_W in the range [1, 3] generally worked well for online RL and [0.5, 1.5] for offline RL."

**Values of b_W used:**

| Setting | b_W |
|---|---|
| π0, Libero | 1.0 |
| π0, AlohaTransferCube | 2.0 |
| π0, real-world DROID | 2.5 |
| Robomimic Lift / Can / Square | 1.5 |
| Robomimic Transport | 1.0 |
| Gym (Hopper / Walker / HalfCheetah) | 1.5 / 2.5 / 1.5 |
| WidowX | 1.5–2 |
| OGBench offline, tuned per task | 0.5–1.5 |

Sources: Table 11 p.26 (π0), Table 4 p.21 (Robomimic), Table 5 p.21 (Gym), Table 10 p.24 (WidowX), Table 7 p.22 (OGBench). π0 runs use target entropy −d/2 in sim and 0 in real (Table 11).

**Smaller policies (DPPO/robomimic, Gym, OGBench).** These use the full noise space with no tiling. W = [−b_W, b_W]^d, where d = action chunk × action dim. Robomimic uses chunk 4 (Transport 8) with 7-d actions, so d is about 28 (56 for Transport). The noise dimension for these is never stated explicitly as a number. Golden Ticket (below) states "7×4 = 28D in Can MLP".

**Ablations.**
- There is no ablation of repeat versus full noise for π0. There is no ablation of b_W, and no comparison of a box versus other regularizers.
- The ablations that do exist (§5.6 and App. A, all on Robomimic Can) vary base-policy width, data quality, training epochs, and train/inference denoising steps (Figs 11–15). None of them touch the noise parametrization.

**Stated limitations and open questions.** §6, p.11 asks: "What are the theoretical properties of diffusion steering? Can we make any claims about the expressivity of the noise-space optimization we employ or the structure of the diffusion policy's noise space and ease of learning a policy over it?" The paper never says that repeated noise is out of distribution for π0. The only distributional argument it makes is the box rationale quoted above. Note that the box is per coordinate, so it does not address the repeat structure.

**Intermediate-step injection.** The paper says nothing about injecting noise after the initial step. For the baselines only (C.1, p.20): "we experiment with injecting additional noise into the DDIM denoising process to help induce additional exploration."

**Relevance to us.** The repeat trick is justified only by SAC stability and by results. The authors call it "expressive enough". They give no in-distribution argument for it, and no ablation.

---

## 2. Noise action-space choices in follow-ups and related work

### 2a. Robot-policy noise-space RL and search (DSRL citers, 2025–2026)

**RLMM-Flow** (Wang et al., arXiv:2607.26460) [FULL]. This is the closest prior work to our decomposition.
- **Setup.** A frozen flow policy for mobile manipulation, with 50×9 actions.
- **"Coarse-to-fine latent steering".** The actor outputs a full H×d_w Gaussian latent. Then:
  - w̄ = AvgPool_H(w_raw), and w_coarse = Repeat(w̄, H).
  - w_res = w_raw − w_coarse, described as the "mean-centered residual".
  - Before step t_cf = 50k, only w_coarse is fed to the policy. After that, w_coarse + λ_res·w_res is used, with λ_res = 0.1 (Alg. 1, Eqs. 3–7, App. hyperparameters).
- **Why.** The authors' rationale: "Its effective search dimension therefore remains d_w, rather than H·d_w." They also note that full-dimensional steering gives "temporally inconsistent corrections".
- **Ablation.** Offline RL, seen-scene success (Table 1): DSRL-NA with horizon-shared 1×d_w noise gets 40%; coarse-to-fine alone gets 43%; together with critic warm-up it gets 46%.
- **Differences from our variant.**
  - The orthogonal residual is agent-controlled and scaled down by 0.1. It is not fresh Gaussian.
  - The coarse vector is not variance-corrected. Average-pooling an N(0,I) latent shrinks its variance by 1/H. Repeating it then puts all the energy in the time-mean subspace and none in the complement.
  - There is no KL regularizer. Instead, the latent critic is trained on a mixture q = ρN(0,I) + (1−ρ)π, with ρ = 0.5.

**Noise Space Learning for VLA Adaptation (NSL)** (Lu, Zhang et al., arXiv:2610.13943) [FULL]. This paper has the most direct evidence on bounds and KL.
- **Parametrization.** App. B.1 says it follows DSRL: "separate 32-dimensional mean and log-standard-deviation heads … The sampled noise vector is repeated across the 50-step prediction horizon. In the main SAC setting, the noise is bounded to (−1, 1) after the initial Gaussian exploration batch." Actors are diagonal Gaussians with "prior-matched initialization". PPO and AWR use unbounded noise.
- **Theory (§3, App. A).**
  - The adaptation cost is defined as C_NSL = E_s KL(π_z(·|s) ‖ ρ), with ρ = N(0,I).
  - By the chain rule and data processing, this cost is ≥ the behavioral KL(q_a ‖ μ_a). The minimum over noise distributions equals the behavioral KL (Eq. 5). They call this "matches the behavioral information lower bound".
  - App. A.7 gives the closed-form KL for a diagonal Gaussian and warns: "Entropy contains no mean penalty and therefore does not control mean drift relative to the pretrained prior". It also warns that a moving reference (as in PPO clipping) does not bound KL to the fixed prior.
  - A degenerate (deterministic) noise policy has infinite KL to a Gaussian prior (App. A). The repeated noise is a degenerate distribution on a 32-d subspace of R^1600, so this applies to it as well.
- **Ablation (Table 2).** Removing the noise limit:
  - SAC on SmolVLA: 93.4% → 32.0%.
  - SAC on π0.5: 66.6% → 56.6%.
  - PPO and AWR are unaffected because they are already unbounded.
  - Fig. 9: without the limit, actor means drift away from initialization and Bellman targets overshoot the return bound. They describe this as "value-driven outward drift in noise space".
- **Real robot (App. B.5).** Bound 5·tanh. Actor outputs pushed to the boundary (max 4.96), and twin-Q disagreement on actor proposals was 6× that on replay data. They had an unintended table contact. Their conclusion: "limiting the noise amplitude alone does not ensure reliable decoded actions".
- **Uncertain.** PPO uses a "reference-KL coefficient 0.005" (Table 4). I could not confirm whether the reference is N(0,I) or the old policy.

**Lagrangian Perturbation Diffusion Steering (LP-DS)** (arXiv:2606.01151) [FULL]. A KL-to-prior trust region.
- **Parametrization.** w = ε + Δ_θ(s, ε), with ε ~ N(0,I). The residual network sees the fresh noise.
- **Constraint.** E_s KL(q_θ(·|s) ‖ N(0,I)) ≤ δ, following Eyring et al. 2025 (HyperNoise). In the small-Jacobian regime this is approximated by ½E‖Δ‖² ≤ δ (Eqs. 5–7). It is enforced with an adaptive Lagrange multiplier (Eqs. 8–10), with δ = 0.35 in most tasks.
- **Ablation on Adroit Pen (§5.3, Fig. 6).**
  - "Removing the Lagrangian penalty leads to unstable learning and degraded success, while removing the noise bound alone does not cause severe performance degradation"
  - Removing both gives "extreme regions of the steering space".
  - Their conclusion: "while hard bounding can improve stability, the Lagrangian trust-region constraint is the primary mechanism preventing out-of-distribution steering."
- **Other results.** LP-DS keeps higher action entropy than DSRL, which collapses to a single mode (Figs 4–5). Fig. 2 shows DSRL drifting to large ‖w‖ that "correlate with low-prior-density decoder inputs".
- **π0.** Includes a π0 LIBERO "cream cheese" result (Fig. 8). The paper does not describe how it handles π0's 1600-d noise. **Uncertain** whether it tiles; possibly it inherits DSRL's code.

**You've Got a Golden Ticket** (Patil et al., arXiv:2603.15757) [FULL]. A constant noise found by search, full-dimensional.
- **Method.** Random search and CEM over a single constant initial noise in the full 1600-d π0 space. CEM starts at μ = 0, σ = 1. No state conditioning.
- **Criticism of tiling.** "DSRL instead tiles the noise space, which departs from the i.i.d. pretraining noise and removes all temporal variation within the chunk."
- **Evidence.**
  - "DSRL on the full 1600D noise (dashed) fails to improve on the base policy" (Fig. 6). This is the only ablation I found of full versus tiled SAC on π0.
  - Random search over constant 1600-d noises matches tiled-space actor-critic on LIBERO π0.
  - Ablation (Fig. 8b): "Tiling the noise vector along the horizon leads to worse performance due to a smaller search space" for CEM.
- **Relevance to us.** For search, the full space is better. For SAC, tiling is needed: full-dimensional SAC fails. This tension is exactly what our structured variant tries to resolve.

**AnchorQ** (Qi et al., arXiv:2610.22852) [FULL, relevant parts].
- **Setup.** π0.5. The noise action is the full d_z = h_a·d_a. Anchors come from flow-reversing coarse reference actions, and local particles z + σ_p·ξ with ξ ~ N(0,I) are value-tilted.
- **Norm analysis (Fig. 4b, §4.3).** "The noise anchors sit at an intermediate magnitude, between the near-zero noise DSRL collapses to and π0.5 prior". Also: "DSRL is a continuous noise-space actor regularized toward the π0.5 prior; however, it collapses to a near-zero noise vector."
- **Relevance to us.** Useful evidence that ‖w‖ for DSRL ends up far from the χ shell (√1600 ≈ 40). **Uncertain** how DSRL was configured there; "regularized toward the prior" may be their own modification.

**FlowDAgger** (Murray et al., arXiv:2607.08877) [FULL, App. A.3].
- Noise targets come from action inversion and are fit by supervised regression on the full chunk noise.
- It notes that DSRL "collapses π0's 1,600-dimensional chunk noise to a single per-step vector repeated across the chunk to keep SAC stable."
- For world-action models (~10^5-d noise), they regress k = 64 PCA coefficients fit on inverted noises: "somewhat more stable". This is a low-rank, data-driven basis.

**Beyond Noise Steering: DLSRL** (arXiv:2609.11270) [FULL, skimmed]. Adds a second latent that modulates intermediate action-token hidden states through adapters, alongside initial-noise steering. It modulates features, not noise, at intermediate steps. I did not extract its noise dimension or bound.

**UniSteer** (arXiv:2605.10821) [FULL, skimmed] and **Flow Reversal Steering** (arXiv:2606.13675) [ABSTRACT]. Both invert actions to noise targets for a noise actor. UniSteer uses a π0-architecture backbone. I did not extract the noise shape or bound from UniSteer.

**RFS: Residual Flow Steering** (arXiv:2602.01789) [FULL, skimmed]. Joint latent-noise steering plus a residual action. Ablations show that DSRL-only latent steering is weaker on out-of-distribution states. The noise shape is not material for us.

**Abstract-only papers:**
- **SCORE** (arXiv:2606.27475): flow steering as a support constraint.
- **AWLS** (arXiv:2610.27736): an "observation-conditioned residual correction to the policy's latent noise".
- **SteerGenPO** (arXiv:2606.16572), **ZPRL** (arXiv:2605.19919, which perturbs a VIB bottleneck latent instead of noise), **GoldenStart** (arXiv:2603.14245, a CVAE Q-guided prior), **ReFORM** (arXiv:2602.05051, a bounded source distribution and a reflected flow that generates bounded noise).
- **LASER** (arXiv:2610.08989): its abstract says "entropy regularization is essential in latent-space RL" to avoid collapse onto critic artifacts.
- **WarmPrior** (arXiv:2605.13959) and **LeaP** (arXiv:2606.17408): these change the source distribution at training time (temporal or proprioceptive priors), so they cannot be used with a frozen π0.

### 2b. Older latent-action RL (non-diffusion)

- **PLAS** (Zhou et al., arXiv:2011.07213) [FULL, method]. Offline RL in a CVAE latent space with prior N(0,1): "we simply define z = π(s) such that z_i ∈ [−σ, σ] for each latent dimension". This is the direct ancestor of DSRL's box.
- **Parrot** (Singh et al., arXiv:2011.10024) [FULL, method]. This is DSRL's ref [70]: RL over the base-Gaussian space of a normalizing-flow behavioral prior. The policy is initialized to p_z. The flow is invertible, so there is no expressivity loss; DSRL contrasts itself with this.
- **SPiRL** (Pertsch et al., arXiv:2010.11944) [KNOWN]. Uses a KL to a learned skill prior instead of a box. This is the classic precedent for "KL instead of a box" in latent-action RL.

### 2c. Initial-noise optimization for image/audio generators

- **ReNO** (Eyring et al., arXiv:2406.04312) [FULL, §3]. Uses the full-dimensional noise. It regularizes the norm toward the χ_d shell: "maximize the log-likelihood of the norm of a noise sample K(ε) = (d−1)log‖ε‖ − ‖ε‖²/2", "as otherwise G_θ might provide unwanted generations". It also uses gradient clipping.
- **D-Flow** (Ben-Hamu et al., arXiv:2402.14017) [FULL, §3.3]. Explicitly rejects the Gaussian negative log-likelihood ½‖x0‖², because it pulls x0 "towards the most likely all zero mean but far from most of the probability mass at norm √d". It uses the χ_d norm likelihood instead (Eq. 10). It also argues that differentiating through the flow implicitly projects onto the data manifold.
  - **Relevance to us.** A KL or ½‖z‖² penalty on a low-dimensional z is fine, but a ½‖w‖² penalty on a full 1600-d deterministic w would shrink it toward 0, which is atypical. AnchorQ's observation that "DSRL collapses to near-zero noise" fits this.
- **InitNO** (Guo et al., arXiv:2404.04650) [FULL, method]. Optimizes a distribution, ε′ = μ + σ⊙ε, with a "distribution alignment loss" KL(N(μ,σ²) ‖ N(0,1)). The authors report that without it, "the optimized noise may deviate from the standard Gaussian distribution, resulting in distorted results" (Fig. 9, t-SNE). This is a direct precedent for a KL-to-N(0,I) regularizer on a learned Gaussian over noise.
- **Noise Hypernetworks / HyperNoise** (Eyring et al., arXiv:2508.09968, NeurIPS 2025) [FULL, §3]. The theoretical basis for KL in noise space.
  - Residual x̂0 = x0 + f_φ(x0). The loss is KL(p^φ_0 ‖ N(0,I)) − E r.
  - The KL is exact up to a log-det term bounded under Lipschitz f_φ, so KL ≈ E ½‖f_φ(x0)‖² (Eq. 10).
  - The data processing inequality gives KL(noise) ≥ KL(outputs) (Eq. 12). So the noise-space KL upper-bounds deviation from the base output distribution.
  - LP-DS borrows this approximation.
- **Inference-time scaling beyond denoising steps** (Ma et al., arXiv:2501.09732) [FULL, §3]. Searches the full noise via random search, zero-order search over a neighborhood of a pivot noise, and "Search over Paths". The last one re-noises intermediate samples with "M i.i.d noises" and re-simulates the forward noising process.
- **Abstract or reference only:** Golden Noise (arXiv:2411.09502), "A noise is worth diffusion guidance" (arXiv:2412.03895), and Mao et al., "Lottery ticket hypothesis in denoising" (ECCV 2024; I did not look up an arXiv ID).

### 2d. Structured (correlated or low-frequency) noise

- **PYoCo** (Ge et al., arXiv:2305.10474) [FULL, §4].
  - Mixed noise model: ε_i = ε_shared + ε_ind,i, with ε_shared ~ N(0, α²/(1+α²) I) and ε_ind ~ N(0, 1/(1+α²) I). Each frame is marginally N(0,I), and frames are correlated through the shared component.
  - **Relevance to us.** This is the same "shared component + independent fill" algebra as ours. But PYoCo uses it at training time, fine-tuning the model on that prior. A frozen model trained on iid noise sees correlated noise as out of distribution. That is why our variant keeps the joint law exactly N(0,I) instead of correlating frames.
- **FreeInit** (Wu et al., arXiv:2312.07537) [FULL, §3]. For a frozen video model at inference, it re-initializes the noise by combining the low-frequency components of a re-noised latent with the high-frequency components of fresh Gaussian noise. It motivates this with an SNR analysis showing low-frequency information leaking through the training noise schedule.
  - **Relevance to us.** This is the closest prior art for "controlled low-frequency component + fresh Gaussian for the rest" on a frozen model. Our time-mean is the DC component of a temporal DCT. The difference: FreeInit's low-frequency part is not constrained to be N(0,I) distributed, and FreeInit is not RL.

---

## 3. Noise injected at later denoising steps

I found no paper that injects RL-chosen noise at an intermediate denoising step of a frozen robot policy. The closest are:
- DLSRL, which uses an RL-chosen intermediate feature modulation, not noise.
- QPILOTS (arXiv:2606.14801, [ABSTRACT via search]), which uses gradient Q-steering at each step, not noise selection.
- ReinFlow (arXiv:2505.22094, [FULL §4.3]), π-StepNFT (arXiv:2603.02083, [ABSTRACT]) and Flow-GRPO-style methods. These inject iid Gaussian noise at every step with a learned or scheduled variance, but they fine-tune the weights, and the policy controls the variance, not the noise realization.

**Theoretical and empirical support for the noise distribution at an intermediate step.** All of it points to fresh iid Gaussian noise drawn from the correct perturbation kernel.

- **SDEdit** (Meng et al., arXiv:2108.01073) [FULL, §3]. Re-noise a guide to time t0 with x(t0) = α(t0)x + σ(t0)z, z ~ N(0,I), then solve in reverse. Larger t0 means more realistic but less faithful output.
- **Restart sampling** (Xu et al., arXiv:2306.14878) [FULL, §3–4].
  - The forward jump x_tmax = x_tmin + ε_{tmin→tmax}, where "the noise ε … is sampled from the corresponding perturbation kernel". They place it late in the trajectory, where accumulated error is largest.
  - Theorem 1: the fresh stochasticity contracts the total-variation error accumulated up to t_max, by a factor (1 − λe^{−U}) < 1.
  - The contraction argument depends on the added noise having the kernel's law, so it says nothing in favor of structured or deterministic injected noise.
- **Adjoint Matching** (Domingo-Enrich et al., arXiv:2409.08861) [FULL, §4.2–4.3].
  - For reward-tilted fine-tuning with a KL penalty, the optimum is p*(X0, X1) ∝ p_base(X0, X1)·exp(r(X1) + V(X0, 0)). This yields the tilted distribution only under a "memoryless" noise schedule, where X0 is independent of X1 (Def. 1). They show this schedule is necessary and sufficient.
  - **Relevance to us (my inference).** If a controller acts mid-trajectory on a deterministic ODE, the result is biased by the initial value function V(X0, 0). A KL-regularized noise policy at the initial step does not suffer from this: the noise-space KL plus the data processing inequality (HyperNoise, NSL) give a clean bound.

**What follows for our case (my derivation, not taken from a paper).** Use π0's convention x_τ = τ·ε + (1−τ)·a, with ε ~ N(0, I_1600) iid. At every τ the network was trained only on noise components that are iid isotropic.

**The repeated noise is far from that distribution.**
- Under iid noise, the time-mean of each of the 32 channels is distributed N(0, 1/50), so its std is about 0.14.
- Under DSRL's repeat, the time-mean of each channel equals z_j. With |z_j| up to 1–2.5, that is about 7–18 std too large.
- The 1568-d time-varying complement is exactly 0, where iid noise would have norm about 39.6.
- The repeated noise distribution is supported on a 32-d subspace, so its KL to N(0, I_1600) is infinite.
- Injecting this at a later τ puts a constant (DC) offset of size τ·z into a partially denoised chunk, with no high-frequency noise. Nothing like that occurs in training. SDEdit, Restart and FreeInit all imply the injected noise should instead be fresh Gaussian drawn from the kernel.

**Our structured parametrization avoids this.** Take w = u zᵀ + (I − u uᵀ)ξ, with u = 1_50/√50 and ξ iid N(0,I):
- If z ~ N(0, I_32), then w ~ N(0, I_1600) exactly.
- If z ~ π(·|s), then KL(law(w) ‖ N(0,I)) = KL(π(·|s) ‖ N(0, I_32)) exactly, because the complement is identical in both. By the data processing inequality, this also bounds KL(steered action chunk ‖ base action chunk) (NSL Eq. 5; HyperNoise Eq. 12).

**Caveat.** In the structured variant, z = 1 shifts each channel's time-mean by only 1/√50 ≈ 0.14. That is about 7× less DC authority than DSRL's repeat at the same |z|. To match DSRL's steering effect, z needs about 7σ, which costs KL ≈ ½·32·49 ≈ 780 nats. So DSRL's effectiveness may partly rely on deliberately out-of-distribution, large-DC inputs. This is untested. Ablate the authority (for example, scale z or anneal the KL weight) rather than assuming equivalence.

---

## 4. Summary table

| Paper (arXiv) | Read | Noise action space | Bound / regularizer | Evidence |
|---|---|---|---|---|
| DSRL 2506.15799 | FULL | π0: 32-d repeated ×50; small policies: full chunk noise | tanh box b_W ∈ [1,3] online, [0.5,1.5] offline | No ablation of either choice |
| RLMM-Flow 2607.26460 | FULL | time-mean (repeated) + agent residual ×0.1, switched on later | none (critic on 0.5·N(0,I) mixture) | 40% → 43% → 46% vs DSRL-NA |
| NSL 2610.13943 | FULL | 32-d repeated (π0 / π0.5 / SmolVLA) | tanh 1 (SAC); unbounded PPO/AWR; KL theory | SAC w/o limit 93.4 → 32.0 |
| LP-DS 2606.01151 | FULL | w = ε + Δ(s,ε), full dim (π0 handling unclear) | Lagrangian ½E‖Δ‖² ≤ δ (≈ KL) | Lagrangian > bound; less collapse |
| Golden Ticket 2603.15757 | FULL | constant full 1600-d (search) | none (CEM σ init 1) | full-dim SAC fails; tiled search worse |
| AnchorQ 2610.22852 | FULL | full d_z, anchors + Gaussian particles | KL-tilt over particles | DSRL ‖z‖ collapses near 0 |
| FlowDAgger 2607.08877 | FULL | full chunk (π0.5); PCA-64 for WAMs | supervised | PCA "somewhat more stable" |
| HyperNoise 2508.09968 | FULL | full residual on noise | KL(noise ‖ N(0,I)) ≈ ½‖f‖²; DPI | theory + image results |
| ReNO 2406.04312 / D-Flow 2402.14017 | FULL | full noise | χ_d norm likelihood | D-Flow: Gaussian NLL pulls to 0 (bad) |
| InitNO 2404.04650 | FULL | μ + σ⊙ε | KL(N(μ,σ²) ‖ N(0,1)) | without it, distorted outputs |
| PYoCo 2305.10474 | FULL | shared + independent noise, marginal N(0,I) | (training-time prior) | algebraic precedent |
| FreeInit 2312.07537 | FULL | low-freq from re-noised latent + fresh high-freq | none | frozen-model precedent |
| SDEdit 2108.01073 / Restart 2306.14878 | FULL | fresh iid kernel noise at intermediate t | n/a | Restart TV-contraction theorem |
| Adjoint Matching 2409.08861 | FULL | per-step control, memoryless schedule | KL | value-bias theorem |
| PLAS 2011.07213 / Parrot 2011.10024 | FULL (method) | CVAE latent box / NF base space | box [−σ,σ] / init to prior | — |
| SPiRL 2010.11944 | KNOWN | skill latent | KL to learned prior | — |
| DLSRL, UniSteer, RFS | skimmed | noise + feature / residual | — | not about noise shape |
| SCORE, AWLS, LASER, ReFORM, WarmPrior, LeaP, ZPRL, GoldenStart, SteerGenPO, QPILOTS, π-StepNFT | ABSTRACT | various | LASER: entropy reg essential | — |

---

## 5. Bottom line

- **Our exact variant is new as far as I found.** No published work combines all three pieces: an agent-controlled time-mean component, an exact fresh-Gaussian orthogonal complement, and a KL-to-N(0,I) penalty.
- **Closest prior work:**
  - RLMM-Flow uses the same mean-plus-residual decomposition, with an agent-controlled residual and no variance correction.
  - PYoCo and FreeInit use the "shared/low-frequency component + independent Gaussian fill" algebra, but at training time and in image/video respectively.
- **Support for KL over a tanh box:**
  - LP-DS ablations show the KL/trust-region constraint, not the hard bound, is what prevents off-prior drift and collapse.
  - InitNO and HyperNoise show KL to N(0,I) on a learned noise distribution works and has a data-processing-inequality guarantee.
  - NSL shows that SAC in noise space drifts outward without some constraint, and that entropy alone does not control mean drift.
- **Support for the structured shape:**
  - Golden Ticket and NSL independently criticize or replicate tiling. Golden Ticket shows full 1600-d SAC fails on π0, while full-dimensional search beats tiled search.
  - So a low-dimensional agent action that keeps the joint noise exactly N(0,I) is well-motivated.
  - Our construction makes the noise-space KL equal to the 32-d KL on z, so it is tractable and exact. DSRL's repeat has infinite KL.
- **Later-step injection:** SDEdit, Restart and Adjoint Matching all imply that injected noise should follow the iid perturbation kernel, which favors the structured/fresh-ξ variant over repeated noise.
- **Main risk:** the structured variant gives the agent about 7× less DC authority per unit |z| than DSRL's repeat. Treat the authority/KL weight as a swept hyperparameter, and compare against DSRL-repeat at matched authority.
