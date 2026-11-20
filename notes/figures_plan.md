# Results figures plan (adaptive-step DSRL)

The story the figures tell, in three claims:
1. **Where you steer matters, and the best step is task-dependent.** The step DSRL uses (k=0) is not always the best one. (Fig 2)
2. **The cause is a tradeoff between how much control a step gives and how far the actions stray from what the base policy knows.** (Fig 3)
3. **A policy that picks k per chunk matches or beats the best fixed k without per-task tuning, and the step it picks follows the phase of the task.** (Figs 4–6)

## Main figures

### Fig 1: Method schematic
- The DDIM chain x_T → … → x_0. Steering at step k means re-noising the current clean-action estimate (x0_hat) with w.
- The adaptive policy has a categorical head for k and per-k Gaussian heads for w.
- The noise critic Q^W(s, x_T, k, w) is distilled from the action critic Q(s, a).
- Drawn by hand, no data needed.

### Fig 2: Fixed-k sweep (core evidence for claim 1)
- **Layout:** one column per task (can / square / transport), two rows.
  - **Top row:** final success (y) vs steering step k (x; k=0 is DSRL, larger k is later in denoising). Dashed line for the base policy. Each seed is a dot, with mean ± SE overlaid.
  - **Bottom row:** sample efficiency (env steps to reach 80% of the best success) vs k.
- **What we hope to see:** the curves have an interior optimum or a plateau, and the optimum moves between tasks. If k=0 wins everywhere, that's a negative result, and Figs 3 and 6 become the explanation.
- **Data:** `eval_summary.csv` from the Step 3 runs.

### Fig 3: Why k matters, control vs staying close to the base (mechanism for claim 2)
- **(a)** Control strength vs k: the action spread over w ~ N(0, I) with x_T fixed (`scripts/authority.py`), one line per task.
- **(b)** Distance from the base vs k: at convergence, the mean ‖a − a_base‖ and the base model's denoising loss on executed actions. These come from the eval rollout dumps.
- **(c)** Each dot is one (task, k) run: control strength (x) vs final success (y), colored by distance from the base. This shows whether success peaks at intermediate control.
- **Data:** the authority CSV and the `eval_rollouts/*.npz` dumps.

### Fig 4: Main learning curves (headline result)
- **Layout:** one panel per task. Success vs env steps, mean ± SE shaded over 3 seeds.
- **Lines:**
  - DSRL (k=0)
  - Best fixed k (oracle)
  - Uniform-random k
  - **Adaptive k (ours)**
  - Residual SAC
- **Horizontal references:** base policy, constant-noise CEM.
- **DPPO:** not plotted (decided 2026-11-04); one sentence of context in the text.
- **What we hope to see:** adaptive ≥ best fixed k, both above DSRL, and adaptive above uniform-random k (so the learned choice matters).

### Fig 5: Summary and compute efficiency
- **(a)** Dot plot: final success and normalized AUC per method × task, with per-seed points shown. This is the "Table 1" in visual form, and the table goes in the appendix with the exact numbers.
- **(b)** Success vs GPU-hours for every method. This is the fair comparison against CEM, which costs almost nothing, and against DPPO, which costs a lot.

### Fig 6: What the adaptive policy learns (claim 3 interpretation)
- **(a)** How π(k) evolves over training: a stacked area chart of the k frequencies vs env steps, one panel per task.
- **(b)** k vs task phase: a heatmap with normalized episode time or a phase label on x, k on y, and frequency as color. Phases are defined from the obs:
  - can / square: reach, grasp, transport, place, using gripper state and object-to-target distance.
  - transport: pick, handover, place, using each arm's gripper and object position.
- **(c)** One example rollout: a filmstrip of frames annotated with the chosen k per chunk.
- **What we hope to see:** early steps (more control) during gross motion, late steps (fine correction) during contact or precision phases. The structure itself is the result, even if it comes out differently.

### Fig 7: Ablations (compact)
- Change in final success relative to adaptive k for:
  - k restricted to the top-3 fixed-k set
  - noise trust region on/off (k=0 and adaptive)
  - best-of-N at eval (DSRL, best fixed k, adaptive)
  - ρ ∈ {0.25, 0.5}
- One grouped bar per task, with seeds shown.

## Appendix figures
- Per-k learning curves for all tasks (the full Step 3 data).
- Transport base success vs number of DDIM steps (100 / 20 / 10), justifying the reduced K.
- Entropy of π(k) and α_k over training (stability of the hybrid actor).
- Stability: per-seed curves and the count of runs that fall more than 20 pts below their peak.
- Episode length of successful episodes per method (speed of task completion).

## Logging these figures need (checked against the implementation)
- `eval_summary.csv` per run: env_steps, wall_s, gpu_s, success, ep_len_success, ‖a − a_base‖, k_entropy. Covers Figs 2, 4, 5, 7.
- `eval_rollouts/step_*.npz`: per-chunk obs, executed action, a_base, k, w, π(k), reward, success. Covers Figs 3b and 6, plus the phase labels.
- `scripts/authority.py` CSV. Covers Fig 3a.
- Video frames for Fig 6c: an offscreen render pass for one seed, done after training on the final checkpoint.
- DPPO numbers: transcribed from the DSRL/DPPO papers into `notes/dppo_reference.csv`.

## Style
- One plotting module, `scripts/plots.py`, reads only the CSV/npz files above, so figures can be regenerated without wandb.
- Use the `nature-figure` / `dataviz` skills when building the final figures.
- Fixed method colors across all figures. Adaptive k is the only saturated accent color, and DSRL is a neutral dark color.
