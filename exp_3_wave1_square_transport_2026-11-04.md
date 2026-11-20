# Experiment: Wave 1 — fixed-k sweep on square/transport + 3-seed adaptive / uniform-k / residual on all tasks
Status: Completed (all runs finished by 2026-11-05; transport continued in exp5 and exp10; Results section never written)

## Goal
Run the main comparison. It has two parts. First, a 1-seed fixed-k scan on square and transport. Second, 3 seeds each of adaptive k (rho=0.5), uniform-random k and residual SAC (beta=0.3) on square and transport, plus seeds 2 and 3 of those methods on can (seed 1 is from Wave 0).

## Setup
- `train_adsrl.py --config-name=<task>`, `env.n_eval_envs=10`, `wandb.mode=offline`, `resume=true`, explicit `name=`, `checkpoint_every_min=120` (new, see below). Budgets come from the task configs: square 2M env steps, transport 3M, can 1M. Transport uses `model.ddim_steps=20` (exp_1 decision: K=20; transport.yaml still says 100).
- Runs (40):
  - Fixed-k, seed 1: square k=0..7 (K=8); transport k ∈ {0,3,5,8,11,14,16,19} (K=20, the same grid as `results/base_eval_wk/transport.csv`). 16 runs.
  - Square and transport, seeds 1-3: `adaptive_rho0.5_s<n>` (`steer.mode=adaptive steer.rho=0.5`), `uniformk_s<n>` (`steer.mode=adaptive steer.freeze_k_uniform=true`), `residual_b0.3_s<n>` (`steer.mode=residual steer.residual_beta=0.3`). 18 runs.
  - Can, seeds 2-3 of the same three methods. 6 runs. Seed 1 is in Wave 0: `adaptive_rho0.5_s1`, `uniformk_s1`, `residual_b0.3/seed1`.
- Script: `scripts/slurm/exp3_wave1.sbatch <task> <run names...>`. It is one generic MPS pack; the run name sets the method and seed. It keeps the Wave 0 fix: `wait` only on training PIDs, never a bare `wait`, which would hang on `nvidia-smi dmon`. Defaults: kempner_h100, kempner_pehlevan_lab, 1 GPU, `-c 24 --mem 128G`, 2-00:00:00 (the kempner MaxTime), exclude holygpu8a12204. `ADSRL_RUNS_PER_GPU` is set to the number of runs.
- Packing: square and can use 6 runs per H100 (5 in S2, which holds the 3 slower adaptive runs). Transport uses 4 or 3 per H100: it is heavier (K=20 sampler, 2048-wide diffusion MLP, 8-step chunks), and it sets the critical path, so it gets more GPU share per run. In total 9 kempner GPUs, against a 16-GPU cap; no other kempner jobs of the user were running at submission.
- Packs:

| Pack | Task | Runs | Runs/GPU |
|---|---|---|---|
| S1 | square | fixed_k0_s1 .. fixed_k5_s1 | 6 |
| S2 | square | fixed_k6_s1 fixed_k7_s1 adaptive_rho0.5_s1 adaptive_rho0.5_s2 adaptive_rho0.5_s3 | 5 |
| S3 | square | uniformk_s1 uniformk_s2 uniformk_s3 residual_b0.3_s1 residual_b0.3_s2 residual_b0.3_s3 | 6 |
| T1 | transport | fixed_k0_s1 fixed_k3_s1 fixed_k5_s1 fixed_k8_s1 | 4 |
| T2 | transport | fixed_k11_s1 fixed_k14_s1 fixed_k16_s1 fixed_k19_s1 | 4 |
| T3 | transport | adaptive_rho0.5_s1 adaptive_rho0.5_s2 residual_b0.3_s1 | 3 |
| T4 | transport | adaptive_rho0.5_s3 uniformk_s1 residual_b0.3_s2 | 3 |
| T5 | transport | uniformk_s2 uniformk_s3 residual_b0.3_s3 | 3 |
| C1 | can | adaptive_rho0.5_s2 adaptive_rho0.5_s3 uniformk_s2 uniformk_s3 residual_b0.3_s2 residual_b0.3_s3 | 6 |

- Logdirs: `logs/adsrl/<task>/<name>` (on scratch through the symlink). Run stdout goes to `logs/exp3/w1_<pack>_<task>_<jobid>/<name>.log`, SLURM logs to `logs/slurm/w1_<pack>_<jobid>.*`.
- **Time-based checkpointing (new).** Transport evaluates and checkpoints only every 8000 vec steps, which is 256k env steps and probably 6+ h of wall time. A walltime kill would lose that much. So `checkpoint_every_min` (default 0, off) was added to the three configs and to `AdsrlCallback._on_step`: it saves `latest.pt` plus the buffer when 120 min have passed since the last save. Writes are atomic (tmp + rename). After a resume, SB3's `n_calls` restarts, so eval env_steps are no longer on the 256k grid. Plots use env_steps, so this is fine.
- Commit: a6bd481 + uncommitted working tree (not committed, per instructions).
- Expected: square ~20 env steps/s per run at 6/GPU, so ~28 h (adaptive ~36 h), which fits one 2-day allocation. Transport is unmeasured. Guess: ~10 env steps/s at 3-4/GPU, so ~3 days, needing one resubmit. Can C1 ~12-14 h.

### How to resubmit a pack (after timeout, preemption or a node failure)
Run from the project root. Each command is identical to the original submission. resume=true continues from `checkpoint/latest.pt`. Runs that already finished find `remaining <= 0`, skip the duplicate final eval and exit within a few minutes. Before resubmitting, check that the old job is gone (`squeue --me`), since two jobs must never train the same logdir.
```bash
cd /n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project
sbatch -J w1_S1 scripts/slurm/exp3_wave1.sbatch square fixed_k0_s1 fixed_k1_s1 fixed_k2_s1 fixed_k3_s1 fixed_k4_s1 fixed_k5_s1
sbatch -J w1_S2 scripts/slurm/exp3_wave1.sbatch square fixed_k6_s1 fixed_k7_s1 adaptive_rho0.5_s1 adaptive_rho0.5_s2 adaptive_rho0.5_s3
sbatch -J w1_S3 scripts/slurm/exp3_wave1.sbatch square uniformk_s1 uniformk_s2 uniformk_s3 residual_b0.3_s1 residual_b0.3_s2 residual_b0.3_s3
sbatch -J w1_T1 scripts/slurm/exp3_wave1.sbatch transport fixed_k0_s1 fixed_k3_s1 fixed_k5_s1 fixed_k8_s1
sbatch -J w1_T2 scripts/slurm/exp3_wave1.sbatch transport fixed_k11_s1 fixed_k14_s1 fixed_k16_s1 fixed_k19_s1
sbatch -J w1_T3 scripts/slurm/exp3_wave1.sbatch transport adaptive_rho0.5_s1 adaptive_rho0.5_s2 residual_b0.3_s1
sbatch -J w1_T4 scripts/slurm/exp3_wave1.sbatch transport adaptive_rho0.5_s3 uniformk_s1 residual_b0.3_s2
sbatch -J w1_T5 scripts/slurm/exp3_wave1.sbatch transport uniformk_s2 uniformk_s3 residual_b0.3_s3
sbatch -J w1_C1 scripts/slurm/exp3_wave1.sbatch can adaptive_rho0.5_s2 adaptive_rho0.5_s3 uniformk_s2 uniformk_s3 residual_b0.3_s2 residual_b0.3_s3
```
- To move a pack off kempner_h100, add the flags on the command line, e.g. `-p gpu_h200 -A olveczky_lab -t 3-00:00:00` (gpu/gpu_h200 allow 3 days). Use one partition per job.
- On the 16-core kempner A100 partition (`-p kempner -c 16 --mem 96G`), use at most 4 runs per pack: split a 6-run pack into two commands with disjoint run lists.
- A resubmit can also drop runs that have finished (`checkpoint/final.pt` exists), so the pack gets more GPU per remaining run.

## Progress Log
- [2026-11-05 16:45] Done: T1 57750667, T2 57750668, S1 57750672, S2 57750673, S3 57750674, C1 57750675 (all COMPLETED, exit 0). Still running: T3-T5 (transport adaptive/uniform at ~2.7M of 3M; residual done). Final det success (mean of last 3 evals): square fixed k0-7 .948/.965/.945/.958/.960/.953/.925/.812, adaptive .937/.955/.975 (3 seeds), uniform .948/.960/.958, residual .005-.10. Transport fixed k0/3/5/8/11/14/16/19 .387/.487/.440/.203/.250/.198/.217/.150 (base ~.19-.30); residual ~0. Transport adaptive latest .315/.635/.455, uniform .44/.525/.49 at 2.69M. Can seeds 2/3: adaptive .923/.940, uniform .913/.928. Transport early-training drop (to 1-10% around 0.9-1.4M) recovered after ~1.6M.
- [2026-11-04 13:00] Measured transport throughput over the first ~35 min of training. SB3 cumulative chunks × 8 / time_elapsed:
  - fixed k (T1, T2; 4/GPU): 33.8-34.2 env steps/s, all 8 runs;
  - adaptive (T3, T4): 22.5-22.8;
  - uniform-k (T4, T5): 22.7-23.1;
  - residual (T3-T5): 62.8-64.6.

  Overheads: the step-0 eval (400 eps, stoch + det) took 20-25 min, and the 640k-step base warm start took ~1 h 10 min. Training started 12:20-12:25.

  Projections assume 2.36M trained env steps plus ~10 evals at ~22 min each, ~3.7 h in total:

  | Runs | Projected finish | Job elapsed at finish |
  |---|---|---|
  | transport fixed (T1, T2) | ~Nov 5 11:30 | ~25 h |
  | transport residual | ~Nov 5 02:30 | — |
  | transport adaptive / uniform-k | ≤ Nov 5 21:00 (upper bound; faster once the residual run in the same pack finishes) | ~34 h |

  Transport therefore fits in one 2-day allocation; the original estimate was 2-3 days. The first trained transport evals land at 896k env steps (warm start + 256k): residual ~13:30, fixed ~14:30, adaptive/uniform-k ~15:30 today. No crashes; 9/9 packs running.
- [2026-11-04 12:15] All 40 runs are alive with no tracebacks. Every run has its step-0 eval (stoch + det). Square and can also have their first trained eval (square at 96k, can at 72k; residual is one eval further along). Transport is still in its 20001-vec-step base warm start (640k env steps), so it has no trained eval yet. Another session's `e4_bound` jobs (57759602, 57759605) bring kempner use to 11/16 GPUs.
  - **Square rates.** Env steps/s while training, between the step-0 eval and the 96k eval, minus ~50 s of warm start. "Eff" includes eval time (~5-7 min per 64k block).

    | Runs (pack) | Train rate (env steps/s) | Eff (env steps/s) | Projected finish |
    |---|---|---|---|
    | fixed k0-k5 (S1) | 23.4 | 20.3 | ~27 h, Nov 5 ~14:00 |
    | fixed k6/k7 (S2) | 20.8 / 21.7 | — | ~27 h, Nov 5 ~14:00 |
    | adaptive (S2) | 19.0 | 17.5 | ~31 h, Nov 5 ~18:00 |
    | uniform-k (S3) | 15.8 | 14.4 | ≤38 h, Nov 6 ~01:00 |
    | residual (S3) | 33 | — | ~17 h, Nov 5 ~04:00 |

    Uniform-k should speed up once the residual runs in S3 finish, so its ~38 h is an upper bound inside the 48 h limit.
  - **Can C1 rates.** adaptive 15.9, uniform-k 16.1, residual ~30 env steps/s. Projected finish Nov 5 ~06:00-07:00 (~19 h).
  - **First evals (det success, step 0 → first trained eval).**
    - Square: k0 .23→.355, k1 .465→.435, k2 .525→.735, k3 .485→.595, k4 .445→.515, k5 .435→.445, k6 .495→.39, k7 .535→.48, adaptive s1/s2/s3 .525/.455/.50 → .50/.465/.545, uniform-k .495/.495/.45 → .51/.545/.53, residual s1/s2/s3 .015/0/0 → .095/.035/.025 at 160k.
    - Can: adaptive s2/s3 .605/.685 → .665/.62, uniform-k s2/s3 .635/.615 → .77/.55, residual s2/s3 .14/.095 → .32/.245 at 120k.
    - Transport step 0 (stoch/det): k0 .355/.015, k3 .37/.165, k5 .34/.295, k8 .335/.295, k11 .27/.19, k14 .33/.275, k16 .29/.27, k19 .22/.17, adaptive s1-s3 det .29/.29/.32, uniform-k det .265/.295/.325, residual det 0/0/.005.
  - **Flag: residual β=0.3 at step 0.** The untrained residual actor is near 0 success on square (≤.045) and transport (≤.01). On can it is .095-.14, against .39 for Wave 0 seed 1. |a−a_base| is about the same across seeds (det ≈0.48 on can, ≈0.96 on transport), so this looks like the random actor-init mean, β·tanh(μ_init), pushing the actions off-distribution, not a code change; residual.py has not changed since 11-03 18:31. Residual is recovering on can (.32 at 120k) but only slowly on square.
- [2026-11-04 10:53] Submitted all 9 packs to kempner_h100 at 10:48: T1=57750667 (holygpu8a13102), T2=57750668 (holygpu8a15202), T3=57750669 (holygpu8a17202), T4=57750670 (holygpu8a17604), T5=57750671 (holygpu8a11501), S1=57750672 (holygpu8a17302), S2=57750673 (holygpu8a17104), S3=57750674 (holygpu8a17202), C1=57750675 (holygpu8a17204). T1-T3 started at once; the other six were Resources/Priority pending for ~4 min and then started. All 9 started the MPS daemon, and the .err files are empty. Kempner GPU use is 10/16: these 9 plus another session's `exp2_base_recheck` (57750992). The step-0 stochastic eval rows are already written for S2, S3 and C1.
- [2026-11-04 10:50] Created the doc and `scripts/slurm/exp3_wave1.sbatch`, and added `checkpoint_every_min`. Hydra composes the overrides (`--cfg job` checked). `squeue --me` is empty: bc_free, ll_art_active and packed_tasks have all finished, so all 16 kempner GPUs are available. kempner_h100 has ~4 idle GPUs and 81 pending jobs; kempner_h200 has 0 idle.

## Results
Pending.

## Analysis
Pending.

## Next Steps
- When runs finish, copy the summaries (eval_summary.csv, run_info.json; no checkpoints) to results/exp3_wave1/.
- Wave 2: 3-seed follow-up of the top fixed k per task, chosen from this scan.
