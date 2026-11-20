# Experiment: Transport, continue training from 3M to 6M env steps
Status: Completed (transport runs finished; continued to 9M in exp10; Results section never written)

## Goal
Transport success was still rising at the 3M-step budget for most methods (fixed k=0, 3, 5; adaptive; random k). Continue the same runs to 6M to see where they converge, and whether adaptive pulls ahead of random k.

## Setup
- Same runs as exp_3 Wave 1, resumed from `checkpoint/latest.pt` + `latest_replay_buffer.pkl` (model, optimizer, replay buffer, callback state) with `train.total_timesteps=6000000`. Same logdirs `logs/adsrl/transport/<run>`, same wandb run ids (wandb resumes the run).
- Runs (14): fixed k ∈ {0, 3, 5, 8, 11, 14, 16, 19} seed 1; adaptive ρ=0.5 seeds 1–3; random k seeds 1–3.
- Not continued: residual SAC β=0.3 (seeds 1–3). It is at 0–1% success after 3M steps, so 3M more steps would not change the comparison.
- Replay buffer stays at 500k chunks (4M env steps). It is loaded from the pickle with its original size, so from ~4M env steps on, the oldest transitions (the 640k-step base-policy warm start first) are overwritten FIFO. A run trained to 6M from scratch with the same config would behave the same way.
- Hardware: kempner_h100, kempner_pehlevan_lab, 1 H100 per pack with MPS, `-c 24 --mem 128G`, 2-day limit. Packs: X1 fixed k0/3/5/8, X2 fixed k11/14/16/19, X3 adaptive s1–3, X4 random k s1–3. X3 and X4 wait (`afterany`) for the current Wave 1 jobs T3–T5 (57750669–71), which are still taking adaptive and random k to 3M.
- Script: `scripts/slurm/exp3_wave1.sbatch` with `ADSRL_EXTRA="train.total_timesteps=6000000"` (new hook: extra hydra overrides appended to every run). Commit a6bd481 + uncommitted working tree.
- Expected time: fixed ~122k env steps/h per run at 4/GPU → ~25 h for 3M more. Adaptive ~74k/h and random k ~83k/h at 3/GPU → ~36–41 h, inside the 2-day limit. If a pack times out, resubmitting the same command continues it.
- Expected result: fixed k=3/5 and DSRL keep rising; late steps (8–19) stay near the base level.

## Progress Log
- [2026-11-06 12:30] Re-estimated ETAs from the last 3 evals. Fixed: ~100k env steps/h, 4.79M at 10:55 → done ~23:00 Nov 6. Adaptive/random k: ~62k/h, 4.02M at 11:15 → would need until ~19:00 Nov 7, the same time X3/X4 hit their 2-day limit (19:04). Queued follow-ups X3b=58336216 and X4b=58336217 (same command, ADSRL_EXTRA=6M, 12 h, afterany on X3/X4); they resume from latest.pt and exit immediately if the runs already finished.
- [2026-11-06 12:40] Report republished (v10). plots.py now reads the raised budget from run_info_resume.json and, while a task is unfinished, compares methods at the furthest env step every full-budget run has reached (transport: 4.0M in the summary, 4.8M in the fixed-k sweep).
- [2026-11-06 12:07] All 4 packs RUNNING, no tracebacks. Fixed k at 4.792M (det; mean of last 3 at 4.28–4.79M): k3 .75, k5 .77, k0 .77, k8 .67, k11 .59, k16 .42, k14 .32, k19 .17. Adaptive/random k at 4.024M (mean of last 2): adaptive .45/.75/.59 (mean .60), random k .64/.76/.67 (mean .69). Fixed ETA ~Nov 7 01:00; adaptive/random ETA ~Nov 7 15:00 (limit 19:04).
- [2026-11-06 01:05] All 4 packs RUNNING, no tracebacks. Fixed k at 3.768M (det): k5 .71, k3 .70, k0 .575, k11 .51, k8 .505, k16 .44, k14 .19, k19 .19. Fixed rate ~102k env steps/h → finish ~Nov 6 22:30. Adaptive/random k at 3.256M (det): adaptive .33/.635/.475, random k .515/.655/.58. Rate ~65k/h → finish ~Nov 7 17:00, ~2 h before the 2-day limit (19:04); if they time out, resubmit X3/X4 to resume.
- [2026-11-05 21:25] Wave 1 T3-T5 COMPLETED (57750669-71). X3=58111534 and X4=58111569 started ~19:04; all 6 adaptive/random-k runs logged `resumed from .../latest.pt`, no tracebacks. 3M results (mean of last 3 det evals): adaptive .328/.615/.485 (mean .48), random k .517/.610/.532 (mean .55); fixed k3 .487, k5 .440, k0 .387. First post-resume fixed-k eval at 3.256M: k0 .535, k3 .495, k5 .49, k11 .34, k8 .21, k16 .265, k14 .13, k19 .13.
- [2026-11-05 17:55] All 8 fixed-k runs (X1, X2) logged `resumed from .../latest.pt: num_timesteps=294996, env_steps=3000000`, with no tracebacks. X3/X4 still held on the Wave 1 dependency.
- [2026-11-05 17:40] Submitted X1=58111511 (fixed k0/3/5/8), X2=58111517 (fixed k11/14/16/19) to kempner_h100; X3=58111534 (adaptive s1-3) and X4=58111569 (random k s1-3) held on afterany:57750669:57750670:57750671. Waiting to confirm the first runs log `resumed from ... latest.pt`.

## Results

## Analysis

## Next Steps
