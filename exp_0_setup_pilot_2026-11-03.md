# Experiment: exp_0 setup + throughput pilot
Status: Completed

## Goal
Build the environment, fetch base-policy checkpoints, and measure DSRL-NA throughput on robomimic can (single run, and packed 1/2/4/6/8 per GPU with and without CUDA MPS) to size walltimes and packing for the fixed-k sweep (plan: notes/plan_adaptive_dsrl.md, Step 0).

## Setup
- venv: `.venv` at project root (uv, Python 3.9). Activate with `source .venv/bin/activate`. Exact versions in `requirements-lock.txt`.
- Data: `dsrl/dppo/log/` (gitignored inside the dppo submodule), from the DSRL Google Drive folder.
- Training script: `scripts/pilot_train_dsrl.py`, a copy of `dsrl/train_dsrl.py` with the d4rl import removed, `+total_timesteps` (SB3 timesteps = action chunks summed over envs, default 20M as upstream) and `+buffer_size` (default 10M as upstream) configurable, and an optional `+profile=true` that wraps `model.train` / `callback.evaluate` with wall-clock timers. Must be run with cwd `dsrl/`.
- Config: `dsrl/cfg/robomimic/dsrl_can.yaml` unchanged (DSRL-NA, 4 train envs, 25 eval envs, UTD 20, 10 noise-critic steps, batch 256, 3x2048 MLPs, DDIM 8 steps).
- SLURM scripts: `scripts/slurm/pilot_single.sbatch`, `scripts/slurm/pilot_pack.sbatch {mps|nomps}`. Account `kempner_pehlevan_lab`, partition `kempner_h100`.
- Expected (plan): ~20 ms per gradient update, ~1.25 updates per env step, ~9 h per 1M env steps on can, ~3x aggregate throughput at 4 runs/GPU.

## Progress Log
- [2026-11-03 13:40] Wrote Results/Analysis. Status -> Completed.
- [2026-11-03 13:28] Pack jobs 57465906 (nomps) and 57465911 (mps) COMPLETED (36 min each). Mid-run sample at N=8: ~9 cores busy in total (about 1.1 per run, with eval off), job MaxRSS ~31 GB (each run had 4 train + 1 eval env procs and a 1M-chunk buffer), 13 GB GPU memory.
- [2026-11-03 12:57] pilot_single 57464151 finished timing (see Results). It then crashed in the final `model.save()` because the timing wrapper put a closure on `model.train` that pickles the SubprocVecEnv; fixed by popping the wrapper before saving. Note: `+profile=true` is still incompatible with mid-run CheckpointCallback saves; use it only for short pilots.
- [2026-11-03 12:51] Resubmitted packs to kempner_h100 with `--gres=gpu:2 -c 46` (only GPU 0 used; the 2nd GPU just buys CPUs under the 24-core/GPU cap): 57465906 (nomps), 57465911 (mps). gpu_h200 jobs 57464172 and 57465697 were stuck on Priority and were cancelled.
- [2026-11-03 12:50] pilot_pack 57464152 (gpu_h200, holygpu8a12204) failed immediately with "CUDA-capable device(s) is/are busy or unavailable"; that node's H200 reported 2 volatile uncorrectable ECC errors. Cancelled, resubmitted with `--exclude=holygpu8a12204` (57465697).
- [2026-11-03 12:47] Submitted pilot_single 57464151 (kempner_h100, 24 CPUs) and pilot_pack 57464152 (nomps) / 57464172 (mps) on gpu_h200 (olveczky_lab, 48 CPUs; kempner partitions cap CPUs per GPU, see below, so the 8-copy pack can't get enough cores there).
- [2026-11-03 12:46] CPU-per-GPU caps found via `sbatch --test-only`: kempner, kempner_h200, kempner_rtx allow <=16 cores per GPU; kempner_h100 allows <=24. gpu / gpu_h200 accepted 48 cores on 1 GPU. This caps how many runs fit per kempner GPU (each run = 1 main + 4 train env procs + 25 eval env procs that are busy during evals).
- [2026-11-03 12:45] Smoke test on H100 (srun 57463183): can/square/transport envs reset and step; driver 610.57.04, torch 2.4.0+cu121 sees the GPU.
- [2026-11-03 12:40] Install issues and fixes: (1) egl-probe needs cmake; the PyPI cmake 4.x rejects its CMakeLists, fixed with `CMAKE_POLICY_VERSION_MINIMUM=3.5` + `--no-build-isolation-package egl-probe`. (2) robosuite 1.4.1 pulls mujoco 3.5.0, which has no cp39 wheel; pinned `mujoco==3.1.6`. (3) robomimic 0.3.0 `env_robosuite.py` does `import mujoco_py` only to reference `mujoco_py.builder.MujocoException`; added a stub at `stubs/mujoco_py/` on sys.path via `.venv/lib/python3.9/site-packages/adsrl_stubs.pth`, so no real mujoco-py/d4rl. (4) robosuite forces `MUJOCO_GL=egl` at import, so `import robosuite` fails on login nodes (no EGL device); it works on GPU nodes.
- [2026-11-03 12:20] venv created, data downloaded, partition access checked (see Results).

## Results

### Environment and data
- venv: `/n/holylfs06/LABS/olveczky_lab/Users/charleszhang/CS2243Project/.venv` (CPython 3.9.25 via uv 0.9.13; `module load python` is not needed). Key pins: torch 2.4.0+cu121, numpy 1.26.4, gym 0.22.0, gymnasium 1.1.1, mujoco 3.1.6, robosuite 1.4.1 (git c848ca8), robomimic 0.3.0, egl-probe 1.0.2, hydra-core 1.3.2, wandb 0.17.7, stable-baselines3 2.6.0a1 (editable fork), dppo 0.8.0 (editable). No d4rl / mujoco-py. Full list: `requirements-lock.txt`.
- Driver 610.57.04 (CUDA 13.3 UMD); cu121 wheels work.
- Downloaded into `dsrl/dppo/log/` (204 MB): `robomimic/{can,square,transport}/{normalization,train}.npz` and `robomimic-pretrain/{can,square,transport}/.../checkpoint/state_*.pt` (the exact paths in `dsrl/cfg/robomimic/dsrl_{can,square,transport}.yaml`) plus their training logs. I also fetched `train.npz` (dataset states, for the authority diagnostic); it wasn't required. The env_meta jsons were already in the submodule (`dsrl/dppo/cfg/robomimic/env_meta/{can,square,transport}.json`). `offline_data_path` in the can/square configs points to files not in the Drive folder; that only matters if `load_offline_data=True`, which defaults to False.

### Partition access
- Associations: `kempner_pehlevan_lab`, `olveczky_lab`, `pehlevan_lab`.
- seas_gpu: AllowGroups=seas (user is in `seas`), and kempner_* accounts are on its DenyAccounts. `sbatch --test-only` accepts `olveczky_lab` and `pehlevan_lab`, and rejects `kempner_pehlevan_lab`. Its estimated start time was ~1 day out.
- Max cores per GPU: kempner, kempner_h200, kempner_rtx <16; kempner_h100 <24; gpu/gpu_h200 accepted 48 on 1 GPU.
- One gpu_h200 node (holygpu8a12204) had an H200 with uncorrectable ECC errors; exclude it.

### Single run (DSRL-NA can, H100, job 57464151, 24 CPUs, unchanged config)
| Metric | Value |
|---|---|
| Training throughput (after init) | **45.2 env steps/s** (20k env steps = 5000 chunks, 25k grad steps, 443 s) |
| Time per gradient step (UTD loop, synced) | **15.7 ms** |
| Time split during `learn` | updates 88.7%, env stepping + rest 11.3% |
| Initial eval (200 eps x 300 steps, 25 envs) | 36.6 s, which is about 3.3% of wall time at eval_interval=3000 vec steps (48k env steps ~= 1060 s of training) |
| Initial base-policy rollout (1501 vec steps = 24k env steps) | 57 s |
| `nvidia-smi dmon` SM% during training | mean 52% (55-58% in update bursts) |
| Job MaxRSS | 17.4 GB (30 env procs, ~0.55 GB each) |

### Packing (H100, eval off, steady state over ~6 min per config after 1 min warmup)
Jobs 57465906 (no MPS) and 57465911 (user-level MPS). Per-run ms/grad is wall time per grad step, so it includes contention.

| N runs | no MPS: agg env steps/s | per run | ms/grad | SM% | MPS: agg env steps/s | per run | ms/grad | SM% | MPS speedup vs N=1 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 46.0 | 46.0 | 17.4 | 51 | 44.3 | 44.3 | 18.1 | 52 | 1.0x |
| 2 | 51.4 | 25.7 | 31.1 | 96 | 83.6 | 41.8 | 19.1 | 80 | 1.9x |
| 4 | 54.7 | 13.7 | 58.6 | 97 | 139.8 | 34.9 | 22.9 | 97 | 3.2x |
| 6 | 54.8 | 9.1 | 87.6 | 99 | 167.4 | 27.9 | 28.7 | 98 | 3.8x |
| 8 | 54.3 | 6.8 | 117.9 | 98 | 179.2 | 22.4 | 35.7 | 99 | 4.0x |

### Revised cost for can (per 1M env steps, includes ~3.5% eval overhead)
| Setting | Hours per run | GPU-hours per run |
|---|---|---|
| Single run | **~6.4 h** (plan: ~9 h) | 6.4 |
| 4/GPU + MPS | ~8.3 h | ~2.1 |
| 6/GPU + MPS | ~10.3 h | ~1.7 |
| 8/GPU + MPS | ~12.8 h | ~1.6 |

## Analysis
- Gradient updates dominate (89% of training wall time), as the plan assumed. Per-update time is 15.7 ms, a bit below the plan's 20 ms, so a single can run is ~6.4 h per 1M env steps rather than ~9 h. Eval overhead is ~3%, not 25%, because 25 parallel envs finish 200 episodes in ~37 s.
- **Without MPS, packing is nearly useless**: time-slicing between contexts caps aggregate throughput at ~1.2x, even though `dmon` reports ~97% "SM" (that column is the fraction of time any kernel is running, not SM occupancy, so it doesn't measure real utilization). **With MPS**, kernels from different runs overlap: 4 runs give 3.2x and 6 give 3.8x, and 8 adds only 5% over 6.
- CPU is not the bottleneck during training (~1.1 busy cores per run with eval off), but the eval bursts (25 env procs per run) and memory (~0.55 GB per env proc) are what limit packing on kempner, where nodes allow only 16-24 cores per GPU.
- Recommendation: **6 runs per GPU with MPS** on kempner_h100 (`-c 24 --mem 96G`; set `env.n_eval_envs=10`, which keeps 200 eval episodes in 20 rounds and cuts each run to ~15 procs / ~8 GB). On the 16-core partitions (kempner, kempner_h200, kempner_rtx) use **4 runs per GPU** with `-c 16 --mem 64G`. Use `--time` of ~12 h for a 6-pack at 1M env steps on can, plus a margin. These ratios were measured on H100 only; A100/H200/RTX-PRO GPUs will differ.
- MPS setup that worked (inside the job): `export CUDA_MPS_PIPE_DIRECTORY=$SCR/mps_pipe CUDA_MPS_LOG_DIRECTORY=$SCR/mps_log; nvidia-cuda-mps-control -d` ... `echo quit | nvidia-cuda-mps-control` at the end (see `scripts/slurm/pilot_pack.sbatch`). Caveat: under MPS, a fatal GPU fault in one client can take down the others in the pack, so checkpoint/resume matters.

## Next Steps
- Re-plan Step 3 GPU-hours with ~1.7 GPU-h per 1M-env-step can run (6-pack + MPS) instead of 9/3 = 3 GPU-h.
- A/B `torch.compile` / TF32 / GPU replay buffer on can under MPS; per-update time now bounds throughput.
- Measure square (DDIM 8, same network size) and transport (obs 59, act 14x8, larger base net) single-run ms/grad before sizing their walltimes; transport at ddim_steps=100 will be much slower.
- Verify throughput with `n_eval_envs=10` and eval on, at 6/GPU under the 24-core cap, as part of the first real can pack.
