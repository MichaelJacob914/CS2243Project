# RoboDojo + pi0.5 feasibility check (2026-11-07)
Status: Running

## Goal
Can we (a) get a RoboDojo-sim-finetuned pi0.5 checkpoint and (b) run RoboDojo's Isaac Sim 5.1 sim on our cluster, to test RL steering of pi0.5 flow-matching noise?

## Findings so far
- Official ckpts: HF dataset RoboDojo-Benchmark/RoboDojo (public, apache-2.0), path ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-{0,1,2}/59999 (openpi JAX/orbax, 3 seeds). Same for Pi_0.
- Isaac Sim 5.1 pip wheels are manylinux_2_35; cluster is RHEL8 glibc 2.28 -> native pip install blocked; use Singularity container.

## Work dirs
- Sources/SIF/assets/venv: /n/netscratch/olveczky_lab/Lab/charleszhang/robodojo/

## Progress Log
- [2026-11-07 15:38] Submitted CPU setup job 58725302 (shared): singularity pull nvcr.io/nvidia/isaac-sim:5.1.0 (public, no NGC login needed), clone IsaacLab/curobo forks, hf download Assets/ (~41 GB).
- [2026-11-07 15:45] Paper: pi0.5 sim = one multi-task model on 34 training tasks x 100 demos (3,500 traj, 1.86M frames, 25 Hz), batch 256, 60k steps from pi05_base, 3 seeds, 50 trials/task/seed. Paper sim throughput: 77 interactions/s total on 8x RTX 4090 (10 envs/GPU, zero action, heterogeneous) -> ~10/s per GPU.
- XPolicyLab Pi_05 adapter calls openpi `policy.infer(obs, noise=...)` via kwargs -> noise injection hook already exists; full 50-step chunk executed open-loop.
- [2026-11-07 16:04] Job 58725302 done: SIF (7.6 GB) at robodojo/sif/isaac-sim_5.1.0.sif, Assets (~41 GB) at robodojo/hfrepo/Assets, submodules cloned. Symlinked RoboDojo/{Assets,XPolicyLab}.
- [2026-11-07 16:06] Submitted CPU install job 58734971 (shared): uv venv at robodojo/venv inside the container (torch 2.7 cu128, isaacsim[all,extscache]==5.1.0, IsaacLab fork, curobo[cu12]). Log: robodojo/logs/rd_install_58734971.out.
- [2026-11-07 16:08] Handed back before install finished; GPU smoke test NOT yet run. Next: once log shows IMPORTS_OK, `sbatch -p kempner_rtx -J rd_smoke_rtx robodojo/jobs/smoke_gpu.sh stack_bowls false`, then the same with -p kempner_h100 (expected to fail: Isaac Sim 5.1 docs say A100/H100 without RT cores are unsupported).
- [2026-11-07 ~16:30] Install job 58734971: isaacsim + IsaacLab OK; cuRobo build FAILED: 'cc' not found inside isaac-sim container (no compiler). Fix: provide a compiler in the venv (e.g. micromamba/conda-forge gcc_linux-64 + CC env var, or ziglang 'zig cc'), then rerun the curobo step only.
- [2026-11-07] Submitted curobo-only fix job 58743627 (shared, olveczky_lab): ziglang as CC; fallback --no-deps.

## 2026-11-07 ~16:45 (main session)
- cuRobo fix job 58743627 FAILED: setuptools-scm could not detect curobo's version (source clone without git metadata; would need SETUPTOOLS_SCM_PRETEND_VERSION). ziglang path not reached past this.
- RTX smoke 58743768 (kempner_rtx, RTX PRO 6000 Blackwell, driver 610.57.04, Vulkan) FAILED: Isaac Sim 5.1 starts (appState=started, ~115 s uptime), then segfaults in libomni.hydra.rtx.plugin.so (the RTX renderer); the eval client restarted 10 times, rc=139 each time. Not yet known whether this is the driver (610 is newer than what Isaac Sim 5.1 was validated on), the Vulkan ICD inside the container, or the missing cuRobo. H100 smoke not run.
