# When to steer

pi0 (and later pi0.5) from [openpi](https://github.com/Physical-Intelligence/openpi), run in MuJoCo simulation (LIBERO, ALOHA sim) with control over the policy's initial noise. This is the base for DSRL-style noise-space steering ([paper](https://arxiv.org/abs/2506.15799)).

## Quickstart (FASRC)

The checkpoints and venvs are already built in a shared folder. You only need your own clone:

```bash
git clone git@github.com:MichaelJacob914/CS2243Project.git && cd CS2243Project
git submodule update --init third_party/openpi third_party/LIBERO
source env/fasrc.sh                                           # LIBERO venv + shared paths
sbatch -t 01:00:00 slurm/eval.sbatch configs/eval/smoke_pi0_libero.yaml   # 1 task x 2 episodes
```

- **Results** go to `$RUNS_ROOT/eval/<config name>/`: one JSON line per episode, plus videos and a `meta_*.json` that records the config and git commits.
- **Summarise** with `python scripts/aggregate.py $RUNS_ROOT/eval/<config name>`.
- **No GitHub SSH key?** The `dsrl` and `dsrl_pi0` submodules use SSH URLs. Run `git config --global url."https://github.com/".insteadOf git@github.com:` once.

## Running evals

```bash
source env/fasrc.sh                       # or: source env/fasrc.sh aloha   (for ALOHA configs)
sbatch slurm/eval.sbatch configs/eval/pi0_libero90_t57.yaml
sbatch slurm/eval.sbatch configs/eval/pi0_libero_spatial.yaml --tag repeat2
sbatch slurm/eval.sbatch configs/eval/pi0_libero90_t57.yaml noise.mode=gaussian --tag gaussian
sbatch --array=0-1 slurm/eval.sbatch configs/eval/pi0_libero_spatial.yaml   # split tasks over 2 jobs
```

- **Overrides** are `key=value`, and dotted keys reach into nested settings (`noise.seed=3`). `--tag` gives a variant its own output folder.
- **Resuming:** resubmitting the same command skips episodes that are already done, which handles the 12 h limit on `gpu_test`.
- **`gpu_test` limits:** 2 jobs at a time, 12 h each, A100 slices with 20 GB. The account is `hankyang_lab`.
- **Interactive use:** `source slurm/common.sh && python scripts/eval.py <config> [key=value ...]` runs the same thing on any node. `common.sh` picks the renderer: GPU EGL if it works, otherwise Mesa's CPU renderer (llvmpipe). The CPU renderer is used on login nodes, and probably on `gpu_test`'s A100 slices.

| Config | What it measures | Reference |
|---|---|---|
| `smoke_pi0_libero.yaml` | Setup works: rendering, checkpoint, loop | runs |
| `pi0_libero_spatial.yaml` | Pipeline matches published numbers | 96.8 % (openpi README) |
| `pi0_libero90_t57.yaml` | DSRL's LIBERO task, base rate | ≈ 0.19 (DSRL, different protocol) |
| `pi0_libero90_t57_fixed.yaml` | Steerability: success under 64 fixed noises | spread = room to steer |
| `pi0_aloha_transfercube.yaml` | DSRL's ALOHA task, base rate | ≈ 0.65 (DSRL) |

**Noise modes** are set with `noise.mode` (see [src/steer/noise/__init__.py](src/steer/noise/__init__.py)):

| Mode | What it does |
|---|---|
| `default` | openpi draws the noise itself |
| `gaussian` | Our seeded draw. It must match `default`, which checks the steering input is wired correctly. |
| `fixed` | The same noise for every query |
| `actor` | A trained DSRL policy (not implemented yet) |

`noise.repeated: true` tiles one 32-dim row across the action chunk, as DSRL does for pi0.

## Layout

```
env/fasrc.sh            the only file with cluster paths; env/template.sh for other machines
src/steer/
  policies/registry.py  policy name -> openpi config, checkpoint, actions per query
  policies/openpi_policy.py   load in-process; act(obs, noise) -> action chunk; noise_shape
  envs/                 libero.py (openpi's official eval protocol), aloha.py
  noise/                noise samplers
  eval/                 rollout.py (one episode), runner.py (tasks x episodes -> results)
configs/eval/           one YAML per experiment
scripts/                eval.py, aggregate.py, setup.sh, download_ckpts.py, fix_mtimes.sh
slurm/                  FASRC job wrappers (they call scripts/eval.py)
third_party/openpi      upstream openpi, pinned; never edited
third_party/LIBERO      nakamotoo's LIBERO fork (works with torch >= 2.6), pinned
dsrl/, dsrl_pi0/        reference code (DSRL authors), not imported
```

Your clone's `src/` is put on `PYTHONPATH` by `env/fasrc.sh`. The shared venvs contain only third-party packages, so everyone runs their own code.

## Shared data (FASRC)

`DATA_ROOT=/n/netscratch/hankyang_lab/Everyone/mjammot/cs2243`

| Path | Contents |
|---|---|
| `openpi_data/` | Checkpoints (`pi0_libero`, `pi0_aloha_sim`) and the tokenizer, in openpi's cache layout |
| `venvs/libero`, `venvs/aloha` | Python 3.11 from openpi's lockfile. LIBERO uses MuJoCo 3.2.3 and robosuite 1.4.1; ALOHA uses MuJoCo 2.3.7 and gym-aloha. |
| `python/`, `src/LIBERO`, `libero_config/` | The interpreter, a pinned LIBERO copy and its config, used by the venvs |
| `mesa/` | Mesa's CPU renderer (llvmpipe) from conda-forge. FASRC has no system software renderer. |
| `runs/<user>/` | Your results and Slurm logs |

**Netscratch deletes files older than 90 days.** Anything downloaded, installed or copied there must have today's date. Run `scripts/fix_mtimes.sh <dir>` once, right after placing the data. FASRC prohibits re-touching files later to avoid the purge. Keep anything you need long-term in git, in W&B, or on holylabs.

## Adding pi0.5

1. Download the checkpoint: `python3 scripts/download_ckpts.py pi05_libero && scripts/fix_mtimes.sh $OPENPI_DATA_HOME`. It's already in the registry.
2. Copy a config and set `policy: pi05_libero`.

Nothing else changes. The noise shape is read from the model: (10, 32) for pi0.5 versus (50, 32) for pi0.

## Rebuilding (new machine, HRT, or a new openpi pin)

```bash
cp env/template.sh env/<site>.sh                 # set DATA_ROOT
source env/<site>.sh none
scripts/setup.sh                                 # venvs from openpi's lockfile, then fix_mtimes
python3 scripts/download_ckpts.py pi0_libero pi0_aloha_sim
```

`scripts/setup.sh` skips `rerun-sdk`, a lerobot visualizer that has no wheel for FASRC's glibc 2.28.
