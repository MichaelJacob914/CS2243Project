# Gaussian Mixture Diffusion

Sample using the reverse probability-flow ODE for an equally weighted Gaussian
mixture, integrated with SciPy's `solve_ivp`. Initialization defaults to an
approximate Gaussian prior; exact forward-mixture sampling is also available.

## Setup and usage

Run from this directory:

```bash
python -m pip install numpy scipy matplotlib pillow tqdm
python main.py --sde VPSDE -n 100 --seed 42
```

Defaults: two 2D Gaussians centered at `(-2, 0)` and `(2, 0)` with identity
covariances, `T=10`, and reversal to `t=0`. Outputs are `samples.npz` and
`output.gif`.

```bash
python main.py --sde VESDE -T 5 -n 200 --seed 42 \
  --gif sampling.gif --steps 50 --fps 10 -o samples.npz
```

- `--sde`: `VESDE` uses `dX = dW`; `VPSDE` uses `dX = -X dt + sqrt(2) dW`.
- `--terminal-distribution`: `approximate` (default) starts from `N(0, I)` for
  VPSDE or `N(0, T I)` for VESDE; `exact` starts from the true forward mixture q_T.
  These priors use diffusion coordinates, standardized when requested.
- `-d` / `--dimensions`: sample dimension (default: 2).
- `-p` / `--params`: Gaussian parameter text file.
- `--standardize`: transform q0 to mean 0 and variance 1 per dimension using
  exact mixture moments, including variation between component means.
- `--end-time`: stop reversal at this time (default: 0).
- `--steps`: ODE intervals per point; the GIF contains `steps + 1` frames.
- `--gif ""`: disable animation.
- `--workers`: number of sampling processes (default: 1); `0` uses available CPUs.

Use `--workers 4` to distribute points across four processes. Each worker computes
a point's entire trajectory using `solve_ivp`; GIF rendering stays in the main
process. Small runs may be slower due to process startup costs.

Animations overlay labeled q_t density contours at each timestep, with fixed
axes and contour levels. Above two dimensions, contours show the marginal
density of the first two coordinates; 1D runs show a density curve with samples
along the x-axis.
Use `python main.py --help` for all options.

Gaussian priors approximate q_T at large T. At finite T, initialization mismatch
can affect the final samples. GIF contours always show the exact forward q_t,
even when starting points come from the approximate prior.

## Custom Gaussian parameters

Each whitespace-separated row contains a mean followed by its covariance,
flattened row by row: `d + d²` values. Covariances must be symmetric and positive
definite. For example, a 2D mixture with identity covariances:

```text
-2 0 1 0 0 1
 2 0 1 0 0 1
```

```bash
python main.py -d 2 -p params.txt --seed 42
```

## Saved data

With `--standardize`, diffusion and saved samples use standardized coordinates.
The GIF transforms points and density contours back to original coordinates
at every timestep.
The archive stores `standardized`, plus `standardization_mean` and
`standardization_scale` when enabled. Recover original coordinates with
`samples * standardization_scale + standardization_mean`. Finite samples may
have empirical means and variances that differ from the exact q0 moments.

The NumPy archive contains `samples`, `terminal_samples`, `means`, `covs`,
`sde`, `start_t`, `end_t`, and `terminal_distribution`. When animation is enabled, it also includes
`times` and `trajectory` with shape `(steps + 1, samples, dimensions)`.

```python
import numpy as np

data = np.load("samples.npz")
samples = data["samples"]
```

## Single-intervention gradient experiment (CPU only)

`RL_finetune.py` implements the proposal's first proof of concept using NumPy
and SciPy, with no PyTorch, GPU, training loop or learned controller:

```bash
python RL_finetune.py -p param.txt --target 5 0 --reward-std 1 \
  --intervention-time 2.5 --strength 0.1 --standardize --workers 4
```

The reward is `r(x) = log N(x; target, reward_std² I)`. Target and reward width
are specified in original coordinates, including when standardizing. The script
computes the remaining sampler's sensitivity `J = dF_t/dx_t` by integrating its
variational ODE, then applies one correction in the direction `Jᵀ grad r(F_t(x_t))`.
Specifically, `u = strength D² Jᵀ grad r`, unbounded by default. Supply
`--epsilon 0.5` to clip the correction to `||D⁻¹u||₂ <= 0.5`, or `--epsilon 0`
to disable the correction. The Python API uses `epsilon=None` for unbounded
corrections. Archives record `bounded` and a numeric `epsilon` (`inf` when unbounded).
`D` contains standard deviations estimated from independent unguided calibration
rollouts (`--calibration-samples`, default 64). The base diffusion is unchanged.

`--workers 1` (default) runs sequentially; `--workers 0` uses available CPUs.
Multiple processes parallelize calibration and each point's full paired
trajectory, including the sensitivity solve. Seeded output and point ordering
match sequential runs. GIF rendering stays in the main process; small runs may
be slower due to process startup costs. Custom reward classes must be defined
at module scope and be picklable when using multiple workers. Python scripts
using the parallel API should call it inside `if __name__ == '__main__':`.

Compare times and strengths using paired initial noise:

```bash
python RL_finetune.py --target 2 0 --intervention-times 0 1 2.5 5 \
  --strengths 0 0.05 0.2 --seed 42 --gif ""
```

Times range from 0 (completed sample) to T (initial noise). Small strengths test
local sensitivity; large corrections can switch modes and need not improve
reward. The Gaussian reward can favor a distant component or a nearby target.
In particular, near mode boundaries the remaining sampler's Jacobian can be
large, so unbounded corrections can move a few points extremely far even if
most rewards improve. Start with small strengths (for example 0.001), compare
predicted versus actual gains, or opt into `--epsilon` for a finite budget.
The CLI reports correction-norm quantiles and the fraction of improved rewards;
archives also store `normalized_correction_norm` and `sampler_jacobian_norm`.

Outputs default to `finetuned_before.npz` / `finetuned_after.npz` and matching
GIFs. Sweeps add `t<index>_s<index>` to filenames. Both trajectories use identical
initial samples; repeated frame times show the instantaneous correction. Use
`--output`, `--gif`, `--steps` and `--fps` as in the sampling CLI. GIF contours
show the frozen model's q_t, while points show the actual corrected trajectories.
Standardized runs display GIFs in original coordinates and save arrays in
diffusion coordinates.

Archives include paired rewards, mode labels/switch frequency (maximum component
density assignment), residuals, latent scales, Jacobians, latent gradients,
predicted and actual reward gains, and predicted and actual final displacements.
For target-error analysis, compute squared distances using saved samples/target;
convert to original coordinates with the saved scale when desired.

The reference density `p*(x) ∝ q0(x) exp(beta r(x))` is exactly a Gaussian mixture.
Its weights, means and covariances are saved as `tilted_weights`, `tilted_means`,
and `tilted_covs`; `--beta` controls this reference only. A single gradient
correction generally does **not** sample p* exactly and need not preserve a
Gaussian-mixture form. The reference is for comparison, not a claimed output law.

For custom smooth objectives, pass a callable with `gradient(point)`:

```python
from RL_finetune import GaussianReward, GradientCorrection

experiment = GradientCorrection(GaussianReward([2., 0.]), model,
                                intervention_time=1., strength=0.1)
times, before, after, diagnostics = experiment.run(count=100, steps=50, seed=42)
```

The previous REINFORCE CLI, controller checkpoints, and DSRL/RFS method options
have been replaced by this experiment. `RL_finetune` and `reward` remain import
aliases for `GradientCorrection` and `GaussianReward`, with the new API above.
