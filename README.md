# Gaussian Mixture Diffusion

Sample an equally weighted Gaussian mixture by drawing from its exact forward
distribution at time `T`, then integrating the reverse probability-flow ODE with
SciPy's `solve_ivp`.

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
- `-d` / `--dimensions`: sample dimension (default: 2).
- `-p` / `--params`: Gaussian parameter text file.
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

The NumPy archive contains `samples`, `terminal_samples`, `means`, `covs`,
`sde`, `start_t`, and `end_t`. When animation is enabled, it also includes
`times` and `trajectory` with shape `(steps + 1, samples, dimensions)`.

```python
import numpy as np

data = np.load("samples.npz")
samples = data["samples"]
```
