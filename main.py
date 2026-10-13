import argparse
import multiprocessing as mp
import numpy as np
import diffusion
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dimensions', type=int, default=2, help='Number of dimensions')
    parser.add_argument('-p', '--params', type=str, default=None,
                        help='Text file: one component per row, mean followed by flattened covariance')
    parser.add_argument('-T', type=float, default=10., help='Maximum time for the SDE')
    parser.add_argument('--sde', type=str.upper, choices=['VESDE', 'VPSDE'], default='VPSDE')
    parser.add_argument('--terminal-distribution', choices=['approximate', 'exact'],
                        default='approximate',
                        help='Initial sampling law: Gaussian prior or exact q_T (default: approximate)')
    parser.add_argument('-n', '--samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--standardize', action='store_true',
                        help='Transform q0 to zero mean and unit variance in each dimension')
    parser.add_argument('--end-time', type=float, default=0., help='Time to reverse to')
    parser.add_argument('-o', '--output', default='samples.npz', help='Output NumPy archive')
    parser.add_argument('--gif', default='output.gif', metavar='PATH', help='Save an animation of reverse sampling')
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of ODE intervals when creating a GIF (default: 50)')
    parser.add_argument('--fps', type=int, default=10, help='GIF frames per second (default: 10)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Worker processes (default: 1; 0 uses available CPUs)')

    args = parser.parse_args()
    if args.dimensions <= 0 or args.samples <= 0:
        parser.error('dimensions and samples must be positive')
    if not np.isfinite(args.T) or args.T <= 0:
        parser.error('T must be finite and positive')
    if not np.isfinite(args.end_time) or not 0 <= args.end_time <= args.T:
        parser.error('end-time must lie between 0 and T')
    if args.seed is not None and args.seed < 0:
        parser.error('seed must be nonnegative')
    if args.steps <= 0 or args.fps <= 0:
        parser.error('steps and fps must be positive')
    if args.workers < 0:
        parser.error('workers must be nonnegative')
    return args


def load_parameters(path, ndim):
    """Load rows [mean, row-major covariance], or use two default Gaussians."""
    if path is None:
        means = np.zeros((2, ndim))
        means[:, 0] = [-2., 2.]
        covs = np.repeat(np.eye(ndim)[None, :, :], 2, axis=0)
    else:
        rows = np.loadtxt(path, ndmin=2)
        if rows.shape[0] == 0 or rows.shape[1] != ndim + ndim**2:
            raise ValueError(f'Each parameter row must contain {ndim + ndim**2} values')
        means = rows[:, :ndim]
        covs = rows[:, ndim:].reshape(-1, ndim, ndim)
    if not (np.all(np.isfinite(means)) and np.all(np.isfinite(covs))):
        raise ValueError('Gaussian parameters must be finite')
    for cov in covs:
        if not np.allclose(cov, cov.T):
            raise ValueError('Covariances must be symmetric')
        try:
            np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError('Covariances must be positive definite') from exc
    return means, covs


def standardize_parameters(means, covs):
    """Standardize using exact moments of the equally weighted mixture."""
    center = means.mean(axis=0)
    centered = means - center
    variance = (np.diagonal(covs, axis1=1, axis2=2) + centered**2).mean(axis=0)
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
        raise ValueError('Mixture variances must be finite and positive')
    scale = np.sqrt(variance)
    return centered / scale, covs / scale[None, :, None] / scale[None, None, :], center, scale


def _point_trajectory(model, times, x):
    positions = np.empty((len(times), model.ndim))
    positions[0] = x
    for step in range(len(times) - 1):
        positions[step + 1] = model.reverse(positions[step], times[step], times[step + 1])
    return positions


def _initialize_worker(model, times):
    global _worker_model, _worker_times
    _worker_model, _worker_times = model, times


def _sample_point(x):
    return _point_trajectory(_worker_model, _worker_times, x)


def sample_trajectory(model, terminal_samples, end_time, steps, workers=1):
    """Solve each point's ODE separately over each of `steps` intervals."""
    times = np.linspace(model.T, end_time, steps + 1)
    count = len(terminal_samples)
    worker_count = min(workers or mp.cpu_count(), count)
    if worker_count == 1:
        results = [_point_trajectory(model, times, x)
                   for x in tqdm(terminal_samples, desc='Sampling', total=count)]
    else:
        # Spawn works on macOS/Windows and avoids inheriting numerical-library state.
        with mp.get_context('spawn').Pool(
            worker_count, initializer=_initialize_worker, initargs=(model, times),
        ) as pool:
            results = list(tqdm(
                pool.imap(_sample_point, terminal_samples,
                          chunksize=max(1, count // (4 * worker_count))),
                total=count, desc='Sampling',
            ))
    # Ordered results preserve point identities and deterministic seeded output.
    trajectory = np.stack(results, axis=1)
    return times, trajectory


def marginal_parameters(model, t):
    """Parameters of q_t marginalized to the displayed dimensions."""
    ndim = min(model.ndim, 2)
    means = model.means[:, :ndim]
    covs = model.covs[:, :ndim, :ndim]
    if model.sde == 'VESDE':
        return means, covs + t * np.eye(ndim)
    decay = np.exp(-t)
    return decay * means, decay**2 * covs + (-np.expm1(-2 * t)) * np.eye(ndim)


def marginal_density(points, means, covs):
    """Evaluate the equally weighted Gaussian mixture on a plotting grid."""
    from scipy.stats import multivariate_normal
    return sum(multivariate_normal.pdf(points, mean=mean, cov=cov)
               for mean, cov in zip(means, covs)) / len(means)


def save_sampling_gif(path, times, trajectory, model, fps, center=None, scale=None):
    """Animate fixed point identities; project onto the first two dimensions."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.animation import FuncAnimation, PillowWriter

    positions = trajectory[..., :2]
    parameters = [marginal_parameters(model, t) for t in times]
    if center is not None and scale is not None:
        ndim = positions.shape[-1]
        display_center = center[:ndim]
        display_scale = scale[:ndim]
        positions = positions * display_scale + display_center
        # Transform both means and covariances so density values include
        # the change-of-coordinates Jacobian through the Gaussian PDF.
        parameters = [
            (means * display_scale + display_center,
             covs * display_scale[None, :, None] * display_scale[None, None, :])
            for means, covs in parameters
        ]
    if positions.shape[-1] == 1:
        positions = np.concatenate([positions, np.zeros_like(positions)], axis=-1)
    fig = Figure(figsize=(6, 6))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    lower = positions.min(axis=(0, 1))
    upper = positions.max(axis=(0, 1))
    for means, covs in parameters:
        spread = 3 * np.sqrt(np.diagonal(covs, axis1=1, axis2=2))
        ndim = means.shape[1]
        lower[:ndim] = np.minimum(lower[:ndim], (means - spread).min(axis=0))
        upper[:ndim] = np.maximum(upper[:ndim], (means + spread).max(axis=0))
    padding = 0.1 * np.maximum(upper - lower, 1.)
    ax.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
    ax.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2' if trajectory.shape[-1] >= 2 else '1D samples (y = 0)')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.25)
    grid_x = np.linspace(*ax.get_xlim(), 120)
    contours = None
    density_line = None
    if model.ndim >= 2:
        grid_y = np.linspace(*ax.get_ylim(), 120)
        xx, yy = np.meshgrid(grid_x, grid_y)
        grid = np.stack([xx, yy], axis=-1)
        densities = [marginal_density(grid, means, covs) for means, covs in parameters]
        # Fixed levels make density values comparable across timesteps.
        peak = max(density.max() for density in densities)
        levels = peak * np.array([.01, .025, .05, .1, .2, .4, .6, .8])
    else:
        densities = [marginal_density(grid_x[:, None], means, covs)
                     for means, covs in parameters]
        ax.set_aspect('auto')
        ax.set_ylim(-.05 * max(d.max() for d in densities),
                    1.1 * max(d.max() for d in densities))
        ax.set_ylabel('Density q_t(x)')
        density_line, = ax.plot(grid_x, densities[0], color='steelblue', label='q_t')
        ax.legend(loc='upper right')
    points = ax.scatter(*positions[0].T, c=np.arange(trajectory.shape[1]),
                        cmap='viridis', s=25, alpha=0.8, zorder=3)
    title = ax.set_title('')

    def update(frame):
        nonlocal contours
        if model.ndim >= 2:
            if contours is not None:
                contours.remove()
            contours = ax.contour(xx, yy, densities[frame], levels=levels,
                                  colors='steelblue', linewidths=0.8, alpha=0.8)
            ax.clabel(contours, inline=True, fontsize=7, fmt='%.3g')
        else:
            density_line.set_ydata(densities[frame])
        points.set_offsets(positions[frame])
        projection = ' (2D marginal density)' if trajectory.shape[-1] > 2 else ''
        title.set_text(f'{model.sde} reverse sampling and q_t{projection}\n'
                       f't = {times[frame]:.4f} | step {frame}/{len(times) - 1}')
        return points, title

    animation = FuncAnimation(fig, update, frames=len(times), interval=1000 / fps)
    animation.save(path, writer=PillowWriter(fps=fps))


def sample_terminal(model, count, rng, distribution='approximate'):
    """Draw from the diffusion Gaussian prior or the exact forward mixture."""
    if distribution == 'approximate':
        std = np.sqrt(model.T) if model.sde == 'VESDE' else 1.
        return std * rng.standard_normal((count, model.ndim))
    if distribution != 'exact':
        raise ValueError('distribution must be approximate or exact')
    components = rng.integers(len(model.means), size=count)
    if model.sde == 'VESDE':
        means = model.means
        covs = model.covs + model.T * np.eye(model.ndim)
    else:
        decay = np.exp(-model.T)
        means = decay * model.means
        covs = decay**2 * model.covs + (-np.expm1(-2 * model.T)) * np.eye(model.ndim)
    samples = np.empty((count, model.ndim))
    for component in range(len(means)):
        mask = components == component
        samples[mask] = rng.multivariate_normal(
            means[component], covs[component], size=int(mask.sum()),
        )
    return samples


def main():
    args = parse_arguments()
    means, covs = load_parameters(args.params, args.dimensions)
    standardization_data = {'standardized': args.standardize}
    center = scale = None
    if args.standardize:
        means, covs, center, scale = standardize_parameters(means, covs)
        standardization_data.update(standardization_mean=center, standardization_scale=scale)
    model = diffusion.Diffusion(means, covs, args.T, ndim=args.dimensions, sde=args.sde)
    rng = np.random.default_rng(args.seed)

    terminal_samples = sample_terminal(model, args.samples, rng, args.terminal_distribution)
    animation_data = {}
    if args.gif:
        times, trajectory = sample_trajectory(
            model, terminal_samples, args.end_time, args.steps, args.workers,
        )
        samples = trajectory[-1]
        animation_data = {'times': times, 'trajectory': trajectory}
        save_sampling_gif(args.gif, times, trajectory, model, args.fps, center, scale)
        print(f'Saved {args.steps + 1} animation frames to {args.gif}')
    else:
        _, trajectory = sample_trajectory(
            model, terminal_samples, args.end_time, 1, args.workers,
        )
        samples = trajectory[-1]
    # Opening the path explicitly preserves the requested filename.
    with open(args.output, 'wb') as output:
        np.savez_compressed(
            output, samples=samples, terminal_samples=terminal_samples,
            means=means, covs=covs, sde=args.sde,
            start_t=args.T, end_t=args.end_time,
            terminal_distribution=args.terminal_distribution,
            **animation_data,
            **standardization_data,
        )
    print(f'Saved {args.samples} samples at t={args.end_time:g} to {args.output}')


if __name__ == "__main__":
    main()
