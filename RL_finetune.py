"""CPU-only single-intervention gradient experiment from the project proposal.

Apply u = strength * D^2 * J^T grad r(F_t(x)), unbounded by default.
Optionally clip to ||D^-1 u|| <= epsilon by supplying --epsilon.
J is computed by a variational ODE alongside the frozen probability-flow ODE.
The Gaussian reward gives an analytic exponentially tilted reference mixture;
this single correction is a transport heuristic, not an exact tilted sampler.
"""
import argparse
import multiprocessing as mp
from functools import partial
from numbers import Integral
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from tqdm import tqdm
from diffusion import Diffusion
from main import load_parameters, standardize_parameters, sample_terminal, save_sampling_gif


def _initialize_worker(function):
    global _worker_function
    _worker_function = function


def _call_worker(point):
    return _worker_function(point)


def _ordered_map(function, points, workers, description):
    """Send model/reward state once per spawned worker, preserving input order."""
    count = len(points)
    processes = min(workers or mp.cpu_count(), count)
    if processes == 1:
        return [function(x) for x in tqdm(points, total=count, desc=description)]
    with mp.get_context('spawn').Pool(
        processes, initializer=_initialize_worker, initargs=(function,),
    ) as pool:
        return list(tqdm(pool.imap(_call_worker, points,
                                  chunksize=max(1, count // (4 * processes))),
                         total=count, desc=description))


class GaussianReward:
    """Smooth log N(target, cov) reward, with an analytic gradient."""
    def __init__(self, target, cov=None):
        self.target = np.asarray(target, dtype=float)
        if self.target.ndim != 1 or not len(self.target) or not np.all(np.isfinite(self.target)):
            raise ValueError('target must be a finite nonempty vector')
        self.cov = np.eye(len(self.target)) if cov is None else np.asarray(cov, dtype=float)
        if self.cov.shape != (len(self.target), len(self.target)) or not np.all(np.isfinite(self.cov)):
            raise ValueError('invalid reward covariance shape or values')
        if not np.allclose(self.cov, self.cov.T):
            raise ValueError('reward covariance must be symmetric')
        np.linalg.cholesky(self.cov)
        self.precision = np.linalg.inv(self.cov)

    def __call__(self, x):
        return multivariate_normal.logpdf(x, mean=self.target, cov=self.cov)

    def gradient(self, x):
        return -(np.asarray(x) - self.target) @ self.precision

    def tilted_mixture(self, model, beta=1.):
        """Exact parameters of p*(x) proportional to q0(x) exp(beta*r(x))."""
        if not np.isfinite(beta) or beta < 0:
            raise ValueError('beta must be finite and nonnegative')
        if beta == 0:
            return np.full(len(model.means), 1 / len(model.means)), model.means.copy(), model.covs.copy()
        likelihood_cov = self.cov / beta
        covs = np.array([np.linalg.inv(np.linalg.inv(c) + beta * self.precision)
                         for c in model.covs])
        means = np.array([c @ (np.linalg.solve(base, m) + beta * self.precision @ self.target)
                          for c, base, m in zip(covs, model.covs, model.means)])
        log_weights = np.array([multivariate_normal.logpdf(self.target, mean=m, cov=c + likelihood_cov)
                                for m, c in zip(model.means, model.covs)])
        return np.exp(log_weights - logsumexp(log_weights)), means, covs


# Keep the previous module's reward name available for imports.
reward = GaussianReward


def forward_parameters(model, t):
    decay = np.exp(-t) if model.sde == 'VPSDE' else 1.
    variance = -np.expm1(-2 * t) if model.sde == 'VPSDE' else t
    return decay * model.means, decay**2 * model.covs + variance * np.eye(model.ndim)


def score_and_jacobian(model, x, t):
    """Analytic Gaussian-mixture score and Hessian of log q_t."""
    means, covs = forward_parameters(model, t)
    precisions = np.linalg.inv(covs)
    component_scores = np.einsum('kij,kj->ki', precisions, means - x)
    logs = np.array([multivariate_normal.logpdf(x, mean=m, cov=c) for m, c in zip(means, covs)])
    weights = np.exp(logs - logsumexp(logs))
    score = weights @ component_scores
    centered = component_scores - score
    hessian = -np.einsum('k,kij->ij', weights, precisions)
    hessian += np.einsum('k,ki,kj->ij', weights, centered, centered)
    return score, hessian


def reverse_with_jacobian(model, x, start_t):
    """Return F_t(x) and dF_t/dx using dJ/dt = (dv/dx) J."""
    x = np.asarray(x, dtype=float)
    if x.shape != (model.ndim,) or not np.all(np.isfinite(x)):
        raise ValueError('x must be a finite point of shape (ndim,)')
    if not np.isfinite(start_t) or not 0 <= start_t <= model.T:
        raise ValueError('start_t must lie between 0 and T')
    n = model.ndim
    if start_t == 0:
        return x.copy(), np.eye(n)

    def rhs(t, state):
        score, hessian = score_and_jacobian(model, state[:n], np.clip(t, 0., start_t))
        if model.sde == 'VESDE':
            velocity, derivative = -.5 * score, -.5 * hessian
        else:
            velocity, derivative = -state[:n] - score, -np.eye(n) - hessian
        return np.concatenate([velocity, (derivative @ state[n:].reshape(n, n)).ravel()])

    result = solve_ivp(rhs, (start_t, 0.), np.concatenate([x, np.eye(n).ravel()]),
                       t_eval=[0.], rtol=1e-7, atol=1e-9)
    if not result.success:
        raise RuntimeError(f'Sensitivity integration failed: {result.message}')
    return result.y[:n, -1], result.y[n:, -1].reshape(n, n)


class GradientCorrection:
    """One gradient correction at time t, with an optional normalized budget.

    D uses empirical marginal standard deviations from independent unguided
    calibration rollouts, shared across strengths. Custom rewards must implement
    __call__(point) and gradient(point), both in diffusion coordinates.
    """
    def __init__(self, reward_fn, model, *, intervention_time=None, strength=.1,
                 epsilon=None, terminal_distribution='approximate', seed=0,
                 calibration_samples=64, workers=1):
        if not isinstance(workers, Integral) or workers < 0:
            raise ValueError('workers must be a nonnegative integer')
        self.workers = workers
        self.reward_fn, self.model = reward_fn, model
        self.time = model.T / 2 if intervention_time is None else intervention_time
        for name, value in [('strength', strength), ('epsilon', epsilon)]:
            if name == 'epsilon' and value is None:
                continue
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not np.isfinite(self.time) or not 0 <= self.time <= model.T:
            raise ValueError('intervention_time must lie between 0 and T')
        if calibration_samples < 2:
            raise ValueError('calibration_samples must be >= 2')
        if not callable(reward_fn) or not callable(getattr(reward_fn, 'gradient', None)):
            raise TypeError('reward_fn must be callable and provide gradient(point)')
        self.strength, self.epsilon = strength, epsilon
        self.terminal_distribution, self.seed = terminal_distribution, seed
        initial = sample_terminal(model, calibration_samples, np.random.default_rng(seed + 1), terminal_distribution)
        latents = np.array(_ordered_map(
            partial(model.reverse, start_t=model.T, end_t=self.time), initial,
            workers, 'Calibrating'))
        self.scale = np.maximum(latents.std(axis=0), 1e-8)

    def correction(self, x):
        endpoint, jacobian = reverse_with_jacobian(self.model, x, self.time)
        grad_reward = np.asarray(self.reward_fn.gradient(endpoint), dtype=float)
        if grad_reward.shape != (self.model.ndim,) or not np.all(np.isfinite(grad_reward)):
            raise ValueError('reward gradient must be a finite vector of shape (ndim,)')
        gradient = jacobian.T @ grad_reward
        # Gradient ascent in coordinates z = D^-1 x, with an optional L2 budget.
        normalized = self.strength * self.scale * gradient
        norm = np.linalg.norm(normalized)
        if self.epsilon is not None and norm > self.epsilon:
            normalized *= self.epsilon / norm
        residual = self.scale * normalized
        return residual, jacobian, gradient, float(gradient @ residual)

    def _run_point(self, initial, times):
        """Keep all ODE intervals and the sensitivity solve for a point together."""
        before, after = [initial.copy()], [initial.copy()]
        frame_times = [times[0]]
        correction = None
        for index, time in enumerate(times):
            if index:
                before.append(self.model.reverse(before[-1], times[index - 1], time))
                after.append(self.model.reverse(after[-1], times[index - 1], time))
                frame_times.append(time)
            if time == self.time:
                correction = self.correction(after[-1])
                before.append(before[-1].copy())
                after.append(after[-1] + correction[0])
                frame_times.append(time)
        return np.asarray(frame_times), np.asarray(before), np.asarray(after), correction

    def run(self, count=100, steps=50, seed=None):
        """Paired trajectories; workers compute complete points independently."""
        if count <= 0 or steps <= 0:
            raise ValueError('count and steps must be positive')
        initial = sample_terminal(self.model, count, np.random.default_rng(self.seed if seed is None else seed),
                                  self.terminal_distribution)
        times = np.unique(np.append(np.linspace(self.model.T, 0., steps + 1), self.time))[::-1]
        results = _ordered_map(partial(self._run_point, times=times), initial,
                               self.workers, 'Gradient correction')
        frame_times = results[0][0]
        before = np.stack([row[1] for row in results], axis=1)
        after = np.stack([row[2] for row in results], axis=1)
        residuals, jacobians, gradients, predicted = map(np.asarray, zip(*(row[3] for row in results)))
        diagnostics = dict(residuals=residuals, jacobians=jacobians,
                           gradients=gradients, predicted_reward_gain=predicted,
                           predicted_displacement=np.einsum('nij,nj->ni', jacobians, residuals),
                           normalized_correction_norm=np.linalg.norm(residuals / self.scale, axis=1),
                           sampler_jacobian_norm=np.linalg.norm(jacobians, ord=2, axis=(1, 2)),
                           latent_scale=self.scale)
        before_rewards = np.array([self.reward_fn(x) for x in before[-1]])
        after_rewards = np.array([self.reward_fn(x) for x in after[-1]])
        if not np.all(np.isfinite([before_rewards, after_rewards])):
            raise ValueError('reward values must be finite')
        diagnostics.update(before_rewards=before_rewards, after_rewards=after_rewards,
                           actual_reward_gain=after_rewards - before_rewards,
                           actual_displacement=after[-1] - before[-1])
        # Assign a mode by maximum component density, accounting for covariances.
        def modes(points):
            return np.argmax(np.array([multivariate_normal.logpdf(points, mean=m, cov=c)
                                       for m, c in zip(self.model.means, self.model.covs)]).reshape(len(self.model.means), count), axis=0)
        before_modes, after_modes = modes(before[-1]), modes(after[-1])
        diagnostics.update(before_modes=before_modes, after_modes=after_modes,
                           mode_switch_frequency=np.mean(before_modes != after_modes))
        return np.asarray(frame_times), before, after, diagnostics


# Existing imports can use this name, but the learned-controller API is replaced.
RL_finetune = GradientCorrection


def staged_path(base, stage, suffix):
    path = Path(base)
    return path.with_name(f'{path.stem}_{stage}{path.suffix or suffix}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p', '--params')
    parser.add_argument('-d', '--dimensions', type=int, default=2)
    parser.add_argument('--sde', choices=['VESDE', 'VPSDE'], default='VPSDE')
    parser.add_argument('-T', type=float, default=5.)
    parser.add_argument('--target', type=float, nargs='+', required=True, help='Reward center in original coordinates')
    parser.add_argument('--reward-std', type=float, default=1., help='Isotropic reward std in original coordinates')
    parser.add_argument('--beta', type=float, default=1., help='Exponent for the analytic tilted reference mixture')
    parser.add_argument('--intervention-time', type=float)
    parser.add_argument('--intervention-times', type=float, nargs='+', help='Sweep multiple intervention times')
    parser.add_argument('--strength', type=float, default=.1)
    parser.add_argument('--strengths', type=float, nargs='+', help='Sweep multiple correction strengths')
    parser.add_argument('--epsilon', type=float, default=None,
                        help='Opt into ||D^-1 u|| <= epsilon (default: unbounded)')
    parser.add_argument('--calibration-samples', type=int, default=64)
    parser.add_argument('--terminal-distribution', choices=['approximate', 'exact'], default='approximate')
    parser.add_argument('--standardize', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-n', '--samples', type=int, default=100)
    parser.add_argument('-o', '--output', default='finetuned.npz')
    parser.add_argument('--gif', default='finetuned.gif', help='GIF base path; empty disables GIFs')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--workers', type=int, default=1,
                        help='Worker processes (default: 1; 0 uses available CPUs)')
    args = parser.parse_args()
    if args.dimensions <= 0 or len(args.target) != args.dimensions:
        parser.error('dimensions must be positive and match target length')
    if min(args.samples, args.steps, args.fps) <= 0:
        parser.error('samples, steps and fps must be positive')
    if args.workers < 0:
        parser.error('workers must be nonnegative')
    if not np.isfinite(args.reward_std) or args.reward_std <= 0:
        parser.error('reward-std must be finite and positive')
    means, covs = load_parameters(args.params, args.dimensions)
    target = np.asarray(args.target)
    reward_cov = args.reward_std**2 * np.eye(args.dimensions)
    center = scale = None
    metadata = dict(standardized=args.standardize)
    if args.standardize:
        means, covs, center, scale = standardize_parameters(means, covs)
        target = (target - center) / scale
        reward_cov = reward_cov / scale[:, None] / scale[None, :]
        metadata.update(standardization_mean=center, standardization_scale=scale)
    model = Diffusion(means, covs, args.T, args.dimensions, args.sde)
    objective = GaussianReward(target, reward_cov)
    weights, reference_means, reference_covs = objective.tilted_mixture(model, args.beta)
    times = args.intervention_times or [args.intervention_time if args.intervention_time is not None else args.T / 2]
    strengths = args.strengths or [args.strength]
    sweep = len(times) * len(strengths) > 1
    for ti, time in enumerate(times):
        for si, strength in enumerate(strengths):
            experiment = GradientCorrection(objective, model, intervention_time=time,
                                            strength=strength, epsilon=args.epsilon, seed=args.seed,
                                            calibration_samples=args.calibration_samples,
                                            workers=args.workers,
                                            terminal_distribution=args.terminal_distribution)
            frame_times, before, after, diagnostics = experiment.run(args.samples, args.steps)
            tag = f't{ti}_s{si}_' if sweep else ''
            for stage, trajectory in [('before', before), ('after', after)]:
                path = staged_path(args.output, tag + stage, '.npz')
                with open(path, 'wb') as output:
                    np.savez_compressed(output, samples=trajectory[-1], terminal_samples=trajectory[0],
                                        trajectory=trajectory, times=frame_times, means=means, covs=covs,
                                        target=target, reward_cov=reward_cov, sde=args.sde, T=args.T,
                                        start_t=args.T, end_t=0., stage=stage, seed=args.seed,
                                        intervention_time=time, strength=strength,
                                        epsilon=np.inf if args.epsilon is None else args.epsilon,
                                        bounded=args.epsilon is not None,
                                        terminal_distribution=args.terminal_distribution, beta=args.beta,
                                        tilted_weights=weights, tilted_means=reference_means, tilted_covs=reference_covs,
                                        **metadata, **diagnostics)
                if args.gif:
                    gif_path = staged_path(args.gif, tag + stage, '.gif')
                    save_sampling_gif(gif_path, frame_times, trajectory, model, args.fps, center, scale)
                print(f't={time:g}, strength={strength:g}: saved {path}')
            print(f"Mean reward: {diagnostics['before_rewards'].mean():.4f} -> "
                  f"{diagnostics['after_rewards'].mean():.4f}; "
                  f"mode switches: {diagnostics['mode_switch_frequency']:.1%}")
            correction_sizes = np.quantile(diagnostics['normalized_correction_norm'], [.5, .99, 1.])
            print('Normalized correction norm (median / p99 / max): '
                  + ' / '.join(f'{value:.4g}' for value in correction_sizes)
                  + f"; reward improved: {np.mean(diagnostics['actual_reward_gain'] > 0):.1%}")


if __name__ == '__main__':
    main()
