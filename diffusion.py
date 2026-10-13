from scipy.stats import multivariate_normal
from scipy.integrate import solve_ivp
from scipy.special import logsumexp
import numpy as np

def gaussian_pdf_gradient(x, mean, cov):
    x = np.asarray(x)
    mean = np.asarray(mean)
    cov = np.asarray(cov)
    
    density = multivariate_normal.pdf(x, mean=mean, cov=cov)
    diff = x - mean
    pseudo_inv_cov = np.linalg.pinv(cov)
    gradient = -density * (pseudo_inv_cov @ diff)
    
    return gradient

class Diffusion:
    """
    Diffusion with q0 an equally weighted mixture of Gaussians.

    VESDE: dX_t = dW_t.
    VPSDE: dX_t = -X_t dt + sqrt(2) dW_t.
    """

    def __init__(
        self,
        means,
        covs,
        T,
        ndim=2,
        sde='VPSDE', # one of 'VESDE' or 'VPSDE'
    ):
        self.ndim  = ndim
        self.means = np.asarray(means, dtype=float)
        self.covs  = np.asarray(covs, dtype=float)
        self.T     = T
        self.sde   = sde.upper()
        if self.sde not in {'VESDE', 'VPSDE'}:
            raise ValueError('sde must be "VESDE" or "VPSDE"')
        if not np.isfinite(T) or T <= 0:
            raise ValueError('T must be finite and positive')
        if self.means.ndim != 2 or self.means.shape[1] != ndim or len(self.means) == 0:
            raise ValueError('means must have shape (n_components, ndim)')
        if self.covs.shape != (len(self.means), ndim, ndim):
            raise ValueError('covs must have shape (n_components, ndim, ndim)')

    def score_fn(self, x, t):
        """
        Return gradient of log q_t(x) for a single point given the
        means, covariances, and diffusion timestep t.
        """
        x = np.asarray(x, dtype=float)
        if x.shape != (self.ndim,) or not np.all(np.isfinite(x)):
            raise ValueError('x must be a finite vector of shape (ndim,)')
        if not np.isfinite(t) or not 0 <= t <= self.T:
            raise ValueError('t must lie between 0 and T')

        identity = np.eye(self.ndim)
        if self.sde == 'VESDE':
            means = self.means
            covs = self.covs + t * identity
        else:
            decay = np.exp(-t)
            means = decay * self.means
            covs = decay**2 * self.covs + (-np.expm1(-2 * t)) * identity

        log_densities = np.array([
            multivariate_normal.logpdf(x, mean=mean, cov=cov)
            for mean, cov in zip(means, covs)
        ])
        weights = np.exp(log_densities - logsumexp(log_densities))
        component_scores = np.linalg.solve(covs, (means - x)[..., None])[..., 0]
        return weights @ component_scores

    def reverse(
        self,
        x_t,
        start_t,
        end_t = 0,
    ):
        """
        Integrate the probability-flow ODE backward and return x at end_t.
        """
        x_t = np.asarray(x_t, dtype=float)
        if x_t.shape != (self.ndim,) or not np.all(np.isfinite(x_t)):
            raise ValueError('x_t must be a finite vector of shape (ndim,)')
        if not (np.isfinite(start_t) and np.isfinite(end_t)
                and 0 <= end_t <= start_t <= self.T):
            raise ValueError('times must satisfy 0 <= end_t <= start_t <= T')
        if start_t == end_t:
            return x_t.copy()

        def velocity(t, x):
            # Runge-Kutta stage times can stray past an endpoint by roundoff.
            # The interval has already been validated; keep score evaluation
            # inside it while retaining strict validation for public calls.
            score = self.score_fn(x, np.clip(t, end_t, start_t))
            if self.sde == 'VESDE':
                return -0.5 * score
            return -x - score

        solution = solve_ivp(
            velocity, (start_t, end_t), x_t,
            t_eval=[end_t], rtol=1e-6, atol=1e-9,
        )
        if not solution.success:
            raise RuntimeError(f'Reverse integration failed: {solution.message}')
        return solution.y[:, -1]
