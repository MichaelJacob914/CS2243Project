"""Run with: python -m unittest discover -s CS2243Project -p 'test_*.py'."""
import unittest
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from diffusion import Diffusion
from RL_finetune import GaussianReward, GradientCorrection, reverse_with_jacobian, score_and_jacobian


class GradientCorrectionTests(unittest.TestCase):
    def model(self, sde):
        return Diffusion([[-1., 0.], [2., 1.]],
                         [np.diag([1., 2.]), np.array([[2., .3], [.3, 1.]])], 1., sde=sde)

    def test_chain_rule_and_sensitivity(self):
        x, t, delta = np.array([.3, -.4]), .7, 1e-5
        objective = GaussianReward([1., 1.], np.diag([.5, 2.]))
        for sde in ['VESDE', 'VPSDE']:
            with self.subTest(sde=sde):
                model = self.model(sde)
                score, hessian = score_and_jacobian(model, x, t)
                np.testing.assert_allclose(score, model.score_fn(x, t))
                numeric_hessian = np.column_stack([
                    (model.score_fn(x + delta*v, t) - model.score_fn(x - delta*v, t))/(2*delta)
                    for v in np.eye(2)])
                np.testing.assert_allclose(hessian, numeric_hessian, atol=1e-8)
                endpoint, jacobian = reverse_with_jacobian(model, x, t)
                numeric_jacobian = np.column_stack([
                    (model.reverse(x + delta*v, t) - model.reverse(x - delta*v, t))/(2*delta)
                    for v in np.eye(2)])
                np.testing.assert_allclose(jacobian, numeric_jacobian, rtol=1e-4, atol=1e-5)
                numerical_gradient = np.array([
                    (objective(model.reverse(x + delta*v, t)) - objective(model.reverse(x - delta*v, t)))/(2*delta)
                    for v in np.eye(2)])
                np.testing.assert_allclose(jacobian.T @ objective.gradient(endpoint),
                                           numerical_gradient, rtol=1e-4, atol=1e-5)

    def test_tilted_density_identity(self):
        model = self.model('VPSDE')
        objective = GaussianReward([1., 1.], np.diag([.5, 2.]))
        weights, means, covs = objective.tilted_mixture(model, .8)
        def density(x, weights, means, covs):
            return logsumexp([np.log(w) + multivariate_normal.logpdf(x, m, c)
                              for w, m, c in zip(weights, means, covs)])
        offsets = [density(x, weights, means, covs)
                   - density(x, [.5, .5], model.means, model.covs) - .8*objective(x)
                   for x in [[0., 0.], [1., 2.], [-3., .5]]]
        np.testing.assert_allclose(offsets, np.repeat(offsets[0], 3), atol=1e-12)

    def test_paired_trajectories_and_budget(self):
        model = self.model('VPSDE')
        objective = GaussianReward([1., 1.])
        for time in [0., .6, 1.]:
            experiment = GradientCorrection(objective, model, intervention_time=time,
                                            strength=0., calibration_samples=3)
            times, before, after, diagnostics = experiment.run(2, steps=2)
            np.testing.assert_array_equal(before, after)
            self.assertEqual(sum(np.diff(times) == 0), 1)
            experiment.strength = 100.
            residual, _, _, predicted = experiment.correction(np.array([.3, -.4]))
            self.assertLessEqual(np.linalg.norm(residual/experiment.scale), experiment.epsilon + 1e-12)
            self.assertGreaterEqual(predicted, 0.)

    def test_parallel_matches_serial(self):
        model = self.model('VESDE')
        objective = GaussianReward([1., 1.])
        options = dict(intervention_time=.6, calibration_samples=3,
                       terminal_distribution='exact', seed=7)
        serial = GradientCorrection(objective, model, workers=1, **options)
        parallel = GradientCorrection(objective, model, workers=2, **options)
        np.testing.assert_array_equal(serial.scale, parallel.scale)
        expected = serial.run(3, steps=2)
        actual = parallel.run(3, steps=2)
        for a, b in zip(expected[:3], actual[:3]):
            np.testing.assert_array_equal(a, b)
        for key in expected[3]:
            np.testing.assert_array_equal(expected[3][key], actual[3][key])

    def test_invalid_workers(self):
        for workers in [-1, 1.5]:
            with self.assertRaises(ValueError):
                GradientCorrection(GaussianReward([1., 1.]), self.model('VPSDE'), workers=workers)


if __name__ == '__main__':
    unittest.main()
