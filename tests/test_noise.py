"""CPU-only checks of the noise samplers (no openpi needed):  python -m pytest tests/"""

import numpy as np
import pytest

from steer.noise import NoiseSampler, make

PI0, PI05 = (50, 32), (10, 32)


def test_default_lets_openpi_sample():
    assert NoiseSampler("default", PI0)() is None


@pytest.mark.parametrize("shape", [PI0, PI05])
def test_shapes_and_dtype(shape):
    for mode in ("gaussian", "fixed"):
        z = NoiseSampler(mode, shape, seed=1)()
        assert z.shape == shape and z.dtype == np.float32


def test_fixed_is_constant_gaussian_is_not():
    f, g = NoiseSampler("fixed", PI0, seed=1), NoiseSampler("gaussian", PI0, seed=1)
    assert np.array_equal(f(), f())
    assert not np.array_equal(g(), g())


def test_seeded_and_reproducible():
    assert np.array_equal(NoiseSampler("gaussian", PI0, seed=3)(), NoiseSampler("gaussian", PI0, seed=3)())
    assert not np.array_equal(NoiseSampler("fixed", PI0, seed=3)(), NoiseSampler("fixed", PI0, seed=4)())


def test_repeated_tiles_one_row():
    z = NoiseSampler("gaussian", PI0, seed=0, repeated=True)()
    assert np.all(z == z[0])


def test_make_from_config():
    z = make({"mode": "fixed", "seed": 5, "scale": 2.0}, PI05, seed=0)()
    ref = 2.0 * np.random.default_rng(5).standard_normal(PI05)
    assert np.allclose(z, ref.astype(np.float32))
    assert make(None, PI0, seed=0)() is None
