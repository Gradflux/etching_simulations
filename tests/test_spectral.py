import numpy as np
import jax
import jax.numpy as jnp
import pytest

from jaxps.experimental import periodic_spectral_derivative, periodic_spectral_laplacian

# Metal does not support complex-number FFT operations at the hardware level.
pytestmark = pytest.mark.skipif(
    jax.default_backend().lower() == "metal",
    reason="Metal backend does not support complex FFT",
)


def test_periodic_spectral_derivative_matches_smooth_periodic_function():
    n = 64
    length = 2.0 * np.pi
    x = jnp.arange(n) * (length / n)
    field = jnp.sin(3.0 * x)
    derivative = periodic_spectral_derivative(field, (length,), axis=0)

    assert np.allclose(np.asarray(derivative), np.asarray(3.0 * jnp.cos(3.0 * x)), atol=2e-5)


def test_periodic_spectral_laplacian_matches_sine_mode():
    n = 64
    length = 2.0 * np.pi
    x = jnp.arange(n) * (length / n)
    y = jnp.arange(n) * (length / n)
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    field = jnp.sin(2.0 * xx) * jnp.cos(3.0 * yy)
    laplacian = periodic_spectral_laplacian(field, (length, length))

    assert np.allclose(np.asarray(laplacian), np.asarray(-13.0 * field), atol=1e-3)
