import jax
import numpy as np

from jaxps.rays import (
    cosine_weighted_directions,
    deterministic_hemisphere_directions,
    normalize_directions,
    polynomial_cosine_directions,
)


def test_direction_normalization():
    directions = normalize_directions(np.asarray([[3.0, 4.0], [0.0, 2.0]]))

    assert np.allclose(np.linalg.norm(np.asarray(directions), axis=-1), 1.0)


def test_cosine_sampling_reproducibility_and_hemisphere():
    key = jax.random.PRNGKey(7)
    dirs_a = cosine_weighted_directions(key, num_rays=128, ndim=3)
    dirs_b = cosine_weighted_directions(key, num_rays=128, ndim=3)
    arr = np.asarray(dirs_a)

    assert np.allclose(arr, np.asarray(dirs_b))
    assert np.all(arr[:, 2] >= 0.0)
    assert np.allclose(np.linalg.norm(arr, axis=-1), 1.0, atol=1e-6)


def test_polynomial_and_deterministic_sampling_shapes():
    key = jax.random.PRNGKey(3)
    poly = polynomial_cosine_directions(key, num_rays=32, exponent=2.0, ndim=2)
    det = deterministic_hemisphere_directions(12, ndim=3)

    assert poly.shape == (32, 2)
    assert det.shape == (12, 3)
    assert np.all(np.asarray(poly)[:, 1] >= 0.0)
    assert np.all(np.asarray(det)[:, 2] >= 0.0)
