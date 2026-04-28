import jax.numpy as jnp
import numpy as np

from jaxps.rays import (
    accumulate_flux,
    accumulate_flux_chunked,
    flux_to_deposition_rate,
    flux_to_etch_rate,
    surface_exposure,
)


def test_flux_nonnegative_and_zero_rays():
    normals = jnp.asarray([[[0.0, 1.0], [1.0, 0.0]]])
    directions = jnp.asarray([[0.0, 1.0], [1.0, 0.0]])
    flux = accumulate_flux(normals, directions)

    assert np.all(np.asarray(flux) >= 0.0)
    assert np.allclose(np.asarray(accumulate_flux(normals, jnp.empty((0, 2)))), 0.0)


def test_surface_exposure_and_flux_to_rate():
    normals = jnp.asarray([[0.0, 1.0]])
    directions = jnp.asarray([[0.0, 1.0], [1.0, 0.0]])
    exposure = surface_exposure(normals, directions)

    assert np.allclose(np.asarray(exposure), [[1.0, 0.0]])
    assert np.allclose(np.asarray(flux_to_deposition_rate(jnp.asarray([2.0]), 0.25)), [0.5])
    assert np.allclose(np.asarray(flux_to_etch_rate(jnp.asarray([2.0]), 0.25)), [-0.5])


def test_symmetric_directions_give_symmetric_flux():
    normals = jnp.asarray([[[1.0, 0.0], [-1.0, 0.0]]])
    directions = jnp.asarray([[1.0, 0.0], [-1.0, 0.0]])
    flux = accumulate_flux(normals, directions)

    assert np.allclose(np.asarray(flux)[0, 0], np.asarray(flux)[0, 1])


def test_chunked_flux_matches_unchunked_flux_for_small_ray_sets():
    normals = jnp.asarray([[[0.0, 1.0], [1.0, 0.0]], [[-1.0, 0.0], [0.0, -1.0]]])
    directions = jnp.asarray([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0], [0.0, -1.0]])
    weights = jnp.asarray([0.2, 0.3, 0.1, 0.4])

    assert np.allclose(
        np.asarray(accumulate_flux_chunked(normals, directions, chunk_size=3)),
        np.asarray(accumulate_flux(normals, directions)),
    )
    assert np.allclose(
        np.asarray(accumulate_flux_chunked(normals, directions, weights=weights, chunk_size=3)),
        np.asarray(accumulate_flux(normals, directions, weights=weights)),
    )
