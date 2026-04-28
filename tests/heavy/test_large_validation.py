import importlib
import time

import jax.numpy as jnp
import numpy as np
import pytest

from jaxps.experimental import periodic_spectral_derivative
from jaxps.geometry import make_grid_2d, make_grid_3d, sdf_circle, sdf_sphere, surface_normals
from jaxps.models import isotropic_deposition_rate, isotropic_etch_rate
from jaxps.rays import accumulate_flux, accumulate_flux_chunked, deterministic_hemisphere_directions
from jaxps.solvers import evolve_level_set, reinitialization_error, reinitialize_signed_distance


pytestmark = pytest.mark.slow


def _positive_x_radius_2d(grid, phi):
    x = np.asarray(grid.axes[0])
    y = np.asarray(grid.axes[1])
    y_idx = int(np.argmin(np.abs(y)))
    values = np.asarray(phi)[:, y_idx]
    start = int(np.argmin(np.abs(x)))
    for idx in range(start, len(x) - 1):
        if values[idx] <= 0.0 <= values[idx + 1]:
            x0, x1 = x[idx], x[idx + 1]
            p0, p1 = values[idx], values[idx + 1]
            return float(x0 - p0 * (x1 - x0) / (p1 - p0))
    raise AssertionError("no positive x-axis zero crossing found")


def test_large_2d_evolution_and_reinitialization_smoke():
    grid = make_grid_2d(((-1.0, 1.0), (-1.0, 1.0)), (257, 257))
    radius0 = 0.55
    rate = 0.08
    t_final = 0.25
    phi0 = sdf_circle(grid, (0.0, 0.0), radius0)

    result = evolve_level_set(
        phi0,
        grid.spacing,
        lambda phi, _t: isotropic_etch_rate(phi, rate),
        t_final=t_final,
        cfl=0.35,
    )
    expected = radius0 - rate * t_final
    assert abs(_positive_x_radius_2d(grid, result.phi) - expected) <= 3.0 * grid.spacing[0]

    x, y = grid.coords
    distorted = phi0 * (1.0 + 0.25 * x - 0.15 * y)
    near = np.abs(np.asarray(phi0)) < 0.12
    before = float(np.mean(np.asarray(reinitialization_error(distorted, grid.spacing))[near]))
    reinit = reinitialize_signed_distance(
        distorted,
        grid.spacing,
        iterations=80,
        cfl=0.3,
        band_width=0.35,
    )
    after = float(np.mean(np.asarray(reinitialization_error(reinit, grid.spacing))[near]))
    assert after < before
    assert after < 0.02


def test_large_3d_sphere_evolution_smoke():
    grid = make_grid_3d(((-0.8, 0.8), (-0.8, 0.8), (-0.8, 0.8)), (73, 73, 73))
    phi0 = sdf_sphere(grid, (0.0, 0.0, 0.0), 0.35)

    result = evolve_level_set(
        phi0,
        grid.spacing,
        lambda phi, _t: isotropic_deposition_rate(phi, 0.05),
        t_final=0.12,
        cfl=0.35,
    )

    assert result.phi.shape == grid.shape
    assert np.isfinite(np.asarray(result.phi)).all()


def test_large_flux_chunking_and_spectral_smoke():
    grid = make_grid_2d(((-1.0, 1.0), (-1.0, 1.0)), (257, 257))
    phi = sdf_circle(grid, (0.0, 0.0), 0.55)
    normals = surface_normals(phi, grid.spacing)
    directions = deterministic_hemisphere_directions(96, ndim=2)

    full = accumulate_flux(normals, directions)
    chunked = accumulate_flux_chunked(normals, directions, chunk_size=17)
    assert np.allclose(np.asarray(full), np.asarray(chunked), atol=1e-6)

    n = 128
    length = 2.0 * np.pi
    x = jnp.arange(n) * (length / n)
    field = jnp.sin(5.0 * x)
    derivative = periodic_spectral_derivative(field, (length,), axis=0)
    assert np.allclose(np.asarray(derivative), np.asarray(5.0 * jnp.cos(5.0 * x)), atol=1e-4)


def test_all_examples_import_under_heavy_validation():
    start = time.perf_counter()
    for module_name in [
        "examples.isotropic_etch_2d",
        "examples.isotropic_deposition_2d",
        "examples.masked_trench_etch_2d",
        "examples.directional_etch_2d",
        "examples.directional_etch_3d",
        "examples.flux_deposition_2d",
        "examples.flux_deposition_3d",
    ]:
        importlib.import_module(module_name)
    assert time.perf_counter() - start < 5.0
