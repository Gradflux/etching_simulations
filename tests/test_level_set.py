import math

import jax.numpy as jnp
import numpy as np

from jaxps.geometry import make_grid_2d, sdf_circle
from jaxps.solvers import cfl_timestep, evolve_level_set, godunov_gradient_magnitude, level_set_step


def test_cfl_timestep_behavior():
    velocity = jnp.ones((4, 4)) * 2.0
    dt = float(cfl_timestep(velocity, spacing=(0.2, 0.1), cfl=0.5))

    assert np.isclose(dt, 0.025)
    assert math.isinf(float(cfl_timestep(jnp.zeros((4, 4)), spacing=(0.1, 0.1))))


def test_one_step_sign_convention_for_etch_and_deposition():
    grid = make_grid_2d(bounds=((-1.0, 1.0), (-1.0, 1.0)), shape=(101, 101))
    phi = sdf_circle(grid, center=(0.0, 0.0), radius=0.5)
    boundary_index = (75, 50)

    etched = level_set_step(phi, -0.1, grid.spacing, dt=0.05)
    deposited = level_set_step(phi, 0.1, grid.spacing, dt=0.05)

    assert np.asarray(etched)[boundary_index] > np.asarray(phi)[boundary_index]
    assert np.asarray(deposited)[boundary_index] < np.asarray(phi)[boundary_index]


def test_godunov_gradient_is_finite_for_corner_like_field():
    grid = make_grid_2d(bounds=((-1.0, 1.0), (-1.0, 1.0)), shape=(51, 51))
    x, y = grid.coords
    phi = jnp.maximum(x, y)
    grad = np.asarray(godunov_gradient_magnitude(phi, 1.0, grid.spacing))

    assert np.isfinite(grad).all()
    assert grad.shape == grid.shape


def test_periodic_bc_evolution():
    # A planar front phi = x - 0.0 with periodic BCs should advance uniformly;
    # the zero crossing wraps when it exits the domain boundary.
    grid = make_grid_2d(bounds=((-1.0, 1.0), (-1.0, 1.0)), shape=(51, 51))
    x, _y = grid.coords
    phi = x  # zero at x=0, interface aligned with grid
    velocity = 0.5
    t_final = 0.4  # front moves 0.2 units, well within domain
    bcs = ("zero_gradient", "periodic")

    result = evolve_level_set(phi, grid.spacing, lambda p, _t: velocity, t_final, bcs=bcs)
    phi_out = np.asarray(result.phi)

    # With zero_gradient on x and periodic on y, a planar x-front should still
    # advance as x - V*T. Check the zero crossing shifted by ~0.2 units.
    expected_shift = velocity * t_final
    # Find zero crossing along centre row (j=25) via linear interpolation
    row = phi_out[:, 25]
    sign_changes = np.where(np.diff(np.sign(row)))[0]
    assert len(sign_changes) >= 1, "expected a zero crossing in output"
    i = sign_changes[0]
    t_interp = -row[i] / (row[i + 1] - row[i])
    x_vals = np.asarray(x[:, 25])
    zero_x = x_vals[i] + t_interp * (x_vals[i + 1] - x_vals[i])
    assert abs(zero_x - expected_shift) < 3 * grid.spacing[0], (
        f"zero crossing at {zero_x:.4f}, expected ~{expected_shift:.4f}"
    )


def test_dense_narrow_band_update_matches_full_update_inside_band_only():
    grid = make_grid_2d(bounds=((-1.0, 1.0), (-1.0, 1.0)), shape=(81, 81))
    x, _y = grid.coords
    phi = x - 0.1
    velocity = -0.2
    dt = 0.02
    band_width = 0.15

    full = level_set_step(phi, velocity, grid.spacing, dt)
    banded = level_set_step(phi, velocity, grid.spacing, dt, band_width=band_width)
    mask = np.abs(np.asarray(phi)) <= band_width

    assert np.allclose(np.asarray(banded)[mask], np.asarray(full)[mask])
    assert np.allclose(np.asarray(banded)[~mask], np.asarray(phi)[~mask])
