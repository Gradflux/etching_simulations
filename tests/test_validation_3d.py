import numpy as np

from jaxps.geometry import make_grid_3d, sdf_plane, sdf_sphere, surface_normals
from jaxps.models import isotropic_deposition_rate, isotropic_etch_rate
from jaxps.solvers import evolve_level_set


def _positive_x_radius(grid, phi):
    x = np.asarray(grid.axes[0])
    y = np.asarray(grid.axes[1])
    z = np.asarray(grid.axes[2])
    y_idx = int(np.argmin(np.abs(y)))
    z_idx = int(np.argmin(np.abs(z)))
    values = np.asarray(phi)[:, y_idx, z_idx]
    start = int(np.argmin(np.abs(x)))
    for idx in range(start, len(x) - 1):
        if values[idx] <= 0.0 <= values[idx + 1]:
            x0, x1 = x[idx], x[idx + 1]
            p0, p1 = values[idx], values[idx + 1]
            return float(x0 - p0 * (x1 - x0) / (p1 - p0))
    raise AssertionError("no positive x-axis zero crossing found")


def _plane_position(grid, phi):
    x = np.asarray(grid.axes[0])
    y = np.asarray(grid.axes[1])
    z = np.asarray(grid.axes[2])
    y_idx = int(np.argmin(np.abs(y)))
    z_idx = int(np.argmin(np.abs(z)))
    values = np.asarray(phi)[:, y_idx, z_idx]
    for idx in range(len(x) - 1):
        if values[idx] <= 0.0 <= values[idx + 1]:
            x0, x1 = x[idx], x[idx + 1]
            p0, p1 = values[idx], values[idx + 1]
            return float(x0 - p0 * (x1 - x0) / (p1 - p0))
    raise AssertionError("no zero crossing found")


def test_sphere_shrink_and_growth_match_analytic_radius():
    grid = make_grid_3d(((-0.8, 0.8), (-0.8, 0.8), (-0.8, 0.8)), (65, 65, 65))
    radius0 = 0.35
    rate = 0.08
    t_final = 0.15
    phi0 = sdf_sphere(grid, (0.0, 0.0, 0.0), radius0)

    etched = evolve_level_set(
        phi0,
        grid.spacing,
        velocity_fn=lambda phi, _t: isotropic_etch_rate(phi, rate),
        t_final=t_final,
        cfl=0.35,
    )
    deposited = evolve_level_set(
        phi0,
        grid.spacing,
        velocity_fn=lambda phi, _t: isotropic_deposition_rate(phi, rate),
        t_final=t_final,
        cfl=0.35,
    )

    assert abs(_positive_x_radius(grid, etched.phi) - (radius0 - rate * t_final)) <= 3.0 * grid.spacing[0]
    assert abs(_positive_x_radius(grid, deposited.phi) - (radius0 + rate * t_final)) <= 3.0 * grid.spacing[0]


def test_3d_plane_front_matches_analytic_motion():
    grid = make_grid_3d(((-0.8, 0.8), (-0.4, 0.4), (-0.4, 0.4)), (65, 33, 33))
    x0 = -0.1
    velocity = 0.12
    t_final = 0.2
    phi0 = sdf_plane(grid, point=(x0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))

    result = evolve_level_set(
        phi0,
        grid.spacing,
        velocity_fn=lambda _phi, _t: velocity,
        t_final=t_final,
        cfl=0.35,
    )

    assert abs(_plane_position(grid, result.phi) - (x0 + velocity * t_final)) <= 2.0 * grid.spacing[0]


def test_sphere_normals_are_radial_in_3d():
    grid = make_grid_3d(((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)), (41, 41, 41))
    phi = sdf_sphere(grid, (0.0, 0.0, 0.0), 0.5)
    normals = np.asarray(surface_normals(phi, grid.spacing))

    assert np.allclose(normals[30, 20, 20], (1.0, 0.0, 0.0), atol=3e-2)
    assert np.allclose(normals[20, 30, 20], (0.0, 1.0, 0.0), atol=3e-2)
    assert np.allclose(normals[20, 20, 30], (0.0, 0.0, 1.0), atol=3e-2)
