import numpy as np

from jaxps.geometry import make_grid_2d, sdf_plane
from jaxps.solvers import evolve_level_set


def _plane_position(grid, phi):
    x = np.asarray(grid.axes[0])
    y = np.asarray(grid.axes[1])
    y_idx = int(np.argmin(np.abs(y)))
    values = np.asarray(phi)[:, y_idx]
    for idx in range(len(x) - 1):
        if values[idx] <= 0.0 <= values[idx + 1]:
            x0, x1 = x[idx], x[idx + 1]
            p0, p1 = values[idx], values[idx + 1]
            return float(x0 - p0 * (x1 - x0) / (p1 - p0))
    raise AssertionError("no zero crossing found")


def test_planar_front_positive_velocity_matches_analytic_motion():
    grid = make_grid_2d(bounds=((-1.0, 1.0), (-1.0, 1.0)), shape=(129, 65))
    x0 = -0.2
    velocity = 0.15
    t_final = 0.3
    phi0 = sdf_plane(grid, point=(x0, 0.0), normal=(1.0, 0.0))

    result = evolve_level_set(
        phi0,
        grid.spacing,
        velocity_fn=lambda _phi, _t: velocity,
        t_final=t_final,
        cfl=0.4,
    )

    assert abs(_plane_position(grid, result.phi) - (x0 + velocity * t_final)) <= 2.0 * grid.spacing[0]


def test_planar_front_negative_velocity_matches_analytic_motion():
    grid = make_grid_2d(bounds=((-1.0, 1.0), (-1.0, 1.0)), shape=(129, 65))
    x0 = 0.2
    velocity = -0.15
    t_final = 0.3
    phi0 = sdf_plane(grid, point=(x0, 0.0), normal=(1.0, 0.0))

    result = evolve_level_set(
        phi0,
        grid.spacing,
        velocity_fn=lambda _phi, _t: velocity,
        t_final=t_final,
        cfl=0.4,
    )

    assert abs(_plane_position(grid, result.phi) - (x0 + velocity * t_final)) <= 2.0 * grid.spacing[0]
