import numpy as np

from jaxps.geometry import make_grid_2d, sdf_circle
from jaxps.models import isotropic_deposition_rate, isotropic_etch_rate
from jaxps.solvers import evolve_level_set


def _positive_x_radius(grid, phi):
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


def test_shrinking_circle_under_isotropic_etch_matches_analytic_radius():
    grid = make_grid_2d(bounds=((-1.0, 1.0), (-1.0, 1.0)), shape=(129, 129))
    radius0 = 0.55
    rate = 0.1
    t_final = 0.2
    phi0 = sdf_circle(grid, center=(0.0, 0.0), radius=radius0)

    result = evolve_level_set(
        phi0,
        grid.spacing,
        velocity_fn=lambda phi, _t: isotropic_etch_rate(phi, rate),
        t_final=t_final,
        cfl=0.4,
    )

    measured = _positive_x_radius(grid, result.phi)
    expected = radius0 - rate * t_final
    assert abs(measured - expected) <= 3.0 * grid.spacing[0]


def test_growing_circle_under_isotropic_deposition_matches_analytic_radius():
    grid = make_grid_2d(bounds=((-1.0, 1.0), (-1.0, 1.0)), shape=(129, 129))
    radius0 = 0.45
    rate = 0.1
    t_final = 0.2
    phi0 = sdf_circle(grid, center=(0.0, 0.0), radius=radius0)

    result = evolve_level_set(
        phi0,
        grid.spacing,
        velocity_fn=lambda phi, _t: isotropic_deposition_rate(phi, rate),
        t_final=t_final,
        cfl=0.4,
    )

    measured = _positive_x_radius(grid, result.phi)
    expected = radius0 + rate * t_final
    assert abs(measured - expected) <= 3.0 * grid.spacing[0]
