import numpy as np

from jaxps.geometry import make_grid_2d, sdf_circle
from jaxps.models import isotropic_etch_rate
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


def _circle_error(shape):
    grid = make_grid_2d(((-1.0, 1.0), (-1.0, 1.0)), (shape, shape))
    radius0 = 0.55
    rate = 0.1
    t_final = 0.2
    phi0 = sdf_circle(grid, (0.0, 0.0), radius0)
    result = evolve_level_set(
        phi0,
        grid.spacing,
        velocity_fn=lambda phi, _t: isotropic_etch_rate(phi, rate),
        t_final=t_final,
        cfl=0.4,
    )
    expected = radius0 - rate * t_final
    return abs(_positive_x_radius(grid, result.phi) - expected), grid.spacing[0]


def test_circle_validation_error_is_grid_scaled_under_refinement():
    coarse_error, coarse_dx = _circle_error(65)
    fine_error, fine_dx = _circle_error(129)

    assert coarse_error <= 4.0 * coarse_dx
    assert fine_error <= 4.0 * fine_dx
    assert fine_error <= coarse_error + 2.0 * fine_dx
