import numpy as np

from jaxps.geometry import gradient_magnitude, make_grid_2d, sdf_circle
from jaxps.solvers import reinitialization_error, reinitialize_signed_distance


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


def test_reinitialization_preserves_sign_and_interface_location():
    grid = make_grid_2d(((-1.0, 1.0), (-1.0, 1.0)), (129, 129))
    phi_sdf = sdf_circle(grid, (0.0, 0.0), 0.45)
    x, y = grid.coords
    phi_distorted = phi_sdf * (1.0 + 0.25 * x * y)

    phi_reinit = reinitialize_signed_distance(
        phi_distorted,
        grid.spacing,
        iterations=60,
        cfl=0.3,
        band_width=0.35,
    )

    inside_band = np.abs(np.asarray(phi_sdf)) < 0.25
    assert np.all(np.sign(np.asarray(phi_reinit)[inside_band]) == np.sign(np.asarray(phi_distorted)[inside_band]))
    assert abs(_positive_x_radius(grid, phi_reinit) - _positive_x_radius(grid, phi_sdf)) <= 3.0 * grid.spacing[0]


def test_reinitialization_improves_gradient_magnitude_near_interface():
    grid = make_grid_2d(((-1.0, 1.0), (-1.0, 1.0)), (129, 129))
    phi_sdf = sdf_circle(grid, (0.0, 0.0), 0.45)
    x, y = grid.coords
    phi_distorted = phi_sdf * (1.0 + 0.35 * x + 0.15 * y)
    near_interface = np.abs(np.asarray(phi_sdf)) < 0.12

    before = np.mean(np.asarray(reinitialization_error(phi_distorted, grid.spacing))[near_interface])
    phi_reinit = reinitialize_signed_distance(
        phi_distorted,
        grid.spacing,
        iterations=80,
        cfl=0.3,
        band_width=0.35,
    )
    after = np.mean(np.asarray(reinitialization_error(phi_reinit, grid.spacing))[near_interface])
    grad_after = np.asarray(gradient_magnitude(phi_reinit, grid.spacing))[near_interface]

    assert after < before
    assert abs(float(np.mean(grad_after)) - 1.0) < 0.12
