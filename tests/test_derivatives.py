import numpy as np

from jaxps.geometry import (
    central_difference,
    forward_difference,
    gradient_magnitude,
    make_grid_2d,
)


def test_linear_derivatives_are_exact():
    grid = make_grid_2d(bounds=((-1.0, 1.0), (-2.0, 2.0)), shape=(41, 81))
    x, y = grid.coords
    phi = 2.0 * x - 3.0 * y + 1.0

    d_x = np.asarray(central_difference(phi, grid.spacing, axis=0))
    d_y = np.asarray(central_difference(phi, grid.spacing, axis=1))

    assert np.allclose(d_x, 2.0, atol=2e-5)
    assert np.allclose(d_y, -3.0, atol=2e-5)


def test_quadratic_central_derivative_is_second_order_inside():
    grid = make_grid_2d(bounds=((-1.0, 1.0), (-1.0, 1.0)), shape=(81, 81))
    x, y = grid.coords
    phi = x**2 + y**2

    d_x = np.asarray(central_difference(phi, grid.spacing, axis=0))
    expected = np.asarray(2.0 * x)

    assert np.allclose(d_x[1:-1, :], expected[1:-1, :], atol=2e-5)


def test_forward_difference_shape_and_gradient_magnitude():
    grid = make_grid_2d(bounds=((-1.0, 1.0), (-1.0, 1.0)), shape=(21, 21))
    x, y = grid.coords
    phi = 3.0 * x + 4.0 * y

    assert forward_difference(phi, grid.spacing, axis=0).shape == grid.shape
    magnitude = np.asarray(gradient_magnitude(phi, grid.spacing))

    assert np.allclose(magnitude, 5.0, atol=2e-5)
