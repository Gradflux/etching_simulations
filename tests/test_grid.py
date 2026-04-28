import numpy as np
import pytest

from jaxps.geometry import make_grid_2d, make_grid_3d


def test_make_grid_2d_shape_spacing_and_endpoints():
    grid = make_grid_2d(bounds=((-1.0, 1.0), (0.0, 2.0)), shape=(5, 3))

    assert grid.ndim == 2
    assert grid.shape == (5, 3)
    assert grid.spacing == (0.5, 1.0)
    assert grid.coords[0].shape == (5, 3)
    assert np.isclose(np.asarray(grid.axes[0])[0], -1.0)
    assert np.isclose(np.asarray(grid.axes[0])[-1], 1.0)
    assert np.isclose(np.asarray(grid.axes[1])[0], 0.0)
    assert np.isclose(np.asarray(grid.axes[1])[-1], 2.0)


def test_make_grid_3d_shape_spacing():
    grid = make_grid_3d(
        bounds=((-1.0, 1.0), (-2.0, 2.0), (0.0, 1.0)),
        shape=(5, 9, 6),
    )

    assert grid.ndim == 3
    assert grid.shape == (5, 9, 6)
    assert np.allclose(grid.spacing, (0.5, 0.5, 0.2))
    assert grid.coords[2].shape == (5, 9, 6)


def test_invalid_grid_inputs_raise():
    with pytest.raises(ValueError):
        make_grid_2d(bounds=((0.0, 0.0), (0.0, 1.0)), shape=(4, 4))
    with pytest.raises(ValueError):
        make_grid_2d(bounds=((0.0, 1.0), (0.0, 1.0)), shape=(1, 4))
    with pytest.raises(ValueError):
        make_grid_3d(bounds=((0.0, 1.0), (0.0, 1.0)), shape=(4, 4, 4))
