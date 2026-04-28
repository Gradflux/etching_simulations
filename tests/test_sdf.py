import numpy as np

from jaxps.geometry import (
    make_grid_2d,
    make_grid_3d,
    sdf_box,
    sdf_circle,
    sdf_difference,
    sdf_intersection,
    sdf_plane,
    sdf_sphere,
    sdf_union,
)


def test_circle_sdf_known_values():
    grid = make_grid_2d(bounds=((-1.0, 1.0), (-1.0, 1.0)), shape=(101, 101))
    phi = sdf_circle(grid, center=(0.0, 0.0), radius=0.5)
    arr = np.asarray(phi)

    assert np.isclose(arr[50, 50], -0.5, atol=1e-6)
    assert np.isclose(arr[75, 50], 0.0, atol=1e-6)
    assert arr[50, 50] < 0.0
    assert arr[0, 0] > 0.0


def test_sphere_sdf_known_values():
    grid = make_grid_3d(
        bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        shape=(21, 21, 21),
    )
    phi = sdf_sphere(grid, center=(0.0, 0.0, 0.0), radius=0.4)
    arr = np.asarray(phi)

    assert np.isclose(arr[10, 10, 10], -0.4, atol=1e-6)
    assert arr[0, 0, 0] > 0.0


def test_plane_sdf_exactness():
    grid = make_grid_2d(bounds=((-1.0, 1.0), (-1.0, 1.0)), shape=(11, 11))
    phi = sdf_plane(grid, point=(0.25, 0.0), normal=(1.0, 0.0))
    x = np.asarray(grid.coords[0])

    assert np.allclose(np.asarray(phi), x - 0.25, atol=1e-6)


def test_box_and_boolean_sdf_signs():
    grid = make_grid_2d(bounds=((-1.0, 1.0), (-1.0, 1.0)), shape=(101, 101))
    box = sdf_box(grid, center=(0.0, 0.0), half_extents=(0.25, 0.5))
    circle = sdf_circle(grid, center=(0.0, 0.0), radius=0.25)

    assert np.asarray(box)[50, 50] < 0.0
    assert np.asarray(box)[0, 0] > 0.0
    assert np.asarray(sdf_union(box, circle))[50, 50] < 0.0
    assert np.asarray(sdf_intersection(box, circle))[50, 50] < 0.0
    assert np.asarray(sdf_difference(box, circle))[50, 50] > 0.0


def test_box_sdf_known_values():
    # 101-point grid from -1 to 1: spacing 0.02, index 50 = x=0, index 75 = x=0.5
    grid = make_grid_2d(bounds=((-1.0, 1.0), (-1.0, 1.0)), shape=(101, 101))
    box = sdf_box(grid, center=(0.0, 0.0), half_extents=(0.5, 0.5))
    arr = np.asarray(box)

    # Center: nearest face is at distance 0.5 → SDF = -0.5
    assert np.isclose(arr[50, 50], -0.5, atol=1e-6)

    # Point on x-face at (0.5, 0): q=(0, -0.5) → SDF = 0
    assert np.isclose(arr[75, 50], 0.0, atol=1e-6)

    # Point at (0.6, 0): q=(0.1, -0.5), outside_norm = 0.1, inside = 0 → SDF = 0.1
    # x=0.6 → index (0.6+1)/0.02 = 80
    assert np.isclose(arr[80, 50], 0.1, atol=1e-6)
