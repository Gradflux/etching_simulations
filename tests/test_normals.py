import numpy as np

from jaxps.geometry import make_grid_2d, sdf_circle, sdf_plane, surface_normals


def test_plane_normals_match_analytic_normal():
    grid = make_grid_2d(bounds=((-1.0, 1.0), (-1.0, 1.0)), shape=(41, 41))
    phi = sdf_plane(grid, point=(0.0, 0.0), normal=(1.0, 0.0))
    normals = np.asarray(surface_normals(phi, grid.spacing))

    assert np.allclose(normals[..., 0], 1.0, atol=2e-5)
    assert np.allclose(normals[..., 1], 0.0, atol=2e-5)


def test_circle_normals_are_radial_away_from_center():
    grid = make_grid_2d(bounds=((-1.0, 1.0), (-1.0, 1.0)), shape=(101, 101))
    phi = sdf_circle(grid, center=(0.0, 0.0), radius=0.5)
    normals = np.asarray(surface_normals(phi, grid.spacing))

    assert np.allclose(normals[75, 50], (1.0, 0.0), atol=2e-2)
    assert np.allclose(normals[50, 75], (0.0, 1.0), atol=2e-2)
    assert not np.isnan(normals).any()
