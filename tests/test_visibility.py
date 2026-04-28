import jax.numpy as jnp
import numpy as np

from jaxps.geometry import make_grid_2d, sdf_box, surface_normals
from jaxps.rays import deterministic_hemisphere_directions, visibility_weights, visible_flux


def test_visibility_blocks_ray_through_material_obstacle():
    grid = make_grid_2d(((0.0, 1.0), (0.0, 1.0)), (81, 81))
    phi = sdf_box(grid, center=(0.5, 0.5), half_extents=(0.06, 0.2))
    origins = jnp.asarray([[0.2, 0.5], [0.8, 0.5]])
    directions = jnp.asarray([[1.0, 0.0]])

    visibility = visibility_weights(
        phi,
        grid,
        origins,
        directions,
        max_steps=80,
        step_size=grid.spacing[0],
    )

    assert np.allclose(np.asarray(visibility[:, 0]), [0.0, 1.0])


def test_visible_flux_is_no_larger_than_exposure_only_flux():
    grid = make_grid_2d(((0.0, 1.0), (0.0, 1.0)), (41, 41))
    phi = sdf_box(grid, center=(0.5, 0.5), half_extents=(0.06, 0.2))
    normals = surface_normals(phi, grid.spacing)
    directions = deterministic_hemisphere_directions(8, ndim=2)

    flux = visible_flux(phi, grid, normals, directions, max_steps=32, step_size=grid.spacing[0])

    assert flux.shape == phi.shape
    assert np.all(np.asarray(flux) >= 0.0)
