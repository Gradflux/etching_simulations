"""Extrusion utilities."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from jaxps.geometry.grid import Grid, make_grid_3d


def extrude_2d_to_3d(phi2d: Array, grid2d: Grid, z_bounds: tuple[float, float], nz: int) -> tuple[Array, Grid]:
    """Extrude a 2D level set into a 3D field by repeating along z."""

    if grid2d.ndim != 2 or phi2d.ndim != 2:
        raise ValueError("extrude_2d_to_3d requires a 2D field and grid")
    grid3d = make_grid_3d(
        bounds=(grid2d.bounds[0], grid2d.bounds[1], z_bounds),
        shape=(grid2d.shape[0], grid2d.shape[1], int(nz)),
    )
    return jnp.repeat(phi2d[..., None], int(nz), axis=2), grid3d
