"""Slicing utilities."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from jaxps.geometry.grid import Grid, make_grid_2d


def slice_3d(phi3d: Array, grid3d: Grid, axis: int, index_or_coordinate: int | float) -> tuple[Array, Grid]:
    """Slice a 3D level-set field into a 2D plane."""

    if grid3d.ndim != 3 or phi3d.ndim != 3:
        raise ValueError("slice_3d requires a 3D field and grid")
    if axis < 0 or axis >= 3:
        raise ValueError("axis must be 0, 1, or 2")
    if isinstance(index_or_coordinate, int):
        index = index_or_coordinate
    else:
        index = int(jnp.argmin(jnp.abs(grid3d.axes[axis] - float(index_or_coordinate))))
    if index < 0 or index >= grid3d.shape[axis]:
        raise ValueError("slice index out of range")
    remaining_axes = [dim for dim in range(3) if dim != axis]
    if axis == 0:
        phi2d = phi3d[index, :, :]
    elif axis == 1:
        phi2d = phi3d[:, index, :]
    else:
        phi2d = phi3d[:, :, index]
    grid2d = make_grid_2d(
        bounds=(grid3d.bounds[remaining_axes[0]], grid3d.bounds[remaining_axes[1]]),
        shape=(grid3d.shape[remaining_axes[0]], grid3d.shape[remaining_axes[1]]),
    )
    return phi2d, grid2d
