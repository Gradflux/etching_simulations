"""Mask helpers for material and protected regions."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from jaxps.geometry.grid import Grid


def material_mask(phi: Array) -> Array:
    """Return a boolean mask for material cells under ``phi < 0``."""

    return phi < 0.0


def void_mask(phi: Array) -> Array:
    """Return a boolean mask for void cells under ``phi > 0``."""

    return phi > 0.0


def rectangular_mask_2d(
    grid: Grid,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> Array:
    """Return a 2D rectangular boolean mask."""

    if grid.ndim != 2:
        raise ValueError("rectangular_mask_2d requires a 2D grid")
    x, y = grid.coords
    return (
        (x >= x_bounds[0])
        & (x <= x_bounds[1])
        & (y >= y_bounds[0])
        & (y <= y_bounds[1])
    )
