"""Analytic signed-distance functions and level-set set operations."""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import Array

from jaxps.geometry.grid import Grid


def _validate_positive(value: float, name: str) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def sdf_circle(grid: Grid, center: tuple[float, float], radius: float) -> Array:
    """Signed distance to a circle, negative inside."""

    if grid.ndim != 2:
        raise ValueError("sdf_circle requires a 2D grid")
    _validate_positive(float(radius), "radius")
    x, y = grid.coords
    cx, cy = center
    return jnp.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius


def sdf_sphere(grid: Grid, center: tuple[float, float, float], radius: float) -> Array:
    """Signed distance to a sphere, negative inside."""

    if grid.ndim != 3:
        raise ValueError("sdf_sphere requires a 3D grid")
    _validate_positive(float(radius), "radius")
    x, y, z = grid.coords
    cx, cy, cz = center
    return jnp.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) - radius


def sdf_plane(grid: Grid, point: tuple[float, ...], normal: tuple[float, ...]) -> Array:
    """Signed distance to a plane passing through ``point`` with outward ``normal``."""

    if len(point) != grid.ndim or len(normal) != grid.ndim:
        raise ValueError("point and normal dimension must match grid dimension")
    normal_norm = math.sqrt(sum(float(component) ** 2 for component in normal))
    if normal_norm == 0.0:
        raise ValueError("normal must be nonzero")
    normal_arr = jnp.asarray(normal, dtype=jnp.result_type(*grid.coords))
    unit = normal_arr / normal_norm
    distance = jnp.zeros(grid.shape, dtype=unit.dtype)
    for axis, coord in enumerate(grid.coords):
        distance = distance + (coord - point[axis]) * unit[axis]
    return distance


def sdf_box(grid: Grid, center: tuple[float, ...], half_extents: tuple[float, ...]) -> Array:
    """Signed distance to an axis-aligned box, negative inside."""

    if len(center) != grid.ndim or len(half_extents) != grid.ndim:
        raise ValueError("center and half_extents dimension must match grid dimension")
    if any(extent <= 0.0 for extent in half_extents):
        raise ValueError("half_extents must be positive")
    q_components = [
        jnp.abs(coord - center[axis]) - half_extents[axis]
        for axis, coord in enumerate(grid.coords)
    ]
    q = jnp.stack(q_components, axis=-1)
    outside = jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1)
    inside = jnp.minimum(jnp.max(q, axis=-1), 0.0)
    return outside + inside


def sdf_union(phi_a: Array, phi_b: Array) -> Array:
    """Level-set union operation."""

    return jnp.minimum(phi_a, phi_b)


def sdf_intersection(phi_a: Array, phi_b: Array) -> Array:
    """Level-set intersection operation."""

    return jnp.maximum(phi_a, phi_b)


def sdf_difference(phi_a: Array, phi_b: Array) -> Array:
    """Level-set difference operation: ``A \\ B``."""

    return jnp.maximum(phi_a, -phi_b)
