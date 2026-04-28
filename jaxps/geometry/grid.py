"""Cartesian grid construction."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
@dataclass(frozen=True)
class Grid:
    """Dense Cartesian grid metadata and coordinate arrays."""

    bounds: tuple[tuple[float, float], ...]
    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    axes: tuple[Array, ...]
    coords: tuple[Array, ...]

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""

        return len(self.shape)

    def tree_flatten(self):
        """Flatten JAX arrays while preserving static metadata."""

        children = self.axes + self.coords
        aux_data = (self.bounds, self.shape, self.spacing, self.ndim)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct a grid from PyTree data."""

        bounds, shape, spacing, ndim = aux_data
        axes = tuple(children[:ndim])
        coords = tuple(children[ndim:])
        return cls(bounds=bounds, shape=shape, spacing=spacing, axes=axes, coords=coords)


def _validate_bounds_shape(
    bounds: tuple[tuple[float, float], ...],
    shape: tuple[int, ...],
    ndim: int,
) -> None:
    if len(bounds) != ndim:
        raise ValueError(f"expected {ndim} bounds entries, got {len(bounds)}")
    if len(shape) != ndim:
        raise ValueError(f"expected {ndim} shape entries, got {len(shape)}")
    for axis, (lo, hi) in enumerate(bounds):
        if not hi > lo:
            raise ValueError(f"bounds for axis {axis} must satisfy hi > lo")
    for axis, n in enumerate(shape):
        if int(n) != n or n < 2:
            raise ValueError(f"shape for axis {axis} must be an integer >= 2")


def _make_grid(bounds: tuple[tuple[float, float], ...], shape: tuple[int, ...], ndim: int) -> Grid:
    _validate_bounds_shape(bounds, shape, ndim)
    clean_bounds = tuple((float(lo), float(hi)) for lo, hi in bounds)
    clean_shape = tuple(int(n) for n in shape)
    spacing = tuple(
        (hi - lo) / float(n - 1) for (lo, hi), n in zip(clean_bounds, clean_shape, strict=True)
    )
    axes = tuple(
        jnp.linspace(lo, hi, n) for (lo, hi), n in zip(clean_bounds, clean_shape, strict=True)
    )
    coords = tuple(jnp.meshgrid(*axes, indexing="ij"))
    return Grid(bounds=clean_bounds, shape=clean_shape, spacing=spacing, axes=axes, coords=coords)


def make_grid_2d(
    bounds: tuple[tuple[float, float], tuple[float, float]],
    shape: tuple[int, int],
) -> Grid:
    """Create a 2D Cartesian grid."""

    return _make_grid(bounds, shape, ndim=2)


def make_grid_3d(
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    shape: tuple[int, int, int],
) -> Grid:
    """Create a 3D Cartesian grid."""

    return _make_grid(bounds, shape, ndim=3)
