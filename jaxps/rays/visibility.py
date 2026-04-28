"""Approximate grid-marched visibility for flux models."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from jaxps.geometry.grid import Grid
from jaxps.rays.flux import surface_exposure
from jaxps.rays.sampling import normalize_directions


def _grid_static_arrays(grid: Grid, dtype) -> tuple[Array, Array, Array]:
    lows = jnp.asarray([bound[0] for bound in grid.bounds], dtype=dtype)
    highs = jnp.asarray([bound[1] for bound in grid.bounds], dtype=dtype)
    spacing = jnp.asarray(grid.spacing, dtype=dtype)
    return lows, highs, spacing


def _sample_nearest(phi: Array, grid: Grid, point: Array) -> tuple[Array, Array]:
    lows, highs, spacing = _grid_static_arrays(grid, point.dtype)
    shape = jnp.asarray(phi.shape)
    in_domain = jnp.all((point >= lows) & (point <= highs))
    indices = jnp.rint((point - lows) / spacing).astype(jnp.int32)
    indices = jnp.clip(indices, 0, shape - 1)
    if phi.ndim == 2:
        value = phi[indices[0], indices[1]]
    elif phi.ndim == 3:
        value = phi[indices[0], indices[1], indices[2]]
    else:
        raise ValueError("visibility sampling supports 2D or 3D fields")
    return value, in_domain


@partial(jax.jit, static_argnames=("max_steps",))
def visibility_weights(
    phi: Array,
    grid: Grid,
    origins: Array,
    directions: Array,
    max_steps: int,
    step_size: float,
) -> Array:
    """Return approximate line-of-sight weights in ``[0, 1]``.

    Rays start one step away from each origin. A ray is blocked when it enters a
    material cell where ``phi < 0`` before leaving the domain.
    """

    directions = normalize_directions(directions)
    origins = jnp.asarray(origins, dtype=phi.dtype)

    def trace_one(origin: Array, direction: Array) -> Array:
        def march(carry: tuple[Array, Array], step_index: Array) -> tuple[tuple[Array, Array], Array]:
            blocked, active = carry
            point = origin + direction * step_size * (step_index.astype(phi.dtype) + 1.0)
            value, in_domain = _sample_nearest(phi, grid, point)
            hit_material = active & in_domain & (value < 0.0)
            next_blocked = blocked | hit_material
            next_active = active & in_domain
            return (next_blocked, next_active), next_blocked

        (blocked, _active), _ = jax.lax.scan(
            march,
            (jnp.asarray(False), jnp.asarray(True)),
            jnp.arange(max_steps),
        )
        return jnp.where(blocked, 0.0, 1.0)

    return jax.vmap(lambda origin: jax.vmap(lambda direction: trace_one(origin, direction))(directions))(
        origins
    )


def visible_flux(
    phi: Array,
    grid: Grid,
    normals: Array,
    directions: Array,
    weights: Array | None = None,
    max_steps: int = 64,
    step_size: float | None = None,
) -> Array:
    """Accumulate exposure flux multiplied by approximate visibility.

    ``normals`` is evaluated at all grid cells. The implementation uses every
    grid point as an origin, reshapes internally, then returns a grid-shaped flux
    field.
    """

    if step_size is None:
        step_size = min(grid.spacing)
    flat_origins = jnp.stack([coord.reshape(-1) for coord in grid.coords], axis=-1).astype(phi.dtype)
    flat_normals = normals.reshape((-1, normals.shape[-1]))
    directions = jnp.asarray(directions, dtype=phi.dtype)
    visibility = visibility_weights(phi, grid, flat_origins, directions, max_steps, step_size)
    exposure = surface_exposure(flat_normals, directions)
    if weights is not None:
        weighted = exposure * visibility * jnp.asarray(weights, dtype=exposure.dtype)
        flux = jnp.sum(weighted, axis=-1)
    else:
        flux = jnp.mean(exposure * visibility, axis=-1)
    return flux.reshape(phi.shape)
