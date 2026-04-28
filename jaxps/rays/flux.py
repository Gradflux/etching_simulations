"""Approximate flux accumulation from direction sets and surface helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from jaxps.rays.sampling import normalize_directions


@jax.jit
def surface_exposure(normals: Array, directions: Array) -> Array:
    """Return nonnegative incidence ``max(dot(normal, direction), 0)``."""

    normals = normalize_directions(normals)
    directions = normalize_directions(directions)
    return jnp.maximum(jnp.einsum("...d,rd->...r", normals, directions), 0.0)


def accumulate_flux(normals: Array, directions: Array, weights: Array | None = None) -> Array:
    """Accumulate scalar flux from normalized directions onto surface normals."""

    directions = jnp.asarray(directions)
    if directions.shape[0] == 0:
        return jnp.zeros(normals.shape[:-1], dtype=normals.dtype)
    exposure = surface_exposure(normals, directions)
    if weights is None:
        return jnp.mean(exposure, axis=-1)
    weights = jnp.asarray(weights, dtype=exposure.dtype)
    if weights.shape[0] != directions.shape[0]:
        raise ValueError("weights length must match number of directions")
    return jnp.sum(exposure * weights, axis=-1)


@jax.jit
def _accumulate_flux_chunked_jit(normals: Array, directions: Array, weights: Array) -> Array:
    direction_chunks = directions
    weight_chunks = weights

    def scan_step(carry: Array, inputs: tuple[Array, Array]) -> tuple[Array, Array]:
        chunk_directions, chunk_weights = inputs
        exposure = surface_exposure(normals, chunk_directions)
        next_carry = carry + jnp.sum(exposure * chunk_weights, axis=-1)
        return next_carry, next_carry

    initial = jnp.zeros(normals.shape[:-1], dtype=normals.dtype)
    total, _ = jax.lax.scan(scan_step, initial, (direction_chunks, weight_chunks))
    return total


def accumulate_flux_chunked(
    normals: Array,
    directions: Array,
    weights: Array | None = None,
    chunk_size: int = 64,
) -> Array:
    """Accumulate flux in ray chunks to reduce peak exposure memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    directions = jnp.asarray(directions)
    num_rays = int(directions.shape[0])
    if num_rays == 0:
        return jnp.zeros(normals.shape[:-1], dtype=normals.dtype)
    pad_count = (-num_rays) % int(chunk_size)
    padded_directions = jnp.pad(directions, ((0, pad_count), (0, 0)))
    if weights is None:
        padded_weights = jnp.pad(jnp.ones((num_rays,), dtype=normals.dtype) / num_rays, (0, pad_count))
    else:
        weights = jnp.asarray(weights, dtype=normals.dtype)
        if weights.shape[0] != num_rays:
            raise ValueError("weights length must match number of directions")
        padded_weights = jnp.pad(weights, (0, pad_count))
    reshaped_directions = padded_directions.reshape((-1, chunk_size, directions.shape[-1]))
    reshaped_weights = padded_weights.reshape((-1, chunk_size))
    return _accumulate_flux_chunked_jit(normals, reshaped_directions, reshaped_weights)


def flux_to_deposition_rate(flux: Array, sticking_coefficient: float) -> Array:
    """Convert flux to positive deposition velocity."""

    if isinstance(sticking_coefficient, (int, float)) and sticking_coefficient < 0.0:
        raise ValueError("sticking_coefficient must be nonnegative")
    return jnp.asarray(flux) * jnp.asarray(sticking_coefficient)


def flux_to_etch_rate(flux: Array, yield_value: Array | float) -> Array:
    """Convert flux and yield to negative etch velocity."""

    return -jnp.asarray(flux) * yield_value


def accumulate_flux_auto(
    normals: Array,
    directions: Array,
    backend: str = "AUTO",
    chunk_size: int = 64,
    weights: Array | None = None,
) -> Array:
    """Dispatch flux accumulation to the backend resolved by ``select_flux_backend``.

    ``JAX_RAYS`` uses chunked scan (``accumulate_flux_chunked``); all other
    implemented backends use the dense einsum (``accumulate_flux``). ``AUTO``
    selects ``JAX_RAYS`` when an accelerator device is visible.
    """

    from jaxps.rays.backends import select_flux_backend

    selection = select_flux_backend(backend)
    if selection.actual_backend == "JAX_RAYS":
        return accumulate_flux_chunked(normals, directions, weights=weights, chunk_size=chunk_size)
    return accumulate_flux(normals, directions, weights=weights)


def approximate_surface_band(phi: Array, width: float) -> Array:
    """Return cells within ``width`` of the zero level set."""

    if width < 0.0:
        raise ValueError("width must be nonnegative")
    return jnp.abs(phi) <= float(width)
