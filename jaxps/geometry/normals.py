"""Surface-normal utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from jaxps.geometry.derivatives import gradient_central


@jax.jit
def _surface_normals_jit(phi: Array, spacing: tuple[float, ...], eps: float) -> Array:
    gradient = gradient_central(phi, spacing)
    norm = jnp.sqrt(jnp.sum(gradient * gradient, axis=-1, keepdims=True))
    return gradient / jnp.maximum(norm, eps)


def surface_normals(phi: Array, spacing: tuple[float, ...], eps: float = 1e-12) -> Array:
    """Return normalized outward normals stacked on the last axis."""

    return _surface_normals_jit(phi, spacing, eps)
