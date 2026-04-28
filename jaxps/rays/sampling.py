"""Direction sampling for approximate JAX-native flux models."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array


def normalize_directions(directions: Array, eps: float = 1e-12) -> Array:
    """Normalize directions along the last axis."""

    directions = jnp.asarray(directions)
    norms = jnp.linalg.norm(directions, axis=-1, keepdims=True)
    return directions / jnp.maximum(norms, eps)


@partial(jax.jit, static_argnames=("num_rays", "ndim"))
def cosine_weighted_directions(key: Array, num_rays: int, ndim: int = 3) -> Array:
    """Sample cosine-weighted directions in the positive final-axis hemisphere."""

    if ndim == 3:
        u1, u2 = jax.random.uniform(key, (2, num_rays))
        radius = jnp.sqrt(u1)
        theta = 2.0 * jnp.pi * u2
        x = radius * jnp.cos(theta)
        y = radius * jnp.sin(theta)
        z = jnp.sqrt(jnp.maximum(1.0 - u1, 0.0))
        return jnp.stack((x, y, z), axis=-1)
    if ndim == 2:
        u = jax.random.uniform(key, (num_rays,))
        sin_theta = 2.0 * u - 1.0
        cos_theta = jnp.sqrt(jnp.maximum(1.0 - sin_theta**2, 0.0))
        return jnp.stack((sin_theta, cos_theta), axis=-1)
    raise ValueError("ndim must be 2 or 3")


@partial(jax.jit, static_argnames=("num_rays", "exponent", "ndim"))
def polynomial_cosine_directions(
    key: Array,
    num_rays: int,
    exponent: float,
    ndim: int = 3,
) -> Array:
    """Sample directions biased by a polynomial cosine lobe."""

    if exponent < 0.0:
        raise ValueError("exponent must be nonnegative")
    if ndim == 3:
        key_mu, key_phi = jax.random.split(key)
        u = jax.random.uniform(key_mu, (num_rays,))
        azimuth = 2.0 * jnp.pi * jax.random.uniform(key_phi, (num_rays,))
        mu = u ** (1.0 / (float(exponent) + 1.0))
        radius = jnp.sqrt(jnp.maximum(1.0 - mu**2, 0.0))
        return jnp.stack((radius * jnp.cos(azimuth), radius * jnp.sin(azimuth), mu), axis=-1)
    if ndim == 2:
        u = jax.random.uniform(key, (num_rays,))
        lateral = jnp.sign(u - 0.5) * jnp.abs(2.0 * u - 1.0) ** (
            1.0 / (float(exponent) + 1.0)
        )
        normal = jnp.sqrt(jnp.maximum(1.0 - lateral**2, 0.0))
        return jnp.stack((lateral, normal), axis=-1)
    raise ValueError("ndim must be 2 or 3")


def deterministic_hemisphere_directions(num_rays: int, ndim: int = 3) -> Array:
    """Return deterministic normalized directions in the positive final-axis hemisphere."""

    if num_rays < 0:
        raise ValueError("num_rays must be nonnegative")
    if num_rays == 0:
        return jnp.empty((0, ndim))
    if ndim == 3:
        i = jnp.arange(num_rays, dtype=jnp.float32)
        mu = (i + 0.5) / float(num_rays)
        radius = jnp.sqrt(jnp.maximum(1.0 - mu**2, 0.0))
        azimuth = i * (jnp.pi * (3.0 - jnp.sqrt(5.0)))
        return normalize_directions(
            jnp.stack((radius * jnp.cos(azimuth), radius * jnp.sin(azimuth), mu), axis=-1)
        )
    if ndim == 2:
        theta = ((jnp.arange(num_rays, dtype=jnp.float32) + 0.5) / float(num_rays) - 0.5) * jnp.pi
        return jnp.stack((jnp.sin(theta), jnp.cos(theta)), axis=-1)
    raise ValueError("ndim must be 2 or 3")
