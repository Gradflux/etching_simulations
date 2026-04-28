"""Shared process-rate helpers."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def validate_nonnegative(value: float, name: str) -> float:
    """Validate and return a nonnegative scalar."""

    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def validate_nonnegative_static(value: object, name: str) -> None:
    """Validate nonnegative Python scalars while allowing traced JAX values."""

    if isinstance(value, (int, float)) and float(value) < 0.0:
        raise ValueError(f"{name} must be nonnegative")


def constant_rate_like(phi: Array, value: float | Array) -> Array:
    """Return a constant velocity field matching ``phi``."""

    return jnp.ones_like(phi) * jnp.asarray(value, dtype=phi.dtype)
