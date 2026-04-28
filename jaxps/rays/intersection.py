"""Lightweight surface-band helpers.

This module intentionally does not implement exact ray tracing. It provides
small utilities for approximate flux models while leaving room for future
external backends.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def approximate_surface_band(phi: Array, width: float) -> Array:
    """Return cells within ``width`` of the zero level set."""

    if width < 0.0:
        raise ValueError("width must be nonnegative")
    return jnp.abs(phi) <= float(width)
