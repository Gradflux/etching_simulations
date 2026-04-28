"""Godunov upwind level-set update kernels."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from jaxps.geometry.derivatives import one_sided_derivatives


def _min_spacing(spacing: tuple[float, ...]) -> float:
    if not spacing:
        raise ValueError("spacing must not be empty")
    if any(float(dx) <= 0.0 for dx in spacing):
        raise ValueError("spacing entries must be positive")
    return min(float(dx) for dx in spacing)


def godunov_gradient_magnitude(
    phi: Array,
    velocity: Array | float,
    spacing: tuple[float, ...],
    bcs: tuple[str, ...] = (),
) -> Array:
    """Return the Godunov upwind approximation to ``|grad phi|``."""

    backward, forward = one_sided_derivatives(phi, spacing, bcs)
    return godunov_gradient_magnitude_from_derivatives(backward, forward, velocity)


@jax.jit
def godunov_gradient_magnitude_from_derivatives(
    backward: Array,
    forward: Array,
    velocity: Array | float,
) -> Array:
    """Return Godunov gradient magnitude from stacked one-sided derivatives."""

    positive_terms = jnp.maximum(backward, 0.0) ** 2 + jnp.minimum(forward, 0.0) ** 2
    negative_terms = jnp.maximum(forward, 0.0) ** 2 + jnp.minimum(backward, 0.0) ** 2
    grad_for_positive_v = jnp.sqrt(jnp.sum(positive_terms, axis=-1))
    grad_for_negative_v = jnp.sqrt(jnp.sum(negative_terms, axis=-1))
    return jnp.where(jnp.asarray(velocity) >= 0.0, grad_for_positive_v, grad_for_negative_v)


@partial(jax.jit, static_argnames=("bcs",))
def _level_set_step_jit(
    phi: Array,
    velocity: Array | float,
    spacing: tuple[float, ...],
    dt: float,
    band_width: float,
    bcs: tuple[str, ...] = (),
) -> Array:
    godunov_norm = godunov_gradient_magnitude(phi, velocity, spacing, bcs)
    next_phi = phi - dt * velocity * godunov_norm
    return jnp.where(jnp.abs(phi) <= band_width, next_phi, phi)


def level_set_step(
    phi: Array,
    velocity: Array | float,
    spacing: tuple[float, ...],
    dt: float,
    band_width: float | None = None,
    bcs: tuple[str, ...] = (),
) -> Array:
    """Advance one explicit Euler step for ``phi_t + V |grad phi| = 0``."""

    if float(dt) < 0.0:
        raise ValueError("dt must be nonnegative")
    if band_width is not None and float(band_width) < 0.0:
        raise ValueError("band_width must be nonnegative when provided")
    _min_spacing(spacing)
    update_width = jnp.inf if band_width is None else float(band_width)
    return _level_set_step_jit(phi, velocity, spacing, dt, update_width, bcs)


@jax.jit
def _cfl_timestep_jit(velocity: Array | float, min_dx: float, cfl: float) -> Array:
    max_velocity = jnp.max(jnp.abs(jnp.asarray(velocity)))
    return jnp.where(max_velocity > 0.0, cfl * min_dx / max_velocity, jnp.inf)


def cfl_timestep(velocity: Array | float, spacing: tuple[float, ...], cfl: float = 0.4) -> Array:
    """Return a CFL-limited timestep, or ``inf`` for identically zero velocity."""

    if not 0.0 < float(cfl) <= 1.0:
        raise ValueError("cfl must be in (0, 1]")
    min_dx = _min_spacing(spacing)
    return _cfl_timestep_jit(velocity, min_dx, cfl)
