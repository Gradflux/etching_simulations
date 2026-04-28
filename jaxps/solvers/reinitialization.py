"""Signed-distance reinitialization by pseudo-time Hamilton-Jacobi evolution."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from jaxps.geometry.derivatives import gradient_magnitude
from jaxps.solvers.level_set import _min_spacing, godunov_gradient_magnitude


def _validate_reinitialization_inputs(
    spacing: tuple[float, ...],
    iterations: int,
    cfl: float,
    band_width: float | None,
) -> float:
    min_dx = _min_spacing(spacing)
    if int(iterations) != iterations or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if not 0.0 < float(cfl) <= 1.0:
        raise ValueError("cfl must be in (0, 1]")
    if band_width is not None and float(band_width) < 0.0:
        raise ValueError("band_width must be nonnegative when provided")
    return min_dx


def reinitialize_signed_distance(
    phi: Array,
    spacing: tuple[float, ...],
    iterations: int = 50,
    cfl: float = 0.3,
    band_width: float | None = None,
    bcs: tuple[str, ...] = (),
) -> Array:
    """Reinitialize ``phi`` toward a signed-distance field.

    The pseudo-time equation is ``phi_tau + S(phi0)(|grad phi| - 1) = 0`` with
    ``S(phi0) = phi0 / sqrt(phi0**2 + min(dx)**2)``. The update is intentionally
    conservative: it preserves the initial sign field and can be restricted to a
    dense narrow band around the interface.
    """

    min_dx = _validate_reinitialization_inputs(spacing, iterations, cfl, band_width)
    if iterations == 0:
        return phi

    phi0 = phi
    smoothed_sign = phi0 / jnp.sqrt(phi0 * phi0 + min_dx * min_dx)
    dtau = float(cfl) * min_dx
    update_width = jnp.inf if band_width is None else float(band_width)
    band_mask = jnp.abs(phi0) <= update_width

    def scan_step(current_phi: Array, _index: Array) -> tuple[Array, Array]:
        grad_norm = godunov_gradient_magnitude(current_phi, smoothed_sign, spacing, bcs)
        next_phi = current_phi - dtau * smoothed_sign * (grad_norm - 1.0)
        next_phi = jnp.where(band_mask, next_phi, current_phi)
        return next_phi, next_phi

    final_phi, _ = jax.lax.scan(scan_step, phi, jnp.arange(iterations))
    return final_phi


@jax.jit
def _reinitialization_error_jit(phi: Array, spacing: tuple[float, ...]) -> Array:
    return jnp.abs(gradient_magnitude(phi, spacing, eps=0.0) - 1.0)


def reinitialization_error(phi: Array, spacing: tuple[float, ...]) -> Array:
    """Return ``abs(|grad phi| - 1)`` as a signed-distance diagnostic."""

    return _reinitialization_error_jit(phi, spacing)
