"""Finite-difference derivative kernels."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array


def _axis_spacing(spacing: tuple[float, ...], axis: int) -> float:
    if axis < 0 or axis >= len(spacing):
        raise ValueError("axis out of range for spacing")
    dx = spacing[axis]
    if isinstance(dx, (int, float)) and dx <= 0.0:
        raise ValueError("spacing entries must be positive")
    return float(dx) if isinstance(dx, (int, float)) else dx


def _neighbor(phi: Array, axis: int, shift: int) -> Array:
    pad_width = [(0, 0)] * phi.ndim
    if shift == 1:
        pad_width[axis] = (0, 1)
        padded = jnp.pad(phi, pad_width, mode="edge")
        indices = jnp.arange(1, phi.shape[axis] + 1)
    elif shift == -1:
        pad_width[axis] = (1, 0)
        padded = jnp.pad(phi, pad_width, mode="edge")
        indices = jnp.arange(0, phi.shape[axis])
    else:
        raise ValueError("shift must be +1 or -1")
    return jnp.take(padded, indices, axis=axis)


@partial(jax.jit, static_argnames=("axis",))
def _forward_difference_jit(phi: Array, spacing: tuple[float, ...], axis: int) -> Array:
    dx = spacing[axis]
    return (_neighbor(phi, axis, 1) - phi) / dx


def forward_difference(phi: Array, spacing: tuple[float, ...], axis: int) -> Array:
    """Forward finite difference with edge replication at the boundary."""

    _axis_spacing(spacing, axis)
    return _forward_difference_jit(phi, spacing, axis)


@partial(jax.jit, static_argnames=("axis",))
def _backward_difference_jit(phi: Array, spacing: tuple[float, ...], axis: int) -> Array:
    dx = spacing[axis]
    return (phi - _neighbor(phi, axis, -1)) / dx


def backward_difference(phi: Array, spacing: tuple[float, ...], axis: int) -> Array:
    """Backward finite difference with edge replication at the boundary."""

    _axis_spacing(spacing, axis)
    return _backward_difference_jit(phi, spacing, axis)


@partial(jax.jit, static_argnames=("axis",))
def _central_difference_jit(phi: Array, spacing: tuple[float, ...], axis: int) -> Array:
    dx = spacing[axis]
    prev_values = _neighbor(phi, axis, -1)
    next_values = _neighbor(phi, axis, 1)
    raw = (next_values - prev_values) / (2.0 * dx)
    fwd = (next_values - phi) / dx
    bwd = (phi - prev_values) / dx
    index_shape = [1] * phi.ndim
    index_shape[axis] = phi.shape[axis]
    indices = jnp.arange(phi.shape[axis]).reshape(index_shape)
    return jnp.where(indices == 0, fwd, jnp.where(indices == phi.shape[axis] - 1, bwd, raw))


def central_difference(phi: Array, spacing: tuple[float, ...], axis: int) -> Array:
    """Central finite difference with first-order one-sided boundary values."""

    _axis_spacing(spacing, axis)
    return _central_difference_jit(phi, spacing, axis)


def one_sided_derivatives(phi: Array, spacing: tuple[float, ...]) -> tuple[Array, Array]:
    """Return stacked backward and forward derivatives for all axes."""

    backward, forward = _one_sided_derivatives_jit(phi, spacing)
    return backward, forward


@jax.jit
def _one_sided_derivatives_jit(phi: Array, spacing: tuple[float, ...]) -> tuple[Array, Array]:
    backward = []
    forward = []
    for axis in range(phi.ndim):
        dx = spacing[axis]
        previous = _neighbor(phi, axis, -1)
        next_values = _neighbor(phi, axis, 1)
        backward.append((phi - previous) / dx)
        forward.append((next_values - phi) / dx)
    return jnp.stack(backward, axis=-1), jnp.stack(forward, axis=-1)


def gradient_central(phi: Array, spacing: tuple[float, ...]) -> Array:
    """Return the central-difference gradient stacked on the last axis."""

    gradients = [central_difference(phi, spacing, axis) for axis in range(phi.ndim)]
    return jnp.stack(gradients, axis=-1)


def gradient_magnitude(phi: Array, spacing: tuple[float, ...], eps: float = 1e-12) -> Array:
    """Return ``sqrt(sum(grad_i^2) + eps)`` using central differences."""

    gradient = gradient_central(phi, spacing)
    return jnp.sqrt(jnp.sum(gradient * gradient, axis=-1) + eps)
