"""Experimental FFT-based operators for smooth periodic diagnostics.

These operators are not monotone and must not be used as the default
Hamilton-Jacobi level-set advection discretization.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def _wave_numbers(size: int, domain_length: float, dtype) -> Array:
    return 2.0 * jnp.pi * jnp.fft.fftfreq(size, d=float(domain_length) / float(size)).astype(dtype)


def periodic_spectral_derivative(field: Array, domain_lengths: tuple[float, ...], axis: int) -> Array:
    """Return a periodic spectral derivative along one axis."""

    if axis < 0 or axis >= field.ndim:
        raise ValueError("axis out of range")
    if len(domain_lengths) != field.ndim:
        raise ValueError("domain_lengths dimension must match field dimension")
    if domain_lengths[axis] <= 0.0:
        raise ValueError("domain lengths must be positive")
    wave_numbers = _wave_numbers(field.shape[axis], domain_lengths[axis], field.dtype)
    reshape = [1] * field.ndim
    reshape[axis] = field.shape[axis]
    multiplier = 1j * wave_numbers.reshape(reshape)
    return jnp.fft.ifft(multiplier * jnp.fft.fft(field, axis=axis), axis=axis).real


@jax.jit
def _periodic_spectral_gradient_jit(field: Array, domain_lengths: tuple[float, ...]) -> Array:
    components = [periodic_spectral_derivative(field, domain_lengths, axis) for axis in range(field.ndim)]
    return jnp.stack(components, axis=-1)


def periodic_spectral_gradient(field: Array, domain_lengths: tuple[float, ...]) -> Array:
    """Return the periodic spectral gradient stacked on the last axis."""

    return _periodic_spectral_gradient_jit(field, domain_lengths)


def periodic_spectral_laplacian(field: Array, domain_lengths: tuple[float, ...]) -> Array:
    """Return the periodic spectral Laplacian."""

    if len(domain_lengths) != field.ndim:
        raise ValueError("domain_lengths dimension must match field dimension")
    spectrum = jnp.fft.fftn(field)
    k_squared = 0.0
    for axis, length in enumerate(domain_lengths):
        if length <= 0.0:
            raise ValueError("domain lengths must be positive")
        wave_numbers = _wave_numbers(field.shape[axis], length, field.dtype)
        reshape = [1] * field.ndim
        reshape[axis] = field.shape[axis]
        k_squared = k_squared + wave_numbers.reshape(reshape) ** 2
    return jnp.fft.ifftn(-k_squared * spectrum).real
