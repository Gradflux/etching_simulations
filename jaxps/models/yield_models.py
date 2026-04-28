"""Simple sputtering yield models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
from jax import Array

from jaxps.models.rates import validate_nonnegative, validate_nonnegative_static


def sputtering_yield(
    cos_theta: Array,
    threshold: float = 0.0,
    scale: float = 1.0,
    exponent: float = 1.0,
) -> Array:
    """Return a nonnegative thresholded polynomial sputtering yield."""

    validate_nonnegative_static(threshold, "threshold")
    validate_nonnegative_static(scale, "scale")
    validate_nonnegative_static(exponent, "exponent")
    active = jnp.maximum(jnp.asarray(cos_theta) - jnp.asarray(threshold), 0.0)
    return jnp.asarray(scale) * active**jnp.asarray(exponent)


def incidence_cosine(direction: Array, normals: Array) -> Array:
    """Return ``mu = max(0, -dot(direction, normal))``.

    ``direction`` points along particle travel toward the surface and
    ``normals`` are outward material normals.
    """

    direction_arr = jnp.asarray(direction, dtype=normals.dtype)
    direction_arr = direction_arr / jnp.maximum(jnp.linalg.norm(direction_arr), 1e-12)
    normals_arr = normals / jnp.maximum(jnp.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
    return jnp.maximum(-jnp.sum(normals_arr * direction_arr, axis=-1), 0.0)


@jax.jit
def _polynomial_cosine_yield_jit(mu: Array, coeffs: Array, clamp_flag: Array) -> Array:
    mu_arr = jnp.asarray(mu, dtype=coeffs.dtype)
    # coeffs[i] is the coefficient for mu^i (ascending order).
    # jnp.polyval expects descending order, so reverse before passing.
    result = jnp.polyval(coeffs[::-1], mu_arr)
    return jnp.where(clamp_flag, jnp.maximum(result, 0.0), result)


def polynomial_cosine_yield(mu: Array, coefficients: Sequence[float] | Array, clamp: bool = True) -> Array:
    """Evaluate ``sum_i coefficients[i] * mu**i``."""

    coeffs = jnp.asarray(coefficients)
    return _polynomial_cosine_yield_jit(mu, coeffs, jnp.asarray(clamp))


@dataclass(frozen=True)
class SputteringYield:
    """Thresholded polynomial sputtering yield."""

    threshold: float = 0.0
    scale: float = 1.0
    exponent: float = 1.0

    def __post_init__(self) -> None:
        validate_nonnegative(self.threshold, "threshold")
        validate_nonnegative(self.scale, "scale")
        validate_nonnegative(self.exponent, "exponent")

    def __call__(self, cos_theta: Array) -> Array:
        """Evaluate the yield."""

        return sputtering_yield(cos_theta, self.threshold, self.scale, self.exponent)


@dataclass(frozen=True)
class PolynomialCosineYield:
    """Polynomial cosine angular yield model."""

    coefficients: tuple[float, ...]
    clamp: bool = True

    def __post_init__(self) -> None:
        if not self.coefficients:
            raise ValueError("coefficients must not be empty")

    def __call__(self, mu: Array) -> Array:
        """Evaluate yield from incidence cosine."""

        return polynomial_cosine_yield(mu, self.coefficients, self.clamp)

    def from_direction(self, direction: Array, normals: Array) -> Array:
        """Evaluate yield from particle direction and outward normals."""

        return self(incidence_cosine(direction, normals))
