"""Deposition velocity models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
from jax import Array

from jaxps.materials.material import Material
from jaxps.materials.registry import MaterialRegistry
from jaxps.models.etch import target_material_velocity
from jaxps.models.rates import constant_rate_like, validate_nonnegative, validate_nonnegative_static


def isotropic_deposition_rate(phi: Array, rate: float) -> Array:
    """Return positive normal velocity for isotropic deposition."""

    validate_nonnegative_static(rate, "rate")
    return constant_rate_like(phi, rate)


def sticking_deposition_rate(flux: Array, sticking_coefficient: float) -> Array:
    """Convert nonnegative flux to deposition velocity."""

    validate_nonnegative_static(sticking_coefficient, "sticking_coefficient")
    return jnp.asarray(flux) * jnp.asarray(sticking_coefficient)


@dataclass(frozen=True)
class IsotropicDeposition:
    """Constant-rate isotropic deposition model."""

    rate: float
    target_materials: tuple[str | Material, ...] | None = None

    def __post_init__(self) -> None:
        validate_nonnegative(self.rate, "rate")

    def velocity(
        self,
        phi: Array,
        t: float | Array = 0.0,
        material_ids: Array | None = None,
        registry: MaterialRegistry | None = None,
    ) -> Array:
        """Return normal velocity for ``phi``."""

        del t
        velocity = isotropic_deposition_rate(phi, self.rate)
        if self.target_materials is not None:
            if material_ids is None:
                raise ValueError("material_ids are required when target_materials are set")
            velocity = target_material_velocity(velocity, material_ids, self.target_materials, registry)
        return velocity


@dataclass(frozen=True)
class SimpleDeposition:
    """Flux deposition model with a scalar sticking coefficient."""

    sticking_coefficient: float

    def __post_init__(self) -> None:
        validate_nonnegative(self.sticking_coefficient, "sticking_coefficient")

    def velocity(self, flux: Array) -> Array:
        """Return deposition velocity from flux."""

        return sticking_deposition_rate(flux, self.sticking_coefficient)
