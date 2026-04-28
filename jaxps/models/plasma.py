"""Conservative plasma process models.

These models are intentionally simple and transparent. They represent ion,
neutral, and passivating flux contributions with explicit sign conventions:
etch terms are negative normal velocities and passivation/deposition terms are
positive normal velocities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array

from jaxps.materials.registry import MaterialRegistry, default_material_registry
from jaxps.materials.value_map import MaterialValueMap
from jaxps.models.rates import validate_nonnegative


def simple_plasma_velocity(
    ion_flux: Array,
    neutral_flux: Array,
    passivation_flux: Array,
    ion_yield: Array | float,
    neutral_coefficient: Array | float,
    passivation_sticking: Array | float,
) -> Array:
    """Return net normal velocity for a simple plasma process.

    ``V = -(Gamma_i Y_i + Gamma_n k_n) + Gamma_p s_p``.
    """

    ion_etch = jnp.asarray(ion_flux) * jnp.asarray(ion_yield)
    neutral_etch = jnp.asarray(neutral_flux) * jnp.asarray(neutral_coefficient)
    passivation = jnp.asarray(passivation_flux) * jnp.asarray(passivation_sticking)
    return -(ion_etch + neutral_etch) + passivation


@dataclass(frozen=True)
class SimplePlasmaProcess:
    """Material-aware ion, neutral, and passivation process model."""

    ion_yield: MaterialValueMap = field(default_factory=lambda: MaterialValueMap(default=1.0))
    neutral_coefficient: MaterialValueMap = field(default_factory=lambda: MaterialValueMap(default=0.0))
    passivation_sticking: MaterialValueMap = field(default_factory=lambda: MaterialValueMap(default=0.0))

    def velocity(
        self,
        material_ids: Array,
        ion_flux: Array,
        neutral_flux: Array,
        passivation_flux: Array,
        registry: MaterialRegistry | None = None,
    ) -> Array:
        """Evaluate the net plasma velocity field."""

        registry = registry or default_material_registry()
        return simple_plasma_velocity(
            ion_flux=ion_flux,
            neutral_flux=neutral_flux,
            passivation_flux=passivation_flux,
            ion_yield=self.ion_yield.evaluate(material_ids, registry),
            neutral_coefficient=self.neutral_coefficient.evaluate(material_ids, registry),
            passivation_sticking=self.passivation_sticking.evaluate(material_ids, registry),
        )


@dataclass(frozen=True)
class PlasmaFluxes:
    """Named scalar fluxes for simple plasma models."""

    ion_flux: float = 0.0
    neutral_flux: float = 0.0
    passivation_flux: float = 0.0

    def __post_init__(self) -> None:
        validate_nonnegative(self.ion_flux, "ion_flux")
        validate_nonnegative(self.neutral_flux, "neutral_flux")
        validate_nonnegative(self.passivation_flux, "passivation_flux")
