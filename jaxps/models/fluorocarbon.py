"""Simplified fluorocarbon plasma etch model.

This is an independent, documented toy model for SF6/C4F8-style behavior:
fluorine-like radicals chemically etch, ions add directional enhancement, and
polymer precursors passivate/deposit. It is not a reproduction of any ViennaPS
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array

from jaxps.materials.registry import MaterialRegistry, default_material_registry
from jaxps.materials.value_map import MaterialValueMap
from jaxps.models.rates import validate_nonnegative


def fluorocarbon_velocity(
    f_flux: Array,
    ion_flux: Array,
    polymer_flux: Array,
    chemical_coefficient: Array | float,
    ion_yield: Array | float,
    polymer_sticking: Array | float,
    angular_yield: Array | float = 1.0,
    coverage: Array | float | None = None,
    coverage_suppression: float = 1.0,
) -> Array:
    """Return net normal velocity for a simplified fluorocarbon process.

    ``V = -(k_F Gamma_F + Y_i Gamma_i A(theta)) exp(-a C) + s_p Gamma_p``.
    Positive velocity deposits/grows material; negative velocity etches.
    """

    chemical = jnp.asarray(chemical_coefficient) * jnp.asarray(f_flux)
    ion = jnp.asarray(ion_yield) * jnp.asarray(ion_flux) * jnp.asarray(angular_yield)
    etch = chemical + ion
    if coverage is not None:
        etch = etch * jnp.exp(-jnp.asarray(coverage_suppression) * jnp.asarray(coverage))
    passivation = jnp.asarray(polymer_sticking) * jnp.asarray(polymer_flux)
    return -etch + passivation


def update_passivation_coverage(
    coverage: Array,
    polymer_flux: Array,
    ion_flux: Array,
    dt: float,
    deposition_coefficient: float,
    sputter_coefficient: float,
) -> Array:
    """Explicitly update a bounded passivation coverage variable."""

    validate_nonnegative(dt, "dt")
    validate_nonnegative(deposition_coefficient, "deposition_coefficient")
    validate_nonnegative(sputter_coefficient, "sputter_coefficient")
    next_coverage = jnp.asarray(coverage) + jnp.asarray(dt) * (
        deposition_coefficient * jnp.asarray(polymer_flux)
        - sputter_coefficient * jnp.asarray(ion_flux) * jnp.asarray(coverage)
    )
    return jnp.clip(next_coverage, 0.0, 1.0)


@dataclass(frozen=True)
class SimplifiedFluorocarbonEtch:
    """Material-aware simplified fluorocarbon process model."""

    chemical_coefficients: MaterialValueMap = field(default_factory=lambda: MaterialValueMap(default=0.0))
    ion_yields: MaterialValueMap = field(default_factory=lambda: MaterialValueMap(default=1.0))
    polymer_sticking: MaterialValueMap = field(default_factory=lambda: MaterialValueMap(default=0.0))
    coverage_suppression: float = 1.0

    def __post_init__(self) -> None:
        validate_nonnegative(self.coverage_suppression, "coverage_suppression")

    def velocity(
        self,
        material_ids: Array,
        f_flux: Array,
        ion_flux: Array,
        polymer_flux: Array,
        angular_yield: Array | float = 1.0,
        coverage: Array | float | None = None,
        registry: MaterialRegistry | None = None,
    ) -> Array:
        """Evaluate net normal velocity."""

        registry = registry or default_material_registry()
        return fluorocarbon_velocity(
            f_flux=f_flux,
            ion_flux=ion_flux,
            polymer_flux=polymer_flux,
            chemical_coefficient=self.chemical_coefficients.evaluate(material_ids, registry),
            ion_yield=self.ion_yields.evaluate(material_ids, registry),
            polymer_sticking=self.polymer_sticking.evaluate(material_ids, registry),
            angular_yield=angular_yield,
            coverage=coverage,
            coverage_suppression=self.coverage_suppression,
        )
