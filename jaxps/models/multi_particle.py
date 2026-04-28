"""Multi-species particle and flux process models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jax import Array

from jaxps.materials.material import Material
from jaxps.materials.registry import MaterialRegistry, default_material_registry
from jaxps.materials.value_map import MaterialValueMap
from jaxps.models.etch import target_material_velocity
from jaxps.models.rates import validate_nonnegative
from jaxps.models.yield_models import PolynomialCosineYield

SpeciesKind = Literal["etch", "deposition"]


def combine_species_rates(species_rates: Array) -> Array:
    """Sum per-species velocity contributions along the first axis."""

    rates = jnp.asarray(species_rates)
    if rates.shape[0] == 0:
        return jnp.asarray(0.0, dtype=rates.dtype)
    return jnp.sum(rates, axis=0)


def linear_species_rate(flux: Array, coefficient: Array | float, sign: Array | float) -> Array:
    """Return a signed linear species velocity contribution."""

    return jnp.asarray(sign) * jnp.asarray(flux) * jnp.asarray(coefficient)


@dataclass(frozen=True)
class ParticleSpecies:
    """One particle species contribution to a process model."""

    name: str
    kind: SpeciesKind = "etch"
    flux: float = 0.0
    coefficient: float = 1.0
    direction: tuple[float, ...] | None = None
    angular_yield: PolynomialCosineYield | None = None
    target_materials: tuple[str | Material, ...] | None = None
    material_response: MaterialValueMap | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("species name must not be empty")
        if self.kind not in ("etch", "deposition"):
            raise ValueError("kind must be 'etch' or 'deposition'")
        validate_nonnegative(self.flux, "flux")
        validate_nonnegative(self.coefficient, "coefficient")
        if self.angular_yield is not None and self.direction is None:
            raise ValueError("direction is required when angular_yield is provided")

    @property
    def sign(self) -> float:
        """Return velocity sign for this species."""

        return -1.0 if self.kind == "etch" else 1.0


def compute_species_rate(
    species: ParticleSpecies,
    normals: Array,
    material_ids: Array | None = None,
    registry: MaterialRegistry | None = None,
) -> Array:
    """Compute one species velocity contribution."""

    registry = registry or default_material_registry()
    coefficient: Array | float = species.coefficient
    if species.material_response is not None:
        if material_ids is None:
            raise ValueError("material_ids are required for material_response")
        coefficient = jnp.asarray(coefficient) * species.material_response.evaluate(material_ids, registry)
    if species.angular_yield is not None:
        coefficient = jnp.asarray(coefficient) * species.angular_yield.from_direction(
            jnp.asarray(species.direction), normals
        )
    velocity = linear_species_rate(species.flux, coefficient, species.sign)
    velocity = jnp.zeros(normals.shape[:-1], dtype=normals.dtype) + velocity
    if species.target_materials is not None:
        if material_ids is None:
            raise ValueError("material_ids are required for target_materials")
        velocity = target_material_velocity(velocity, material_ids, species.target_materials, registry)
    return velocity


@dataclass(frozen=True)
class MultiParticleProcess:
    """Additive multi-species process model."""

    species: tuple[ParticleSpecies, ...]
    return_diagnostics: bool = False

    def velocity(
        self,
        normals: Array,
        material_ids: Array | None = None,
        registry: MaterialRegistry | None = None,
    ) -> Array | tuple[Array, Array]:
        """Return total velocity, optionally with per-species diagnostics."""

        if not self.species:
            total = jnp.zeros(normals.shape[:-1], dtype=normals.dtype)
            diagnostics = jnp.zeros((0,) + normals.shape[:-1], dtype=normals.dtype)
        else:
            diagnostics = jnp.stack(
                [
                    compute_species_rate(item, normals, material_ids=material_ids, registry=registry)
                    for item in self.species
                ],
                axis=0,
            )
            total = combine_species_rates(diagnostics)
        if self.return_diagnostics:
            return total, diagnostics
        return total
