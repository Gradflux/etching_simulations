"""Etching velocity models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
from jax import Array

from jaxps.materials.material import Material
from jaxps.materials.registry import MaterialRegistry, default_material_registry
from jaxps.models.rates import constant_rate_like, validate_nonnegative, validate_nonnegative_static


def isotropic_etch_rate(phi: Array, rate: float) -> Array:
    """Return negative normal velocity for isotropic etching."""

    validate_nonnegative_static(rate, "rate")
    return constant_rate_like(phi, -jnp.asarray(rate))


def directional_etch_rate(normals: Array, direction: Array, rate: float) -> Array:
    """Return nonpositive velocity from an incoming directional etch field.

    ``direction`` is the incoming direction of flux travel. Incidence is
    ``max(dot(normal, -direction), 0)``.
    """

    validate_nonnegative_static(rate, "rate")
    direction_arr = jnp.asarray(direction, dtype=normals.dtype)
    direction_arr = direction_arr / jnp.maximum(jnp.linalg.norm(direction_arr), 1e-12)
    incidence = jnp.maximum(jnp.sum(normals * (-direction_arr), axis=-1), 0.0)
    return -jnp.asarray(rate, dtype=normals.dtype) * incidence


def masked_etch_rate(base_velocity: Array, mask: Array) -> Array:
    """Set protected cells to zero velocity where ``mask`` is true."""

    return jnp.where(mask, 0.0, base_velocity)


def material_id_mask(
    material_ids: Array,
    materials: Sequence[str | Material] | None,
    registry: MaterialRegistry | None = None,
) -> Array:
    """Return a boolean mask for selected materials."""

    if materials is None:
        return jnp.ones_like(material_ids, dtype=bool)
    registry = registry or default_material_registry()
    selected_ids = []
    for material in materials:
        selected = material if isinstance(material, Material) else registry.get(str(material))
        if selected.id is None:
            raise ValueError(f"material {selected.name} has no id")
        selected_ids.append(int(selected.id))
    if not selected_ids:
        return jnp.zeros_like(material_ids, dtype=bool)
    mask = jnp.zeros_like(material_ids, dtype=bool)
    for material_id in selected_ids:
        mask = mask | (material_ids == material_id)
    return mask


def target_material_velocity(
    velocity: Array,
    material_ids: Array,
    target_materials: Sequence[str | Material] | None,
    registry: MaterialRegistry | None = None,
    fallback_velocity: float = 0.0,
) -> Array:
    """Apply velocity only to selected target materials."""

    mask = material_id_mask(material_ids, target_materials, registry)
    return jnp.where(mask, velocity, fallback_velocity)


def masked_material_etch_rate(
    material_ids: Array,
    registry: MaterialRegistry,
    base_rate: float,
    mask_materials: Sequence[str | Material],
    mask_rate: float = 0.0,
) -> Array:
    """Return material-dependent etch velocity with slower/protected masks."""

    validate_nonnegative_static(base_rate, "base_rate")
    validate_nonnegative_static(mask_rate, "mask_rate")
    mask = material_id_mask(material_ids, mask_materials, registry)
    rates = jnp.where(mask, float(mask_rate), float(base_rate))
    return -rates


def ion_enhanced_etch_rate(chemical_rate: float, ion_flux: Array, ion_yield: Array | float) -> Array:
    """Return a nonpositive ion-enhanced etch velocity."""

    validate_nonnegative_static(chemical_rate, "chemical_rate")
    validate_nonnegative_static(ion_yield, "ion_yield")
    return -(jnp.asarray(chemical_rate) + jnp.asarray(ion_flux) * ion_yield)


@dataclass(frozen=True)
class IsotropicEtch:
    """Constant-rate isotropic etch model."""

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
        velocity = isotropic_etch_rate(phi, self.rate)
        if self.target_materials is not None:
            if material_ids is None:
                raise ValueError("material_ids are required when target_materials are set")
            velocity = target_material_velocity(velocity, material_ids, self.target_materials, registry)
        return velocity


@dataclass(frozen=True)
class DirectionalEtch:
    """Directional etch model driven by surface normals."""

    rate: float
    direction: tuple[float, ...]

    def __post_init__(self) -> None:
        validate_nonnegative(self.rate, "rate")
        if not self.direction:
            raise ValueError("direction must not be empty")

    def velocity(self, normals: Array) -> Array:
        """Return normal velocity from precomputed normals."""

        return directional_etch_rate(normals, jnp.asarray(self.direction), self.rate)


@dataclass(frozen=True)
class MaskedEtch:
    """Apply a protected-cell mask to a base etch velocity."""

    protected_mask: Array

    def velocity(self, base_velocity: Array) -> Array:
        """Return masked velocity."""

        return masked_etch_rate(base_velocity, self.protected_mask)


@dataclass(frozen=True)
class MaterialMaskedEtch:
    """Etch model with explicit mask material rates."""

    base_rate: float
    mask_materials: tuple[str | Material, ...]
    mask_rate: float = 0.0

    def __post_init__(self) -> None:
        validate_nonnegative(self.base_rate, "base_rate")
        validate_nonnegative(self.mask_rate, "mask_rate")

    def velocity(self, material_ids: Array, registry: MaterialRegistry | None = None) -> Array:
        """Return material-mask velocity field."""

        return masked_material_etch_rate(
            material_ids,
            registry or default_material_registry(),
            self.base_rate,
            self.mask_materials,
            self.mask_rate,
        )


@dataclass(frozen=True)
class IonEnhancedEtch:
    """Simple chemical plus ion-enhanced etch model."""

    chemical_rate: float
    ion_yield: float

    def __post_init__(self) -> None:
        validate_nonnegative(self.chemical_rate, "chemical_rate")
        validate_nonnegative(self.ion_yield, "ion_yield")

    def velocity(self, ion_flux: Array) -> Array:
        """Return nonpositive velocity from ion flux."""

        return ion_enhanced_etch_rate(self.chemical_rate, ion_flux, self.ion_yield)
