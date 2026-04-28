"""Immutable material registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from jaxps.materials.builtins import BUILTIN_MATERIALS
from jaxps.materials.material import Material


@dataclass(frozen=True)
class MaterialRegistry:
    """Small immutable registry for material lookup by name or id."""

    materials: tuple[Material, ...] = ()
    custom_id_start: int = 1000

    def __post_init__(self) -> None:
        names: set[str] = set()
        ids: set[int] = set()
        for material in self.materials:
            for name in (material.name, *material.aliases):
                if name in names:
                    raise ValueError(f"duplicate material name or alias: {name}")
                names.add(name)
            if material.id is not None:
                if material.id in ids:
                    raise ValueError(f"duplicate material id: {material.id}")
                ids.add(material.id)

    def register(
        self,
        material: Material | str,
        *,
        id: int | None = None,
        category: str = "custom",
        density: float | None = None,
        metadata: dict[str, Any] | None = None,
        color: tuple[float, float, float] | None = None,
        aliases: Iterable[str] = (),
        properties: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> "MaterialRegistry":
        """Return a new registry with ``material`` added."""

        if isinstance(material, str):
            material = Material(
                name=material,
                id=id if id is not None else self.next_custom_id(),
                category=category,
                density=density,
                metadata=metadata or {},
                color=color,
                aliases=tuple(aliases),
                properties=properties or {},
            )
        elif material.id is None:
            material = Material(
                name=material.name,
                id=id if id is not None else self.next_custom_id(),
                category=material.category,
                density=material.density,
                metadata=material.metadata,
                color=material.color,
                aliases=material.aliases,
                properties=material.properties,
                builtin=material.builtin,
            )

        remaining = []
        for existing in self.materials:
            same_name = existing.name == material.name or material.name in existing.aliases
            same_id = material.id is not None and existing.id == material.id
            if same_name or same_id:
                if not overwrite:
                    if same_name:
                        raise ValueError(f"duplicate material name: {material.name}")
                    raise ValueError(f"duplicate material id: {material.id}")
                continue
            if any(alias == existing.name or alias in existing.aliases for alias in material.aliases):
                raise ValueError(f"duplicate material alias in {material.aliases}")
            remaining.append(existing)
        return MaterialRegistry(materials=tuple(remaining) + (material,), custom_id_start=self.custom_id_start)

    def next_custom_id(self) -> int:
        """Return the next deterministic custom material ID."""

        used = {material.id for material in self.materials if material.id is not None}
        candidate = self.custom_id_start
        while candidate in used:
            candidate += 1
        return candidate

    def get(self, name: str) -> Material:
        """Look up a material by name."""

        for material in self.materials:
            if material.name == name or name in material.aliases:
                return material
        raise KeyError(name)

    def get_by_id(self, material_id: int) -> Material:
        """Look up a material by numeric id."""

        for material in self.materials:
            if material.id == material_id:
                return material
        raise KeyError(material_id)

    def list_materials(self) -> tuple[Material, ...]:
        """Return registered materials."""

        return self.materials

    def names(self) -> tuple[str, ...]:
        """Return canonical registered material names."""

        return tuple(material.name for material in self.materials)

    def to_dict(self) -> dict[str, Any]:
        """Serialize registry to JSON-compatible data."""

        return {
            "custom_id_start": self.custom_id_start,
            "materials": [material.to_dict() for material in self.materials],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MaterialRegistry":
        """Deserialize a registry from JSON-compatible data."""

        return cls(
            materials=tuple(Material.from_dict(item) for item in data.get("materials", [])),
            custom_id_start=int(data.get("custom_id_start", 1000)),
        )


def default_material_registry() -> MaterialRegistry:
    """Return a registry containing built-in materials."""

    return MaterialRegistry(materials=BUILTIN_MATERIALS)


def get_material(name: str, registry: MaterialRegistry | None = None) -> Material:
    """Convenience lookup from the default or supplied registry."""

    return (registry or default_material_registry()).get(name)


def serialize_material_registry(registry: MaterialRegistry) -> dict[str, Any]:
    """Serialize registry to JSON-compatible data."""

    return registry.to_dict()


def deserialize_material_registry(data: dict[str, Any]) -> MaterialRegistry:
    """Deserialize registry data."""

    return MaterialRegistry.from_dict(data)
