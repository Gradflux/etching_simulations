"""Material dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class Material:
    """Simple immutable material description."""

    name: str
    id: int | None = None
    category: str = "custom"
    density: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    color: tuple[float, float, float] | None = None
    aliases: tuple[str, ...] = ()
    properties: Mapping[str, Any] = field(default_factory=dict)
    builtin: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("material name must not be empty")
        if self.id is not None and self.id < 0:
            raise ValueError("material id must be nonnegative when provided")
        if self.density is not None and self.density < 0.0:
            raise ValueError("density must be nonnegative when provided")
        if self.color is not None and len(self.color) != 3:
            raise ValueError("color must be an RGB tuple when provided")
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "properties", MappingProxyType(dict(self.properties)))

    def to_dict(self) -> dict[str, Any]:
        """Serialize material to JSON-compatible data."""

        return {
            "name": self.name,
            "id": self.id,
            "category": self.category,
            "density": self.density,
            "metadata": dict(self.metadata),
            "color": list(self.color) if self.color is not None else None,
            "aliases": list(self.aliases),
            "properties": dict(self.properties),
            "builtin": self.builtin,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Material":
        """Deserialize a material from JSON-compatible data."""

        color = data.get("color")
        return cls(
            name=str(data["name"]),
            id=data.get("id"),
            category=str(data.get("category", "custom")),
            density=data.get("density"),
            metadata=data.get("metadata", {}),
            color=tuple(color) if color is not None else None,
            aliases=tuple(data.get("aliases", ())),
            properties=data.get("properties", {}),
            builtin=bool(data.get("builtin", False)),
        )
