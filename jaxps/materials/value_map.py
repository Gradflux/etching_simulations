"""Material-to-value mapping utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
from jax import Array

from jaxps.materials.material import Material
from jaxps.materials.registry import MaterialRegistry, default_material_registry


def _material_name(material: str | Material) -> str:
    return material.name if isinstance(material, Material) else str(material)


@dataclass(frozen=True)
class MaterialValueMap:
    """Map material names to scalar or vector values with a default fallback."""

    default: Any = 0.0
    values: dict[str, Any] = field(default_factory=dict)

    def set(self, material: str | Material, value: Any) -> "MaterialValueMap":
        """Return a new map with one value set."""

        next_values = dict(self.values)
        next_values[_material_name(material)] = value
        return MaterialValueMap(default=self.default, values=next_values)

    def evaluate(self, material_ids: Array, registry: MaterialRegistry | None = None) -> Array:
        """Convert a material ID field to a JAX value field."""

        registry = registry or default_material_registry()
        default_arr = jnp.asarray(self.default)
        value_arrays = {name: jnp.asarray(value) for name, value in self.values.items()}
        value_shape = default_arr.shape
        if value_shape == ():
            for value_arr in value_arrays.values():
                if value_arr.shape != ():
                    value_shape = value_arr.shape
                    break
        for value_arr in value_arrays.values():
            if value_arr.shape not in ((), value_shape):
                raise ValueError("all non-scalar material values must have the same shape")
        dtype = jnp.result_type(default_arr, *value_arrays.values()) if value_arrays else default_arr.dtype
        result = jnp.zeros(material_ids.shape + value_shape, dtype=dtype) + default_arr.astype(dtype)
        for name, value in self.values.items():
            material = registry.get(name)
            if material.id is None:
                raise ValueError(f"material {name} has no id")
            value_arr = jnp.asarray(value, dtype=result.dtype)
            condition = material_ids == material.id
            if value_shape:
                condition = condition[(...,) + (None,) * len(value_shape)]
            result = jnp.where(condition, value_arr, result)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""

        return {"default": self.default, "values": self.values}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MaterialValueMap":
        """Deserialize a value map."""

        return cls(default=data.get("default", 0.0), values=dict(data.get("values", {})))
