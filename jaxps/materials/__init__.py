"""Material metadata utilities."""

from jaxps.materials.builtins import (
    BUILTIN_MATERIALS,
    builtin_material_by_name,
    is_builtin_material,
    list_builtin_materials,
)
from jaxps.materials.material import Material
from jaxps.materials.registry import (
    MaterialRegistry,
    default_material_registry,
    deserialize_material_registry,
    get_material,
    serialize_material_registry,
)
from jaxps.materials.value_map import MaterialValueMap

__all__ = [
    "BUILTIN_MATERIALS",
    "Material",
    "MaterialRegistry",
    "MaterialValueMap",
    "builtin_material_by_name",
    "default_material_registry",
    "deserialize_material_registry",
    "get_material",
    "is_builtin_material",
    "list_builtin_materials",
    "serialize_material_registry",
]
