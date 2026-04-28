"""Built-in semiconductor process materials.

The list is an independently chosen compact set from general semiconductor
process knowledge. It is not copied from ViennaPS source code.
"""

from __future__ import annotations

from jaxps.materials.material import Material


BUILTIN_MATERIALS: tuple[Material, ...] = (
    Material("Void", 0, "void", metadata={"description": "empty/ambient region"}, aliases=("Air", "Vacuum"), color=(0.9, 0.95, 1.0), builtin=True),
    Material("Si", 1, "semiconductor", density=2.329, aliases=("Silicon",), color=(0.45, 0.45, 0.5), builtin=True),
    Material("SiO2", 2, "dielectric", density=2.2, aliases=("Oxide", "SiliconDioxide"), color=(0.8, 0.9, 1.0), builtin=True),
    Material("Si3N4", 3, "dielectric", density=3.17, aliases=("Nitride",), color=(0.6, 0.75, 0.9), builtin=True),
    Material("Photoresist", 4, "mask", aliases=("Resist", "PR"), color=(0.8, 0.35, 0.15), builtin=True),
    Material("Mask", 5, "mask", color=(0.15, 0.15, 0.15), builtin=True),
    Material("Polymer", 6, "passivation", color=(0.3, 0.7, 0.4), builtin=True),
    Material("W", 7, "metal", density=19.25, aliases=("Tungsten",), color=(0.55, 0.55, 0.6), builtin=True),
    Material("TiN", 8, "metal", density=5.4, color=(0.55, 0.45, 0.25), builtin=True),
    Material("Al2O3", 9, "dielectric", density=3.95, aliases=("Alumina",), color=(0.9, 0.85, 0.75), builtin=True),
    Material("HfO2", 10, "dielectric", density=9.68, color=(0.85, 0.75, 0.55), builtin=True),
    Material("Cu", 11, "metal", density=8.96, aliases=("Copper",), color=(0.9, 0.45, 0.2), builtin=True),
    Material("Al", 12, "metal", density=2.7, aliases=("Aluminum",), color=(0.75, 0.75, 0.8), builtin=True),
    Material("Au", 13, "metal", density=19.32, aliases=("Gold",), color=(1.0, 0.78, 0.2), builtin=True),
    Material("Cr", 14, "metal", density=7.19, aliases=("Chromium",), color=(0.5, 0.5, 0.55), builtin=True),
)


def list_builtin_materials() -> tuple[Material, ...]:
    """Return built-in canonical materials."""

    return BUILTIN_MATERIALS


def builtin_material_by_name(name: str) -> Material:
    """Return a built-in material by canonical name or alias."""

    for material in BUILTIN_MATERIALS:
        if material.name == name or name in material.aliases:
            return material
    raise KeyError(name)


def is_builtin_material(name: str) -> bool:
    """Return whether ``name`` resolves to a built-in material."""

    try:
        builtin_material_by_name(name)
    except KeyError:
        return False
    return True
