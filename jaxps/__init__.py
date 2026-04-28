"""Clean-room JAX process simulation kernels."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "EvolutionResult",
    "Grid",
    "evolve_level_set",
    "level_set_step",
    "make_grid_2d",
    "make_grid_3d",
    "sdf_circle",
    "sdf_plane",
    "sdf_sphere",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazily resolve public symbols to keep lightweight imports dependency-safe."""

    geometry_exports = {
        "Grid",
        "make_grid_2d",
        "make_grid_3d",
        "sdf_circle",
        "sdf_plane",
        "sdf_sphere",
    }
    solver_exports = {"EvolutionResult", "evolve_level_set", "level_set_step"}

    if name in geometry_exports:
        module = import_module("jaxps.geometry")
        return getattr(module, name)
    if name in solver_exports:
        module = import_module("jaxps.solvers")
        return getattr(module, name)
    raise AttributeError(f"module 'jaxps' has no attribute {name!r}")
