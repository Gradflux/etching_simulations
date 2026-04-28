"""Clean-room JAX process simulation kernels."""

from jaxps.geometry import Grid, make_grid_2d, make_grid_3d, sdf_circle, sdf_plane, sdf_sphere
from jaxps.solvers import EvolutionResult, evolve_level_set, level_set_step

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
