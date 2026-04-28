"""Geometry, SDF, derivative, and normal utilities."""

from jaxps.geometry.derivatives import (
    backward_difference,
    central_difference,
    forward_difference,
    gradient_central,
    gradient_magnitude,
    one_sided_derivatives,
)
from jaxps.geometry.grid import Grid, make_grid_2d, make_grid_3d
from jaxps.geometry.extrude import extrude_2d_to_3d
from jaxps.geometry.masks import material_mask, rectangular_mask_2d, void_mask
from jaxps.geometry.mesh import (
    BoundaryMesh2D,
    SurfaceMesh3D,
    extract_hull_mesh,
    extract_surface_mesh_3d,
    extract_zero_contour_2d,
)
from jaxps.geometry.normals import surface_normals
from jaxps.geometry.slice import slice_3d
from jaxps.geometry.sdf import (
    sdf_box,
    sdf_circle,
    sdf_difference,
    sdf_intersection,
    sdf_plane,
    sdf_sphere,
    sdf_union,
)

__all__ = [
    "Grid",
    "BoundaryMesh2D",
    "SurfaceMesh3D",
    "backward_difference",
    "central_difference",
    "extract_hull_mesh",
    "extract_surface_mesh_3d",
    "extract_zero_contour_2d",
    "extrude_2d_to_3d",
    "forward_difference",
    "gradient_central",
    "gradient_magnitude",
    "make_grid_2d",
    "make_grid_3d",
    "material_mask",
    "one_sided_derivatives",
    "rectangular_mask_2d",
    "sdf_box",
    "sdf_circle",
    "sdf_difference",
    "sdf_intersection",
    "sdf_plane",
    "sdf_sphere",
    "sdf_union",
    "slice_3d",
    "surface_normals",
    "void_mask",
]
