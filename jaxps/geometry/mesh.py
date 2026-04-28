"""Independent level-set contour and surface extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from jaxps.geometry.grid import Grid


@dataclass(frozen=True)
class BoundaryMesh2D:
    """Polyline-like zero-contour representation."""

    points: Array
    segments: Array


@dataclass(frozen=True)
class SurfaceMesh3D:
    """Simple point/facet surface representation."""

    vertices: Array
    faces: Array


def _edge_point(p0: Array, p1: Array, v0: Array, v1: Array) -> Array:
    denom = jnp.where(jnp.abs(v1 - v0) > 1e-12, v1 - v0, 1.0)
    t = jnp.clip(-v0 / denom, 0.0, 1.0)
    return p0 + t * (p1 - p0)


def extract_zero_contour_2d(phi: Array, grid: Grid) -> BoundaryMesh2D:
    """Extract approximate 2D zero-contour points and segments.

    This is an independent marching-squares-style cell scan. Ambiguous cases are
    represented by connecting the first two edge crossings in a cell.
    """

    if grid.ndim != 2 or phi.ndim != 2:
        raise ValueError("extract_zero_contour_2d requires a 2D field and grid")
    points: list[Array] = []
    segments: list[tuple[int, int]] = []
    x, y = grid.coords
    for i in range(phi.shape[0] - 1):
        for j in range(phi.shape[1] - 1):
            values = [phi[i, j], phi[i + 1, j], phi[i + 1, j + 1], phi[i, j + 1]]
            coords = [
                jnp.asarray([x[i, j], y[i, j]]),
                jnp.asarray([x[i + 1, j], y[i + 1, j]]),
                jnp.asarray([x[i + 1, j + 1], y[i + 1, j + 1]]),
                jnp.asarray([x[i, j + 1], y[i, j + 1]]),
            ]
            if all(float(v) > 0.0 for v in values) or all(float(v) < 0.0 for v in values):
                continue
            crossings = []
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
                va = float(values[a])
                vb = float(values[b])
                if va == 0.0:
                    crossings.append(coords[a])
                elif va * vb < 0.0 or vb == 0.0:
                    crossings.append(_edge_point(coords[a], coords[b], values[a], values[b]))
            if len(crossings) >= 2:
                start = len(points)
                points.extend(crossings[:2])
                segments.append((start, start + 1))
    point_array = jnp.asarray(points) if points else jnp.empty((0, 2))
    segment_array = jnp.asarray(segments, dtype=jnp.int32) if segments else jnp.empty((0, 2), dtype=jnp.int32)
    return BoundaryMesh2D(points=point_array, segments=segment_array)


def extract_surface_mesh_3d(phi: Array, grid: Grid) -> SurfaceMesh3D:
    """Extract approximate 3D surface points from sign-changing grid edges.

    Faces are left empty in this first clean-room implementation. The vertices
    form a surface point cloud suitable for validation and lightweight export.
    """

    if grid.ndim != 3 or phi.ndim != 3:
        raise ValueError("extract_surface_mesh_3d requires a 3D field and grid")
    points: list[Array] = []
    coords = grid.coords
    for axis in range(3):
        slices_a = [slice(None)] * 3
        slices_b = [slice(None)] * 3
        slices_a[axis] = slice(0, -1)
        slices_b[axis] = slice(1, None)
        va = phi[tuple(slices_a)]
        vb = phi[tuple(slices_b)]
        changed = jnp.asarray((va <= 0.0) & (vb >= 0.0) | (va >= 0.0) & (vb <= 0.0))
        indices = jnp.argwhere(changed)
        for raw_index in indices:
            idx_a = [int(k) for k in raw_index]
            idx_b = list(idx_a)
            idx_b[axis] += 1
            p0 = jnp.asarray([coord[tuple(idx_a)] for coord in coords])
            p1 = jnp.asarray([coord[tuple(idx_b)] for coord in coords])
            points.append(_edge_point(p0, p1, phi[tuple(idx_a)], phi[tuple(idx_b)]))
    vertices = jnp.asarray(points) if points else jnp.empty((0, 3))
    return SurfaceMesh3D(vertices=vertices, faces=jnp.empty((0, 3), dtype=jnp.int32))


def extract_hull_mesh(phi: Array, grid: Grid, include_interfaces: bool = False) -> BoundaryMesh2D | SurfaceMesh3D:
    """Extract the current outer zero surface.

    ``include_interfaces`` is accepted for API compatibility. Multi-material
    interface extraction is not implemented yet.
    """

    if include_interfaces:
        raise NotImplementedError("material interface hull extraction is not implemented yet")
    if grid.ndim == 2:
        return extract_zero_contour_2d(phi, grid)
    if grid.ndim == 3:
        return extract_surface_mesh_3d(phi, grid)
    raise ValueError("only 2D and 3D grids are supported")
