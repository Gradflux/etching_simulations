"""Independent level-set contour and surface extraction helpers.

These functions run on the host (CPU) and are not JIT-compatible. They convert
JAX arrays to NumPy once at the start to avoid per-element device-to-host
synchronisations, then use vectorised NumPy operations for the geometric scan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


def _edge_crossing(
    v0: np.ndarray,
    v1: np.ndarray,
    px0: np.ndarray,
    py0: np.ndarray,
    px1: np.ndarray,
    py1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised edge crossing: returns (has_crossing, x, y) for sign-changing edges."""
    has = ((v0 <= 0.0) & (v1 >= 0.0)) | ((v0 >= 0.0) & (v1 <= 0.0))
    denom = np.where(np.abs(v1 - v0) > 1e-12, v1 - v0, 1.0)
    t = np.clip(-v0 / denom, 0.0, 1.0)
    return has, px0 + t * (px1 - px0), py0 + t * (py1 - py0)


def extract_zero_contour_2d(phi: Array, grid: Grid) -> BoundaryMesh2D:
    """Extract approximate 2D zero-contour points and segments.

    Marching-squares-style scan; connects the first two edge crossings per cell.
    Fully vectorised with NumPy — no Python loop over grid cells.
    Not JIT-compatible; runs on the host.
    """

    if grid.ndim != 2 or phi.ndim != 2:
        raise ValueError("extract_zero_contour_2d requires a 2D field and grid")

    phi_np = np.asarray(phi)
    x_np = np.asarray(grid.coords[0])
    y_np = np.asarray(grid.coords[1])

    Ni, Nj = phi_np.shape
    if Ni < 2 or Nj < 2:
        return BoundaryMesh2D(points=jnp.empty((0, 2)), segments=jnp.empty((0, 2), dtype=jnp.int32))

    # Cell corners (shape: Ni-1, Nj-1)
    c00 = phi_np[:-1, :-1]
    c10 = phi_np[1:, :-1]
    c11 = phi_np[1:, 1:]
    c01 = phi_np[:-1, 1:]

    x00, y00 = x_np[:-1, :-1], y_np[:-1, :-1]
    x10, y10 = x_np[1:, :-1], y_np[1:, :-1]
    x11, y11 = x_np[1:, 1:], y_np[1:, 1:]
    x01, y01 = x_np[:-1, 1:], y_np[:-1, 1:]

    # 4 edges per cell: bottom, right, top (reversed), left (reversed)
    has0, cx0, cy0 = _edge_crossing(c00, c10, x00, y00, x10, y10)
    has1, cx1, cy1 = _edge_crossing(c10, c11, x10, y10, x11, y11)
    has2, cx2, cy2 = _edge_crossing(c11, c01, x11, y11, x01, y01)
    has3, cx3, cy3 = _edge_crossing(c01, c00, x01, y01, x00, y00)

    # Stack edge data along last axis: shape (Ni-1, Nj-1, 4)
    has_all = np.stack([has0, has1, has2, has3], axis=-1)
    cx_all = np.stack([cx0, cx1, cx2, cx3], axis=-1)
    cy_all = np.stack([cy0, cy1, cy2, cy3], axis=-1)

    has_seg = has_all.sum(axis=-1) >= 2  # cells with ≥2 crossings
    if not has_seg.any():
        return BoundaryMesh2D(points=jnp.empty((0, 2)), segments=jnp.empty((0, 2), dtype=jnp.int32))

    cumsum = np.cumsum(has_all, axis=-1)
    first_edge = np.argmax((cumsum == 1) & has_all, axis=-1)   # index of 1st crossing per cell
    second_edge = np.argmax((cumsum == 2) & has_all, axis=-1)  # index of 2nd crossing per cell

    ii, jj = np.meshgrid(np.arange(Ni - 1), np.arange(Nj - 1), indexing="ij")
    pt1 = np.stack([cx_all[ii, jj, first_edge], cy_all[ii, jj, first_edge]], axis=-1)[has_seg]
    pt2 = np.stack([cx_all[ii, jj, second_edge], cy_all[ii, jj, second_edge]], axis=-1)[has_seg]

    n_segs = pt1.shape[0]
    points = np.empty((n_segs * 2, 2))
    points[0::2] = pt1
    points[1::2] = pt2
    seg_idx = np.stack([np.arange(0, n_segs * 2, 2), np.arange(1, n_segs * 2, 2)], axis=-1)

    return BoundaryMesh2D(
        points=jnp.asarray(points),
        segments=jnp.asarray(seg_idx, dtype=jnp.int32),
    )


def extract_surface_mesh_3d(phi: Array, grid: Grid) -> SurfaceMesh3D:
    """Extract approximate 3D surface points from sign-changing grid edges.

    Fully vectorised with NumPy — interpolates all crossing points along each
    axis in one pass without a Python loop over individual grid cells.
    Faces are left empty; vertices form a point cloud for validation and export.
    Not JIT-compatible; runs on the host.
    """

    if grid.ndim != 3 or phi.ndim != 3:
        raise ValueError("extract_surface_mesh_3d requires a 3D field and grid")

    phi_np = np.asarray(phi)
    coords_np = [np.asarray(c) for c in grid.coords]

    all_points: list[np.ndarray] = []
    for axis in range(3):
        sl_a: list = [slice(None)] * 3
        sl_b: list = [slice(None)] * 3
        sl_a[axis] = slice(0, -1)
        sl_b[axis] = slice(1, None)
        va = phi_np[tuple(sl_a)]
        vb = phi_np[tuple(sl_b)]
        changed = ((va <= 0.0) & (vb >= 0.0)) | ((va >= 0.0) & (vb <= 0.0))
        if not changed.any():
            continue
        denom = np.where(np.abs(vb - va) > 1e-12, vb - va, 1.0)
        t = np.clip(-va / denom, 0.0, 1.0)[..., np.newaxis]
        ca = np.stack([c[tuple(sl_a)] for c in coords_np], axis=-1)
        cb = np.stack([c[tuple(sl_b)] for c in coords_np], axis=-1)
        all_points.append((ca + t * (cb - ca))[changed])

    if not all_points:
        return SurfaceMesh3D(vertices=jnp.empty((0, 3)), faces=jnp.empty((0, 3), dtype=jnp.int32))

    return SurfaceMesh3D(
        vertices=jnp.asarray(np.concatenate(all_points, axis=0)),
        faces=jnp.empty((0, 3), dtype=jnp.int32),
    )


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
