"""Optional VTK output boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from jax import Array


def write_vtk_structured_grid(path: str | Path, phi: Array, spacing: tuple[float, ...]) -> None:
    """Write ``phi`` as a legacy ASCII VTK structured-points file.

    The writer is intentionally small and dependency-free. It stores only the
    scalar level-set field with origin ``0`` because the current API receives
    spacing but not full grid bounds. Use ``save_simulation`` for complete
    metadata-preserving output.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(phi)
    if arr.ndim not in (2, 3):
        raise ValueError("VTK structured-grid output supports 2D or 3D arrays")
    if len(spacing) != arr.ndim:
        raise ValueError("spacing length must match phi dimension")
    dims = arr.shape if arr.ndim == 3 else (arr.shape[0], arr.shape[1], 1)
    vtk_spacing = spacing if len(spacing) == 3 else (spacing[0], spacing[1], 1.0)
    values = arr.reshape(-1, order="F")
    with path.open("w") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write("jaxps level-set field\n")
        handle.write("ASCII\n")
        handle.write("DATASET STRUCTURED_POINTS\n")
        handle.write(f"DIMENSIONS {dims[0]} {dims[1]} {dims[2]}\n")
        handle.write("ORIGIN 0 0 0\n")
        handle.write(f"SPACING {vtk_spacing[0]} {vtk_spacing[1]} {vtk_spacing[2]}\n")
        handle.write(f"POINT_DATA {values.size}\n")
        handle.write("SCALARS phi float 1\n")
        handle.write("LOOKUP_TABLE default\n")
        np.savetxt(handle, values, fmt="%.9g")
