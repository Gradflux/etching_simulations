"""Optional VTK output boundary."""

from __future__ import annotations

from pathlib import Path

from jax import Array


def write_vtk_structured_grid(path: str | Path, phi: Array, spacing: tuple[float, ...]) -> None:
    """Write a structured grid when optional mesh output support is added."""

    del path, phi, spacing
    raise NotImplementedError("VTK output is not implemented yet; save arrays with save_npz")
