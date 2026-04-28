"""Simulation and domain serialization helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxps.geometry.grid import Grid, make_grid_2d, make_grid_3d
from jaxps.materials.registry import (
    MaterialRegistry,
    default_material_registry,
    deserialize_material_registry,
    serialize_material_registry,
)

BOUNDARY_CONDITIONS = {"reflective", "periodic", "open", "fixed", "zero_gradient"}


@dataclass(frozen=True)
class DomainSetup:
    """Serializable domain setup metadata."""

    dim: int
    bounds: tuple[tuple[float, float], ...]
    shape: tuple[int, ...]
    boundary_conditions: tuple[str, ...] = field(default_factory=tuple)
    initial_geometry: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        if len(self.bounds) != self.dim or len(self.shape) != self.dim:
            raise ValueError("bounds and shape must match dim")
        conditions = self.boundary_conditions or tuple("open" for _ in range(self.dim))
        if len(conditions) != self.dim:
            raise ValueError("boundary_conditions must match dim")
        for condition in conditions:
            if condition not in BOUNDARY_CONDITIONS:
                raise ValueError(f"unknown boundary condition: {condition}")
        object.__setattr__(self, "bounds", tuple((float(lo), float(hi)) for lo, hi in self.bounds))
        object.__setattr__(self, "shape", tuple(int(n) for n in self.shape))
        object.__setattr__(self, "boundary_conditions", tuple(conditions))

    @property
    def spacing(self) -> tuple[float, ...]:
        """Return grid spacing."""

        return tuple((hi - lo) / (n - 1) for (lo, hi), n in zip(self.bounds, self.shape, strict=True))

    def make_grid(self) -> Grid:
        """Create a Grid from this setup."""

        if self.dim == 2:
            return make_grid_2d(self.bounds, self.shape)  # type: ignore[arg-type]
        return make_grid_3d(self.bounds, self.shape)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""

        return {
            "dim": self.dim,
            "bounds": [list(bound) for bound in self.bounds],
            "shape": list(self.shape),
            "spacing": list(self.spacing),
            "boundary_conditions": list(self.boundary_conditions),
            "initial_geometry": self.initial_geometry,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DomainSetup":
        """Deserialize domain setup."""

        return cls(
            dim=int(data["dim"]),
            bounds=tuple(tuple(bound) for bound in data["bounds"]),
            shape=tuple(int(n) for n in data["shape"]),
            boundary_conditions=tuple(data.get("boundary_conditions", ())),
            initial_geometry=data.get("initial_geometry"),
        )


@dataclass(frozen=True)
class SimulationState:
    """Serializable simulation arrays and metadata."""

    phi: Array
    grid: Grid
    material_ids: Array | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def domain(self) -> DomainSetup:
        """Return domain metadata for this state."""

        return DomainSetup(dim=self.grid.ndim, bounds=self.grid.bounds, shape=self.grid.shape)


def save_simulation(
    path: str | Path,
    state: SimulationState,
    registry: MaterialRegistry | None = None,
) -> None:
    """Save simulation arrays and material registry to a directory."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    registry = registry or default_material_registry()
    arrays = {"phi": np.asarray(state.phi)}
    if state.material_ids is not None:
        arrays["material_ids"] = np.asarray(state.material_ids)
    np.savez(path / "arrays.npz", **arrays)
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "format": "jaxps-simulation",
                "version": 1,
                "domain": state.domain.to_dict(),
                "metadata": state.metadata,
                "material_names": registry.names(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (path / "registry.json").write_text(
        json.dumps(serialize_material_registry(registry), indent=2, sort_keys=True) + "\n"
    )


def load_simulation(path: str | Path) -> tuple[SimulationState, MaterialRegistry]:
    """Load a simulation saved by ``save_simulation``."""

    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    registry = deserialize_material_registry(json.loads((path / "registry.json").read_text()))
    domain = DomainSetup.from_dict(metadata["domain"])
    grid = domain.make_grid()
    with np.load(path / "arrays.npz") as arrays:
        phi = jnp.asarray(arrays["phi"])
        material_ids = jnp.asarray(arrays["material_ids"]) if "material_ids" in arrays.files else None
    if phi.shape != grid.shape:
        raise ValueError("loaded phi shape does not match serialized domain")
    if material_ids is not None and material_ids.shape != grid.shape:
        raise ValueError("loaded material field shape does not match serialized domain")
    return SimulationState(phi=phi, grid=grid, material_ids=material_ids, metadata=metadata.get("metadata", {})), registry
