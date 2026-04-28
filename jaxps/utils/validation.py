"""Serializable parameter dataclasses with lightweight validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _validate_positive(value: float, name: str) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _validate_nonnegative(value: float, name: str) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True)
class AdvectionParameters:
    """Parameters for explicit level-set advection."""

    cfl: float = 0.4
    max_steps: int | None = None
    band_width: float | None = None

    def __post_init__(self) -> None:
        _validate_positive(float(self.cfl), "cfl")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")
        if self.band_width is not None:
            _validate_positive(float(self.band_width), "band_width")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible data."""

        return {"cfl": self.cfl, "max_steps": self.max_steps, "band_width": self.band_width}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdvectionParameters":
        """Deserialize parameters."""

        return cls(
            cfl=float(data.get("cfl", 0.4)),
            max_steps=data.get("max_steps"),
            band_width=data.get("band_width"),
        )


@dataclass(frozen=True)
class RayTracingParameters:
    """Parameters for approximate JAX ray/flux calculations."""

    num_rays: int = 64
    max_steps: int = 64
    step_size: float | None = None
    backend: str = "AUTO"

    def __post_init__(self) -> None:
        if self.num_rays < 0:
            raise ValueError("num_rays must be nonnegative")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.step_size is not None:
            _validate_positive(float(self.step_size), "step_size")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible data."""

        return {
            "num_rays": self.num_rays,
            "max_steps": self.max_steps,
            "step_size": self.step_size,
            "backend": self.backend,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RayTracingParameters":
        """Deserialize parameters."""

        return cls(
            num_rays=int(data.get("num_rays", 64)),
            max_steps=int(data.get("max_steps", 64)),
            step_size=data.get("step_size"),
            backend=str(data.get("backend", "AUTO")),
        )


@dataclass(frozen=True)
class CoverageParameters:
    """Parameters for simple passivation coverage dynamics."""

    initial: float = 0.0
    deposition_rate: float = 0.0
    sputter_rate: float = 0.0

    def __post_init__(self) -> None:
        _validate_nonnegative(float(self.initial), "initial")
        _validate_nonnegative(float(self.deposition_rate), "deposition_rate")
        _validate_nonnegative(float(self.sputter_rate), "sputter_rate")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible data."""

        return {
            "initial": self.initial,
            "deposition_rate": self.deposition_rate,
            "sputter_rate": self.sputter_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CoverageParameters":
        """Deserialize parameters."""

        return cls(
            initial=float(data.get("initial", 0.0)),
            deposition_rate=float(data.get("deposition_rate", 0.0)),
            sputter_rate=float(data.get("sputter_rate", 0.0)),
        )


@dataclass(frozen=True)
class ProcessParameters:
    """Generic named process parameter bundle."""

    name: str = "process"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible data."""

        return {"name": self.name, "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessParameters":
        """Deserialize parameters."""

        return cls(name=str(data.get("name", "process")), metadata=dict(data.get("metadata", {})))
