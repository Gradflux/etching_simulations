"""Shared typing aliases."""

from typing import Any, Sequence, Tuple

try:
	from jax import Array
except ModuleNotFoundError:  # pragma: no cover - depends on optional runtime dependency
	Array = Any

Bounds = Tuple[Tuple[float, float], ...]
Shape = Tuple[int, ...]
Spacing = Tuple[float, ...]
FloatSequence = Sequence[float]

__all__ = ["Array", "Bounds", "FloatSequence", "Shape", "Spacing"]
