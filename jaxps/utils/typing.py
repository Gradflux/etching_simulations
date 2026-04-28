"""Shared typing aliases."""

from typing import Sequence, Tuple

from jax import Array

Bounds = Tuple[Tuple[float, float], ...]
Shape = Tuple[int, ...]
Spacing = Tuple[float, ...]
FloatSequence = Sequence[float]

__all__ = ["Array", "Bounds", "FloatSequence", "Shape", "Spacing"]
