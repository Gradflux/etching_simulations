"""Array file helpers."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import Array


def save_npz(path: str | Path, **arrays: Array) -> None:
    """Save arrays to an ``.npz`` file using NumPy serialization."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{name: np.asarray(value) for name, value in arrays.items()})


def load_npz(path: str | Path) -> dict[str, Array]:
    """Load arrays from an ``.npz`` file as JAX arrays."""

    with np.load(path) as data:
        return {name: jnp.asarray(data[name]) for name in data.files}
