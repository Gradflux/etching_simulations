"""Shared example helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def output_dir() -> Path:
    """Return and create the example output directory."""

    path = Path(__file__).resolve().parent / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_optional_contour(path: Path, phi, title: str) -> None:
    """Save a contour plot when matplotlib is installed."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    arr = np.asarray(phi)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.contour(arr.T, levels=[0.0], colors="black")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
