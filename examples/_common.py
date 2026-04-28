"""Shared example helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def output_dir(path: str | Path | None = None) -> Path:
    """Return and create the example output directory."""

    resolved = Path(path) if path is not None else Path(__file__).resolve().parent / "outputs"
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def example_parser(description: str) -> argparse.ArgumentParser:
    """Return a small common CLI parser for examples."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for .npz and optional plot outputs; defaults to examples/outputs",
    )
    return parser


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
