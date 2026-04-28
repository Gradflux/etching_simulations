from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jaxps.geometry import make_grid_2d, sdf_circle, surface_normals
from jaxps.io import save_npz
from jaxps.models import directional_etch_rate
from jaxps.solvers import evolve_level_set
from jaxps.utils import describe_devices

try:
    from ._common import example_parser, output_dir, save_optional_contour
except ImportError:
    from _common import example_parser, output_dir, save_optional_contour


def main(output_path=None) -> None:
    grid = make_grid_2d(((-1.0, 1.0), (-1.0, 1.0)), (256, 256))
    phi0 = sdf_circle(grid, (0.0, 0.0), 0.5)

    def velocity_fn(phi, _t):
        normals = surface_normals(phi, grid.spacing)
        return directional_etch_rate(normals, direction=(0.0, -1.0), rate=0.12)

    result = evolve_level_set(phi0, grid.spacing, velocity_fn=velocity_fn, t_final=0.25)
    out = output_dir(output_path)
    save_npz(out / "directional_etch_2d.npz", phi0=phi0, phi=result.phi)
    save_optional_contour(out / "directional_etch_2d.png", result.phi, "Directional etch 2D")
    print(f"devices: {describe_devices()}")
    print(f"steps: {result.num_steps}, dt: {result.dt:.6g}")


if __name__ == "__main__":
    args = example_parser("run directional 2D etch example").parse_args()
    main(args.output_dir)
