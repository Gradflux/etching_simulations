from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jaxps.geometry import make_grid_2d, sdf_circle, surface_normals
from jaxps.io import save_npz
from jaxps.rays import accumulate_flux, deterministic_hemisphere_directions, flux_to_deposition_rate
from jaxps.solvers import evolve_level_set
from jaxps.utils import describe_devices

try:
    from ._common import example_parser, output_dir, save_optional_contour
except ImportError:
    from _common import example_parser, output_dir, save_optional_contour


def main(output_path=None) -> None:
    grid = make_grid_2d(((-1.0, 1.0), (-1.0, 1.0)), (256, 256))
    phi0 = sdf_circle(grid, (0.0, 0.0), 0.35)
    directions = deterministic_hemisphere_directions(32, ndim=2)

    def velocity_fn(phi, _t):
        normals = surface_normals(phi, grid.spacing)
        flux = accumulate_flux(normals, directions)
        return flux_to_deposition_rate(flux, 0.08)

    result = evolve_level_set(phi0, grid.spacing, velocity_fn=velocity_fn, t_final=0.25)
    out = output_dir(output_path)
    save_npz(out / "flux_deposition_2d.npz", phi0=phi0, phi=result.phi)
    save_optional_contour(out / "flux_deposition_2d.png", result.phi, "Flux deposition 2D")
    print(f"devices: {describe_devices()}")
    print(f"steps: {result.num_steps}, dt: {result.dt:.6g}")


if __name__ == "__main__":
    args = example_parser("run flux deposition 2D example").parse_args()
    main(args.output_dir)
