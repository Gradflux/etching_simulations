from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jaxps.geometry import make_grid_2d, rectangular_mask_2d, sdf_box
from jaxps.io import save_npz
from jaxps.models import isotropic_etch_rate, masked_etch_rate
from jaxps.solvers import evolve_level_set
from jaxps.utils import describe_devices

try:
    from ._common import example_parser, output_dir, save_optional_contour
except ImportError:
    from _common import example_parser, output_dir, save_optional_contour


def main(output_path=None) -> None:
    grid = make_grid_2d(((-1.0, 1.0), (-0.2, 1.2)), (256, 180))
    phi0 = sdf_box(grid, center=(0.0, 0.45), half_extents=(0.8, 0.55))
    protected = rectangular_mask_2d(grid, (-1.0, -0.25), (-0.2, 1.2)) | rectangular_mask_2d(
        grid, (0.25, 1.0), (-0.2, 1.2)
    )

    def velocity_fn(phi, _t):
        return masked_etch_rate(isotropic_etch_rate(phi, 0.08), protected)

    result = evolve_level_set(phi0, grid.spacing, velocity_fn=velocity_fn, t_final=0.4)
    out = output_dir(output_path)
    save_npz(out / "masked_trench_etch_2d.npz", phi0=phi0, phi=result.phi, protected=protected)
    save_optional_contour(out / "masked_trench_etch_2d.png", result.phi, "Masked trench etch 2D")
    print(f"devices: {describe_devices()}")
    print(f"steps: {result.num_steps}, dt: {result.dt:.6g}")


if __name__ == "__main__":
    args = example_parser("run masked trench 2D etch example").parse_args()
    main(args.output_dir)
