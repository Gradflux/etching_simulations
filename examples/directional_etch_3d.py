from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jaxps.geometry import make_grid_3d, sdf_sphere, surface_normals
from jaxps.io import save_npz
from jaxps.models import directional_etch_rate
from jaxps.solvers import evolve_level_set
from jaxps.utils import describe_devices


def main() -> None:
    grid = make_grid_3d(((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)), (96, 96, 96))
    phi0 = sdf_sphere(grid, (0.0, 0.0, 0.0), 0.55)

    def velocity_fn(phi, _t):
        normals = surface_normals(phi, grid.spacing)
        return directional_etch_rate(normals, direction=(0.0, 0.0, -1.0), rate=0.1)

    result = evolve_level_set(phi0, grid.spacing, velocity_fn=velocity_fn, t_final=0.2)
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    save_npz(out / "directional_etch_3d.npz", phi0=phi0, phi=result.phi)
    print(f"devices: {describe_devices()}")
    print(f"steps: {result.num_steps}, dt: {result.dt:.6g}")


if __name__ == "__main__":
    main()
