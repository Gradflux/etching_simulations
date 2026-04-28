from __future__ import annotations

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

from common import device_metadata, emit_result, parser
from jaxps.geometry import make_grid_2d, sdf_circle, surface_normals
from jaxps.rays import accumulate_flux_chunked, deterministic_hemisphere_directions


def main() -> None:
    argument_parser = parser("benchmark chunked flux accumulation")
    argument_parser.add_argument("--num-rays", type=int, default=128)
    argument_parser.add_argument("--chunk-size", type=int, default=64)
    args = argument_parser.parse_args()
    grid = make_grid_2d(((-1.0, 1.0), (-1.0, 1.0)), (args.size, args.size))
    phi = sdf_circle(grid, (0.0, 0.0), 0.5)
    normals = surface_normals(phi, grid.spacing)
    directions = deterministic_hemisphere_directions(args.num_rays, ndim=2)
    fn = jax.jit(lambda n: accumulate_flux_chunked(n, directions, chunk_size=args.chunk_size))

    start = time.perf_counter()
    flux = fn(normals).block_until_ready()
    compile_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(args.repeat):
        flux = fn(normals).block_until_ready()
    run_time = (time.perf_counter() - start) / float(args.repeat)

    result = {
        "benchmark": "flux_accumulation",
        **device_metadata(args.requested_backend),
        "shape": list(phi.shape),
        "dtype": str(phi.dtype),
        "num_rays": args.num_rays,
        "compile_plus_first_run_s": compile_time,
        "steady_run_s": run_time,
        "throughput_per_s": phi.size * args.num_rays / run_time,
        "mean_flux": float(flux.mean()),
    }
    emit_result(result, args.format, args.output)


if __name__ == "__main__":
    main()
