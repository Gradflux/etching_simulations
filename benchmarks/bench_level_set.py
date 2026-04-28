from __future__ import annotations

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

from common import device_metadata, emit_result, parser
from jaxps.geometry import make_grid_2d, sdf_circle
from jaxps.solvers import level_set_step


def main() -> None:
    args = parser("benchmark one level-set step").parse_args()
    grid = make_grid_2d(((-1.0, 1.0), (-1.0, 1.0)), (args.size, args.size))
    phi = sdf_circle(grid, (0.0, 0.0), 0.5)
    fn = jax.jit(lambda arr: level_set_step(arr, -0.1, grid.spacing, 0.001))

    start = time.perf_counter()
    out = fn(phi).block_until_ready()
    compile_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(args.repeat):
        out = fn(phi).block_until_ready()
    run_time = (time.perf_counter() - start) / float(args.repeat)

    result = {
        "benchmark": "level_set",
        **device_metadata(args.requested_backend),
        "shape": list(phi.shape),
        "dtype": str(phi.dtype),
        "compile_plus_first_run_s": compile_time,
        "steady_run_s": run_time,
        "throughput_per_s": phi.size / run_time,
        "checksum": float(out.mean()),
    }
    emit_result(result, args.format, args.output)


if __name__ == "__main__":
    main()
