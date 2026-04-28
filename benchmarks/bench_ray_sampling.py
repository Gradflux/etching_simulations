from __future__ import annotations

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

from common import device_metadata, emit_result, parser
from jaxps.rays import cosine_weighted_directions


def main() -> None:
    argument_parser = parser("benchmark cosine-weighted ray sampling")
    argument_parser.add_argument("--num-rays", type=int, default=1_000_000)
    args = argument_parser.parse_args()
    key = jax.random.PRNGKey(0)
    fn = lambda k: cosine_weighted_directions(k, num_rays=args.num_rays, ndim=3)

    start = time.perf_counter()
    dirs = fn(key).block_until_ready()
    compile_time = time.perf_counter() - start

    start = time.perf_counter()
    for i in range(args.repeat):
        dirs = fn(jax.random.fold_in(key, i)).block_until_ready()
    run_time = (time.perf_counter() - start) / float(args.repeat)

    result = {
        "benchmark": "ray_sampling",
        **device_metadata(args.requested_backend),
        "shape": list(dirs.shape),
        "dtype": str(dirs.dtype),
        "compile_plus_first_run_s": compile_time,
        "steady_run_s": run_time,
        "throughput_per_s": args.num_rays / run_time,
        "mean_z": float(dirs[:, 2].mean()),
    }
    emit_result(result, args.format, args.output)


if __name__ == "__main__":
    main()
