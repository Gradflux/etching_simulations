from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jaxps.utils import available_backends, has_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="run all jaxps benchmarks")
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--require-backend", choices=("cpu", "metal", "mps", "cuda", "gpu"), default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    requested_backend = args.require_backend or args.device
    if args.require_backend is not None and not has_backend(args.require_backend):
        print(
            f"required backend '{args.require_backend}' is not available; available backends: "
            f"{', '.join(available_backends()) or 'none'}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if requested_backend != "auto":
        print(
            "Device selection is controlled by the installed JAX backend and environment; "
            f"requested backend hint: {requested_backend}",
            file=sys.stderr,
        )

    root = Path(__file__).resolve().parent
    commands = []
    for size in args.sizes:
        commands.extend(
            [
                [sys.executable, str(root / "bench_derivatives.py"), "--size", str(size)],
                [sys.executable, str(root / "bench_level_set.py"), "--size", str(size)],
                [sys.executable, str(root / "bench_flux_accumulation.py"), "--size", str(size)],
            ]
        )
    commands.append([sys.executable, str(root / "bench_ray_sampling.py"), "--num-rays", "100000"])

    results = []
    for command in commands:
        completed = subprocess.run(
            command
            + [
                "--repeat",
                str(args.repeat),
                "--format",
                "json",
                "--requested-backend",
                requested_backend,
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        results.append(json.loads(completed.stdout))

    text = json.dumps(results, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
