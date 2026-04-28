from __future__ import annotations

import json
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jaxps.utils import has_mps_hardware, safe_mps_status


def main() -> int:
    """Check whether the optional Apple Metal/MPS JAX backend is available."""

    parser = argparse.ArgumentParser(description="check Apple Metal/MPS hardware and JAX backend")
    parser.add_argument(
        "--hardware-only",
        action="store_true",
        help="exit successfully when Metal-capable hardware is present, even if JAX is CPU-only",
    )
    args = parser.parse_args()

    if args.hardware_only:
        hardware_present = has_mps_hardware()
        print(json.dumps({"hardware_mps_present": hardware_present}, indent=2, sort_keys=True))
        if hardware_present:
            return 0
        print("Apple Metal/MPS hardware was not detected.", file=sys.stderr)
        return 2

    info = safe_mps_status()
    print(json.dumps(info, indent=2, sort_keys=True))
    if info["jax_metal_backend_available"]:
        return 0
    if info["hardware_mps_present"]:
        print(
            "Apple Metal-capable hardware is present, but JAX does not expose a Metal backend "
            "in this environment. Install/enable jax-metal in a separate environment and rerun.",
            file=sys.stderr,
        )
        return 2
    print(
        "Apple Metal/MPS hardware was not detected, and JAX does not expose a Metal backend.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
