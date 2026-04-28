"""Shared benchmark helpers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import jax

from jaxps.rays import detect_external_optix


def parser(description: str) -> argparse.ArgumentParser:
    """Return the common benchmark argument parser."""

    argument_parser = argparse.ArgumentParser(description=description)
    argument_parser.add_argument("--size", type=int, default=512)
    argument_parser.add_argument("--repeat", type=int, default=5)
    argument_parser.add_argument("--format", choices=("text", "json", "csv"), default="text")
    argument_parser.add_argument("--output", type=Path, default=None)
    argument_parser.add_argument("--requested-backend", default="auto")
    return argument_parser


def device_metadata(requested_backend: str = "auto") -> dict[str, object]:
    """Return compact metadata for the active default JAX device."""

    device = jax.devices()[0]
    optix_status = detect_external_optix()
    return {
        "requested_backend": requested_backend,
        "actual_backend": str(jax.default_backend()),
        "device_platform": str(device.platform),
        "device_kind": str(device.device_kind),
        "external_optix_available": bool(optix_status["available"]),
    }


def emit_result(result: dict[str, Any], output_format: str, output: Path | None = None) -> None:
    """Emit one benchmark result as text, JSON, or CSV."""

    if output_format == "json":
        text = json.dumps(result, sort_keys=True)
    elif output_format == "csv":
        keys = list(result)
        row = {key: json.dumps(value) if isinstance(value, (list, tuple)) else value for key, value in result.items()}
        if output is not None:
            with output.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                writer.writerow(row)
            return
        text = ",".join(keys) + "\n" + ",".join(str(row[key]) for key in keys)
    else:
        text = "\n".join(f"{key}: {value}" for key, value in result.items())

    if output is None:
        print(text)
    else:
        output.write_text(text + "\n")
