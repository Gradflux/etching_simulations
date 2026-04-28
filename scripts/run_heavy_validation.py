from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "isotropic_etch_2d.py": "shrink",
    "isotropic_deposition_2d.py": "grow",
    "masked_trench_etch_2d.py": "shrink",
    "directional_etch_2d.py": "shrink",
    "directional_etch_3d.py": "shrink",
    "flux_deposition_2d.py": "grow",
    "flux_deposition_3d.py": "grow",
}


def run_command(command: list[str], cwd: Path = ROOT) -> dict[str, object]:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def require_success(result: dict[str, object]) -> None:
    if result["returncode"] != 0:
        command = " ".join(str(part) for part in result["command"])
        raise RuntimeError(
            f"command failed: {command}\nstdout:\n{result['stdout']}\nstderr:\n{result['stderr']}"
        )


def material_count(path: Path) -> tuple[int, int]:
    with np.load(path) as data:
        phi0 = np.asarray(data["phi0"])
        phi = np.asarray(data["phi"])
        if not np.all(np.isfinite(phi0)) or not np.all(np.isfinite(phi)):
            raise ValueError(f"{path.name} contains non-finite values")
        if phi0.shape != phi.shape:
            raise ValueError(f"{path.name} changed array shape")
        return int(np.count_nonzero(phi0 < 0.0)), int(np.count_nonzero(phi < 0.0))


def validate_example_output(path: Path, expectation: str) -> dict[str, object]:
    initial, final = material_count(path)
    if expectation == "shrink" and final >= initial:
        raise ValueError(f"{path.name} expected shrinkage, got material cells {initial} -> {final}")
    if expectation == "grow" and final <= initial:
        raise ValueError(f"{path.name} expected growth, got material cells {initial} -> {final}")
    with np.load(path) as data:
        extra_keys = sorted(name for name in data.files if name not in {"phi0", "phi"})
    return {"file": path.name, "initial_material_cells": initial, "final_material_cells": final, "extra_keys": extra_keys}


def run_examples(output_dir: Path) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for script_name, expectation in EXAMPLES.items():
        script = ROOT / "examples" / script_name
        require_success(run_command([sys.executable, str(script), "--output-dir", str(output_dir)]))
        npz_path = output_dir / script_name.replace(".py", ".npz")
        if not npz_path.exists():
            raise FileNotFoundError(f"example did not write expected output: {npz_path}")
        results.append(validate_example_output(npz_path, expectation))
    return results


def run_direct_diagnostics(output_dir: Path) -> dict[str, object]:
    from jaxps.geometry import make_grid_2d, sdf_circle, surface_normals
    from jaxps.io import SimulationState, load_simulation, save_simulation
    from jaxps.materials import default_material_registry
    from jaxps.rays import deterministic_hemisphere_directions, visible_flux
    from jaxps.solvers import reinitialization_error, reinitialize_signed_distance

    grid = make_grid_2d(((-1.0, 1.0), (-1.0, 1.0)), (129, 129))
    phi = sdf_circle(grid, (0.0, 0.0), 0.45)
    distorted = phi + 0.08 * phi * phi
    band = jnp.abs(phi) < 0.2
    before = float(jnp.mean(reinitialization_error(distorted, grid.spacing)[band]))
    reinitialized = reinitialize_signed_distance(distorted, grid.spacing, iterations=40, band_width=0.4)
    after = float(jnp.mean(reinitialization_error(reinitialized, grid.spacing)[band]))
    if not after < before:
        raise ValueError(f"reinitialization did not improve error: {before} -> {after}")

    normals = surface_normals(phi, grid.spacing)
    directions = deterministic_hemisphere_directions(16, ndim=2)
    flux = visible_flux(phi, grid, normals, directions, max_steps=16)
    if not bool(jnp.all(jnp.isfinite(flux)) and jnp.all(flux >= 0.0)):
        raise ValueError("visible flux contains invalid values")

    registry = default_material_registry().register("HeavyValidationMaterial")
    state = SimulationState(phi=phi, grid=grid, metadata={"case": "heavy_validation"})
    case_dir = output_dir / "serialization_case"
    save_simulation(case_dir, state, registry)
    loaded_state, loaded_registry = load_simulation(case_dir)
    if loaded_state.phi.shape != grid.shape:
        raise ValueError("serialized field shape changed")
    if loaded_registry.get("HeavyValidationMaterial").id != registry.get("HeavyValidationMaterial").id:
        raise ValueError("material registry ID changed during round trip")

    return {
        "reinitialization_error_before": before,
        "reinitialization_error_after": after,
        "visible_flux_mean": float(jnp.mean(flux)),
        "serialization_shape": list(loaded_state.phi.shape),
    }


def run_benchmarks(sizes: list[int], repeat: int) -> list[dict[str, object]]:
    result = run_command(
        [
            sys.executable,
            str(ROOT / "benchmarks" / "run_all.py"),
            "--sizes",
            *[str(size) for size in sizes],
            "--repeat",
            str(repeat),
        ]
    )
    require_success(result)
    data = json.loads(str(result["stdout"]))
    required = {
        "benchmark",
        "requested_backend",
        "actual_backend",
        "device_platform",
        "device_kind",
        "shape",
        "dtype",
        "compile_plus_first_run_s",
        "steady_run_s",
        "throughput_per_s",
    }
    for item in data:
        missing = required - set(item)
        if missing:
            raise ValueError(f"benchmark result missing keys: {sorted(missing)}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="run heavy jaxps validation")
    parser.add_argument("--all", action="store_true", help="run normal tests, slow tests, examples, and benchmarks")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/jaxps-validation"))
    parser.add_argument("--benchmark-sizes", type=int, nargs="+", default=[64])
    parser.add_argument("--benchmark-repeat", type=int, default=1)
    args = parser.parse_args()

    summary: dict[str, object] = {}
    if args.all:
        normal = run_command([sys.executable, "-m", "pytest", "-q"])
        require_success(normal)
        slow = run_command([sys.executable, "-m", "pytest", "-q", "-m", "slow"])
        require_success(slow)
        summary["normal_tests"] = "passed"
        summary["slow_tests"] = "passed"
    else:
        slow = run_command([sys.executable, "-m", "pytest", "-q", "-m", "slow"])
        require_success(slow)
        summary["slow_tests"] = "passed"

    summary["examples"] = run_examples(args.output_dir / "examples")
    summary["diagnostics"] = run_direct_diagnostics(args.output_dir)
    summary["benchmarks"] = run_benchmarks(args.benchmark_sizes, args.benchmark_repeat)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
