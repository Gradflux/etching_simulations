import json
import subprocess
import sys
from pathlib import Path


def test_benchmark_json_output_schema():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks" / "bench_ray_sampling.py"),
            "--num-rays",
            "16",
            "--repeat",
            "1",
            "--format",
            "json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    # Metal driver prints device info lines directly to stdout; extract the JSON line.
    json_line = next(
        (line for line in completed.stdout.splitlines() if line.startswith("{")),
        None,
    )
    assert json_line is not None, f"no JSON line found in stdout: {completed.stdout!r}"
    data = json.loads(json_line)

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
    assert required <= set(data)
    assert data["benchmark"] == "ray_sampling"
    assert data["requested_backend"] == "auto"
    assert data["shape"] == [16, 3]


def test_benchmark_csv_output_schema():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks" / "bench_ray_sampling.py"),
            "--num-rays",
            "8",
            "--repeat",
            "1",
            "--format",
            "csv",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    # Metal driver prints device info lines directly to stdout before the CSV header;
    # find the first comma-separated line that contains the expected column names.
    header_line = next(
        (line for line in completed.stdout.splitlines() if "," in line and "benchmark" in line),
        None,
    )
    assert header_line is not None, f"no CSV header found in stdout: {completed.stdout!r}"
    header = header_line.split(",")

    assert "benchmark" in header
    assert "device_platform" in header
    assert "throughput_per_s" in header
