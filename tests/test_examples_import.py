import importlib
import subprocess
import sys
from pathlib import Path


def test_examples_import_without_running_main():
    for module_name in [
        "examples.isotropic_etch_2d",
        "examples.isotropic_deposition_2d",
        "examples.masked_trench_etch_2d",
        "examples.directional_etch_2d",
        "examples.directional_etch_3d",
        "examples.flux_deposition_2d",
        "examples.flux_deposition_3d",
    ]:
        module = importlib.import_module(module_name)
        assert hasattr(module, "main")


def test_example_output_dir_argument_writes_to_requested_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "examples" / "isotropic_etch_2d.py"),
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "steps:" in completed.stdout
    assert (tmp_path / "isotropic_etch_2d.npz").exists()
