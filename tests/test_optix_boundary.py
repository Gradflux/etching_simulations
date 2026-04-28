from pathlib import Path

import jaxps.rays.backends as backends
from jaxps.rays import detect_external_optix, select_flux_backend


def test_external_optix_detection_falls_back_without_required_runtime(monkeypatch, tmp_path):
    fake_root = tmp_path / "optix"
    (fake_root / "include").mkdir(parents=True)
    (fake_root / "include" / "optix.h").write_text("/* fake test-only header */\n")
    monkeypatch.setenv("OPTIX_ROOT", str(fake_root))
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(backends.shutil, "which", lambda _name: None)
    monkeypatch.setattr(backends.importlib.util, "find_spec", lambda _name: None)

    status = detect_external_optix()
    assert status["optix_header_exists"]
    assert not status["available"]

    selection = select_flux_backend("EXTERNAL_OPTIX")
    assert selection.requested_backend == "EXTERNAL_OPTIX"
    assert selection.actual_backend in {"JAX_GRID", "JAX_RAYS"}
    assert not selection.external_optix_available


def test_project_does_not_vendor_optix_headers_or_cuda_sources():
    root = Path(__file__).resolve().parents[1] / "jaxps"
    forbidden_suffixes = {".h", ".hpp", ".cuh", ".cu", ".ptx", ".so", ".dylib", ".dll"}
    offenders = [
        path
        for path in root.rglob("*")
        if "optix" in path.name.lower() and path.suffix.lower() in forbidden_suffixes
    ]

    assert offenders == []
