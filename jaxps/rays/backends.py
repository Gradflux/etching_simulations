"""Flux backend selection utilities.

The backend names are high-level execution choices for the clean-room JAX
implementation. They do not load CUDA drivers or external ray tracing SDKs.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import shutil
from typing import Literal

from jaxps.utils.devices import available_backends, has_backend

FluxBackendName = Literal["AUTO", "CPU_GRID", "JAX_GRID", "JAX_RAYS", "EXTERNAL_OPTIX"]

IMPLEMENTED_FLUX_BACKENDS: tuple[str, ...] = ("CPU_GRID", "JAX_GRID", "JAX_RAYS")
KNOWN_FLUX_BACKENDS: tuple[str, ...] = IMPLEMENTED_FLUX_BACKENDS + ("AUTO", "EXTERNAL_OPTIX")


@dataclass(frozen=True)
class FluxBackendSelection:
    """Resolved flux backend metadata."""

    requested_backend: str
    actual_backend: str
    device_backends: tuple[str, ...]
    reason: str
    external_optix_available: bool = False
    external_optix_status: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible backend metadata."""

        data = {
            "requested_backend": self.requested_backend,
            "actual_backend": self.actual_backend,
            "device_backends": list(self.device_backends),
            "reason": self.reason,
            "external_optix_available": self.external_optix_available,
        }
        if self.external_optix_status is not None:
            data["external_optix_status"] = self.external_optix_status
        return data


def detect_external_optix() -> dict[str, object]:
    """Return status for an optional user-installed OptiX environment.

    Detection is deliberately conservative and does not import NVIDIA headers,
    SDK files, or binaries from this project. ``available`` means the user has
    provided enough external runtime hints for a future adapter to try using
    OptiX; it does not mean this package vendors or implements OptiX tracing.
    """

    optix_root_value = os.environ.get("OPTIX_ROOT") or os.environ.get("OptiX_ROOT")
    cuda_root_value = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    optix_root = Path(optix_root_value).expanduser() if optix_root_value else None
    cuda_root = Path(cuda_root_value).expanduser() if cuda_root_value else None
    include_dir = optix_root / "include" if optix_root is not None else None
    optix_header = include_dir / "optix.h" if include_dir is not None else None
    nvidia_smi = shutil.which("nvidia-smi")
    python_extension = importlib.util.find_spec("jaxps_optix")
    status = {
        "optix_root": str(optix_root) if optix_root is not None else None,
        "optix_root_exists": bool(optix_root is not None and optix_root.exists()),
        "optix_include_exists": bool(include_dir is not None and include_dir.exists()),
        "optix_header_exists": bool(optix_header is not None and optix_header.exists()),
        "cuda_root": str(cuda_root) if cuda_root is not None else None,
        "cuda_root_exists": bool(cuda_root is not None and cuda_root.exists()),
        "nvidia_smi_available": nvidia_smi is not None,
        "python_extension_present": python_extension is not None,
    }
    status["available"] = bool(
        status["optix_header_exists"]
        and (status["nvidia_smi_available"] or status["python_extension_present"])
    )
    return status


def normalize_flux_backend(name: str) -> str:
    """Normalize and validate a flux backend name."""

    normalized = name.upper()
    if normalized not in KNOWN_FLUX_BACKENDS:
        raise ValueError(
            f"unknown flux backend '{name}'; expected one of {', '.join(KNOWN_FLUX_BACKENDS)}"
        )
    return normalized


def select_flux_backend(preferred: str = "AUTO") -> FluxBackendSelection:
    """Select an implemented flux backend.

    ``AUTO`` chooses JAX-based rays when a non-CPU JAX backend is visible and
    otherwise chooses the dense JAX grid implementation on CPU.
    """

    requested = normalize_flux_backend(preferred)
    device_backends = available_backends()
    optix_status = detect_external_optix()
    if requested == "EXTERNAL_OPTIX":
        if optix_status["available"]:
            return FluxBackendSelection(
                requested_backend=requested,
                actual_backend="EXTERNAL_OPTIX",
                device_backends=device_backends,
                reason="external OptiX environment detected",
                external_optix_available=True,
                external_optix_status=optix_status,
            )
        accelerated = has_backend("gpu") or has_backend("metal") or has_backend("tpu")
        fallback = "JAX_RAYS" if accelerated else "JAX_GRID"
        return FluxBackendSelection(
            requested_backend=requested,
            actual_backend=fallback,
            device_backends=device_backends,
            reason="external OptiX unavailable; falling back to implemented JAX backend",
            external_optix_available=False,
            external_optix_status=optix_status,
        )
    if requested != "AUTO":
        return FluxBackendSelection(
            requested_backend=requested,
            actual_backend=requested,
            device_backends=device_backends,
            reason="explicit implemented backend requested",
            external_optix_available=bool(optix_status["available"]),
            external_optix_status=optix_status,
        )
    accelerated = has_backend("gpu") or has_backend("metal") or has_backend("tpu")
    actual = "JAX_RAYS" if accelerated else "JAX_GRID"
    reason = "accelerator backend available" if accelerated else "CPU JAX backend fallback"
    return FluxBackendSelection(
        requested_backend=requested,
        actual_backend=actual,
        device_backends=device_backends,
        reason=reason,
        external_optix_available=bool(optix_status["available"]),
        external_optix_status=optix_status,
    )
