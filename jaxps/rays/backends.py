"""Flux backend selection utilities.

The backend names are high-level execution choices for the clean-room JAX
implementation. They do not load CUDA drivers or external ray tracing SDKs.
"""

from __future__ import annotations

from dataclasses import dataclass
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

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible backend metadata."""

        return {
            "requested_backend": self.requested_backend,
            "actual_backend": self.actual_backend,
            "device_backends": list(self.device_backends),
            "reason": self.reason,
        }


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
    if requested == "EXTERNAL_OPTIX":
        raise NotImplementedError("EXTERNAL_OPTIX is reserved for a future optional backend")
    if requested != "AUTO":
        return FluxBackendSelection(
            requested_backend=requested,
            actual_backend=requested,
            device_backends=device_backends,
            reason="explicit implemented backend requested",
        )
    accelerated = has_backend("gpu") or has_backend("metal") or has_backend("tpu")
    actual = "JAX_RAYS" if accelerated else "JAX_GRID"
    reason = "accelerator backend available" if accelerated else "CPU JAX backend fallback"
    return FluxBackendSelection(
        requested_backend=requested,
        actual_backend=actual,
        device_backends=device_backends,
        reason=reason,
    )
