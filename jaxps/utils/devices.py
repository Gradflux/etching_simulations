"""JAX device reporting."""

from __future__ import annotations

import importlib.util
import importlib.metadata
import json
import os
import platform
import subprocess
import sys


def _import_jax():
    """Import jax lazily so hardware-only checks can run without jax installed."""

    try:
        import jax  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "jax is not installed in this environment. Install project dependencies first, "
            "for example: python -m pip install -e '.[test]'"
        ) from exc
    return jax


def _normalize_backend(name: str) -> str:
    normalized = name.lower()
    if normalized == "mps":
        return "metal"
    if normalized == "cuda":
        return "gpu"
    return normalized


def available_backends() -> tuple[str, ...]:
    """Return normalized backend/platform names exposed by JAX devices."""

    jax = _import_jax()
    backends = {_normalize_backend(str(device.platform)) for device in jax.devices()}
    return tuple(sorted(backends))


def has_backend(name: str) -> bool:
    """Return whether a backend is available.

    ``mps`` is treated as an alias for Apple's JAX ``metal`` backend. ``cuda``
    is treated as an alias for JAX's ``gpu`` platform.
    """

    return _normalize_backend(name) in available_backends()


def jax_metal_plugin_installed() -> bool:
    """Return whether a known JAX Metal plugin module is import-discoverable."""

    for distribution in ("jax-metal", "jax_metal"):
        try:
            importlib.metadata.version(distribution)
            return True
        except importlib.metadata.PackageNotFoundError:
            pass
    if importlib.util.find_spec("jax_plugins") is not None:
        if importlib.util.find_spec("jax_plugins.xla_metal") is not None:
            return True
    return importlib.util.find_spec("jax_metal") is not None


def has_mps_hardware() -> bool:
    """Return whether this machine appears to have Apple Metal-capable hardware.

    This is hardware detection only. JAX still needs the optional Metal plugin
    to expose a usable ``metal`` backend.
    """

    if platform.system() != "Darwin":
        return False

    try:
        completed = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            check=False,
            text=True,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return platform.machine() == "arm64"

    output = f"{completed.stdout}\n{completed.stderr}".lower()
    if "metal: supported" in output:
        return True
    return platform.machine() == "arm64"


def mps_status() -> dict[str, object]:
    """Return hardware and JAX-backend status for Apple Metal/MPS usage."""

    jax = _import_jax()
    return {
        "hardware_mps_present": has_mps_hardware(),
        "jax_metal_plugin_installed": jax_metal_plugin_installed(),
        "jax_metal_backend_available": has_backend("metal"),
        "default_backend": jax.default_backend(),
        "available_backends": available_backends(),
        "devices": describe_devices(),
    }


def probe_jax_backend_subprocess(force_cpu: bool = False, timeout: int = 20) -> dict[str, object]:
    """Probe JAX backend state in a subprocess.

    A broken accelerator plugin can abort the interpreter during backend
    initialization. Probing in a child process keeps diagnostic scripts alive.
    """

    code = """
import json
import jax
print(json.dumps({
    "jax_version": jax.__version__,
    "default_backend": jax.default_backend(),
    "devices": [f"{device.platform}:{device.device_kind}" for device in jax.devices()],
}))
"""
    env = os.environ.copy()
    if force_cpu:
        env["JAX_PLATFORMS"] = "cpu"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    result: dict[str, object] = {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "ok": completed.returncode == 0,
    }
    if completed.returncode == 0:
        payload = None
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            # Some backends (notably jax-metal) print informational lines to
            # stdout before/after the JSON payload. Recover by parsing a JSON
            # object from one of the output lines.
            for line in completed.stdout.splitlines():
                candidate = line.strip()
                if not (candidate.startswith("{") and candidate.endswith("}")):
                    continue
                try:
                    payload = json.loads(candidate)
                    break
                except json.JSONDecodeError:
                    continue
        if payload is None:
            result["parse_error"] = "could not parse JAX probe JSON"
        else:
            result.update(payload)
    return result


def safe_mps_status() -> dict[str, object]:
    """Return MPS status without risking an in-process JAX Metal abort."""

    hardware_present = has_mps_hardware()
    plugin_installed = jax_metal_plugin_installed()
    cpu_probe = probe_jax_backend_subprocess(force_cpu=True)
    accelerator_probe = probe_jax_backend_subprocess(force_cpu=False)
    accelerator_devices = tuple(str(device) for device in accelerator_probe.get("devices", ()))
    metal_available = accelerator_probe.get("ok") and any(
        device.split(":", 1)[0].lower() == "metal" for device in accelerator_devices
    )
    return {
        "hardware_mps_present": hardware_present,
        "jax_metal_plugin_installed": plugin_installed,
        "jax_metal_backend_available": bool(metal_available),
        "cpu_probe": cpu_probe,
        "accelerator_probe": accelerator_probe,
    }


def assert_backend(name: str) -> None:
    """Raise ``RuntimeError`` if the requested backend is not available."""

    if not has_backend(name):
        raise RuntimeError(
            f"requested backend '{name}' is not available; available backends: "
            f"{', '.join(available_backends()) or 'none'}"
        )


def describe_devices() -> str:
    """Return a compact description of available JAX devices."""

    jax = _import_jax()
    devices = jax.devices()
    if not devices:
        return "No JAX devices available."
    return ", ".join(f"{device.platform}:{device.device_kind}" for device in devices)


def device_report() -> dict[str, object]:
    """Return structured JAX device and backend metadata."""

    jax = _import_jax()
    devices = jax.devices()
    backends = available_backends()
    return {
        "jax_version": jax.__version__,
        "jaxlib_version": getattr(jax.lib, "__version__", "unknown"),
        "default_backend": jax.default_backend(),
        "available_backends": backends,
        "has_gpu": has_backend("gpu"),
        "has_metal": has_backend("metal"),
        "has_tpu": has_backend("tpu"),
        "devices": [
            {
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
                "id": int(getattr(device, "id", 0)),
            }
            for device in devices
        ],
        "warning": None if any(backend != "cpu" for backend in backends) else "Only CPU backend is available.",
    }
