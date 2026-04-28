import platform

import pytest

from jaxps.utils import (
    assert_backend,
    available_backends,
    describe_devices,
    has_backend,
    has_mps_hardware,
    jax_metal_plugin_installed,
    mps_status,
    safe_mps_status,
)


def test_available_backends_and_device_description_are_consistent():
    backends = available_backends()

    assert isinstance(backends, tuple)
    assert len(backends) >= 1
    # Every reported backend must be recognized by has_backend().
    for backend in backends:
        assert has_backend(backend)
    # describe_devices() must mention at least one of the active backends.
    description = describe_devices().lower()
    assert any(backend in description for backend in backends)


def test_assert_backend_raises_for_missing_backend():
    with pytest.raises(RuntimeError):
        assert_backend("__definitely_missing_backend__")


def test_mps_hardware_status_is_reported_separately_from_jax_backend():
    status = mps_status()

    assert isinstance(status["hardware_mps_present"], bool)
    assert isinstance(status["jax_metal_plugin_installed"], bool)
    assert isinstance(status["jax_metal_backend_available"], bool)
    assert isinstance(has_mps_hardware(), bool)
    assert isinstance(jax_metal_plugin_installed(), bool)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        assert status["hardware_mps_present"]


def test_safe_mps_status_uses_subprocess_probe():
    status = safe_mps_status()

    assert isinstance(status["hardware_mps_present"], bool)
    assert isinstance(status["jax_metal_plugin_installed"], bool)
    assert isinstance(status["jax_metal_backend_available"], bool)
    assert "cpu_probe" in status
    assert "accelerator_probe" in status
