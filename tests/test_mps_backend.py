import pytest

from jaxps.utils import assert_backend, safe_mps_status


pytestmark = pytest.mark.mps


def test_mps_backend_available_when_requested():
    if not safe_mps_status()["jax_metal_backend_available"]:
        pytest.skip("Apple Metal/MPS backend is not available in this environment")
    assert_backend("metal")
