import jax.numpy as jnp
import numpy as np
import pytest

from jaxps.models import (
    DirectionalEtch,
    IonEnhancedEtch,
    IsotropicDeposition,
    IsotropicEtch,
    SimpleDeposition,
    SputteringYield,
    directional_etch_rate,
    ion_enhanced_etch_rate,
    isotropic_deposition_rate,
    isotropic_etch_rate,
    masked_etch_rate,
    sticking_deposition_rate,
    sputtering_yield,
)


def test_isotropic_rate_signs():
    phi = jnp.zeros((3, 3))

    assert np.all(np.asarray(isotropic_etch_rate(phi, 0.2)) < 0.0)
    assert np.all(np.asarray(isotropic_deposition_rate(phi, 0.2)) > 0.0)
    assert np.all(np.asarray(IsotropicEtch(0.1).velocity(phi)) == -0.1)
    assert np.all(np.asarray(IsotropicDeposition(0.1).velocity(phi)) == 0.1)


def test_mask_and_directional_behavior():
    normals = jnp.asarray([[[0.0, 1.0], [1.0, 0.0]]])
    velocity = directional_etch_rate(normals, direction=jnp.asarray([0.0, -1.0]), rate=0.5)

    assert np.isclose(np.asarray(velocity)[0, 0], -0.5)
    assert np.isclose(np.asarray(velocity)[0, 1], 0.0)

    base = jnp.asarray([-1.0, -1.0])
    masked = masked_etch_rate(base, jnp.asarray([True, False]))
    assert np.allclose(np.asarray(masked), [0.0, -1.0])
    assert DirectionalEtch(0.5, (0.0, -1.0)).velocity(normals).shape == (1, 2)


def test_flux_models_and_yield():
    flux = jnp.asarray([0.0, 2.0])

    assert np.allclose(np.asarray(ion_enhanced_etch_rate(0.1, flux, 0.5)), [-0.1, -1.1])
    assert np.allclose(np.asarray(IonEnhancedEtch(0.1, 0.5).velocity(flux)), [-0.1, -1.1])
    assert np.allclose(np.asarray(sticking_deposition_rate(flux, 0.25)), [0.0, 0.5])
    assert np.allclose(np.asarray(SimpleDeposition(0.25).velocity(flux)), [0.0, 0.5])
    assert np.allclose(np.asarray(sputtering_yield(jnp.asarray([0.2, 0.8]), 0.5)), [0.0, 0.3])
    assert np.allclose(np.asarray(SputteringYield(threshold=0.5)(jnp.asarray([0.2, 0.8]))), [0.0, 0.3])


def test_invalid_model_parameters_raise():
    with pytest.raises(ValueError):
        IsotropicEtch(-1.0)
    with pytest.raises(ValueError):
        IsotropicDeposition(-1.0)
    with pytest.raises(ValueError):
        SimpleDeposition(-0.1)
    with pytest.raises(ValueError):
        SputteringYield(exponent=-1.0)
