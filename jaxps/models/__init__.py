"""Process velocity and yield models."""

from jaxps.models.deposition import (
    IsotropicDeposition,
    SimpleDeposition,
    isotropic_deposition_rate,
    sticking_deposition_rate,
)
from jaxps.models.etch import (
    DirectionalEtch,
    IonEnhancedEtch,
    IsotropicEtch,
    MaterialMaskedEtch,
    MaskedEtch,
    directional_etch_rate,
    ion_enhanced_etch_rate,
    isotropic_etch_rate,
    material_id_mask,
    masked_material_etch_rate,
    masked_etch_rate,
    target_material_velocity,
)
from jaxps.models.fluorocarbon import (
    SimplifiedFluorocarbonEtch,
    fluorocarbon_velocity,
    update_passivation_coverage,
)
from jaxps.models.multi_particle import (
    MultiParticleProcess,
    ParticleSpecies,
    combine_species_rates,
    compute_species_rate,
    linear_species_rate,
)
from jaxps.models.plasma import PlasmaFluxes, SimplePlasmaProcess, simple_plasma_velocity
from jaxps.models.rates import constant_rate_like, validate_nonnegative
from jaxps.models.yield_models import (
    PolynomialCosineYield,
    SputteringYield,
    incidence_cosine,
    polynomial_cosine_yield,
    sputtering_yield,
)

__all__ = [
    "DirectionalEtch",
    "IonEnhancedEtch",
    "IsotropicDeposition",
    "IsotropicEtch",
    "MaterialMaskedEtch",
    "MaskedEtch",
    "MultiParticleProcess",
    "ParticleSpecies",
    "PlasmaFluxes",
    "SimpleDeposition",
    "SimplePlasmaProcess",
    "SimplifiedFluorocarbonEtch",
    "PolynomialCosineYield",
    "SputteringYield",
    "combine_species_rates",
    "constant_rate_like",
    "compute_species_rate",
    "directional_etch_rate",
    "fluorocarbon_velocity",
    "ion_enhanced_etch_rate",
    "isotropic_deposition_rate",
    "isotropic_etch_rate",
    "incidence_cosine",
    "linear_species_rate",
    "material_id_mask",
    "masked_material_etch_rate",
    "masked_etch_rate",
    "polynomial_cosine_yield",
    "simple_plasma_velocity",
    "sputtering_yield",
    "sticking_deposition_rate",
    "target_material_velocity",
    "update_passivation_coverage",
    "validate_nonnegative",
]
