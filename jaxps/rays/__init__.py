"""JAX-native ray direction and flux approximations."""

from jaxps.rays.backends import (
    IMPLEMENTED_FLUX_BACKENDS,
    KNOWN_FLUX_BACKENDS,
    FluxBackendSelection,
    normalize_flux_backend,
    select_flux_backend,
)
from jaxps.rays.flux import (
    accumulate_flux_chunked,
    accumulate_flux,
    flux_to_deposition_rate,
    flux_to_etch_rate,
    surface_exposure,
)
from jaxps.rays.intersection import approximate_surface_band
from jaxps.rays.sampling import (
    cosine_weighted_directions,
    deterministic_hemisphere_directions,
    normalize_directions,
    polynomial_cosine_directions,
)
from jaxps.rays.visibility import visibility_weights, visible_flux

__all__ = [
    "FluxBackendSelection",
    "IMPLEMENTED_FLUX_BACKENDS",
    "KNOWN_FLUX_BACKENDS",
    "accumulate_flux",
    "accumulate_flux_chunked",
    "approximate_surface_band",
    "cosine_weighted_directions",
    "deterministic_hemisphere_directions",
    "flux_to_deposition_rate",
    "flux_to_etch_rate",
    "normalize_directions",
    "normalize_flux_backend",
    "polynomial_cosine_directions",
    "select_flux_backend",
    "surface_exposure",
    "visibility_weights",
    "visible_flux",
]
