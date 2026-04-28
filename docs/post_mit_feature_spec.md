# Post-MIT Public Feature Specification

This file restates the post-MIT feature goals in clean-room implementation
language. It is not derived from GPL source code.

## Runtime Acceleration

Use JAX device discovery. CPU is the baseline. GPU/TPU/Metal backends are used
only when JAX exposes them. Benchmarks and device utilities report the requested
and actual backend.

Implemented clean-room modules:

- `jaxps.utils.devices`
- `jaxps.rays.backends`
- `benchmarks/*.py`

## Polynomial Cosine Yield

The incidence cosine is:

```text
mu = max(0, -dot(d, n))
```

where `d` is the particle travel direction and `n` is the outward material
normal. A polynomial yield is:

```text
Y(mu) = c0 + c1 mu + c2 mu^2 + ... + ck mu^k
```

Negative yields are clamped to zero by default.

Implemented clean-room module:

- `jaxps.models.yield_models`

## Mesh And Geometry Utilities

Provide independent zero-contour and surface-point extraction from level-set
arrays, plus extrusion and slicing helpers. These routines are simple geometry
utilities, not copies of upstream mesh internals.

Implemented clean-room modules:

- `jaxps.geometry.mesh`
- `jaxps.geometry.extrude`
- `jaxps.geometry.slice`

## Material System

Provide deterministic built-in materials, custom runtime registration, material
name serialization, material value maps, target-material process masks, and mask
material behavior.

Implemented clean-room modules:

- `jaxps.materials.material`
- `jaxps.materials.builtins`
- `jaxps.materials.registry`
- `jaxps.materials.value_map`
- `jaxps.io.serialization`

## Process Models

Process models may depend on material, flux, incidence angle, sticking
coefficient, and yield. Multi-species and fluorocarbon-style models are
simplified transparent formulas that combine flux terms into one normal velocity
field under the documented sign convention.

Implemented clean-room modules:

- `jaxps.models.etch`
- `jaxps.models.deposition`
- `jaxps.models.plasma`
- `jaxps.models.fluorocarbon`
- `jaxps.models.multi_particle`

## Serialization

Persist material names and registries in metadata. Numeric IDs are stored as
array values but must be interpretable through the serialized registry.

## Time Integration

Use robust integer step counts, explicit `t_final`, safe zero-velocity handling,
and `max_steps` failure rather than open-ended loops.
