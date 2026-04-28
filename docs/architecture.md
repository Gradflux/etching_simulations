# Architecture

## Package Structure

- `jaxps.geometry`: Cartesian grids, signed-distance functions, derivatives,
  normals, masks, contour/surface extraction, extrusion, and slicing.
- `jaxps.solvers`: level-set update kernels, CFL logic, time integration, dense
  narrow-band masking, and PDE reinitialization.
- `jaxps.models`: normal-velocity models for etching, deposition, masks, ion
  enhancement, angular yield, plasma, fluorocarbon-style, and multi-particle
  workflows.
- `jaxps.rays`: JAX-native direction sampling, flux accumulation, chunked flux,
  and approximate grid-marched visibility.
- `jaxps.materials`: immutable material objects, built-ins, registries, and
  material-to-value maps.
- `jaxps.io`: lightweight array output, simulation serialization, domain setup
  metadata, and optional future VTK/mesh output.
- `jaxps.utils`: typing aliases, device reporting, and validated parameter
  dataclasses.

## Core Data

`Grid` stores static domain metadata plus JAX coordinate arrays. Level-set state
is a JAX array `phi` with the same shape as the grid. Materials and model
parameters are Python dataclasses; numerical kernels receive arrays and scalar
parameters.

`MaterialRegistry` stores canonical material identity by name and numeric ID.
Built-in materials use deterministic low IDs and custom materials start at a
configurable high offset. `MaterialValueMap` converts material-ID arrays to
JAX scalar or vector fields for rates, yields, sticking coefficients, and
density-like properties.

## Level-Set Evolution

Models produce a normal velocity field `V_n` on the grid. The solver evaluates
a Godunov upwind approximation to `|grad phi|` and advances:

```text
phi_next = phi - dt * V_n * |grad phi|_upwind
```

The timestep is constrained by:

```text
dt <= cfl * min(dx) / max(abs(V_n))
```

## Ray And Flux Flow

Ray sampling creates normalized direction sets. Flux utilities project those
directions onto surface normals using nonnegative incidence factors and convert
the resulting scalar flux to deposition or etch velocities. The visibility path
marches rays through the SDF on the Cartesian grid and blocks directions that hit
material before leaving the domain. This is still an approximation and not an
OptiX-like industrial ray tracer.

`jaxps.rays.backends` provides high-level backend selection. `AUTO` resolves to
an implemented JAX path and records the requested backend, actual backend, and
visible JAX device platforms. `EXTERNAL_OPTIX` is reserved for a future optional
backend and is not implemented or vendored.

## Post-MIT Clean-Room Feature Layer

The post-MIT public behavior spec is implemented as independent modules:

- polynomial angular yields in `jaxps.models.yield_models`,
- material-target and mask-material velocity selection in `jaxps.models.etch`
  and `jaxps.models.deposition`,
- simplified plasma and fluorocarbon formulas in `jaxps.models.plasma` and
  `jaxps.models.fluorocarbon`,
- additive species workflows in `jaxps.models.multi_particle`,
- contour/surface extraction in `jaxps.geometry.mesh`,
- domain and material-name serialization in `jaxps.io.serialization`.

These modules are based on the public clean-room specification and documented
formulas, not GPL implementation source.

## JAX Compilation

Finite differences, Godunov gradients, level-set steps, direction sampling, and
flux accumulation are written with JAX primitives. Repeated evolution uses
`jax.lax.scan`. Python objects stay outside JIT-critical loops unless treated as
static configuration.
