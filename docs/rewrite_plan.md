# Rewrite Plan

## Scope

`jaxps` will provide a Python-native, JAX-based process simulation library with
dense Cartesian grids, signed-distance geometry, finite-difference derivatives,
explicit level-set evolution, process-rate models, and approximate ray/flux
models.

The implementation does not include sparse HRLE-style data structures,
industrial ray tracing, OptiX integration, or exact ViennaPS parity. It now
includes a basic PDE reinitialization path, dense masked narrow-band updates,
material-aware process models, and simple post-MIT public-behavior features
implemented from clean-room specifications.

## Clean-Room Rule

Implementation is written independently from mathematical principles and public
numerical methods. Pinned MIT-era ViennaTools repositories are used only for
audit and high-level attribution. GPL versions and later forbidden versions must
not be inspected, copied, translated, or paraphrased.

## Mathematical Basis

The core equation is:

```text
phi_t + V_n |grad phi| = 0
```

with `phi < 0` in material, `phi > 0` in void, and outward normal
`n = grad(phi) / |grad(phi)|`. Negative normal velocity etches material inward;
positive normal velocity deposits material outward.

## Milestones

1. Project setup, license audit, notices, and architecture documentation.
2. Dense Cartesian grids, SDFs, derivatives, normals, CFL timestep, and explicit
   level-set evolution.
3. Analytic validation for shrinking/growing circles and planar fronts.
4. Process-rate models for etch, deposition, masks, ion enhancement, sticking,
   and sputtering yield.
5. JAX-native direction sampling and approximate flux accumulation.
6. Runnable examples that save arrays and optional plots.
7. Benchmarks that separate JIT compile time from steady-state execution.
8. Clean-room post-MIT public behavior layer: material built-ins/custom
   registries, material-name serialization, polynomial cosine yield, mesh
   extraction, flux backend selection, and simplified plasma/multi-particle
   process models.

## Test Plan

Unit tests cover grid construction, SDF signs and values, finite differences,
normals, level-set sign convention, process models, material registry behavior,
ray sampling, and flux accumulation. Validation tests compare against analytic
circle and plane motion.

Post-MIT clean-room tests cover material serialization, material value maps,
target and mask material behavior, polynomial cosine yields, contour/surface
extraction, extrusion/slicing, backend selection, parameter validation, and
simplified plasma/fluorocarbon/multi-particle formulas.

## Performance Plan

Numerical kernels use `jax.numpy`, `jax.jit`, `jax.vmap`, and `jax.lax.scan`.
Benchmarks report available devices, warm up compiled functions, and time
steady-state execution with result synchronization.

## Known Limitations

The current version uses dense grids and approximate flux models. It is intended
as a correct, extensible baseline, not a complete industrial process simulator.
Sparse/narrow-band storage, production mesh generation, calibrated plasma
chemistry, exact visibility/occlusion, and optional external ray tracing
backends remain future work.
