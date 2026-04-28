# After MIT License Changes

This document records the clean-room adaptation boundary for public features
described after the local MIT-era ViennaTools references. It is intended as an
implementation handoff note: a reviewer can describe public behavior from
release notes or documentation, and a separate implementer can implement that
behavior without reading GPL source code.

## Source Boundary

Allowed source-code references remain limited to the locally pinned MIT-era
repositories:

- ViennaPS `v4.2.2`
- ViennaCore `v1.10.0`
- ViennaLS `v5.5.1`
- ViennaHRLE `v0.8.0`
- ViennaRay `v3.11.1`
- ViennaCS `v1.1.1`

Forbidden source-code references:

- ViennaPS `>= v4.3.0`
- ViennaCore `>= v2.0.0`
- ViennaLS `>= v5.6.0`
- ViennaHRLE `>= v1.0.0`
- ViennaRay `>= v4.0.0`
- ViennaCS `>= v2.0.0`

No GPL implementation code, GPL tests, GPL examples, or GPL internal algorithms
were copied, translated, ported, or paraphrased for these changes.

## Adapted Public Behaviors

The following behavior was implemented independently from the user-provided
public feature specification, mathematical principles, and general semiconductor
process-simulation knowledge:

| Public behavior requested | Independent implementation files | Clean-room implementation basis |
|---|---|---|
| Runtime acceleration and CPU fallback | `jaxps/utils/devices.py`, `benchmarks/` | JAX device discovery, `jax.jit`, `jax.vmap`, `jax.lax.scan`; no custom CUDA loader |
| Optional Apple Metal/MPS diagnostics | `jaxps/utils/devices.py`, `scripts/check_mps_backend.py`, `docs/mps_metal.md` | Apple/JAX runtime probing only; Metal is optional and experimental |
| Polynomial cosine yield | `jaxps/models/yield_models.py` | `mu = max(0, -dot(d, n))`; Horner polynomial evaluation; optional nonnegative clamp |
| Built-in and runtime custom materials | `jaxps/materials/material.py`, `builtins.py`, `registry.py` | General semiconductor material names and deterministic IDs chosen for this project |
| Material-name persistence | `jaxps/io/serialization.py` | `.npz` arrays plus JSON registry metadata storing names and IDs |
| Mask and target material models | `jaxps/models/etch.py`, `deposition.py` | JAX boolean masks generated from project-local material registries |
| Material value maps | `jaxps/materials/value_map.py` | Explicit mapping from material ID arrays to scalar/vector JAX fields |
| Hull/surface extraction | `jaxps/geometry/mesh.py` | Independent marching-squares-style 2D contours and 3D sign-change point clouds |
| Extrusion and slicing utilities | `jaxps/geometry/extrude.py`, `slice.py` | Direct Cartesian array construction and indexing |
| Modular flux backend selection | `jaxps/rays/backends.py` | High-level backend names resolved to implemented JAX paths; OptiX reserved only |
| Multi-particle process workflow | `jaxps/models/multi_particle.py` | Additive species rates with explicit signs, optional angular yields, and material response |
| Plasma and fluorocarbon-style models | `jaxps/models/plasma.py`, `fluorocarbon.py` | Transparent simplified formulas from general plasma-etch physics |
| Domain setup serialization | `jaxps/io/serialization.py` | JSON-compatible dimension, bounds, shape, spacing, and boundary metadata |
| Parameter objects | `jaxps/utils/validation.py` | Small validated dataclasses with JSON-compatible `to_dict`/`from_dict` |

No CUDA library, NVIDIA OptiX SDK, OptiX headers, samples, binaries, GPL source
files, GPL tests, or GPL examples are part of these changes.

## Implemented Formulas

Polynomial cosine yield:

```text
mu = max(0, -dot(d, n))
Y(mu) = c0 + c1 mu + c2 mu^2 + ... + ck mu^k
Y_clamped = max(Y, 0)
```

Simple plasma velocity:

```text
V_n = -(Gamma_i Y_i + Gamma_n k_n) + Gamma_p s_p
```

Simplified fluorocarbon velocity:

```text
V_n = -(k_F Gamma_F + Y_i Gamma_i A(theta)) exp(-a C) + s_p Gamma_p
```

These formulas are deliberately compact and documented so future changes can be
reviewed scientifically without relying on implementation details from GPL
projects.

## How To Update From This Point

For future post-MIT/GPL-era public features:

1. Record the public behavior in `docs/post_mit_feature_spec.md` without source
   code details.
2. Implement from equations, public docs, release notes, and tests written from
   expected behavior.
3. Add tests that validate physical behavior and API contracts, not internal
   parity with GPL code.
4. Update this document with what behavior was adapted and which implementation
   modules changed.
5. Re-run `pytest -q`, slow validation, and backend checks.

The clean-room implementation may be functionally inspired by public feature
descriptions, but it must remain structurally and textually independent.

## Current Verification

The post-MIT clean-room feature tests cover:

- material built-ins, aliases, custom IDs, and serialization,
- material value maps for scalar and vector values,
- material-target and mask-material process behavior,
- polynomial cosine yield convention and JIT equivalence,
- independent contour/surface extraction plus extrusion/slicing,
- modular flux backend selection,
- parameter dataclass validation,
- plasma, fluorocarbon, and multi-particle velocity formulas.
