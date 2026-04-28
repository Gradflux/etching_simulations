# jaxps

`jaxps` is an independent Python/JAX implementation of core process-simulation
ideas: Cartesian level-set geometry, etching and deposition velocity models,
finite-difference evolution, reinitialization, and simple JAX-native ray/flux
approximations.

The project is a clean-room rewrite. It is not a fork of GPL ViennaPS, does not
wrap ViennaPS, and does not vendor NVIDIA OptiX. The pinned MIT-era ViennaTools
repositories are used only as permissive reference context and attribution
sources. New implementation code is written from mathematical principles and
public numerical methods.

## Licensing Boundary

Only the following local MIT references are in scope:

- ViennaPS `v4.2.2`
- ViennaCore `v1.10.0`
- ViennaLS `v5.5.1`
- ViennaHRLE `v0.8.0`
- ViennaRay `v3.11.1`
- ViennaCS `v1.1.1`

GPL-era versions and later license-transition versions must not be inspected,
copied, translated, or paraphrased. See `docs/license_audit.md`.

## Installation

```bash
cd etching_simulations
python3 -m venv .venv
.venv/bin/python -m pip install -U pip setuptools wheel
.venv/bin/python -m pip install -e ".[test]"
```

If editable install fails with `Cannot import 'setuptools.build_meta'`, the
virtual environment is missing build tooling. Re-run:

```bash
.venv/bin/python -m pip install -U setuptools wheel
.venv/bin/python -m pip install -e ".[test]"
```

## Tests

```bash
cd etching_simulations
.venv/bin/python -m pytest -q
```

The validation tests check that the documented sign convention is respected:
etching shrinks material and deposition grows it.

Heavy validation and benchmark runs:

```bash
.venv/bin/python -m pytest -q -m slow
.venv/bin/python scripts/run_heavy_validation.py --all --output-dir /tmp/jaxps-validation
.venv/bin/python benchmarks/run_all.py --sizes 128 256 512 --repeat 5
```

## Examples

Examples run on CPU by default and use a GPU automatically when the installed
JAX backend exposes one:

```bash
python3 examples/isotropic_etch_2d.py
```

Each example writes arrays to `outputs/`. If `matplotlib` is installed, simple
plots are saved as well.

## Folder Structure

```text
jaxps/geometry/
  grid.py          Cartesian 2D/3D grid metadata and coordinates
  sdf.py           Circle, sphere, plane, box, and Boolean SDFs
  derivatives.py   Finite-difference derivative kernels
  normals.py       Surface normal utilities
  masks.py         Geometric/material masks
  mesh.py          Independent contour/surface extraction
  extrude.py       2D-to-3D extrusion
  slice.py         3D-to-2D slicing
jaxps/solvers/     Level-set evolution, CFL logic, reinitialization
jaxps/models/      Etch, deposition, angular yield, plasma, fluorocarbon,
                  material-target, mask-material, and multi-particle models
jaxps/rays/        Direction sampling, exposure flux, visibility approximation,
                  and backend selection
jaxps/materials/   Built-ins, immutable materials, registries, value maps
jaxps/io/          Array I/O, simulation/domain serialization, legacy VTK scalar output
jaxps/utils/       Device reporting, parameter dataclasses, shared typing
tests/             Unit tests and analytic validation cases
examples/          Runnable process examples
benchmarks/        JIT-aware CPU/GPU benchmark scripts
docs/              License audit, physics, numerics, examples, performance,
                  and post-MIT clean-room feature tracking
```

## Physics And Sign Convention

The simulator evolves a level-set field `phi(x, t)` whose zero contour is the
material interface. The convention is:

- `phi < 0`: material
- `phi > 0`: void
- `n = grad(phi) / |grad(phi)|`: outward normal
- `phi_t + V_n |grad(phi)| = 0`

Positive `V_n` grows material outward. Negative `V_n` etches material inward.
For a circle or sphere with `phi = r - R`, the analytic radius law is
`R(t) = R0 + V_n t`.

The default evolution method is monotone upwind finite differences. Spectral
methods are documented as possible future diagnostics for smooth periodic fields,
not as the production advection method for masked or nonsmooth interfaces.

Further reading:

- `docs/physics.md`
- `docs/numerical_methods.md`
- `docs/examples.md`
- `docs/performance.md`
- `docs/mps_metal.md`
- `docs/optix_external.md`
- `docs/after_mit_license_changes.md`
- `docs/post_mit_feature_spec.md`

## Benchmarks

Benchmarks separate JIT compile time from steady-state execution and can emit
JSON/CSV:

```bash
.venv/bin/python benchmarks/run_all.py --sizes 128 256 --repeat 3
.venv/bin/python benchmarks/bench_level_set.py --size 256 --format json
```

Optional Apple Metal/MPS checks must be run in a separate `jax-metal`
environment:

```bash
.venv-metal/bin/python scripts/check_mps_backend.py
.venv-metal/bin/python -m pytest -q -m mps
.venv-metal/bin/python benchmarks/run_all.py --sizes 128 256 --repeat 5 --require-backend metal
```

Optional external OptiX detection never vendors NVIDIA files:

```bash
OPTIX_ROOT=/path/to/external/optix .venv/bin/python -m pytest -q tests/test_optix_boundary.py
```

## Validation Matrix

| Check | Command | Required for baseline |
|---|---|---|
| Editable install | `.venv/bin/python -m pip install -e ".[test]"` | Yes |
| Unit and analytic validation | `.venv/bin/python -m pytest -q` | Yes |
| Heavy CPU validation | `.venv/bin/python -m pytest -q -m slow` | Yes |
| Full integration smoke | `.venv/bin/python scripts/run_heavy_validation.py --all --output-dir /tmp/jaxps-validation` | Yes |
| Benchmarks | `.venv/bin/python benchmarks/run_all.py --sizes 64 128 256 --repeat 3` | Yes |
| Apple Metal/MPS | `.venv-metal/bin/python -m pytest -q -m mps` | Optional |
| External OptiX boundary | `OPTIX_ROOT=/path/to/optix .venv/bin/python -m pytest -q tests/test_optix_boundary.py` | Optional |

## Current Status

- Dense Cartesian grids support 2D/3D level-set evolution with dense
  narrow-band masking.
- Reinitialization uses a validated PDE-based redistancing pass near smooth
  interfaces.
- Ray/flux modeling includes JAX exposure, chunked accumulation, grid-marched
  visibility approximation, and an external-only OptiX backend boundary.
- VTK legacy structured-point output is available for scalar level-set fields;
  `.npz`/simulation serialization remains the metadata-preserving format.
- JAX kernels use `jit`, `vmap`, and `lax.scan` in the derivative, solver,
  ray/flux, reinitialization, and benchmark paths.
- GPU acceleration is automatic only when the installed JAX backend exposes a
  non-CPU device.

## Next Iteration Targets

- Add sparse HRLE-style storage or a real sparse narrow-band data structure.
- Build a native optional OptiX adapter against user-installed OptiX/CUDA
  without vendoring NVIDIA files.
- Add calibrated plasma chemistry and stronger 3D process validation.
- Improve production redistancing and mesh export beyond simple VTK scalar
  fields.
