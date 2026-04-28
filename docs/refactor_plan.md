# Refactor Plan — jaxps etching_simulations

_Date: 2026-04-28. Written after a full read of every source file, test, example,
benchmark, and documentation page._

---

## 1. Current Problems

### 1.1 Performance: device-to-host transfers in hot loops

**`geometry/mesh.py`** — `extract_zero_contour_2d` iterates over every grid cell in
a nested Python loop and calls `float(v)` on each JAX scalar. This causes one
host-side synchronization per cell value. For a 256×256 grid that is ~65 000
blocking transfers before any geometry is extracted. `extract_surface_mesh_3d`
does the same via `jnp.argwhere` + a Python loop that indexes JAX arrays element
by element.

Fix: convert the full phi and coordinate arrays to NumPy once at function entry
and work with NumPy throughout the loop. Only convert the result back to JAX at
the end.

### 1.2 Performance: unnecessary allocations in derivative hot paths

**`geometry/derivatives.py`** — `_neighbor` allocates a padded array with
`jnp.pad` and then immediately discards the pad with `jnp.take`. Two allocations
instead of one.

Fix: compute clipped gather indices with `jnp.clip(jnp.arange(n) ± 1, 0, n-1)`
and call `jnp.take` once directly — no pad array needed.

### 1.3 Performance: overkill scan for small polynomial evaluation

**`models/yield_models.py`** — `_polynomial_cosine_yield_jit` uses
`jax.lax.scan` to implement Horner's rule. Scan adds per-step overhead that is
measurable when the polynomial is only degree 1–4, which is the universal case.
`jnp.polyval` implements the same algorithm with lower constant overhead and
clearer intent.

Fix: replace the scan with `jnp.polyval(coeffs[::-1], mu_arr)`.

### 1.4 Performance: line-by-line Python loop in VTK writer

**`io/vtk.py`** — values are written one per Python `handle.write` call. For a
256×256 grid that is 65 536 format+write calls. `np.savetxt` does the same in a
single NumPy-managed batch.

Fix: replace loop with `np.savetxt(handle, values, fmt="%.9g")`.

### 1.5 Code clarity: vectorizable Python loop in `sdf_plane`

**`geometry/sdf.py`** — `sdf_plane` builds the dot product `(x - p) · n̂` by
accumulating into a zero array inside a Python `for` loop over axes. This is a
natural stack + sum operation and should be written that way.

Fix: `jnp.sum((jnp.stack(grid.coords, axis=-1) - point_arr) * unit, axis=-1)`.

### 1.6 Over-engineering: one-function file `rays/intersection.py`

`approximate_surface_band` is `jnp.abs(phi) <= width`, a one-liner exported from
its own module. It belongs either inlined at call sites or moved into
`rays/flux.py`.

Decision: keep the function (it appears in the public __all__), move it into
`rays/flux.py` and delete `rays/intersection.py`. Update the `rays/__init__.py`
import.

### 1.7 Over-engineering: elaborate backend selection for a single backend

**`rays/backends.py`** — `FluxBackendSelection`, `select_flux_backend`,
`normalize_flux_backend`, `IMPLEMENTED_FLUX_BACKENDS`, `KNOWN_FLUX_BACKENDS` are
elaborate machinery to select between JAX_GRID, JAX_RAYS, and EXTERNAL_OPTIX.
In practice only JAX_GRID and JAX_RAYS exist, and they both call the same
`accumulate_flux` kernel. `EXTERNAL_OPTIX` is an explicitly non-functional
boundary. The selection logic (15 lines) is correct but the surrounding
dataclass and its `to_dict()` serializer add more code than they deserve for a
feature with no real switching.

Decision: keep `backends.py` and its public API intact (tests and users depend on
it). Add a docstring noting that EXTERNAL_OPTIX is a detection boundary, not an
implemented backend. No code deletion; the module is small.

### 1.8 Over-engineering: serializable parameter dataclasses with no callers

**`utils/validation.py`** — `AdvectionParameters`, `RayTracingParameters`,
`CoverageParameters`, `ProcessParameters` are frozen dataclasses with
`to_dict`/`from_dict` and validation. They appear in one test, nowhere in
simulation code. They are pre-emptive infrastructure for a configuration layer
that does not yet exist.

Decision: keep them (removing them would break the test and the public API).
Add a docstring explaining they are serializable parameter containers intended for
future workflow tooling.

### 1.9 Missing value-level test for `sdf_box`

`test_box_and_boolean_sdf_signs` checks only that the center is negative and the
corner is positive. The box SDF is a non-trivial formula; a value test at a
known point is needed.

Fix: add `test_box_sdf_known_values` that checks the center value (−0.5) and an
outside-point value (0.25) analytically.

### 1.10 Stale documentation

`docs/validation.md` reports "47 passed" — the suite now has 62 passed + 2
skipped. Update the snapshot section.

---

## 2. Simplified Architecture (Proposed — Mostly Already in Place)

The package structure is already close to the target. No module moves are
required. The target is:

```
jaxps/
  geometry/    grid, sdf, derivatives, normals, masks, mesh (host-side), extrude, slice
  materials/   material, registry, value_map, builtins
  models/      etch, deposition, rates, yield_models, plasma, fluorocarbon, multi_particle
  solvers/     level_set, time_integration, reinitialization
  rays/        backends, flux (+approximate_surface_band merged in), sampling, visibility
  io/          arrays, serialization, vtk
  utils/       devices, typing, validation (parameter dataclasses)
  experimental/  spectral (FFT, non-production)
```

One file to delete: `rays/intersection.py` (content merged into `rays/flux.py`).

---

## 3. Files to Change

| File | Change |
|---|---|
| `geometry/derivatives.py` | Simplify `_neighbor` (clip+take, no pad) |
| `geometry/sdf.py` | Vectorize `sdf_plane` inner loop |
| `geometry/mesh.py` | Host-side numpy loop; one-shot JAX conversion |
| `models/yield_models.py` | Replace `lax.scan` Horner with `jnp.polyval` |
| `io/vtk.py` | Replace Python loop with `np.savetxt` |
| `rays/flux.py` | Move `approximate_surface_band` here |
| `rays/intersection.py` | Delete after merge |
| `rays/__init__.py` | Update import path |
| `tests/test_sdf.py` | Add `test_box_sdf_known_values` |
| `docs/validation.md` | Update stale test count |
| `docs/refactor_plan.md` | This file |

---

## 4. APIs to Preserve

All public `__all__` symbols in every `__init__.py` must remain importable at the
same path. The only structural move is `approximate_surface_band` from
`rays.intersection` to `rays.flux`; it continues to be exported from
`jaxps.rays` so no user import breaks.

---

## 5. APIs to Change

None. All changes are internal implementation improvements.

---

## 6. Expected Benefits

- `extract_zero_contour_2d` on a 256×256 grid: eliminates ~65 000 blocking
  device-to-host synchronizations. Expected 10–100× faster on GPU.
- `_neighbor`: halves intermediate allocations in every derivative call.
- `_polynomial_cosine_yield_jit`: lower JIT overhead for small polynomials.
- `sdf_plane`: marginally fewer intermediate arrays; much clearer code.
- `write_vtk_structured_grid`: ~100× faster for large grids.
- `test_box_sdf_known_values`: catches sign-convention regressions that the
  existing sign-only test would miss.

---

## 7. Risks

- **`jnp.polyval` behaviour**: the coefficient ordering must be reversed
  (`coeffs[::-1]`) since `polyval` expects highest-degree first. The test
  `test_polynomial_cosine_yield_convention_and_jit` must pass unchanged.
- **`np.savetxt` format**: must produce one value per line to stay VTK-compatible.
  `np.savetxt(handle, values, fmt="%.9g")` produces one value per line for a 1D
  array. Verified before commit.
- **`mesh.py` dtype**: NumPy floats are `float64` by default; the existing tests
  compare numpy norms against tolerances and will tolerate the dtype change.

---

## 8. Migration Notes

No user-facing API changes. Any code using `from jaxps.rays import
approximate_surface_band` continues to work (re-exported from `rays/__init__.py`).

---

## 9. Remaining Technical Debt (Not Addressed Here)

- `extract_zero_contour_2d` and `extract_surface_mesh_3d` are still O(N²) and
  O(N³) respectively. A proper marching-cubes implementation using JAX-native
  vectorized edge classification would scale better for production grids.
- `boundary_conditions` in `DomainSetup` declares 5 types but the solver always
  uses edge replication (`jnp.pad(mode="edge")`). When non-periodic boundary
  conditions are needed, `level_set_step` must be updated to honour the domain
  metadata.
- The backend selection in `rays/backends.py` dispatches to the same JAX
  implementation regardless of `JAX_GRID` vs `JAX_RAYS`. A real split between a
  vmap ray loop and a dense grid einsum would make the distinction meaningful.
- Reinitialization uses a fixed `cfl=0.3` default; high-curvature interfaces near
  corners may need fewer than `iterations=50` pseudo-time steps to avoid
  over-redistancing.

---

## 10. Recommended Next Steps (Post This Refactor)

1. Implement a vectorized marching-squares / marching-cubes extraction that stays
   on device (outputs fixed-size padded arrays).
2. Connect `boundary_conditions` in `DomainSetup` to actual boundary handling in
   `level_set_step`.
3. Add second-order ENO/WENO spatial discretisation option for smoother high-speed
   front propagation.
4. Add calibrated Si/SiO₂ etch selectivity tests based on public experimental data.
