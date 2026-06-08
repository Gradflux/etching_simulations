# Example Physics

## `isotropic_etch_2d.py`

Initial geometry is a circle:

```text
phi(x, y) = sqrt(x^2 + y^2) - R0
```

The model uses `V_n = -R`, so the expected radius is:

```text
R(t) = R0 - R t
```

## `isotropic_deposition_2d.py`

The same circular SDF is evolved with `V_n = +R`, giving:

```text
R(t) = R0 + R t
```

## `masked_trench_etch_2d.py`

The base material is an axis-aligned box SDF. Protected mask cells set velocity
to zero:

```text
V_masked = 0       in protected cells
V_masked = -R      elsewhere
```

The result is a simplified trench-like evolution, not a complete resist or
surface-chemistry model.

## `sf6_sinusoid_etch_2d.py`

The initial stack is a `260 nm` PPA layer above `500 nm` of silicon. The default
profile includes flat outer plateaus, a central patterned window from `1 um` to
`19 um`, and a recessed sinusoid that mimics the measured edge where the scratch
starts:

```text
depth(x) = edge_window(x) [D_ridge + 0.5 A (1 - cos(2 pi x / pitch))]
surface(x) = T_ppa - depth(x)
phi(x, y) = y - surface(x)
```

The material boundary is fixed at `y = 0`. The moving exposed surface etches at
the PPA rate while it is above that boundary and at the silicon rate once it has
transferred into silicon:

```text
R(y) = R_ppa   y >= 0
R(y) = R_si    -T_si <= y < 0
isotropic:  V_n = -R(y)
anisotropic: V_n = -R(y) max(n dot (0, 1), 0)
```

Defaults are `T_ppa = 260 nm`, `T_si = 500 nm`, `R_ppa = 40 nm/min`,
`R_si = 30 nm/min`, primary pitch `1 um`, comparison pitch `0.5 um`,
`D_ridge = 20 nm`, sinusoid relief `A = 100 nm`, and a `20 um` wide window.
The deepest initial trough is therefore `120 nm` below the PPA top surface,
leaving `140 nm` of PPA above the PPA/Si interface. The example rejects
geometries where
`D_ridge + A > T_ppa`, so the initial topography cannot start inside silicon.
Each run simulates both pitches across the same physical x-window. For each
pitch, it writes one combined profile/material-stack figure showing the
normalized initial/final profiles and the initial/final PPA/Si cross sections.
It also writes a separate `1500 nm` wide final-profile PNG with equal x/y
nanometer scaling. Use `--comparison-pitch-um` to change the second pitch,
`--actual-scale-window-nm` to change the actual-scale width, and
`--actual-scale-center-um` to move the actual-scale window.

Use `--etch-mode isotropic` for etching along the local surface normal. Use
`--etch-mode anisotropic` for downward directional etching through
`jaxps.models.directional_etch_rate` with incoming direction `(0, -1)`. This
example intentionally has no angular ion spread, shadowing, or line-of-sight
blocking.

## `directional_etch_2d.py` And `directional_etch_3d.py`

The directional model computes:

```text
I = max(n dot (-d), 0)
V_n = -R I
```

Only parts of the surface facing the incoming beam direction etch.

## `flux_deposition_2d.py` And `flux_deposition_3d.py`

The examples use deterministic hemisphere directions and approximate:

```text
Gamma = sum_i w_i max(n dot omega_i, 0)
V_n = s Gamma
```

This is an exposure/visibility approximation. It does not model scattering,
re-emission, detailed sticking chemistry, or exact ray tracing.

## Material-Aware Extensions

The library also supports material-dependent process examples through the
model APIs, even where a standalone example script has not yet been added.
The governing form is:

```text
V_n(x) = model(phi, material_id(x), flux(x), n(x))
```

For target-material etching:

```text
V_n = -R   material in targets
V_n = 0    material outside targets
```

For mask-material etching:

```text
V_n = -R_base    etchable material
V_n = -R_mask    mask material
```

For the simplified fluorocarbon model:

```text
V_n = -(k_F Gamma_F + Y_i Gamma_i A(theta)) exp(-a C) + s_p Gamma_p
```

These formulas are documented in `docs/physics.md` and tested in
`tests/test_post_mit_features.py`.
