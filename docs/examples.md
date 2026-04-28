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
