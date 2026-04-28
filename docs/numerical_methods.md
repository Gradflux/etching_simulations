# Numerical Methods

## Sign Convention

The level-set function `phi(x, t)` represents the interface:

```text
Gamma(t) = {x : phi(x, t) = 0}
```

The convention is:

- `phi < 0`: material
- `phi > 0`: void/outside
- outward normal: `n = grad(phi) / |grad(phi)|`
- level-set equation: `phi_t + V_n |grad phi| = 0`
- positive `V_n`: interface moves along outward normal
- negative `V_n`: interface moves inward

Etching therefore uses negative velocity. Deposition uses positive velocity.

## Cartesian Grids

2D grids use shape `(nx, ny)` and 3D grids use `(nx, ny, nz)`. For each axis:

```text
dx = (x_max - x_min) / (n - 1)
```

Coordinate arrays are generated with `meshgrid(indexing="ij")`.

## Signed-Distance Functions

Implemented analytic SDFs include circles, spheres, planes, and axis-aligned
boxes. Boolean operations use standard level-set combinations:

```text
union        = min(phi_a, phi_b)
intersection = max(phi_a, phi_b)
difference   = max(phi_a, -phi_b)
```

## Finite Differences

Forward, backward, and central differences are computed along each axis. Central
differences use first-order one-sided values at physical boundaries. Gradient
magnitudes use a small epsilon only for safe normalization.

## Godunov Upwind Hamiltonian

For:

```text
phi_t + V |grad phi| = 0
```

let `D-` be the backward derivative and `D+` the forward derivative. For
`V >= 0`, the upwind magnitude is:

```text
sqrt(sum(max(D-, 0)^2 + min(D+, 0)^2))
```

For `V < 0`, it is:

```text
sqrt(sum(max(D+, 0)^2 + min(D-, 0)^2))
```

The explicit Euler update is:

```text
phi_next = phi - dt * V * |grad phi|_G
```

## CFL Condition

The stable timestep is:

```text
dt <= cfl * min(dx, dy, dz) / max(abs(V))
```

If velocity is identically zero, evolution returns the input unchanged.

## Dense Narrow-Band Updates

Evolution and reinitialization can receive `band_width`. The full dense update
is computed with JAX-compatible static shapes, then applied only where:

```text
abs(phi) <= band_width
```

This reduces unnecessary far-field changes but is not sparse HRLE storage.

## Reinitialization

The implemented reinitialization equation is:

```text
phi_tau + S(phi0)(|grad phi| - 1) = 0
S(phi0) = phi0 / sqrt(phi0^2 + min(dx)^2)
```

Godunov one-sided derivatives are selected from the sign of `S(phi0)`.
Validation checks that the zero contour drifts by only a few grid cells and that
`|grad phi|` near smooth interfaces moves toward one.

## Spectral Methods

Spectral derivatives are not used for production level-set advection. They can
be attractive for smooth periodic fields but are non-monotone and can create
oscillations near masks, corners, shocks, or discontinuous process rates.

## Analytic Validation

For `phi = sqrt((x - cx)^2 + (y - cy)^2) - R0`, a constant velocity gives:

```text
R(T) = R0 + V_n T
```

Thus etching with `V_n = -rate` shrinks the circle and deposition with
`V_n = +rate` grows it. For a planar front `phi = x - x0`, the interface moves
to `x0 + V_n T`.
