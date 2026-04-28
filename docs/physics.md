# Physics Model

## Interface Representation

The material boundary is represented implicitly by a level-set field:

```text
Gamma(t) = {x : phi(x, t) = 0}
```

The sign convention is fixed:

```text
phi < 0  material
phi > 0  void
n = grad(phi) / |grad(phi)|
```

The normal velocity law is:

```text
phi_t + V_n |grad(phi)| = 0
```

Positive `V_n` moves the interface along the outward normal. Negative `V_n`
moves it inward and therefore etches material.

## Isotropic Etching And Deposition

For a constant etch rate `R > 0`:

```text
V_n = -R
```

For a constant deposition rate:

```text
V_n = +R
```

For circles and spheres initialized as `phi = r - R0`, the exact radius is:

```text
R(t) = R0 + V_n t
```

## Directional Etching

For a beam direction `d` describing particle travel, the incidence factor is:

```text
I = max(n dot (-d), 0)
```

The directional etch velocity is:

```text
V_n = -R I
```

Only surface orientations facing the incoming beam etch.

## Ion-Enhanced Etching

The simplified ion-enhanced model combines chemical etching and ion flux:

```text
V_n = -(R_chem + Gamma_i Y(theta))
```

`Gamma_i` is ion flux and `Y(theta)` is a sputtering or enhancement yield. The
current yield implementation is a thresholded polynomial:

```text
Y = scale * max(cos(theta) - threshold, 0)^exponent
```

For the polynomial cosine yield model, the incoming particle direction `d`
points along travel toward the surface and the outward normal is `n`:

```text
mu = max(0, -dot(d, n))
Y(mu) = c0 + c1 mu + c2 mu^2 + ... + ck mu^k
```

Negative polynomial values are clamped to zero by default because negative
physical yields are not meaningful in the simple rate models.

## Deposition With Sticking

For scalar incoming flux `Gamma` and sticking coefficient `s`:

```text
V_n = s Gamma
```

This assumes deposited material grows the existing interface outward.

## Flux Integral

For angular distribution `f(omega)`, the local exposure flux is approximated by:

```text
Gamma(x) = integral max(n(x) dot omega, 0) f(omega) d omega
```

The code approximates this integral with sampled or deterministic directions.
The optional visibility approximation multiplies each directional contribution
by a grid-marched line-of-sight weight in `[0, 1]`.

## Material-Dependent Rates

Material fields are stored as integer IDs interpreted through a material
registry. A material value map converts IDs to rate coefficients:

```text
k(x) = value_map(material_id(x))
V_n(x) = -k(x) Gamma(x)
```

Target-material models set `V_n = 0` outside the selected materials. Mask
materials can use a separate reduced mask rate:

```text
V_n = -R_base          unmasked material
V_n = -R_mask          mask material
```

with `R_mask = 0` for an ideal protective mask.

## Plasma And Fluorocarbon-Style Models

The simple plasma model combines ion etch, neutral chemical etch, and
passivating deposition:

```text
V_n = -(Gamma_i Y_i + Gamma_n k_n) + Gamma_p s_p
```

The simplified fluorocarbon model is:

```text
V_n = -(k_F Gamma_F + Y_i Gamma_i A(theta)) exp(-a C) + s_p Gamma_p
```

Here `Gamma_F` is fluorine-like radical flux, `Gamma_i` is ion flux,
`Gamma_p` is polymer precursor flux, `A(theta)` is angular yield, and `C` is an
optional passivation coverage. These are deliberately transparent first-order
models, not calibrated industrial plasma chemistry.

## Spectral Methods

FFT/spectral derivatives can be fast for smooth periodic fields, curvature
diagnostics, or filtering. They are not used as the default level-set advection
method because the Hamilton-Jacobi equation needs monotone upwind discretization
near sharp corners, masks, and discontinuous rates. A future
`jaxps.experimental.spectral` contains periodic diagnostic operators for smooth
fields. They are not part of the default process-evolution path.
