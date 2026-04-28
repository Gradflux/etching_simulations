# Validation

## Circle Shrink

The initial signed-distance field is:

```text
phi(x, y) = sqrt((x - cx)^2 + (y - cy)^2) - R0
```

With `V_n = -rate`, the expected radius is:

```text
R(T) = R0 - rate * T
```

The test estimates the final radius from a grid-axis zero crossing and accepts
an error proportional to grid spacing.

## Circle Growth

With the same initial field and `V_n = +rate`, the expected radius is:

```text
R(T) = R0 + rate * T
```

This verifies that deposition expands material under the documented sign
convention.

## Planar Front

For:

```text
phi(x, y) = x - x0
```

a constant velocity gives:

```text
x_interface(T) = x0 + V_n * T
```

Positive and negative velocities are tested.

## Expected Errors

The dense-grid first-order upwind update introduces numerical diffusion. Tests
therefore use tolerances tied to grid spacing, not machine precision.

## Latest Local Validation Snapshot

Environment:

- `jax 0.10.0`
- `jaxlib 0.10.0`
- backend: `cpu`
- device: `cpu:cpu`
- no Apple Metal/MPS backend detected in the default `.venv`

Results from the heavy validation smoke pass:

- Standard unit/validation suite: `47 passed`.
- 257x257 circle etch radius error: approximately `1.04e-6` at grid spacing
  `0.0078125`.
- Reinitialization mean near-interface error improved from approximately
  `0.1053` to `0.00227`.
- 73x73x73 sphere deposition completed with finite output.
- Chunked flux matched full flux within tolerance.
- Benchmarks emitted valid JSON for sizes `64`, `128`, and `256`.

The exact timing and throughput numbers are backend- and machine-dependent.
