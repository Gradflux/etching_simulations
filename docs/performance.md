# Performance And GPU Notes

## JAX Backends

`jaxps` uses JAX arrays and JIT-compiled kernels. It runs on CPU by default and
uses GPU only when the installed JAX backend exposes a GPU device.

Check devices with:

```bash
.venv/bin/python -c "from jaxps.utils import describe_devices; print(describe_devices())"
```

CUDA-enabled JAX installation is platform-specific. Install the CUDA JAX wheel
following official JAX instructions for the target CUDA version, then rerun the
device check.

## Benchmark Timing

JAX compilation can dominate the first call. Benchmarks therefore report:

- `compile_plus_first_run_s`
- `steady_run_s`
- `throughput_per_s`
- device platform and device kind
- shape and dtype

Use JSON output for automated tracking:

```bash
.venv/bin/python benchmarks/bench_level_set.py --size 256 --format json
```

Run all benchmarks:

```bash
.venv/bin/python benchmarks/run_all.py --sizes 128 256 512 --repeat 5
```

Run the full validation smoke, including examples and benchmark schema checks:

```bash
.venv/bin/python scripts/run_heavy_validation.py --all --output-dir /tmp/jaxps-validation
```

Require a backend for benchmark acceptance:

```bash
.venv/bin/python benchmarks/run_all.py --sizes 128 256 --repeat 5 --require-backend cpu
.venv-metal/bin/python benchmarks/run_all.py --sizes 128 256 --repeat 5 --require-backend metal
```

When `--require-backend` is unavailable, the runner exits nonzero and reports
the backends currently exposed by JAX.

## Optional External OptiX

`EXTERNAL_OPTIX` is a boundary for a future adapter, not a vendored backend.
Detection looks for user-provided runtime hints such as `OPTIX_ROOT`, CUDA/NVIDIA
driver visibility, and an optional future `jaxps_optix` extension. If the
external environment is absent, backend selection falls back to implemented JAX
paths.

```bash
OPTIX_ROOT=/path/to/external/optix .venv/bin/python -m pytest -q tests/test_optix_boundary.py
```

## Current Optimization Choices

- Derivative hot paths use fused one-sided and central-gradient derivative
  stacks for Godunov norms, normals, and diagnostics.
- Polynomial cosine yield uses a JAX `lax.scan` Horner evaluation.
- Flux accumulation supports chunking to reduce peak exposure memory.
- Dense masked narrow-band updates avoid changing far-away cells while keeping
  JAX array shapes static.
- Examples are CPU-safe and automatically use GPU when JAX sees one.
- `jaxps.experimental.spectral` provides FFT-based periodic diagnostics for
  smooth fields only.

## Spectral Methods

Spectral derivatives can be faster for smooth periodic diagnostics, but they are
not monotone. They are not appropriate as the default advection discretization
for masked etching, corners, discontinuous velocities, or topological changes.
