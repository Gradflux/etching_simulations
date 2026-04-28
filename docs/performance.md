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

Require a backend for benchmark acceptance:

```bash
.venv/bin/python benchmarks/run_all.py --sizes 128 256 --repeat 5 --require-backend cpu
.venv-metal/bin/python benchmarks/run_all.py --sizes 128 256 --repeat 5 --require-backend metal
```

When `--require-backend` is unavailable, the runner exits nonzero and reports
the backends currently exposed by JAX.

## Current Optimization Choices

- Derivative hot paths use fused one-sided derivative stacks for Godunov norms.
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
