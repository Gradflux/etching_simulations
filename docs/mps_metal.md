# Optional Apple Metal/MPS Backend

`jaxps` keeps CPU JAX as the supported baseline. Apple Metal/MPS acceleration is
optional and should be tested in a separate environment because Apple's JAX Metal
plugin is experimental.

Apple's JAX Metal plugin documentation states that the plugin uses OpenXLA/PjRT
to lower JAX programs through StableHLO and MPSGraph/Metal runtime APIs. It also
states that the plugin is experimental and does not support all JAX tests or
some dtypes such as `float64`, `complex64`, and `complex128`.

Official references:

- Apple JAX Metal plugin: https://developer.apple.com/metal/jax/
- JAX installation guide: https://docs.jax.dev/en/latest/installation.html

## Separate Environment

Do not install `jax-metal` into the default `.venv` used for CPU validation.
Create a separate environment:

```bash
cd etching_simulations
python3 -m venv .venv-metal
.venv-metal/bin/python -m pip install -U pip wheel numpy
.venv-metal/bin/python -m pip install -e ".[test]"
.venv-metal/bin/python -m pip install jax-metal
```

Depending on the current plugin and JAX compatibility matrix, you may need to
pin `jax`, `jaxlib`, or set `ENABLE_PJRT_COMPATIBILITY=1`. Follow Apple's
current compatibility table for the target macOS and plugin version.

## Verify

```bash
.venv-metal/bin/python -c "import jax; print(jax.devices()); print(jax.default_backend())"
.venv-metal/bin/python scripts/check_mps_backend.py
```

The check script reports two separate facts:

- `hardware_mps_present`: the Mac appears to have Metal-capable Apple GPU
  hardware.
- `jax_metal_backend_available`: JAX exposes a usable Metal backend in the
  current Python environment.

It is possible, and common, for `hardware_mps_present` to be `true` while
`jax_metal_backend_available` is `false`. That means the machine has MPS/Metal
hardware, but the current Python environment is still CPU-only. Expected backend
names vary by JAX/plugin version, but `jaxps` treats `mps` as an alias for
`metal`.

To check only whether the machine has Metal-capable hardware, use:

```bash
python scripts/check_mps_backend.py --hardware-only
```

## Run MPS Tests And Benchmarks

```bash
.venv-metal/bin/python -m pytest -q -m mps
.venv-metal/bin/python benchmarks/run_all.py --sizes 128 256 --repeat 5 --require-backend metal
```

Do not claim Metal acceleration unless these commands report a Metal backend and
the MPS tests pass.

## Interpreting A Failed Backend Probe

If `scripts/check_mps_backend.py` reports:

```json
"hardware_mps_present": true,
"jax_metal_plugin_installed": true,
"jax_metal_backend_available": false
```

then the machine has Metal/MPS-capable hardware and the Python package is
installed, but JAX could not initialize a usable Metal backend. One observed
failure mode is a plugin abort containing:

```text
No supported GPU was found.
platform->VisibleDeviceCount() == 1 (0 vs. 1)
```

In that state, run CPU validation by forcing JAX to ignore the plugin:

```bash
JAX_PLATFORMS=cpu .venv-metal/bin/python -m pytest -q
```

Metal benchmark acceptance still requires:

```bash
.venv-metal/bin/python benchmarks/run_all.py --sizes 128 256 --repeat 5 --require-backend metal
```
