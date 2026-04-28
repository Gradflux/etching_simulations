# Optional External NVIDIA OptiX Boundary

`jaxps` does not vendor NVIDIA OptiX. It does not include OptiX headers, SDK
files, samples, binaries, PTX, or copied implementation code.

OptiX support is represented only as an optional external backend boundary:

- Users install and license OptiX outside this repository.
- `OPTIX_ROOT` can point to a local external OptiX installation.
- `jaxps` can report whether an external OptiX-looking environment is present.
- If OptiX is absent, backend selection falls back to implemented JAX paths.
- A native OptiX adapter is future work and must be developed without vendoring
  NVIDIA SDK contents.

Official public references:

- NVIDIA OptiX downloads: https://developer.nvidia.com/designworks/optix/download
- NVIDIA `optix-dev` headers repository: https://github.com/NVIDIA/optix-dev

Detection checks are intentionally conservative. They look for external runtime
hints such as:

- `OPTIX_ROOT/include/optix.h`
- `nvidia-smi` on `PATH`
- an optional future Python extension named `jaxps_optix`

These checks do not load or copy OptiX files into the project.
