# Third-Party Notices

`jaxps` is an independent implementation. It does not vendor code from the
projects listed below unless explicitly stated in future audit entries.

## MIT-Era ViennaTools References

- ViennaPS `v4.2.2`, MIT License, Institute for Microelectronics, TU Wien.
- ViennaCore `v1.10.0`, MIT License, Institute for Microelectronics, TU Wien.
- ViennaLS `v5.5.1`, MIT License, Institute for Microelectronics, TU Wien.
- ViennaHRLE `v0.8.0`, MIT License, Institute for Microelectronics, TU Wien.
- ViennaRay `v3.11.1`, MIT License, Institute for Microelectronics, TU Wien.
- ViennaCS `v1.1.1`, MIT License, Institute for Microelectronics, TU Wien.

These repositories are permissive references for audit and high-level
understanding. `jaxps` implementation code is written independently.

## Optional Future Dependencies

- pybind11, BSD 3-Clause License, only if later used.
- VTK, BSD 3-Clause License, only if later used.
- Embree, Apache License 2.0, only if later used.
- NVIDIA OptiX may be supported only as an optional external proprietary SDK in
  the future. OptiX headers, samples, SDK files, binaries, and copied code are
  not vendored in this project.
