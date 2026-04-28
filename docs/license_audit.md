# License Audit

Audit date: 2026-04-28

This file records the local permissive reference boundary for `jaxps`. The new
package is an independent Python/JAX implementation and is not a fork of GPL
ViennaPS or any GPL ViennaTools component.

## Clean-Room Rules

- Do not inspect, copy, port, translate, or paraphrase GPL implementation code.
- Do not inspect or copy implementation code from later forbidden versions.
- Use only the pinned MIT-era references listed below for attribution and
  high-level understanding.
- Prefer independent implementation from mathematical principles and public
  numerical methods.
- Do not vendor NVIDIA OptiX headers, samples, SDK files, binaries, or copied
  code.
- Treat ViennaPS `v4.3.0+` and related forbidden versions as GPL-era source-code
  boundaries. Public release-note behavior may be re-specified in clean-room
  documents, but GPL source code, tests, and examples must not be inspected.

## Audited Local References

| Repo | Pinned version | Audited local version | Commit | License summary | Allowed use | Forbidden future versions |
|---|---:|---:|---|---|---|---|
| ViennaPS | `v4.2.2` | `v4.2.2` | `9e92af6` | MIT; use, copy, modify, publish, distribute, sublicense, and sell with notice preservation | Permissive reference and attribution only; implementation remains independent | `>= v4.3.0` |
| ViennaCore | `v1.10.0` | `v1.10.0` | `3b8078e` | MIT; use, copy, modify, publish, distribute, sublicense, and sell with notice preservation | Permissive reference and attribution only; implementation remains independent | `>= v2.0.0` |
| ViennaLS | `v5.5.1` | `v5.5.1` | `8833d18` | MIT; use, copy, modify, publish, distribute, sublicense, and sell with notice preservation | Permissive reference and attribution only; implementation remains independent | `>= v5.6.0` |
| ViennaHRLE | `v0.8.0` | `v0.8.0` | `c7a5224` | MIT; use, copy, modify, publish, distribute, sublicense, and sell with notice preservation | Permissive reference and attribution only; implementation remains independent | `>= v1.0.0` |
| ViennaRay | `v3.11.1` | `v3.11.1` | `ed83898` | MIT; use, copy, modify, publish, distribute, sublicense, and sell with notice preservation | Permissive reference and attribution only; implementation remains independent | `>= v4.0.0` |
| ViennaCS | `v1.1.1` | `v1.1.1` | `dc9d54b` | MIT; use, copy, modify, publish, distribute, sublicense, and sell with notice preservation | Permissive reference and attribution only; implementation remains independent | `>= v2.0.0` |

The local license headers inspected for these tags contain MIT license text and
copyright notices for the Institute for Microelectronics, TU Wien.

## Project License Assumption

`jaxps` is released under the MIT License. The package is intended for commercial
use and does not impose source disclosure obligations. This audit is not legal
advice; downstream distributors should perform their own review.

## Post-MIT Feature Tracking

Clean-room adaptations of public post-MIT behavior are tracked in:

- `docs/post_mit_feature_spec.md`
- `docs/after_mit_license_changes.md`
- `docs/mit_license_changes.md`
