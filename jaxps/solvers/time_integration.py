"""Time integration for level-set evolution."""

from __future__ import annotations

import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from jaxps.solvers.level_set import cfl_timestep, level_set_step


class EvolutionResult(NamedTuple):
    """Result of a level-set evolution."""

    phi: Array
    t_final: float
    num_steps: int
    dt: float
    trajectory: Array | None


VelocityFn = Callable[[Array, float | Array], Array | float]


def evolve_level_set(
    phi0: Array,
    spacing: tuple[float, ...],
    velocity_fn: VelocityFn,
    t_final: float,
    cfl: float = 0.4,
    max_steps: int | None = None,
    return_trajectory: bool = False,
    band_width: float | None = None,
) -> EvolutionResult:
    """Evolve ``phi0`` to ``t_final`` using fixed-step explicit Euler scans."""

    if t_final < 0.0:
        raise ValueError("t_final must be nonnegative")
    if t_final == 0.0:
        empty = jnp.empty((0,) + phi0.shape, dtype=phi0.dtype) if return_trajectory else None
        return EvolutionResult(phi=phi0, t_final=0.0, num_steps=0, dt=0.0, trajectory=empty)

    initial_velocity = velocity_fn(phi0, 0.0)
    stable_dt = float(cfl_timestep(initial_velocity, spacing, cfl))
    if math.isinf(stable_dt):
        empty = jnp.empty((0,) + phi0.shape, dtype=phi0.dtype) if return_trajectory else None
        return EvolutionResult(phi=phi0, t_final=t_final, num_steps=0, dt=0.0, trajectory=empty)

    num_steps = max(1, int(math.ceil(t_final / stable_dt)))
    if max_steps is not None and num_steps > max_steps:
        raise ValueError(f"required {num_steps} steps exceeds max_steps={max_steps}")
    dt = float(t_final / num_steps)

    def scan_step(carry: tuple[Array, Array], _index: Array) -> tuple[tuple[Array, Array], Array]:
        phi, time = carry
        velocity = velocity_fn(phi, time)
        next_phi = level_set_step(phi, velocity, spacing, dt, band_width=band_width)
        return (next_phi, time + dt), next_phi

    (phi_final, _), trajectory = jax.lax.scan(
        scan_step,
        (phi0, jnp.asarray(0.0, dtype=phi0.dtype)),
        jnp.arange(num_steps),
    )
    return EvolutionResult(
        phi=phi_final,
        t_final=float(t_final),
        num_steps=num_steps,
        dt=dt,
        trajectory=trajectory if return_trajectory else None,
    )
