import numpy as np

from jaxps.geometry import make_grid_2d, sdf_plane
from jaxps.solvers import evolve_level_set


def test_evolve_level_set_reaches_final_time_and_shape():
    grid = make_grid_2d(bounds=((-1.0, 1.0), (-1.0, 1.0)), shape=(41, 41))
    phi0 = sdf_plane(grid, point=(0.0, 0.0), normal=(1.0, 0.0))

    result = evolve_level_set(
        phi0=phi0,
        spacing=grid.spacing,
        velocity_fn=lambda phi, _t: 0.1 * np.ones(phi.shape, dtype=np.float32),
        t_final=0.1,
        cfl=0.4,
    )

    assert result.phi.shape == grid.shape
    assert result.t_final == 0.1
    assert result.num_steps >= 1
    assert result.dt > 0.0


def test_evolve_level_set_returns_trajectory_when_requested():
    grid = make_grid_2d(bounds=((-1.0, 1.0), (-1.0, 1.0)), shape=(41, 41))
    phi0 = sdf_plane(grid, point=(0.0, 0.0), normal=(1.0, 0.0))

    result = evolve_level_set(
        phi0=phi0,
        spacing=grid.spacing,
        velocity_fn=lambda _phi, _t: 0.2,
        t_final=0.1,
        cfl=0.4,
        return_trajectory=True,
    )

    assert result.trajectory is not None
    assert result.trajectory.shape == (result.num_steps,) + grid.shape
