from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jaxps.geometry import make_grid_2d, surface_normals
from jaxps.io import save_npz
from jaxps.models import directional_etch_rate, isotropic_etch_rate
from jaxps.solvers import evolve_level_set
from jaxps.utils import describe_devices

try:
    from ._common import example_parser, output_dir
except ImportError:
    from _common import example_parser, output_dir


def sinusoidal_surface_phi(
    grid,
    pitch_um: float,
    ppa_thickness_um: float,
    sinusoid_relief_um: float,
    ridge_depth_um: float,
    pattern_start_um: float,
    pattern_end_um: float,
    edge_transition_um: float,
    phase_rad: float = 0.0,
):
    """Return an SDF-like field for a recessed sinusoidal PPA/Si transfer profile."""

    x, y = grid.coords
    phase = 2.0 * jnp.pi * (x - pattern_start_um) / pitch_um + phase_rad
    central_depth = ridge_depth_um + 0.5 * sinusoid_relief_um * (1.0 - jnp.cos(phase))
    if edge_transition_um > 0.0:
        left_t = jnp.clip(
            (x - (pattern_start_um - edge_transition_um)) / edge_transition_um,
            0.0,
            1.0,
        )
        right_t = jnp.clip((x - pattern_end_um) / edge_transition_um, 0.0, 1.0)
        left_weight = left_t * left_t * (3.0 - 2.0 * left_t)
        right_weight = 1.0 - right_t * right_t * (3.0 - 2.0 * right_t)
        window = jnp.minimum(left_weight, right_weight)
    else:
        window = ((x >= pattern_start_um) & (x <= pattern_end_um)).astype(x.dtype)
    surface_depth = central_depth * window
    surface = ppa_thickness_um - surface_depth
    return y - surface, surface


def top_surface_profile(phi, grid) -> np.ndarray:
    """Extract the highest zero crossing for each x-column."""

    values = np.asarray(phi)
    y_axis = np.asarray(grid.axes[1])
    profile = np.full((values.shape[0],), np.nan, dtype=float)
    for x_index in range(values.shape[0]):
        column = values[x_index, :]
        crossings = np.nonzero((column[:-1] <= 0.0) & (column[1:] >= 0.0))[0]
        if crossings.size == 0:
            continue
        y_index = int(crossings[-1])
        phi0 = float(column[y_index])
        phi1 = float(column[y_index + 1])
        y0 = float(y_axis[y_index])
        y1 = float(y_axis[y_index + 1])
        denom = phi1 - phi0
        profile[x_index] = y0 if abs(denom) < 1e-12 else y0 - phi0 * (y1 - y0) / denom
    return profile


def depth_profile_nm(profile_um: np.ndarray, top_reference_um: float) -> np.ndarray:
    """Convert a surface-height profile to depth below a top reference."""

    return (top_reference_um - profile_um) * 1000.0


def pitch_label(pitch_um: float) -> str:
    """Return a filename-safe pitch label in nanometers."""

    pitch_nm = 1000.0 * pitch_um
    if abs(pitch_nm - round(pitch_nm)) < 1e-9:
        return f"{int(round(pitch_nm))}nm"
    return f"{pitch_nm:g}nm".replace(".", "p")


def plot_depth_overlay(ax, x_axis, initial_depth_nm, final_depth_nm) -> None:
    """Plot initial and final profiles using the AFM-style depth convention."""

    max_depth = float(np.nanmax([np.nanmax(initial_depth_nm), np.nanmax(final_depth_nm)]))
    ax.plot(x_axis, initial_depth_nm, color="#263494", linewidth=2.0)
    ax.plot(x_axis, final_depth_nm, color="#c81924", linewidth=1.2)
    ax.set_xlabel("Position (um)", fontsize=24)
    ax.set_ylabel("Depth (nm)", fontsize=24)
    ax.set_xlim(float(x_axis[0]), float(x_axis[-1]))
    ax.set_ylim(max(150.0, max_depth * 1.15), -max(5.0, max_depth * 0.05))
    ax.set_yticks(np.arange(0.0, 151.0, 30.0))
    ax.tick_params(axis="both", labelsize=20, width=1.2, length=7)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def plot_material_stack(
    ax,
    x_axis,
    surface_depth_nm,
    ppa_thickness_nm: float,
    si_thickness_nm: float,
    title: str,
) -> None:
    """Plot PPA and Si regions below an exposed surface-depth profile."""

    bottom_nm = ppa_thickness_nm + si_thickness_nm
    finite = np.isfinite(surface_depth_nm)
    si_top_nm = np.maximum(surface_depth_nm, ppa_thickness_nm)
    ax.fill_between(
        x_axis,
        si_top_nm,
        bottom_nm,
        where=finite,
        color="#8da0cb",
        alpha=0.85,
        label="Si",
    )
    ppa_mask = finite & (surface_depth_nm < ppa_thickness_nm)
    ax.fill_between(
        x_axis,
        surface_depth_nm,
        ppa_thickness_nm,
        where=ppa_mask,
        color="#66c2a5",
        alpha=0.85,
        label="PPA",
    )
    ax.plot(x_axis, surface_depth_nm, color="black", linewidth=1.0)
    ax.axhline(ppa_thickness_nm, color="0.25", linewidth=0.8)
    ax.set_title(title, fontsize=15, pad=8)
    ax.set_xlim(float(x_axis[0]), float(x_axis[-1]))
    ax.set_ylim(bottom_nm, -max(5.0, 0.04 * bottom_nm))
    ax.set_ylabel("depth (nm)", fontsize=13)
    ax.tick_params(axis="both", labelsize=12, width=1.0, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def plot_final_actual_scale(
    ax,
    x_axis_um,
    final_depth_nm,
    center_um: float,
    window_nm: float,
) -> None:
    """Plot a final-profile window with one x-nm equal to one y-nm."""

    x_nm = (np.asarray(x_axis_um) - center_um) * 1000.0
    half_window_nm = 0.5 * window_nm
    finite = np.isfinite(final_depth_nm)
    in_window = finite & (x_nm >= -half_window_nm) & (x_nm <= half_window_nm)
    if not np.any(in_window):
        raise ValueError("actual-scale window does not overlap the simulated x-domain")

    depth_max = float(np.nanmax(final_depth_nm[in_window]))
    ax.plot(x_nm[in_window], final_depth_nm[in_window], color="#c81924", linewidth=1.3)
    ax.set_title("after etch, actual scale", fontsize=15, pad=6)
    ax.set_xlabel("Position from center (nm)", fontsize=14)
    ax.set_ylabel("Depth (nm)", fontsize=14)
    ax.set_xlim(-half_window_nm, half_window_nm)
    ax.set_ylim(max(10.0, depth_max * 1.05), 0.0)
    ax.set_aspect("equal", adjustable="box", anchor="C")
    ax.tick_params(axis="both", labelsize=12, width=1.0, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def save_optional_profile_stack_plot(
    profile_stack_path: Path,
    grid,
    initial_profile,
    final_profile,
    top_reference_um: float,
    ppa_thickness_nm: float,
    si_thickness_nm: float,
) -> list[Path]:
    """Save a combined profile and material-stack plot when matplotlib is installed."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    x_axis = np.asarray(grid.axes[0])
    initial_depth_nm = depth_profile_nm(initial_profile, float(np.nanmax(initial_profile)))
    final_depth_nm = depth_profile_nm(final_profile, float(np.nanmax(final_profile)))
    initial_stack_depth_nm = depth_profile_nm(initial_profile, top_reference_um)
    final_stack_depth_nm = depth_profile_nm(final_profile, top_reference_um)

    fig = plt.figure(figsize=(13.0, 6.4), constrained_layout=True)
    gridspec = fig.add_gridspec(2, 2, width_ratios=(1.45, 1.0), hspace=0.04, wspace=0.08)
    profile_ax = fig.add_subplot(gridspec[:, 0])
    initial_stack_ax = fig.add_subplot(gridspec[0, 1])
    final_stack_ax = fig.add_subplot(gridspec[1, 1], sharex=initial_stack_ax)
    plot_depth_overlay(profile_ax, x_axis, initial_depth_nm, final_depth_nm)
    plot_material_stack(
        initial_stack_ax,
        x_axis,
        initial_stack_depth_nm,
        ppa_thickness_nm,
        si_thickness_nm,
        "initial stack",
    )
    plot_material_stack(
        final_stack_ax,
        x_axis,
        final_stack_depth_nm,
        ppa_thickness_nm,
        si_thickness_nm,
        "after etch",
    )
    initial_stack_ax.tick_params(labelbottom=False)
    final_stack_ax.set_xlabel("Position (um)", fontsize=13)
    handles, labels = final_stack_ax.get_legend_handles_labels()
    final_stack_ax.legend(handles[:2], labels[:2], loc="lower right", fontsize=11, frameon=False)
    fig.savefig(profile_stack_path, dpi=180)
    plt.close(fig)
    return [profile_stack_path]


def save_optional_actual_scale_plot(
    actual_scale_path: Path,
    grid,
    final_profile,
    top_reference_um: float,
    actual_scale_center_um: float,
    actual_scale_window_nm: float,
) -> list[Path]:
    """Save a final-profile plot using equal x/y nanometer scale."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    x_axis = np.asarray(grid.axes[0])
    final_depth_nm = depth_profile_nm(final_profile, top_reference_um)
    fig, ax = plt.subplots(figsize=(10.0, 2.6), constrained_layout=True)
    plot_final_actual_scale(
        ax,
        x_axis,
        final_depth_nm,
        actual_scale_center_um,
        actual_scale_window_nm,
    )
    fig.savefig(actual_scale_path, dpi=180)
    plt.close(fig)
    return [actual_scale_path]


def _positive(value: float, name: str) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def build_parser() -> argparse.ArgumentParser:
    parser = example_parser("run a PPA-to-Si transfer etch for a 1 um sinusoidal layer")
    parser.add_argument(
        "--pitch-um",
        type=float,
        default=1.0,
        help="primary sinusoid pitch in micrometers",
    )
    parser.add_argument(
        "--comparison-pitch-um",
        type=float,
        default=0.5,
        help="second sinusoid pitch to simulate in the same run",
    )
    parser.add_argument(
        "--ppa-thickness-nm",
        type=float,
        default=260.0,
        help="PPA layer thickness in nanometers",
    )
    parser.add_argument(
        "--si-thickness-nm",
        type=float,
        default=500.0,
        help="silicon thickness below the PPA layer in nanometers",
    )
    parser.add_argument(
        "--ppa-etch-rate-nm-per-min",
        type=float,
        default=40.0,
        help="PPA etch rate in nanometers per minute",
    )
    parser.add_argument(
        "--si-etch-rate-nm-per-min",
        type=float,
        default=30.0,
        help="silicon etch rate in nanometers per minute",
    )
    parser.add_argument(
        "--sinusoid-relief-nm",
        type=float,
        default=100.0,
        help="central sinusoid peak-to-valley relief in nanometers",
    )
    parser.add_argument(
        "--ridge-depth-nm",
        type=float,
        default=20.0,
        help="depth of the sinusoid ridge peaks below the outer top plateau",
    )
    parser.add_argument(
        "--pattern-start-um",
        type=float,
        default=1.0,
        help="x-position where the central sinusoidal field starts",
    )
    parser.add_argument(
        "--pattern-end-um",
        type=float,
        default=19.0,
        help="x-position where the central sinusoidal field ends",
    )
    parser.add_argument(
        "--edge-transition-um",
        type=float,
        default=0.12,
        help="smooth transition width between outer plateau and sinusoidal field",
    )
    parser.add_argument(
        "--etch-time-min",
        type=float,
        default=3.0,
        help="etch time in minutes",
    )
    parser.add_argument(
        "--etch-mode",
        choices=("isotropic", "anisotropic"),
        default="isotropic",
        help="isotropic normal etch or anisotropic downward directional etch",
    )
    parser.add_argument(
        "--actual-scale-window-nm",
        type=float,
        default=1500.0,
        help="x-width of the final-profile actual-scale PNG in nanometers",
    )
    parser.add_argument(
        "--actual-scale-center-um",
        type=float,
        default=None,
        help="x-center of the actual-scale PNG; default is a central trough",
    )
    parser.add_argument(
        "--num-periods",
        type=int,
        default=20,
        help="number of sinusoid periods in the simulated 2D window",
    )
    parser.add_argument(
        "--points-per-pitch",
        type=int,
        default=96,
        help="grid samples per sinusoid pitch",
    )
    parser.add_argument(
        "--air-buffer-um",
        type=float,
        default=0.08,
        help="void height above the initial peaks",
    )
    parser.add_argument(
        "--substrate-buffer-um",
        type=float,
        default=0.05,
        help="extra material depth below the expected etched surface",
    )
    parser.add_argument(
        "--cfl",
        type=float,
        default=0.35,
        help="CFL number for explicit level-set evolution",
    )
    return parser


def _run_single_pitch(args: argparse.Namespace) -> None:
    """Run one configured sinusoid pitch case."""

    _positive(args.pitch_um, "pitch_um")
    if hasattr(args, "comparison_pitch_um"):
        _positive(args.comparison_pitch_um, "comparison_pitch_um")
    _positive(args.ppa_thickness_nm, "ppa_thickness_nm")
    _positive(args.si_thickness_nm, "si_thickness_nm")
    _positive(args.ppa_etch_rate_nm_per_min, "ppa_etch_rate_nm_per_min")
    _positive(args.si_etch_rate_nm_per_min, "si_etch_rate_nm_per_min")
    if args.sinusoid_relief_nm is not None:
        _positive(args.sinusoid_relief_nm, "sinusoid_relief_nm")
    if args.ridge_depth_nm < 0.0:
        raise ValueError("ridge_depth_nm must be nonnegative")
    if args.edge_transition_um < 0.0:
        raise ValueError("edge_transition_um must be nonnegative")
    _positive(args.etch_time_min, "etch_time_min")
    _positive(args.actual_scale_window_nm, "actual_scale_window_nm")
    if args.num_periods < 1:
        raise ValueError("num_periods must be at least 1")
    if args.points_per_pitch < 16:
        raise ValueError("points_per_pitch must be at least 16")

    ppa_thickness_um = args.ppa_thickness_nm / 1000.0
    si_thickness_um = args.si_thickness_nm / 1000.0
    ppa_rate_um_per_min = args.ppa_etch_rate_nm_per_min / 1000.0
    si_rate_um_per_min = args.si_etch_rate_nm_per_min / 1000.0
    sinusoid_relief_nm = (
        args.ppa_thickness_nm - args.ridge_depth_nm
        if args.sinusoid_relief_nm is None
        else args.sinusoid_relief_nm
    )
    if sinusoid_relief_nm < 0.0:
        raise ValueError("ridge_depth_nm must not exceed ppa_thickness_nm")
    if args.ridge_depth_nm + sinusoid_relief_nm > args.ppa_thickness_nm + 1e-9:
        raise ValueError(
            "initial sinusoid must stay within the PPA layer: "
            "ridge_depth_nm + sinusoid_relief_nm must be <= ppa_thickness_nm"
        )
    sinusoid_relief_um = sinusoid_relief_nm / 1000.0
    ridge_depth_um = args.ridge_depth_nm / 1000.0
    dx = args.pitch_um / float(args.points_per_pitch)
    x_extent = args.pitch_um * float(args.num_periods)
    if args.pattern_start_um < 0.0:
        raise ValueError("pattern_start_um must be nonnegative")
    if args.pattern_end_um <= args.pattern_start_um:
        raise ValueError("pattern_end_um must be greater than pattern_start_um")
    if args.pattern_end_um > x_extent:
        raise ValueError("pattern_end_um must not exceed num_periods * pitch_um")
    if args.actual_scale_center_um is not None:
        if not 0.0 <= args.actual_scale_center_um <= x_extent:
            raise ValueError("actual_scale_center_um must be inside the x-domain")
    pattern_mid_um = 0.5 * (args.pattern_start_um + args.pattern_end_um)
    trough_index = int(np.floor((pattern_mid_um - args.pattern_start_um) / args.pitch_um))
    default_actual_scale_center_um = args.pattern_start_um + (trough_index + 0.5) * args.pitch_um
    default_actual_scale_center_um = float(
        np.clip(default_actual_scale_center_um, args.pattern_start_um, args.pattern_end_um)
    )
    actual_scale_center_um = (
        default_actual_scale_center_um
        if args.actual_scale_center_um is None
        else args.actual_scale_center_um
    )
    x_bounds = (0.0, x_extent)
    expected_vertical_drop = ppa_thickness_um + si_rate_um_per_min * args.etch_time_min
    y_min = -si_thickness_um - expected_vertical_drop - args.substrate_buffer_um
    y_max = ppa_thickness_um + args.air_buffer_um
    ny = max(32, int(np.ceil((y_max - y_min) / dx)) + 1)
    nx = args.num_periods * args.points_per_pitch + 1
    grid = make_grid_2d((x_bounds, (y_min, y_max)), (nx, ny))
    phi0, initial_surface = sinusoidal_surface_phi(
        grid,
        pitch_um=args.pitch_um,
        ppa_thickness_um=ppa_thickness_um,
        sinusoid_relief_um=sinusoid_relief_um,
        ridge_depth_um=ridge_depth_um,
        pattern_start_um=args.pattern_start_um,
        pattern_end_um=args.pattern_end_um,
        edge_transition_um=args.edge_transition_um,
    )
    y_coord = grid.coords[1]

    def velocity_fn(phi, _t):
        material_rate = jnp.where(
            y_coord >= 0.0,
            ppa_rate_um_per_min,
            jnp.where(y_coord >= -si_thickness_um, si_rate_um_per_min, 0.0),
        )
        if args.etch_mode == "anisotropic":
            normals = surface_normals(phi, grid.spacing)
            return directional_etch_rate(normals, direction=(0.0, -1.0), rate=material_rate)
        return isotropic_etch_rate(phi, material_rate)

    result = evolve_level_set(
        phi0,
        grid.spacing,
        velocity_fn=velocity_fn,
        t_final=args.etch_time_min,
        cfl=args.cfl,
        band_width=expected_vertical_drop + 3.0 * max(grid.spacing),
    )

    initial_profile = top_surface_profile(phi0, grid)
    final_profile = top_surface_profile(result.phi, grid)
    top_reference_um = ppa_thickness_um
    initial_depth_nm = depth_profile_nm(initial_profile, float(np.nanmax(initial_profile)))
    final_depth_nm = depth_profile_nm(final_profile, float(np.nanmax(final_profile)))
    initial_stack_depth_nm = depth_profile_nm(initial_profile, top_reference_um)
    final_stack_depth_nm = depth_profile_nm(final_profile, top_reference_um)
    out = output_dir(args.output_dir)
    output_suffix = getattr(args, "output_suffix", "")
    save_npz(
        out / f"sf6_sinusoid_etch{output_suffix}_2d.npz",
        phi0=phi0,
        phi=result.phi,
        x_um=grid.axes[0],
        y_um=grid.axes[1],
        initial_surface_um=initial_surface[:, 0],
        initial_profile_um=jnp.asarray(initial_profile),
        final_profile_um=jnp.asarray(final_profile),
        initial_depth_nm=jnp.asarray(initial_depth_nm),
        final_depth_nm=jnp.asarray(final_depth_nm),
        initial_stack_depth_nm=jnp.asarray(initial_stack_depth_nm),
        final_stack_depth_nm=jnp.asarray(final_stack_depth_nm),
        ppa_thickness_nm=jnp.asarray(args.ppa_thickness_nm),
        si_thickness_nm=jnp.asarray(args.si_thickness_nm),
        sinusoid_relief_nm=jnp.asarray(sinusoid_relief_nm),
        ridge_depth_nm=jnp.asarray(args.ridge_depth_nm),
        pattern_start_um=jnp.asarray(args.pattern_start_um),
        pattern_end_um=jnp.asarray(args.pattern_end_um),
        edge_transition_um=jnp.asarray(args.edge_transition_um),
        ppa_etch_rate_nm_per_min=jnp.asarray(args.ppa_etch_rate_nm_per_min),
        si_etch_rate_nm_per_min=jnp.asarray(args.si_etch_rate_nm_per_min),
        etch_time_min=jnp.asarray(args.etch_time_min),
        pitch_um=jnp.asarray(args.pitch_um),
        etch_mode=np.asarray(args.etch_mode),
        actual_scale_center_um=jnp.asarray(actual_scale_center_um),
        actual_scale_window_nm=jnp.asarray(args.actual_scale_window_nm),
    )
    plot_paths = save_optional_profile_stack_plot(
        out / f"sf6_sinusoid_profile_stack{output_suffix}_2d.png",
        grid,
        initial_profile,
        final_profile,
        top_reference_um,
        args.ppa_thickness_nm,
        args.si_thickness_nm,
    )
    plot_paths += save_optional_actual_scale_plot(
        out / f"sf6_sinusoid_actual_scale{output_suffix}_2d.png",
        grid,
        final_profile,
        top_reference_um,
        actual_scale_center_um,
        args.actual_scale_window_nm,
    )

    finite_drop = initial_profile - final_profile
    print(f"devices: {describe_devices()}")
    print(f"pitch: {1000.0 * args.pitch_um:.6g} nm")
    print(f"steps: {result.num_steps}, dt: {result.dt:.6g} min")
    print(f"PPA thickness: {args.ppa_thickness_nm:.6g} nm")
    print(f"Si thickness: {args.si_thickness_nm:.6g} nm")
    print(f"initial sinusoid relief: {sinusoid_relief_nm:.6g} nm")
    print(f"initial ridge depth: {args.ridge_depth_nm:.6g} nm")
    print(f"PPA etch rate: {args.ppa_etch_rate_nm_per_min:.6g} nm/min")
    print(f"Si etch rate: {args.si_etch_rate_nm_per_min:.6g} nm/min")
    print(f"etch mode: {args.etch_mode}")
    print(f"actual-scale center: {actual_scale_center_um:.6g} um")
    print(f"actual-scale width: {args.actual_scale_window_nm:.6g} nm")
    print(f"mean simulated vertical drop: {np.nanmean(finite_drop):.6g} um")
    print(f"max simulated vertical drop: {np.nanmax(finite_drop):.6g} um")
    for path in plot_paths:
        print(f"plot: {path}")


def run(args: argparse.Namespace) -> None:
    """Run the configured etch for the primary and comparison pitches."""

    _positive(args.pitch_um, "pitch_um")
    _positive(args.comparison_pitch_um, "comparison_pitch_um")
    base_x_extent_um = args.pitch_um * float(args.num_periods)
    pitch_cases: list[float] = []
    for pitch_um in (args.pitch_um, args.comparison_pitch_um):
        if not any(np.isclose(pitch_um, existing) for existing in pitch_cases):
            pitch_cases.append(float(pitch_um))

    for case_index, pitch_um in enumerate(pitch_cases):
        if case_index:
            print()
        case_args = argparse.Namespace(**vars(args))
        periods_float = base_x_extent_um / pitch_um
        num_periods = max(1, int(round(periods_float)))
        if not np.isclose(num_periods * pitch_um, base_x_extent_um):
            num_periods = max(1, int(np.ceil(periods_float)))
        case_args.pitch_um = pitch_um
        case_args.num_periods = num_periods
        case_args.output_suffix = f"_{pitch_label(pitch_um)}"
        _run_single_pitch(case_args)


def main(output_path=None, **overrides) -> None:
    """Run the example with defaults, optionally overriding parsed argument names."""

    args = build_parser().parse_args([])
    if output_path is not None:
        args.output_dir = output_path
    for name, value in overrides.items():
        if not hasattr(args, name):
            raise ValueError(f"unknown sf6_sinusoid_etch_2d option: {name}")
        setattr(args, name, value)
    run(args)


def main_from_cli(argv: list[str] | None = None) -> None:
    """Run the example from command-line arguments."""

    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main_from_cli()
