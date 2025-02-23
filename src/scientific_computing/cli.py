from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import typer
from tqdm import tqdm

from scientific_computing.time_dependent_diffusion import (
    Cylinder,
    Rectangle,
    RunMode,
    animate_diffusion,
    plot_solution_comparison,
)
from scientific_computing.vibrating_strings_1d.utils.animation import animate_wave
from scientific_computing.vibrating_strings_1d.utils.discretize_pde import (
    discretize_pde,
)
from scientific_computing.vibrating_strings_1d.utils.grid_initialisation import (
    Initialisation,
)

FONT_SIZE_TINY = 7
FONT_SIZE_SMALL = 8
FONT_SIZE_DEFAULT = 10
FONT_SIZE_LARGE = 12

plt.rc("font", family="Georgia")
plt.rc("font", weight="normal")  # controls default font
plt.rc("mathtext", fontset="stix")
plt.rc("font", size=FONT_SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=FONT_SIZE_DEFAULT)  # fontsize of the axes title
plt.rc("axes", labelsize=FONT_SIZE_DEFAULT)  # fontsize of the x and y labels
plt.rc("figure", labelsize=FONT_SIZE_DEFAULT)

sns.set_context(
    "paper",
    rc={
        "axes.linewidth": 0.5,
        "axes.labelsize": FONT_SIZE_LARGE,
        "axes.titlesize": FONT_SIZE_DEFAULT,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "ytick.minor.width": 0.4,
        "xtick.labelsize": FONT_SIZE_SMALL,
        "ytick.labelsize": FONT_SIZE_SMALL,
    },
)

app = typer.Typer(
    rich_markup_mode="rich",
    no_args_is_help=True,
    help="Command line interface to assignment 1 code",
)


vibrating_string = typer.Typer(
    name="string1d", no_args_is_help=True, help="Vibrating string question."
)

td_diffusion = typer.Typer(
    name="td-diffusion", no_args_is_help=True, help="Time-dependent diffusion questions"
)

app.add_typer(vibrating_string)
app.add_typer(td_diffusion)


def parse_rect_sinks(
    rectangle_sinks: list[str] | None,
) -> list[Rectangle]:
    if rectangle_sinks:
        sinks = []
        for s in rectangle_sinks:
            s = s.split(" ")
            if len(s) != 4:
                raise ValueError(
                    f"Rectangles should be in format 'x y w h', found: {s}"
                )
            x, y, w, h = s
            sinks.append((float(x), float(y), float(w), float(h)))
    else:
        sinks = []
    return sinks


@app.command()
def hello(name: str):
    """Prints 'Hello <NAME>'."""
    print(f"Hello {name}.")


@vibrating_string.command(name="plot")
def plot_vibrating_string(
    measurements: Annotated[
        list[int],
        typer.Option("--measurement", "-m", help="Timepoints to plot system state."),
    ],
    velocity: Annotated[
        float, typer.Option("--velocity", "-c", help="Propagation velocity.")
    ],
    length: Annotated[
        int, typer.Option("--length", "-s", help="Length of the string (L).")
    ] = 1,
    dt: Annotated[
        float,
        typer.Option(help="Step size in temporal dimension.", min=1e-6),
    ] = 0.001,
    dx: Annotated[
        float,
        typer.Option(
            help="Step size in spatial dimension. Must divide the string length.",
            min=1e-6,
        ),
    ] = 0.01,
    save_path: Annotated[
        Path | None,
        typer.Option(
            "--save-path",
            "-o",
            help="Filepath to save plot to (including extension)",
        ),
    ] = None,
):
    measurements = sorted(list(set(measurements)))
    if not measurements:
        raise ValueError("Measurements list should be non-empty.")

    spatial_intervals = int(length / dx)
    timesteps = max(measurements) + 1
    runtime = timesteps * dt

    fig, axes = plt.subplots(
        3, 1, figsize=(2.8, 3), layout="constrained", sharex=True, sharey=True
    )
    string_points = np.linspace(0, length, spatial_intervals + 1)
    for init, ax in zip(Initialisation, axes.flatten(), strict=True):
        result = discretize_pde(
            spatial_intervals,
            timesteps,
            length,
            runtime,
            velocity,
            init,
        )

        for frame in measurements:
            ax.plot(string_points, result[frame], label=f"t={frame * dt:.2f}")

        ax.set_xlim(0, 1)
        ax.set_ylabel(r"$\Psi(x;t)$")
        ax.set_title(init.as_equation_str())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.supxlabel(r"$x$")
    fig.legend(
        [r"$t=$" + f"{m * dt:.2f}" for m in measurements],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    if save_path:
        fig.savefig(save_path, dpi=500, bbox_inches="tight")
    else:
        plt.show()


@vibrating_string.command(name="animate")
def animate_vibrating_string(
    case: Annotated[
        Initialisation, typer.Option(help="String initialisation")
    ] = Initialisation.LowFreq,
    spatial_intervals: Annotated[
        int,
        typer.Option(
            "--spatial-intervals",
            "-si",
            help="Number of intervals to split spatial dimension.",
        ),
    ] = 50,
    temporal_intervals: Annotated[
        int,
        typer.Option(
            "--temporal-intervals",
            "-ti",
            help="Number of intervals to split time dimension.",
        ),
    ] = 100,
    runtime: Annotated[
        float,
        typer.Option(
            "--runtime",
            "-t",
            help="Clock time length of simulation. Equal to temporal intervals * dt",
        ),
    ] = 15.0,
    string_length: Annotated[
        float,
        typer.Option(
            "--string-length",
            "-s",
            help="Length of the string. Equal to spatial intervals * ds",
        ),
    ] = 1.0,
    propagation_velocity: Annotated[
        float,
        typer.Option("-c", help="Propagation velocity"),
    ] = 1.0,
    save_path: Annotated[
        Path | None,
        typer.Option(
            "--save-path",
            "-o",
            help="Filepath to save animation to (including extension)",
        ),
    ] = None,
    framerate: Annotated[
        int,
        typer.Option(
            "--framerate", help="Number of frames to show per second in the animation"
        ),
    ] = 5,
):
    """Animate a 1D vibrating string."""
    animation = animate_wave(
        spatial_intervals=spatial_intervals,
        time_steps=temporal_intervals,
        runtime=runtime,
        string_length=string_length,
        propagation_velocity=propagation_velocity,
        case=case,
        frame_rate=framerate,
    )
    if save_path is not None:
        animation.save(save_path)
    else:
        plt.show()


@td_diffusion.command()
def jacobi(
    n: Annotated[
        int,
        typer.Option("-n", help="Number of intervals to divide spatial domain into."),
    ] = 50,
    diffusivity: Annotated[
        float,
        typer.Option("--diffusivity", "-d", help="Diffusivity coefficient"),
    ] = 1.0,
    epsilon: Annotated[
        float,
        typer.Option("--epsilon", "-e", help="Convergence threshold."),
    ] = 1e-6,
    max_iters: Annotated[
        int,
        typer.Option("--max-iters", help="Maximum iterations before termination."),
    ] = 100_000,
    rectangle_sinks: Annotated[
        list[str] | None,
        typer.Option("--sink-rect", help="Location of a rectangular sink: 'x y w h'."),
    ] = None,
):
    sinks = parse_rect_sinks(rectangle_sinks)
    cylinder = Cylinder(spatial_intervals=n, diffusivity=diffusivity, sinks=sinks)
    iters = cylinder.solve_jacobi(epsilon=epsilon, max_iters=max_iters)
    print(f"Converged in {iters} iterations" if iters else "Didn't converge")
    plt.imshow(cylinder.grid)
    plt.show()


@td_diffusion.command()
def gauss_seidel(
    n: Annotated[
        int,
        typer.Option("-n", help="Number of intervals to divide spatial domain into."),
    ] = 50,
    diffusivity: Annotated[
        float,
        typer.Option("--diffusivity", "-d", help="Diffusivity coefficient"),
    ] = 1.0,
    epsilon: Annotated[
        float,
        typer.Option("--epsilon", "-e", help="Convergence threshold."),
    ] = 1e-6,
    max_iters: Annotated[
        int,
        typer.Option("--max-iters", help="Maximum iterations before termination."),
    ] = 100_000,
    rectangle_sinks: Annotated[
        list[str] | None,
        typer.Option("--sink-rect", help="Location of a rectangular sink: 'x y w h'."),
    ] = None,
):
    sinks = parse_rect_sinks(rectangle_sinks)
    cylinder = Cylinder(spatial_intervals=n, diffusivity=diffusivity, sinks=sinks)
    iters = cylinder.solve_gauss_seidel(epsilon=epsilon, max_iters=max_iters)
    print(f"Converged in {iters} iterations" if iters else "Didn't converge")
    plt.imshow(cylinder.grid)
    plt.show()


@td_diffusion.command()
def sor(
    omega: Annotated[
        float,
        typer.Option(
            "--omega",
            "-w",
        ),
    ],
    n: Annotated[
        int,
        typer.Option("-n", help="Number of intervals to divide spatial domain into."),
    ] = 50,
    diffusivity: Annotated[
        float,
        typer.Option("--diffusivity", "-d", help="Diffusivity coefficient"),
    ] = 1.0,
    epsilon: Annotated[
        float,
        typer.Option("--epsilon", "-e", help="Convergence threshold."),
    ] = 1e-5,
    max_iters: Annotated[
        int,
        typer.Option("--max-iters", help="Maximum iterations before termination."),
    ] = 100_000,
    rectangle_sinks: Annotated[
        list[str] | None,
        typer.Option("--sink-rect", help="Location of a rectangular sink: 'x y w h'."),
    ] = None,
):
    sinks = parse_rect_sinks(rectangle_sinks)
    cylinder = Cylinder(spatial_intervals=n, diffusivity=diffusivity, sinks=sinks)
    iters = cylinder.solve_sor(omega=omega, epsilon=epsilon, max_iters=max_iters)
    print(f"Converged in {iters} iterations" if iters else "Didn't converge")
    plt.imshow(cylinder.grid)
    plt.show()


@td_diffusion.command()
def optimal_sor(
    min_omega: Annotated[
        float, typer.Option("--min-omega", help="Minimum omega to test")
    ],
    max_omega: Annotated[
        float, typer.Option("--max-omega", help="Maximum omega to test")
    ],
    n_omegas: Annotated[
        int, typer.Option("--n-omega", help="Number of omega values to test")
    ],
    grid_sizes: Annotated[list[int], typer.Option("-n", help="Grid sizes to test")],
    diffusivity: Annotated[
        float,
        typer.Option("--diffusivity", "-d", help="Diffusivity coefficient"),
    ] = 1.0,
    epsilon: Annotated[
        float,
        typer.Option("--epsilon", "-e", help="Convergence threshold."),
    ] = 1e-6,
    max_iters: Annotated[
        int,
        typer.Option("--max-iters", help="Maximum iterations before termination."),
    ] = 100_000,
    rectangle_sinks: Annotated[
        list[str] | None,
        typer.Option("--sink-rect", help="Location of a rectangular sink: 'x y w h'."),
    ] = None,
):
    sinks = parse_rect_sinks(rectangle_sinks)
    omega_range = np.linspace(min_omega, max_omega, n_omegas)
    results = []
    with tqdm(total=len(grid_sizes) * n_omegas) as bar:
        for grid_size in grid_sizes:
            iters_till_convergence = []
            for w in omega_range:
                cylinder = Cylinder(
                    spatial_intervals=grid_size, diffusivity=diffusivity, sinks=sinks
                )
                iters = cylinder.solve_sor(
                    omega=w, epsilon=epsilon, max_iters=max_iters
                )
                iters_till_convergence.append(iters)
                bar.update()
            results.append(iters_till_convergence)

    fig, ax = plt.subplots(layout="constrained")
    for i, grid_size in enumerate(grid_sizes):
        ax.plot(omega_range, results[i], label=str(grid_size))

    ax.legend()
    ax.set_title(
        r"Number of iterations till convergence for varying grid size and $\omega$"
    )
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("Iterations till convergence")
    plt.show()


@td_diffusion.command()
def plot_timesteps(
    measurement_times: Annotated[
        list[float],
        typer.Option(
            "--measurement",
            "-m",
            help="Timepoints to plot system state, in range [0,1].",
        ),
    ],
    diffusivity: Annotated[
        float, typer.Option("--diffusivity", "-d", help="Diffusivity coefficient")
    ],
    width: Annotated[
        int, typer.Option("--width", "-w", help="Width (and height) of the surface.")
    ] = 1,
    dt: Annotated[
        float,
        typer.Option(help="Step size in temporal dimension.", min=1e-8),
    ] = 0.001,
    dx: Annotated[
        float,
        typer.Option(
            help="Step size in spatial dimension. Must divide surface width.", min=1e-8
        ),
    ] = 0.01,
    save_path: Annotated[
        Path | None,
        typer.Option(
            "--save-path",
            "-o",
            help="Filepath to save plot to (including extension)",
        ),
    ] = None,
    mode: Annotated[
        RunMode,
        typer.Option(help="Simulation mode."),
    ] = RunMode.Numba,
    rectangle_sinks: Annotated[
        list[str] | None,
        typer.Option("--sink-rect", help="Location of a rectangular sink: 'x y w h'."),
    ] = None,
):
    # Check stability condition
    if (stability_cond := (4 * dt * diffusivity) / (dx**2)) > 1:
        typer.confirm(
            f"Stability condition not met: 4*dt*D/dx^2 = {stability_cond:.2f} > 1. "
            "Do you want to proceed?",
            abort=True,
        )

    # Get sorted list of unique measurement times
    measurement_times = sorted(list(set(measurement_times)))
    if not measurement_times:
        raise ValueError("Measurements list should be non-empty.")

    spatial_intervals = int(width / dx)
    sinks = parse_rect_sinks(rectangle_sinks)

    cylinder = Cylinder(
        spatial_intervals=spatial_intervals,
        diffusivity=diffusivity,
        sinks=sinks,
    )
    grids = cylinder.measure(measurement_times=measurement_times, dt=dt, mode=mode)

    ncol = 5
    nrow = len(measurement_times) // ncol + len(measurement_times) % ncol
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(6.5, 3), layout="constrained", sharex=True, sharey=True
    )
    for i, (grid, ax) in enumerate(zip(grids, axes.flatten(), strict=False)):
        heatmap = ax.imshow(grid, extent=[0, width, 0, width], vmin=0, vmax=1)
        ax.set_title(r"$t=$ " + str(measurement_times[i]))
        ax.set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$y$")
    fig.colorbar(heatmap, ax=axes[-1], shrink=0.3)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


@td_diffusion.command(name="compare")
def compare_simulation_to_analytical(
    measurement_times: Annotated[
        list[float] | None,
        typer.Option(
            "--measurement",
            "-m",
            help=(
                "Timepoints to plot system state, in range [0,1]. "
                "Default: [0.001, 0.01, 0.1, 1.0]"
            ),
        ),
    ] = None,
    dt: Annotated[
        float,
        typer.Option(help="Time step size"),
    ] = 0.001,
    intervals: Annotated[
        int,
        typer.Option(help="Number of spatial intervals"),
    ] = 10,
    terms: Annotated[
        int,
        typer.Option(help="Number of terms to use in analytical solution"),
    ] = 100,
    mode: Annotated[
        RunMode,
        typer.Option(help="Simulation mode."),
    ] = RunMode.Numba,
    save_path: Annotated[
        Path | None,
        typer.Option(
            "--save-path",
            "-o",
            help="Filepath to save plot to (including extension)",
        ),
    ] = None,
):
    """Plot simulation vs analytical solution for time dependent diffusion."""
    measurement_times = measurement_times or [0.001, 0.01, 0.1, 1.0]
    fig = plot_solution_comparison(dt, measurement_times, intervals, terms, mode)
    if save_path:
        fig.savefig(save_path, dpi=DPI)
    else:
        plt.show()


@td_diffusion.command(name="animate")
def animate_time_dependent_diffusion(
    diffusivity: Annotated[
        float, typer.Option("--diffusivity", "-d", help="Diffusivity coefficient")
    ] = 1.0,
    dt: Annotated[float, typer.Option(help="Time step size")] = 0.00001,
    time_steps: Annotated[
        int, typer.Option(help="Number of time steps in the simulation")
    ] = 1000,
    intervals: Annotated[int, typer.Option(help="Number of spatial intervals")] = 100,
    measure_every: Annotated[
        int, typer.Option(help="Number of steps to record information.")
    ] = 5,
    mode: Annotated[
        RunMode,
        typer.Option(help="Simulation mode."),
    ] = RunMode.Numba,
    rectangle_sinks: Annotated[
        list[str] | None,
        typer.Option("--sink-rect", help="Location of a rectangular sink: 'x y w h'."),
    ] = None,
    rectangle_ins: Annotated[
        list[str] | None,
        typer.Option(
            "--ins-rect", help="Location of a rectangular insulator: 'x y w h'."
        ),
    ] = None,
):
    """Animate time dependent diffusion on a cylinder."""
    if (stability_cond := (4 * dt * diffusivity) / ((1 / intervals) ** 2)) > 1:
        typer.confirm(
            f"Stability condition not met: 4*dt*D/dx^2 = {stability_cond:.2f} > 1. "
            "Do you want to proceed?",
            abort=True,
        )
    cylinder = Cylinder(
        spatial_intervals=intervals,
        diffusivity=diffusivity,
        sinks=parse_rect_sinks(rectangle_sinks),
        insulators=parse_rect_sinks(rectangle_ins),
    )
    measurements = cylinder.measure_all(
        run_time=time_steps, dt=dt, mode=mode, measure_every=measure_every
    )

    animate_diffusion(measurements)


if __name__ == "__main__":
    app()
