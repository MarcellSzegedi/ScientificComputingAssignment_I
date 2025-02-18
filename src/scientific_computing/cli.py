from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer

from scientific_computing.animation import (
    animate_diffusion,
)
from scientific_computing.time_dependent_diffusion import (
    Cylinder,
    Rectangle,
    RunMode,
    plot_solution_comparison,
    time_dependent_diffusion_numba,
)
from scientific_computing.vibrating_strings_1d.utils.animation import animate_wave
from scientific_computing.vibrating_strings_1d.utils.discretize_pde import (
    discretize_pde,
)
from scientific_computing.vibrating_strings_1d.utils.grid_initialisation import (
    Initialisation,
)

DPI = 500

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
    initialisation: Annotated[
        Initialisation, typer.Option("--init", "-i", help="Initial conditions.")
    ] = Initialisation.LowFreq,
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
    result = discretize_pde(
        spatial_intervals, timesteps, length, runtime, velocity, initialisation
    )

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    string_points = np.linspace(0, length, spatial_intervals + 1)
    for frame in measurements:
        ax.plot(string_points, result[frame], label=f"t={frame * dt:.2f}")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\Psi(x, t)$")
    ax.set_title(
        f"1D wave amplitude at varying times, c={velocity:.2f}, "
        f"{initialisation} initialisation"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=500)
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
    cylinder.solve_jacobi(epsilon=epsilon, max_iters=max_iters)
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
    cylinder.solve_gauss_seidel(epsilon=epsilon, max_iters=max_iters)
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
    cylinder.solve_sor(omega=omega, epsilon=epsilon, max_iters=max_iters)
    plt.imshow(cylinder.grid)
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

    ncol = 2
    nrow = len(measurement_times) // ncol + len(measurement_times) % ncol
    fig, axes = plt.subplots(nrow, ncol, layout="constrained", sharex=True, sharey=True)
    for grid, ax in zip(grids, axes.flatten(), strict=True):
        ax.imshow(grid, extent=[0, width, 0, width], vmin=0, vmax=1)

    if save_path:
        fig.savefig(save_path, dpi=DPI)
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
):
    """Plot simulation vs analytical solution for time dependent diffusion."""
    measurement_times = measurement_times or [0.001, 0.01, 0.1, 1.0]
    plot_solution_comparison(dt, measurement_times, intervals, terms, mode)


@td_diffusion.command(name="animate")
def animate_time_dependent_diffusion(
    diffusivity: Annotated[
        float, typer.Option("--diffusivity", "-d", help="Diffusivity coefficient")
    ] = 1.0,
    dt: Annotated[float, typer.Option(help="Time step size")] = 0.001,
    time_steps: Annotated[
        int, typer.Option(help="Number of time steps in the simulation")
    ] = 1000,
    intervals: Annotated[int, typer.Option(help="Number of spatial intervals")] = 10,
):
    """Animate time dependent diffusion on a cylinder."""
    _, grid_history = time_dependent_diffusion_numba(
        time_steps, intervals, dt, diffusivity
    )
    animate_diffusion(grid_history)


if __name__ == "__main__":
    app()
