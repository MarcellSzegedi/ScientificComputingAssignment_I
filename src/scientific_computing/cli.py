from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer

from scientific_computing.time_dependent_diffusion import (
    one_step_diffusion,
    plot_solution_comparison,
)
from scientific_computing.vibrating_strings_1d.utils.animation import animate_wave

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


@app.command()
def hello(name: str):
    """Prints 'Hello <NAME>'."""
    print(f"Hello {name}.")


@vibrating_string.command(name="animate")
def animate_vibrating_string(
    case: Annotated[int, typer.Option(help="String initialisation")] = 1,
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
def plot_timesteps(
    measurements: Annotated[
        list[int],
        typer.Option("--measurement", "-m", help="Timepoints to plot system state."),
    ],
    diffusivity: Annotated[
        float, typer.Option("--diffusivity", "-d", help="Diffusivity coefficient")
    ],
    width: Annotated[
        int, typer.Option("--width", "-w", help="Width (and height) of the surface.")
    ] = 1,
    dt: Annotated[
        float,
        typer.Option(help="Step size in temporal dimension.", min=1e-6),
    ] = 0.001,
    dx: Annotated[
        float,
        typer.Option(
            help="Step size in spatial dimension. Must divide surface width.", min=1e-6
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
    # Check stability condition
    if (stability_cond := (4 * dt * diffusivity) / (dx**2)) > 1:
        typer.confirm(
            f"Stability condition not met: 4*dt*D/dx^2 = {stability_cond:.2f} > 1. "
            "Do you want to proceed?",
            abort=True,
        )

    # Get sorted list of unique measurement times
    measurements = sorted(list(set(measurements)))
    if not measurements:
        raise ValueError("Measurements list should be non-empty.")

    # Initialise grid
    # TODO: What if dx doesn't divide width? How do we tell with floating point
    #   arithmetic?
    spatial_intervals = int(width / dx)
    grid = np.zeros((spatial_intervals, spatial_intervals), dtype=np.float64)
    grid[0] = 1

    ncol = 2
    nrow = len(measurements) // ncol + len(measurements) % ncol
    fig, axes = plt.subplots(nrow, ncol, layout="constrained", sharex=True, sharey=True)
    flat_axes = axes.flatten()
    next_measure_idx = 0
    for frame in range(max(measurements) + 1):
        if frame == measurements[next_measure_idx]:
            flat_axes[next_measure_idx].imshow(
                grid, extent=[0, width, 0, width], vmin=0, vmax=1
            )
            next_measure_idx += 1
        grid = one_step_diffusion(grid, dt, dx, diffusivity)

    if save_path:
        fig.savefig(save_path, dpi=DPI)
    else:
        plt.show()


@td_diffusion.command(name="compare")
def compare_simulation_to_analytical(
    dt: Annotated[float, typer.Option(help="Time step size")] = 0.001,
    time_steps: Annotated[
        int, typer.Option(help="Number of time steps in the simulation")
    ] = 1000,
    intervals: Annotated[int, typer.Option(help="Number of spatial intervals")] = 10,
    terms: Annotated[
        int, typer.Option(help="Number of terms to use in analytical solution")
    ] = 100,
):
    """Plot simulation vs analytical solution for time dependent diffusion."""
    plot_solution_comparison(dt, time_steps, intervals, terms)


if __name__ == "__main__":
    app()
