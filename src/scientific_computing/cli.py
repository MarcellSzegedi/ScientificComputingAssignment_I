from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import typer

from scientific_computing.time_dependent_diffusion.utils.diffusion_sim import (
    plot_solution_comparison,
)
from scientific_computing.vibrating_strings_1d.utils.animation import animate_wave

app = typer.Typer(
    rich_markup_mode="rich",
    no_args_is_help=True,
    help="Command line interface to assignment 1 code",
)


vibrating_string = typer.Typer(
    name="string1d", no_args_is_help=True, help="Vibrating string question."
)

app.add_typer(vibrating_string)


@app.command()
def hello(name: str):
    """Prints 'Hello <NAME>'."""
    print(f"Hello {name}.")


@app.command()
def plot_time_dependent_diffusion_solution(
    time_steps: Annotated[
        int, typer.Option("--time-steps", "-ts", help="Number of time steps")
    ] = 1000,
    intervals: Annotated[
        int,
        typer.Option(
            "--intervals", "-i", help="Number of intervals to split spatial dimension,"
        ),
    ] = 50,
    dt: Annotated[float, typer.Option("--dt", "-dt", help="Time step size")] = 0.0001,
    terms: Annotated[
        int,
        typer.Option(
            "--terms",
            "-t",
            help="Number of terms to sum to approximate the analytical solution",
        ),
    ] = 100,
):
    plot_solution_comparison(dt, time_steps, intervals, terms)


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
        float, typer.Option("-c", help="Propagation velocity")
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


if __name__ == "__main__":
    app()
