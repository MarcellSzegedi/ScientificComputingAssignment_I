from pathlib import Path

import matplotlib.pyplot as plt
import typer

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


@vibrating_string.command(name="animate")
def animate_vibrating_string(
    case: int,
    intervals: int,
    timesteps: int,
    save_path: Path | None = None,
    framerate: int = 5,
):
    """Animate a 1D vibrating string."""
    animation = animate_wave(intervals, timesteps, case, framerate)
    if save_path is not None:
        animation.save(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    app()
