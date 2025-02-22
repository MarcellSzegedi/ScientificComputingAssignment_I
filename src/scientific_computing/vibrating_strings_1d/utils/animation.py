import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from .discretize_pde import discretize_pde


def animate_wave(
    spatial_intervals,
    time_steps,
    runtime: float,
    case: int,
    string_length: float = 1.0,
    frame_rate: int = 5,
    propagation_velocity: float = 1.0,
):
    """
    Animates the propagation of a wave using a numerical solution to the wave equation.
    :return: Animation object displaying wave evolution over time.
    """
    result = discretize_pde(
        spatial_intervals,
        time_steps,
        string_length,
        runtime,
        propagation_velocity,
        case,
    )

    fig, ax = plt.subplots()
    x_axis = np.linspace(0, string_length, spatial_intervals + 1)

    (plot_axis,) = ax.plot(x_axis, result[0])

    def update_plot(frame: int):
        plot_axis.set_ydata(result[frame])
        return [plot_axis]

    ani = animation.FuncAnimation(
        fig,
        update_plot,
        frames=result.shape[0],
        interval=1000 / frame_rate,
        repeat=False,
        blit=True,
    )

    return ani
