import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from vibrating_strings_1d.utils.discretize_pde import discretize_pde


def animate_wave(spatial_intervals, time_steps, case, frame_rate: int):
    """

    :return:
    """
    result = discretize_pde(spatial_intervals, time_steps, case)

    fig, ax = plt.subplots()
    x_axis = np.arange(result.shape[1])

    plot_axis, = ax.plot(x_axis, result[0])

    def update_plot(frame: int):
        plot_axis.set_ydata(result[frame])
        return [plot_axis]

    ani = animation.FuncAnimation(
        fig,
        update_plot,
        frames = result.shape[0],
        interval = 1000 / frame_rate,
        repeat = False,
        blit = True
    )

    return ani
