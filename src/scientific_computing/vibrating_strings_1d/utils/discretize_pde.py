from vibrating_strings_1d.utils.constants import c, dt, dx
from vibrating_strings_1d.utils.grid_initialisation import initialize_grid


def discretize_pde(spatial_intervals: int, time_steps: int, case: int):
    """
    Discretizes the second order vibration PDE in order to solve numerically.
    :return: Computed wave grid over time.
    """
    grid = initialize_grid(spatial_intervals, time_steps, case)
    r = (c * dt / dx) ** 2

    for n in range(1, time_steps - 1):
        for i in range(1, spatial_intervals):
            grid[n + 1, i] = (
                r * (grid[n, i + 1] - 2 * grid[n, i] + grid[n, i - 1])
                + 2 * grid[n, i]
                - grid[n - 1, i]
            )

    return grid
