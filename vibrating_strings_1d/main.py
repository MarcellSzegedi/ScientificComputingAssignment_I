from vibrating_strings_1d.utils.constants import c, dt, dx, TIME_STEPS, SPATIAL_POINTS
from vibrating_strings_1d.utils.grid_initialisation import initialize_grid


def discretize_pde():
    """
    Discretizes the second order vibration PDE in order to solve numerically.
    :return: Computed wave grid over time.
    """
    grid = initialize_grid(case)
    r = (c * dt / dx) ** 2

    for n in range(1, TIME_STEPS - 1):
        for i in range(1, SPATIAL_POINTS):
            grid[n + 1, i] = r * (grid[n, i + 1] - 2 * grid[n, i] + grid[n, i - 1]) + 2 * grid[n, i] - grid[n - 1, i]

    return grid


if __name__ == "__main__":
    case = 1
    result = discretize_pde()
    print("Simulation complete. Grid: ", result)
