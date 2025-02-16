import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numba import njit
from scipy.special import erfc


@njit
def one_step_diffusion(
    grid: npt.NDArray[np.float64],
    buffer: npt.NDArray[np.float64],
    dt: float,
    dx: float,
    D: float,
):
    diffusion_coeff = (dt * D) / (dx**2)
    for i in range(1, grid.shape[0] - 1):
        for j in range(0, grid.shape[1]):
            neighbor_concentration_diff = (
                grid[i + 1, j]  # Lower neighbor
                + grid[i - 1, j]  # Upper neighbor
                + grid[i, (j + 1) % grid.shape[1]]  # Right neighbor
                + grid[i, (j - 1) % grid.shape[1]]  # Left neighbor
                - 4 * grid[i, j]  # Self
            )

            buffer[i, j] = grid[i, j] + diffusion_coeff * neighbor_concentration_diff

    for i in range(1, grid.shape[0] - 1):
        for j in range(0, grid.shape[1]):
            grid[i, j] = buffer[i, j]

    return grid


# @njit
def time_dependent_diffusion(time_steps: int, intervals: int, dt: float, D: float):
    if time_steps < 1:
        raise ValueError("Time steps must be greater than 2.")

    grid_history = []
    grid = np.zeros((intervals, intervals), dtype=np.float64)
    grid[0] = 1
    dx = 1 / intervals

    buffer = np.zeros((intervals, intervals), dtype=np.float64)
    for _ in range(1, time_steps):
        grid = one_step_diffusion(grid, buffer, dt, dx, D)
        grid_history.append(grid)

    return grid, grid_history


def analytical_solution(y: float, D: float, t: float, terms: int):
    solution = 0
    for i in range(terms):
        arg1 = (1 - y + 2 * i) / (2 * np.sqrt(D * t))
        arg2 = (1 + y + 2 * i) / (2 * np.sqrt(D * t))
        solution += erfc(arg1) - erfc(arg2)
    return solution


def plot_solution_comparison(dt: float, time_steps: int, intervals: int, terms: int):
    t_range = [0.001, 0.01, 0.1, 1.0]
    y_range = np.linspace(0, 1, intervals)
    D = 1

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    simulation_result, grid_history = time_dependent_diffusion(
        time_steps, intervals, dt, D
    )

    for idx, t in enumerate(t_range):
        time_step_idx = int(t / dt)
        if time_step_idx >= len(grid_history):
            grid_at_t = grid_history[-1]
        else:
            grid_at_t = grid_history[time_step_idx]
        simulation_result_y = grid_at_t[::-1, 0]
        analytical_result_y = [analytical_solution(y, D, t, terms) for y in y_range]

        ax = axes[idx]
        ax.plot(
            y_range,
            analytical_result_y,
            label=f"Analytical (t={t})",
            color="grey",
            linestyle="dashed",
        )
        ax.plot(y_range, simulation_result_y, label=f"Simulation (t={t})", color="blue")
        ax.set_title(f"Diffusion Profile at t={t}")
        ax.set_xlabel("y")
        ax.set_ylabel("Concentration (c)")
        ax.legend()
    plt.show()


def is_stable_scheme(dt: float, dx: float, diffusivity: float) -> bool:
    """Test if discretized diffusion scheme is stable."""
    return 4 * dt * diffusivity / dx**2 <= 1
