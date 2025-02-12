import numpy as np
import numpy.typing as npt


def one_step_diffusion(grid: npt.NDArray[np.float64], dt: float, dx: float, D: float):
    new_grid = grid.copy()
    diffusion_coeff = (dt * D)/(dx**2)
    for i in range(1, grid.shape[0] - 1):
        for j in range(0, grid.shape[1]):
            neighbor_concentration_diff = (
                grid[i+1, j] + # Lower neighbor
                grid[i-1, j] + # Upper neighbor
                grid[i, (j+1) % grid.shape[1]] + # Right neighbor
                grid[i, (j-1) % grid.shape[1]] - # Left neighbor
                4*grid[i, j] # Self
            )

            new_grid[i,j] = grid[i,j] + diffusion_coeff*neighbor_concentration_diff
    return new_grid

def time_dependent_diffusion(time_steps: int, intervals: int, dt: float, D: float):
    if time_steps < 1:
        raise ValueError("Time steps must be greater than 2.")

    grid = np.zeros((intervals, intervals), dtype=np.float64)
    grid[0] = 1
    dx = 1 / intervals

    for _ in range(1, time_steps):
        grid = one_step_diffusion(grid, dt, dx, D)

    return grid
