import numpy as np

from stationary_diffusion.utils.grid_initialisation import initialize_grid


def jacobi_iteration(delta: float, iters: int = 100, tol: float = 1e-5):
    """
    Implements the Jacobi iteration to solve the diffusion equation.

    :param iters: Maximum number of iterations to achieve convergence.
    :param tol: The maximum difference of the current iteration and the previous iteration for convegence.
    :return: The computed solution grid of the diffusion PDE solved by the Jacobi iteration.
    """
    grid = initialize_grid(delta)
    N = grid.shape[0]
    new_grid = grid.copy()
    for _ in range(iters):
        old_grid = new_grid.copy()
        for i in range(1, N-1):
            for j in range(1, N-1):
                new_grid[i, j] = 0.25 * (old_grid[i+1, j] + old_grid[i-1, j] + old_grid[i, j+1] + old_grid[i, j-1])
        if np.max(np.abs(new_grid - old_grid)) < tol:
            break
    return new_grid
