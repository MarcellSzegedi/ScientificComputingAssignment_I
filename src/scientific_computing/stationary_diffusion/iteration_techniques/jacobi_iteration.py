import numpy as np


def jacobi_iteration(grid: np.ndarray) -> (np.ndarray, float):
    """
    Implements the Jacobi iteration to solve the diffusion equation.

    :param iters: Maximum number of iterations to achieve convergence.
    :return: The computed solution grid of the diffusion PDE solved by the
        Jacobi iteration.
    """

    # grid = initialize_grid(delta)
    # N = grid.shape[0]
    # new_grid = grid.copy()
    # for _ in range(iters):
    #     old_grid = new_grid.copy()
    #     for i in range(1, N - 1):
    #         for j in range(1, N - 1):
    #             new_grid[i, j] = 0.25 * (
    #                 old_grid[i + 1, j]
    #                 + old_grid[i - 1, j]
    #                 + old_grid[i, j + 1]
    #                 + old_grid[i, j - 1]
    #             )
    #     max_diff = np.max(np.abs(new_grid - old_grid))
    #     if max_diff < tol:
    #         break
    # return new_grid, max_diff


    # Creating neighbor arrays
    upper_neighbor = grid[:-2, 1:-1]
    lower_neighbor = grid[2:, 1:-1]
    left_neighbor = grid[1:-1, :-2]
    right_neighbor = grid[1:-1, 2:]

    # Calculating the new grid
    new_grid = np.mean([upper_neighbor, lower_neighbor, left_neighbor, right_neighbor], axis=0)

    # Calculate the maximum deviation between grid cell values at 't+1' and 't'
    max_cell_diff = float(np.max(np.abs(new_grid - grid[1:-1, 1:-1])))

    return new_grid, max_cell_diff
