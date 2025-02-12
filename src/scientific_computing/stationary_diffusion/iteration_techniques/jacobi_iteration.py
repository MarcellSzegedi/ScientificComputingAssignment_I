import numpy as np

from scientific_computing.stationary_diffusion.utils.constants import TOLERANCE as tol


def jacobi_iteration(grid: np.ndarray, iters: int):
    """
    Implements the Jacobi iteration to solve the diffusion equation.

    :param iters: Maximum number of iterations to achieve convergence.
    :return: The computed solution grid of the diffusion PDE solved by the
        Jacobi iteration.
    """

    # grid = initialize_grid(delta)
    N = grid.shape[0]
    new_grid = grid.copy()
    for _ in range(iters):
        old_grid = new_grid.copy()
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                new_grid[i, j] = 0.25 * (
                    old_grid[i + 1, j]
                    + old_grid[i - 1, j]
                    + old_grid[i, j + 1]
                    + old_grid[i, j - 1]
                )
        max_diff = np.max(np.abs(new_grid - old_grid))
        if max_diff < tol:
            break
    return new_grid, max_diff

    # Creating neighbor arrays

    # req_met = False

    # upper_neighbor = grid[:-2, 1:-1]
    # lower_neighbor = grid[2:, 1:-1]
    # left_neighbor = grid[1:-1, :-2]
    # right_neighbor = grid[1:-1, 2:]

    # print(grid.shape)
    # print(upper_neighbor.shape)
    # new_grid = np.mean(upper_neighbor, lower_neighbor, left_neighbor, right_neighbor)
    # if np.max(np.abs(new_grid - grid)) < tol:
    #     req_met = True

    # return new_grid, req_met
