import numpy as np


def reset_grid_wrapping(grid: np.ndarray) -> np.ndarray:
    """
    Resets the grid wrapping based on the follwoing rules:
    - Wrap the top with ones
    - Wrap the bottom with zeros
    _ Wrap the left and right with each other

    Args:
        grid: Existing grid (2D numpy array) where every cell value represents a delta
            step in the discretised square field with side interval [0, 1].

    Returns:
        Updated grid (2D numpy array)
    """
    _checks_grid_shape(grid)

    # Reset top and bottom rows
    grid[0, :] = 1
    grid[-1, :] = 0

    # Reset left and right columns
    grid[:, 0] = grid[:, -2]
    grid[:, -1] = grid[:, 1]

    return grid


def _checks_grid_shape(grid: np.ndarray) -> None:
    """
    Checks if the grid is a 2D square

    Args:
        grid: Existing grid (2D numpy array) where every cell value represents a delta
            step in the discretised square field with side interval [0, 1].

    Returns:
        None
    """
    if len(grid.shape) != 2:
        raise ValueError(f"Grid must be a 2D, found {grid.ndim} dimensions.")
    if grid.shape[0] != grid.shape[1]:
        raise ValueError(
            f"Grid must be a square, while the current grid shape is {grid.shape}."
        )
