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


def add_grid_wrapping(grid: np.ndarray) -> np.ndarray:
    """
    Adds a wrapping around the existing grid based on the following rules:
    - Wrap the top with ones
    - Wrap the bottom with zeros
    _ Wrap the left and right with each other

    Args:
        grid: Existing grid (2D numpy array)

    Returns:
        Wrapped grid (2D numpy array)
    """
    _checks_grid_shape(grid)

    # Create a shell for the existing grid
    wrapped_grid = np.zeros((grid.shape[0] + 2, grid.shape[1] + 2))

    # Add the existing grid to the center of the shell leaving a layer of 0s around it
    wrapped_grid[1:-1, 1:-1] = grid

    # Change the top row
    wrapped_grid[0, :] = 1

    # Change the right and left columns
    wrapped_grid[:, 0] = wrapped_grid[:, -2]
    wrapped_grid[:, -1] = wrapped_grid[:, 1]

    return wrapped_grid


def check_new_grid(new_grid: np.ndarray, old_grid: np.ndarray) -> None:
    """
    Checks if the new computed grid has the same shape as the previous one.

    Args:
        new_grid: 2D numpy array consisting of grid values at time 't'
        old_grid: 2D numpy array consisting of grid values at time 't+1'

    Returns:
        None
    """
    if new_grid.shape != old_grid.shape:
        raise ValueError(f"The shape of new grid at 't+1' is not equal to the shape of old grid at 't'. The shape of "
                         f"the new grid is {new_grid.shape}, while the shape of the old one is {old_grid.shape}.")



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
