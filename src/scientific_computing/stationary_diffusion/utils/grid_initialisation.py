import numpy as np


LATTICE_LENGTH = 1


def initialize_grid(delta: float) -> (np.ndarray, int):
    """
    Creates a grid (2D numpy array) where every value is going to represent a delta
    space step (in this case distance along x and y coordinates)

    Args:
        delta: (float) The length of discretization step of the continuous field.

    Returns:
        The final grid (2D numpy array).
    """
    # Checks delta parameter
    _check_delta_param(delta)

    # Calculate grid size
    grid_size = calc_grid_size(delta)

    # Initialise grid
    grid = create_grid(grid_size)

    return grid, grid_size


def calc_grid_size(delta: float) -> int:
    """
    Calculates the size of the grid based on the given value of the length of the
    discretization step (delta).

    Args:
        delta: (float) The length of discretization step of the continuous field.

    Returns:
        The size of the grid.
    """
    return (
        int(LATTICE_LENGTH // delta)
        if LATTICE_LENGTH / delta == LATTICE_LENGTH // delta
        else int(LATTICE_LENGTH // delta + 1)
    )


def create_grid(grid_size: int) -> np.ndarray:
    """
    Create a grid based on the following rules:
    - Every value in the grid is set to 0
    - Wrap the top with ones

    Args:
        grid_size: (int) size of the grid.

    Returns:
        2D numpy array representing the grid built based on the initialisation rules.
    """
    # Initialise a 2D numpy array of the proper size
    grid = np.zeros(shape=(grid_size, grid_size))

    # Set top row to 1
    grid[0, :] = 1

    return grid


def _check_delta_param(delta: float) -> None:
    """
    Checks if the delta parameter would produce at least a 3 by 3 grid.

    Args:
        delta: (float) The length of discretization step of the continuous field.

    Returns:
        None
    """
    if delta > LATTICE_LENGTH / 3:
        raise ValueError(
            "The delta parameter would NOT produce at least a 3 by 3 grid."
        )
