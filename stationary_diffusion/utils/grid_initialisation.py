import numpy as np

from stationary_diffusion.utils.constants import LATTICE_LENGTH


def initialize_grid(delta: float) -> np.ndarray:
    """
    Creates a grid (2D numpy array) where every value if going to represent a delta time step in the
    corresponding measure (in this case distance along x and y coordinates) from the previous value. The 'previous'
    values are in the left and down directions, while the 'future' values ind the right and up directions.

    Args:
        delta: (float) The length of discretization step of the continuous field.

    Returns:
        The final grid (2D numpy array).
    """
    # Checks delta parameter
    _check_delta_param(delta)

    # Calculate grid size
    grid_size = _calc_grid_size(delta)

    # Initialise grid
    grid = create_grid(grid_size)

    return grid


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
    grid = np.zeros(shape=(grid_size + 2, grid_size + 2))

    # Set top row to 1
    grid[0, :] = 1

    return grid


def _calc_grid_size(delta: float) -> int:
    """
    Calculates the size of the grid based on the given value of the length of the discretization step (delta).

    Args:
        delta: (float) The length of discretization step of the continuous field.

    Returns:
        The size of the grid

    """
    return (int(LATTICE_LENGTH // delta)
            if LATTICE_LENGTH / delta == LATTICE_LENGTH // delta
            else int(LATTICE_LENGTH // delta + 1))


def _check_delta_param(delta: float) -> None:
    """
    Checks if the delta parameter would produce at least a 3 by 3 grid.

    Args:
        delta: (float) The length of discretization step of the continuous field.

    Returns:
        None
    """
    if delta > LATTICE_LENGTH / 3:
        raise ValueError("The delta parameter would NOT produce at least a 3 by 3 grid.")
