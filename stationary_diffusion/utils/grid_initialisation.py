import numpy as np

from stationary_diffusion.utils.constants import LATTICE_LENGTH


def initialize_grid(delta: float) -> np.ndarray:
    """
    Creates a grid (2 dimensional numpy array) where every value if going to be represent a delta time step in the
    corresponding measure (in this case distance along x and y coordinates) from the previous value. The 'previous'
    values are in the left and down directions, while the 'future' values ind the right and up directions.

    Args:
        delta: (float) The length of discretization step of the continuous field.

    Returns:
        The final grid (2 dimensional numpy array).
    """
    # Calculate grid size
    grid_size = _calc_grid_size(delta)


def _calc_grid_size(delta: float) -> float:
    """
    Calculates the size of the grid based on the given value of the length of the discretization step (delta).

    Args:
        delta: (float) The length of discretization step of the continuous field.

    Returns:
        The size of the grid

    """
    return LATTICE_LENGTH // delta + 1
