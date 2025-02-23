from typing import Optional

import numba as nb
import numpy as np


@nb.njit
def apply_jacobi_iter_step(
    old_grid: np.ndarray, sink: Optional[np.ndarray] = None
) -> (np.ndarray, float):
    """
    Applies one Jacobi iteration to an existing grid.

    Args:
        old_grid: Existing grid (2D numpy array) where every cell value represents a delta
                    step in the discretised square field with side interval [0, 1].
        sink: 2D numpy array filled with boolean values which are True if the cell in the corresponding index is part
                    of the sink, False otherwise.

    Returns:
        2D NumPy array containing the updated values after a SOR iteration and the maximum difference between the new
        and the old grid
    """
    # Copying the grid not to overwrite existing values
    new_grid = old_grid.copy()

    # Apply Jacobi iteration approach
    for row_idx in range(1, old_grid.shape[0] - 1):
        for col_idx in range(0, old_grid.shape[1]):
            # Check if the cell is part of the sink
            if sink is not None and sink[row_idx, col_idx]:
                new_grid[row_idx, col_idx] = 0
            else:
                new_grid[row_idx, col_idx] = 0.25 * (
                    old_grid[row_idx - 1, col_idx]
                    + old_grid[row_idx, (col_idx - 1) % old_grid.shape[1]]
                    + old_grid[row_idx + 1, col_idx]
                    + old_grid[row_idx, (col_idx + 1) % new_grid.shape[1]]
                )

    # Calculate the maximum deviation between grid cell values at 't+1' and 't'
    max_cell_diff = np.max(np.abs(old_grid - new_grid))

    return new_grid, max_cell_diff
