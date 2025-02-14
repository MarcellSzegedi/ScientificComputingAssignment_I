import numpy as np

from scientific_computing.stationary_diffusion.iteration_techniques.gauss_seidel_iteration import gauss_seidel_iteration


def sor_iteration(
        old_grid: np.ndarray,
        iter_trans_mat: np.ndarray,
        output_vec_trans_mat: np.ndarray,
        omega: float
) -> [np.ndarray, float]:
    """
    Applies one Successive Over Relaxation iteration to an existing grid.

    Args:
        old_grid: Existing grid (2D numpy array) where every cell value represents a delta
                    step in the discretised square field with side interval [0, 1].
        iter_trans_mat: Matrix used to compute the new grid by multiplying the previous one, applying the Gauss Seidel
                        iteration
        output_vec_trans_mat: Matrix used to compute the output vector by multiplying with the flattened old grid
        omega: Weight parameter for the SOR iteration

    Returns:
        2D NumPy array containing the updated values after a SOR iteration.
    """
    # Calculate the Gauss-Seidel subresult
    gs_result = gauss_seidel_iteration(old_grid, iter_trans_mat, output_vec_trans_mat)

    # Calculate the linear combination of the current (at time 't') grid and the gs_result (at time 't+1')
    new_grid = omega / 4 * gs_result + (1 - omega) * old_grid

    # Calculate the maximum deviation between grid cell values at 't+1' and 't'
    max_cell_diff = np.max(np.abs(old_grid - new_grid))

    return new_grid, max_cell_diff
