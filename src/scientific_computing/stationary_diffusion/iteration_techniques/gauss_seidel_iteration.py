import logging

import numpy as np
import scipy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def gauss_seidel_iteration(
        old_grid: np.ndarray,
        iter_trans_mat: np.ndarray,
        output_vec_trans_mat: np.ndarray
) -> [np.ndarray, float]:
    """
    Applies one Gauss Seidel iteration to an existing grid.

    Args:
        old_grid: Existing grid (2D numpy array) where every cell value represents a delta
                    step in the discretised square field with side interval [0, 1].
        iter_trans_mat: Matrix used to compute the new grid by multiplying the previous one, applying the Gauss Seidel
                        iteration
        output_vec_trans_mat: Matrix used to compute the output vector by multiplying with the flattened old grid

    Returns:
        2D NumPy array containing the updated values after a GS iteration.
    """
    _check_trans_matrix(iter_trans_mat)

    # Calculate 'b' output vector using grid at time 't' to solve x=inv(A)b
    output_vec = calc_output_vector(old_grid, output_vec_trans_mat)
    _check_output_matrix(iter_trans_mat, output_vec)

    # Calculating the next step 't+1' in the iteration
    cell_solutions = np.matmul(iter_trans_mat, output_vec)

    # Reshape the 1D gird cell value vector to a 2D grid
    new_grid = restructure_grid(old_grid, cell_solutions)
    _check_new_grid(new_grid, old_grid)

    # Calculate the maximum deviation between grid cell values at 't+1' and 't'
    max_cell_diff = np.max(np.abs(old_grid - new_grid))

    return new_grid, max_cell_diff


def calc_iter_transformation(n: int) -> np.ndarray:
    """
    Creates matrix 'A' from the linear equation system of Ax = b for the t+1 time step
    of the gauss_seidel iteration. 'x' is the vector of variables in the grid in t+1.
    x_1 = g_11, x_2 = g_12, x_3 = g_13 etc. and b is calculated from the grid at time=t

    Args:
        n: Length of the side of the grid

    Returns:
        'A' matrix
    """
    # Initialising the matrix
    var_vector_len = n * n
    matrix = np.eye(var_vector_len, var_vector_len) * 4

    # Create repeating patters for the diagonals
    pattern_1 = np.array([-1] * n)
    pattern_2 = np.array([-1] * (n - 1) + [0])
    pattern_3 = np.array([-1] + [0] * (n - 1))

    # Creating the diagonal values
    diag_1 = np.tile(pattern_1, (var_vector_len // len(pattern_1) + 1))[
        : var_vector_len - n
    ]
    diag_2 = np.tile(pattern_2, (var_vector_len // len(pattern_2) + 1))[
        : var_vector_len - 1
    ]
    diag_3 = np.tile(pattern_3, (var_vector_len // len(pattern_3) + 1))[
        : var_vector_len - n + 1
    ]

    # Adding the diagonal values to the existing matrix
    matrix += np.diag(diag_1, k=-n)
    matrix += np.diag(diag_2, k=-1)
    matrix += np.diag(diag_3, k=n-1)

    # Calculating the inverse matrix
    inverse_matrix = scipy.sparse.linalg.inv(matrix)
    logging.info("Transformation matrix is calculated successfully for Gauss Seidel iteration.")
    return inverse_matrix


def calc_output_vector_trans_matrix(n: int) -> np.ndarray:
    """
    Computes the matrix 'B' that allows obtaining the output vector by solving the following linear equation system:
    b = Bz, where the 'b' is the output vector, and 'z' is the flattened grid cell value vector at time 't'.

    Args:
        n: Length of the side of the grid

    Returns:
        'B' matrix
    """
    # Initialising the matrix
    var_vector_len = n * n
    matrix = np.eye(var_vector_len, var_vector_len) * 4

    # Create repeating patters for the diagonals
    pattern_1 = np.array([1] * (n - 1) + [0])
    pattern_2 = np.array([1] * n)
    pattern_3 = np.array([1] + [0] * (n - 1))

    # Creating the diagonal values
    diag_1 = np.tile(pattern_1, (var_vector_len // len(pattern_1) + 1))[:var_vector_len - 1]
    diag_2 = np.tile(pattern_2, (var_vector_len // len(pattern_2) + 1))[:var_vector_len - n]
    diag_3 = np.tile(pattern_3, (var_vector_len // len(pattern_3) + 1))[:var_vector_len - n + 1]

    # Adding the diagonal values to the existing matrix
    matrix += np.diag(diag_1, k=1)
    matrix += np.diag(diag_2, k=n)
    matrix += np.diag(diag_3, k=-n + 1)

    return matrix


def calc_output_vector(grid: np.ndarray, output_trans_matrix: np.ndarray) -> np.ndarray:
    """
    Calculates the output vector ('b' in the Ax=b) for the iteration calculation for 't+1' given a grid at 't'.

    Args:
        grid: Existing grid (2D numpy array) where every cell value represents a delta
                step in the discretised square field with side interval [0, 1].

    Returns:
        Output vector (1D numpy array)
    """
    # Reshaping the grid to a 1D array to obtain 'z' vector in the b=Bz equation
    cell_vector = grid.ravel(order='C')

    # Calculating the output vector
    output_vector = np.matmul(output_trans_matrix, cell_vector)

    # Adding +1 to the first n (length of the grid) value, due to the upper boundary condition of the problem
    output_vector[:grid.shape[0]] += 1

    return output_vector


def restructure_grid(old_grid: np.ndarray, cell_solutions: np.ndarray) -> np.ndarray:
    """
    Reshapes the previously calculated 1D cell value vector into a 2D dimensional grid the same shape (not necessarily
    square) as the previous grid was at time 't'.

    Args:
        old_grid: 2D numpy array consisting of grid values at time 't'
        cell_solutions: 1D numpy array consisting of grid values at time 't+1'

    Returns:
        2D numpy array containing the updated values after a GS iteration.
    """
    # Check if reshaping is possible based on size compatibility
    if old_grid.shape[0] * old_grid.shape[1] != cell_solutions.shape[0]:
        raise ValueError(f"Reshaping the iterated grid value 1D vector is not possible because its size "
                         f"{cell_solutions.shape[0]} does not match the required "
                         f"{old_grid.shape[0] * old_grid.shape[1]} values for the grid shape {old_grid.shape}.")

    return cell_solutions.reshape(old_grid.shape[0], old_grid.shape[1])


def _check_trans_matrix(trans_matrix: np.ndarray) -> None:
    """
    Checks if the transformation matrix is squared. This requirement is applicable due to the specific lattice given in
    the problem.

    Args:
        trans_matrix: Matrix used to compute the new grid by multiplying the previous one, applying the Gauss Seidel
                        iteration

    Returns:
        None
    """
    if trans_matrix.shape[0] != trans_matrix.shape[1]:
        raise ValueError(f"Transformation matrix is not square. Its height is {trans_matrix.shape[0]}, while the width "
                         f"is {trans_matrix.shape[1]}.")


def _check_output_matrix(trans_matrix: np.ndarray, output_vec: np.ndarray) -> None:
    """
    Checks if the output vector 'b' has the same length as the width of the inverse of matrix 'A' in the linear
    equation Ax = b, where 'x' is the 1D vector consisting of the new values of the grid at time 't+1'. The match is
    required due to the matrix multiplication operation.

    Args:
        trans_matrix: Matrix used to compute the new grid by multiplying the previous one, applying the Gauss Seidel
                        iteration
        output_vec: 'b' vector of the equation Ax = b, where 'x' is the 1D vector consisting of the new values of the
                        grid at time 't+1'

    Returns:
        None
    """
    if output_vec.shape[0] != trans_matrix.shape[1]:
        raise ValueError(f"The output vector ('b' in Ax=b) has different height than the transformation matrix width. "
                         f"Output vector's height is {output_vec.shape[0]}, while the transformation matrix width is "
                         f"{trans_matrix.shape[1]}.")


def _check_new_grid(new_grid: np.ndarray, old_grid: np.ndarray) -> None:
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
