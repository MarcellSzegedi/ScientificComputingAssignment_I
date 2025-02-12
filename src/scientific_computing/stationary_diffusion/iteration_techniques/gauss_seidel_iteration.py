import numpy as np


def gauss_seidel_iteration(grid: np.ndarray, trans_matrix: np.ndarray):
    """
    Applies one Gauss Seidel iteration to an existing grid.

    Args:
        grid: Existing grid (2D numpy array) where every cell value represents a delta
            step in the discretised square field with side interval [0, 1].

    Returns:
        2D NumPy array containing the updated values after a GS iteration.
    """

    raise NotImplementedError


def det_iter_transformation(n: int) -> np.ndarray:
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
    matrix += np.diag(diag_3, k=n - 1)

    return matrix
