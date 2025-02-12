import numpy as np

from .iteration_techniques.jacobi_iteration import jacobi_iteration
from .utils.common_functions import reset_grid_wrapping
from .utils.grid_initialisation import initialize_grid


def main(
    delta: float, iteration_technique: callable, max_iter: int = 100
) -> np.ndarray:
    # Create containers
    grid_history = []

    # Initialise grid
    grid_history.append(initialize_grid(delta))

    for _ in range(max_iter):
        new_grid, req_met = iteration_technique(reset_grid_wrapping(grid_history[-1]))
        if not req_met:
            break
        grid_history.append(new_grid)

    return np.array(grid_history)


if __name__ == "__main__":
    # Testing the function

    # Settings
    delta = 0.01
    iteration_func = jacobi_iteration

    # Running the model
    results = main(delta, iteration_func)
