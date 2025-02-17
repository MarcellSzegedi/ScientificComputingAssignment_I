import logging

import numpy as np

from iteration_techniques.jacobi_iteration import jacobi_iteration
from iteration_techniques.gauss_seidel_iteration import (gauss_seidel_iteration, calc_iter_transformation,
                                                         calc_output_vector_trans_matrix)
from iteration_techniques.sor_iteration import sor_iteration
from utils.common_functions import reset_grid_wrapping, add_grid_wrapping
from utils.grid_initialisation import initialize_grid
from scientific_computing.stationary_diffusion.plotting.diffusion_animation import (create_stat_diff_animation,
                                                                                    create_stat_diff_animation_for_all_iter)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")


MAX_ITER = 2000
TOLERANCE = 1e-6


def main_jacobi_iter(delta: float) -> (np.ndarray, int):
    logging.info("___________________________________________________________________________________________")
    logging.info("Jacobi Simulation Started")
    # Create containers
    grid_history = []

    # Initialise grid
    initial_grid, _ = initialize_grid(delta)
    grid_history.append(initial_grid)

    iter_num = 0
    for i in range(MAX_ITER):
        new_grid, max_cell_diff = jacobi_iteration(grid_history[-1])
        grid_history.append(add_grid_wrapping(new_grid))

        if i % 100 == 0:
            logging.info(f"Iteration: {i} DONE")

        if max_cell_diff < TOLERANCE:
            logging.info(f"The maximum difference between iteration steps is negligible; therefore, the iteration loop "
                         f"has stopped. The last iteration was {i}.")
            iter_num = i
            break

    logging.info("Diffusion simulation DONE")

    return np.array(grid_history), iter_num if iter_num != 0 else MAX_ITER


def main_gs_iter(delta: float) -> (np.ndarray, int):
    logging.info("___________________________________________________________________________________________")
    logging.info("Gauss - Seidel Simulation Started")
    # Create container
    grid_history = []

    # Initialise grid
    initial_grid, grid_size = initialize_grid(delta)
    grid_history.append(initial_grid)

    # Calculate transformation matrix
    iter_trans_mat = calc_iter_transformation(grid_size)
    output_vec_trans_mat = calc_output_vector_trans_matrix(grid_size)
    logging.info("Initialisation DONE")

    iter_num = 0
    for i in range(MAX_ITER):
        new_grid, max_cell_diff = gauss_seidel_iteration(grid_history[-1][1:-1, 1:-1], iter_trans_mat, output_vec_trans_mat)
        grid_history.append(add_grid_wrapping(new_grid))

        if i % 100 == 0:
            logging.info(f"Iteration: {i} DONE")

        if max_cell_diff < TOLERANCE:
            logging.info(f"The maximum difference between iteration steps is negligible; therefore, the iteration loop "
                         f"has stopped. The last iteration was {i}.")
            iter_num = i
            break

    logging.info("Diffusion simulation DONE")

    return np.array(grid_history), iter_num if iter_num != 0 else MAX_ITER


def main_sor_iter(delta: float, omega: float) -> (np.ndarray, int):
    logging.info("___________________________________________________________________________________________")
    logging.info("Successive Over Relaxation Simulation Started")
    # Create container
    grid_history = []

    # Initialise grid
    initial_grid, grid_size = initialize_grid(delta)
    grid_history.append(initial_grid)

    # Calculate transformation matrix
    iter_trans_mat = calc_iter_transformation(grid_size)
    output_vec_trans_mat = calc_output_vector_trans_matrix(grid_size)
    logging.info("Initialisation DONE")

    iter_num = 0
    for i in range(MAX_ITER):
        new_grid, max_cell_diff = sor_iteration(grid_history[-1][1:-1, 1:-1], iter_trans_mat, output_vec_trans_mat, omega)
        grid_history.append(add_grid_wrapping(new_grid))

        if i % 100 == 0:
            logging.info(f"Iteration: {i} DONE")

        if max_cell_diff < TOLERANCE:
            logging.info(f"The maximum difference between iteration steps is negligible; therefore, the iteration loop "
                         f"has stopped. The last iteration was {i}.")
            iter_num = i
            break

    logging.info("Diffusion simulation DONE")

    return np.array(grid_history), iter_num if iter_num != 0 else MAX_ITER


if __name__ == "__main__":
    # Settings
    delta = 0.02
    omega = 1.9

    jac_grid_hist, _ = main_jacobi_iter(delta)
    gs_grid_hist, _ = main_gs_iter(delta)
    sor_grid_hist, _ = main_sor_iter(delta, omega)
    create_stat_diff_animation_for_all_iter(
        max(jac_grid_hist.shape[0], gs_grid_hist.shape[0], sor_grid_hist.shape[0]),
        "figures",
        "Iteration_Approaches_Comparison.html",
        jac_grid_hist,
        gs_grid_hist,
        sor_grid_hist
    )
