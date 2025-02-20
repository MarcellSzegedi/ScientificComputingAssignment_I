import logging

import numpy as np

from scientific_computing.stationary_diffusion.iteration_techniques.jacobi_iteration import jacobi_iteration
from scientific_computing.stationary_diffusion.iteration_techniques.gauss_seidel_iteration import (
    gs_iteration_sink_comp,
    gauss_seidel_iteration,
    calc_iter_transformation,
    calc_output_vector_trans_matrix
)
from scientific_computing.stationary_diffusion.iteration_techniques.sor_iteration import (sor_iteration,
                                                                                          sor_iteration_numba_imp)
from scientific_computing.stationary_diffusion.utils.common_functions import reset_grid_wrapping, add_grid_wrapping
from scientific_computing.stationary_diffusion.utils.grid_initialisation import initialize_grid, initialize_grid_numba
from scientific_computing.stationary_diffusion.plotting.diffusion_animation import (create_stat_diff_animation,
                                                                                    create_stat_diff_animation_for_all_iter)
from scientific_computing.stationary_diffusion.plotting.plotting_functions import (plot_max_cell_diff_measure,
                                                                                   plot_delta_omega_connection,
                                                                                   plot_convergence)
from scientific_computing.stationary_diffusion.objects.sinks import (create_hor_single_sink_filter,
                                                                     create_hor_double_sink_filter)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")


MAX_ITER = 10000
TOLERANCE = 1e-5


def main_jacobi_iter(delta: float) -> (np.ndarray, np.ndarray, int):
    logging.info("___________________________________________________________________________________________")
    logging.info("Jacobi Simulation Started")

    # Create containers
    grid_history = []
    max_cell_diff_hist = []

    # Initialise grid
    initial_grid, _ = initialize_grid(delta)
    grid_history.append(initial_grid)

    iter_num = 0
    for i in range(MAX_ITER):
        new_grid, max_cell_diff = jacobi_iteration(grid_history[-1])
        grid_history.append(add_grid_wrapping(new_grid))
        max_cell_diff_hist.append(max_cell_diff)

        if i % 100 == 0:
            logging.info(f"Iteration: {i} DONE")

        if max_cell_diff < TOLERANCE:
            logging.info(f"The maximum difference between iteration steps is negligible; therefore, the iteration loop "
                         f"has stopped. The last iteration was {i}.")
            iter_num = i
            break

    logging.info("Diffusion simulation DONE")

    return np.array(grid_history), np.array(max_cell_diff_hist), iter_num if iter_num != 0 else MAX_ITER


def main_gs_iter(delta: float) -> (np.ndarray, np.ndarray, int):
    logging.info("___________________________________________________________________________________________")
    logging.info("Gauss - Seidel Simulation Started")

    # Create container
    grid_history = []
    max_cell_diff_hist = []

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
        max_cell_diff_hist.append(max_cell_diff)

        if i % 100 == 0:
            logging.info(f"Iteration: {i} DONE")

        if max_cell_diff < TOLERANCE:
            logging.info(f"The maximum difference between iteration steps is negligible; therefore, the iteration loop "
                         f"has stopped. The last iteration was {i}.")
            iter_num = i
            break

    logging.info("Diffusion simulation DONE")

    return np.array(grid_history), np.array(max_cell_diff_hist), iter_num if iter_num != 0 else MAX_ITER


def main_sor_iter(delta: float, omega: float) -> (np.ndarray, np.ndarray, int):
    logging.info("___________________________________________________________________________________________")
    logging.info("Successive Over Relaxation Simulation Started")

    # Create container
    grid_history = []
    max_cell_diff_hist = []

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
        max_cell_diff_hist.append(max_cell_diff)

        if not bool(np.all(np.abs(grid_history[-1] - grid_history[-1][:, [0]]) < 1)):
            logging.info(f"At step {i} the simulation broke.")

        if i % 100 == 0:
            logging.info(f"Iteration: {i} DONE")

        if max_cell_diff < TOLERANCE:
            logging.info(f"The maximum difference between iteration steps is negligible; therefore, the iteration loop "
                         f"has stopped. The last iteration was {i}.")
            iter_num = i
            break

    logging.info("Diffusion simulation DONE")

    return np.array(grid_history), np.array(max_cell_diff_hist), iter_num if iter_num != 0 else MAX_ITER


def main_gs_sink_comp(delta: float) -> (np.ndarray, np.ndarray, int):
    logging.info("___________________________________________________________________________________________")
    logging.info("Gauss - Seidel Simulation Started")

    # Create container
    grid_history = []
    max_cell_diff_hist = []

    # Initialise grid and sink filter
    initial_grid, grid_size = initialize_grid(delta)
    grid_history.append(initial_grid)
    sink_filter = create_hor_double_sink_filter(grid_history[-1])
    logging.info("Initialisation DONE")

    iter_num = 0
    for i in range(MAX_ITER):
        new_grid, max_cell_diff = gs_iteration_sink_comp(grid_history[-1], sink_filter)
        grid_history.append(reset_grid_wrapping(new_grid))
        max_cell_diff_hist.append(max_cell_diff)

        if i % 100 == 0:
            logging.info(f"Iteration: {i} DONE")

        if max_cell_diff < TOLERANCE:
            logging.info(f"The maximum difference between iteration steps is negligible; therefore, the iteration loop "
                         f"has stopped. The last iteration was {i}.")
            iter_num = i
            break

    logging.info("Diffusion simulation DONE")

    return np.array(grid_history), np.array(max_cell_diff_hist), iter_num if iter_num != 0 else MAX_ITER


def main_sor_numba_imp(delta: float, omega: float) -> (np.ndarray, np.ndarray, int):
    logging.info("___________________________________________________________________________________________")
    logging.info("Successive Over Relaxation Simulation Started")

    # Create container
    grid_history = []
    max_cell_diff_hist = []

    # Initialise grid
    initial_grid, grid_size = initialize_grid_numba(delta)
    grid_history.append(initial_grid)
    logging.info("Initialisation DONE")

    iter_num = 0
    for i in range(MAX_ITER):
        new_grid, max_cell_diff = sor_iteration_numba_imp(grid_history[-1], omega)
        grid_history.append(reset_grid_wrapping(new_grid))
        max_cell_diff_hist.append(max_cell_diff)

        if not bool(np.all(np.abs(grid_history[-1] - grid_history[-1][:, [0]]) < 1)):
            logging.info(f"At step {i} the simulation broke.")

        if i % 100 == 0:
            logging.info(f"Iteration: {i} DONE")

        if max_cell_diff < TOLERANCE:
            logging.info(f"The maximum difference between iteration steps is negligible; therefore, the iteration loop "
                         f"has stopped. The last iteration was {i}.")
            iter_num = i
            break

    logging.info("Diffusion simulation DONE")

    return np.array(grid_history), np.array(max_cell_diff_hist), iter_num if iter_num != 0 else MAX_ITER




if __name__ == "__main__":
    # # Settings
    # delta = 0.02
    # omega = (1.7 + 2) / 2
    #
    # jac_grid_hist, jac_max_cell_diff, _ = main_jacobi_iter(delta)
    # gs_grid_hist, gs_max_cell_diff, _ = main_gs_iter(delta)
    # sor_grid_hist, sor_max_cell_diff, _ = main_sor_iter(delta, omega)
    #
    # plot_convergence(
    #     np.mean(jac_grid_hist, axis=2),
    #     np.mean(gs_grid_hist, axis=2),
    #     np.mean(sor_grid_hist, axis=2),
    #     "convergence_plot.png",
    #     "figures"
    # )

    # # Settings
    # delta = np.linspace(0.01, 0.04, 4)
    # omega = np.linspace(1.01, 1.99, 5)
    #
    # sor_results = {}
    # for delta_val in delta:
    #     temp_results = {}
    #     for omega_val in omega:
    #         _, _, max_iter = main_sor_iter(delta_val, omega_val)
    #         temp_results[omega_val] = max_iter
    #     sor_results[delta_val] = temp_results
    #
    # plot_delta_omega_connection(sor_results,
    #                             "delta_omega_connection.png",
    #                             "figures")

    # # Settings
    # delta = 0.02
    # omega = np.linspace(1.71, 1.99, 4)
    # sor_results = {}
    #
    # jac_grid_hist, jac_max_cell_diff, _ = main_jacobi_iter(delta)
    # gs_grid_hist, gs_max_cell_diff, _ = main_gs_iter(delta)
    #
    # for omega_val in omega:
    #     sor_grid_hist, sor_max_cell_diff, _ = main_sor_iter(delta, omega_val)
    #     sor_results[omega_val] = sor_max_cell_diff
    #
    # plot_max_cell_diff_measure(
    #     jac_max_cell_diff,
    #     gs_max_cell_diff,
    #     sor_results,
    #     "max_cell_diff_plot.png",
    #     "figures"
    # )

    # create_stat_diff_animation_for_all_iter(
    #     max(jac_grid_hist.shape[0], gs_grid_hist.shape[0], sor_grid_hist.shape[0]),
    #     "figures",
    #     "Iteration_Approaches_Comparison_50_by_50.html",
    #     jac_grid_hist,
    #     gs_grid_hist,
    #     sor_grid_hist
    # )

    # # Settings
    # delta = 0.01
    #
    # gs_grid_hist, gs_max_cell_diff, _ = main_gs_sink_comp(delta)
    #
    # create_stat_diff_animation(
    #     gs_grid_hist,
    #     "figures",
    #     "GS_sink_anim.html",
    #     "Gauss - Seidel Simulation with Sink"
    # )

    delta = 0.01
    omega = 1.95

    sor_grid_hist, _, _ = main_sor_numba_imp(delta, omega)

    alma = 1
