import logging
from typing import Optional

import numpy as np

from scientific_computing.stationary_diffusion.iteration_techniques.jacobi_iteration import apply_jacobi_iter_step
from scientific_computing.stationary_diffusion.iteration_techniques.gauss_seidel_iteration import apply_gauss_seidel_iter_step
from scientific_computing.stationary_diffusion.iteration_techniques.sor_iteration import apply_sor_iter_step
from scientific_computing.stationary_diffusion.utils.grid_initialisation import initialize_grid
from scientific_computing.stationary_diffusion.plotting.diffusion_animation import (create_stat_diff_animation,
                                                                                    create_stat_diff_animation_for_all_iter)
from scientific_computing.stationary_diffusion.plotting.plotting_functions import (plot_max_cell_diff_measure,
                                                                                   plot_delta_omega_connection,
                                                                                   plot_convergence,
                                                                                   plot_delta_omega_connection_with_sink)
from scientific_computing.stationary_diffusion.objects.sinks import (create_hor_single_sink_filter,
                                                                     create_hor_double_sink_filter)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")


MAX_ITER = 100000
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
        new_grid, max_cell_diff = apply_jacobi_iter_step(grid_history[-1])
        grid_history.append(new_grid)
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
    logging.info("Initialisation DONE")

    iter_num = 0
    for i in range(MAX_ITER):
        new_grid, max_cell_diff = apply_gauss_seidel_iter_step(grid_history[-1])
        grid_history.append(new_grid)
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


def main_sor_iter(delta: float, omega: float, sink: Optional[callable]=None) -> (np.ndarray, np.ndarray, int):
    logging.info("___________________________________________________________________________________________")
    logging.info("Successive Over Relaxation Simulation Started")

    # Create container
    grid_history = []
    max_cell_diff_hist = []

    # Initialise grid and sink(s) if applicable
    initial_grid, grid_size = initialize_grid(delta)
    grid_history.append(initial_grid)
    sink_filter = sink(initial_grid) if sink is not None else None
    logging.info("Initialisation DONE")

    iter_num = 0
    for i in range(MAX_ITER):
        new_grid, max_cell_diff = apply_sor_iter_step(grid_history[-1], omega, sink_filter)
        grid_history.append(new_grid)
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


if __name__ == "__main__":
    # # Settings
    # delta = 0.01
    # omega = 1.9
    #
    # jac_grid_hist, jac_max_cell_diff, _ = main_jacobi_iter(delta)
    # gs_grid_hist, gs_max_cell_diff, _ = main_gs_iter(delta)
    # sor_grid_hist, sor_max_cell_diff, _ = main_sor_iter(delta, omega)
    #
    # plot_convergence(
    #     np.mean(jac_grid_hist, axis=2),
    #     np.mean(gs_grid_hist, axis=2),
    #     np.mean(sor_grid_hist, axis=2),
    #     "convergence_plot_with_sink.png",
    #     "figures"
    # )

    # # Settings
    # delta = [1/200, 1/150, 1/100, 1/50, 1/25]
    # omega = np.linspace(1, 1.99, 100)
    #
    # sor_results = {}
    # for delta_val in delta:
    #     temp_results = {}
    #     for omega_val in omega:
    #         _, _, max_iter = main_sor_iter(delta_val, omega_val)
    #         temp_results[omega_val] = max_iter
    #     sor_results[delta_val] = temp_results
    #
    # for delta, omega_dict in sor_results.items():
    #     min_omega = min(omega_dict, key=omega_dict.get)
    #     print(f"For {delta}, the omega with the smallest number is {min_omega} with value {omega_dict[min_omega]}")
    #
    # plot_delta_omega_connection(
    #     sor_results,
    #     "delta_omega_connection_new.png",
    #     "figures"
    # )

    # Settings
    delta = [1/200, 1/150, 1/100, 1/50, 1/25]
    omega = np.linspace(1, 1.99, 100)

    sor_results = {}
    for delta_val in delta:
        temp_results = {}
        for omega_val in omega:
            _, _, max_iter = main_sor_iter(delta_val, omega_val)
            temp_results[omega_val] = max_iter
        sor_results[delta_val] = temp_results

    sor_results_sink = {}
    for delta_val in delta:
        temp_results = {}
        for omega_val in omega:
            _, _, max_iter = main_sor_iter(delta_val, omega_val, create_hor_single_sink_filter)
            temp_results[omega_val] = max_iter
        sor_results_sink[delta_val] = temp_results

    print("Results without the sink:")
    for delta, omega_dict in sor_results.items():
        min_omega = min(omega_dict, key=omega_dict.get)
        print(f"For {delta}, the omega with the smallest number is {min_omega} with value {omega_dict[min_omega]}")
    print("____________________________________________________________________________________________")
    print("Results with the sink:")
    for delta, omega_dict in sor_results_sink.items():
        min_omega = min(omega_dict, key=omega_dict.get)
        print(f"For {delta}, the omega with the smallest number is {min_omega} with value {omega_dict[min_omega]}")

    plot_delta_omega_connection_with_sink(
        sor_results,
        sor_results_sink,
        "delta_omega_connection_new_with_sink.png",
        "figures"
    )


    # # Settings
    # delta = 0.02
    # omega = np.linspace(1.7, 1.99, 5)
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
    #     "max_cell_diff_plot_new.png",
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
    # omega = 1.9
    #
    # gs_grid_hist, gs_max_cell_diff, _ = main_sor_iter(delta, omega, create_hor_single_sink_filter)
    #
    # create_stat_diff_animation(
    #     gs_grid_hist,
    #     "figures",
    #     "SOR_single_sink_anim.html",
    #     "SOR Simulation with Sink"
    # )
