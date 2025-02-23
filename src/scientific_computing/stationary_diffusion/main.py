import logging
from typing import Optional, Union

import numpy as np

from scientific_computing.stationary_diffusion.iteration_techniques.jacobi_iteration import apply_jacobi_iter_step
from scientific_computing.stationary_diffusion.iteration_techniques.gauss_seidel_iteration import apply_gauss_seidel_iter_step
from scientific_computing.stationary_diffusion.iteration_techniques.sor_iteration import apply_sor_iter_step
from scientific_computing.stationary_diffusion.utils.grid_initialisation import initialize_grid
from scientific_computing.stationary_diffusion.plotting.diffusion_animation import (create_stat_diff_animation,
                                                                                    create_stat_diff_animation_for_all_iter)
from scientific_computing.stationary_diffusion.plotting.plotting_functions import (plot_convergence_speed,
                                                                                   plot_delta_omega_connection,
                                                                                   plot_convergence,
                                                                                   plot_delta_omega_connection_with_sink,
                                                                                   plot_convergence_speed_with_sink,
                                                                                   plot_convergence_diff_sinks)
from scientific_computing.stationary_diffusion.objects.sinks import (create_hor_single_sink_filter,
                                                                     create_hor_double_sink_filter)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")


class IterationApproaches:
    @staticmethod
    def jacobi_iteration(
            delta: float,
            sink: Optional[callable]=None,
            sink_width: Optional[float] = 0.5,
            max_iter: int=100000,
            atol: float=1e-5
    ) -> (np.ndarray, np.ndarray, int):
        logging.info("___________________________________________________________________________________________")
        logging.info("Jacobi Simulation Started")

        # Create containers
        grid_history = []
        max_cell_diff_hist = []

        # Initialise grid
        initial_grid, _ = initialize_grid(delta)
        grid_history.append(initial_grid)
        sink_filter = sink(initial_grid, sink_width) if sink is not None else None
        logging.info("Initialisation DONE")

        iter_num = 0
        for i in range(max_iter):
            new_grid, max_cell_diff = apply_jacobi_iter_step(grid_history[-1], sink_filter)
            grid_history.append(new_grid)
            max_cell_diff_hist.append(max_cell_diff)

            if i % 100 == 0:
                logging.info(f"Iteration: {i} DONE")

            if max_cell_diff < atol:
                logging.info(
                    f"The maximum difference between iteration steps is negligible; therefore, the iteration loop "
                    f"has stopped. The last iteration was {i}.")
                iter_num = i
                break

        logging.info("Diffusion simulation DONE")

        return np.array(grid_history), np.array(max_cell_diff_hist), iter_num if iter_num != 0 else max_iter

    @staticmethod
    def gauss_seidel_iteration(
            delta: float,
            sink: Optional[callable]=None,
            sink_width: Optional[float] = 0.5,
            max_iter: int=100000,
            atol: float=1e-5
    ) -> (np.ndarray, np.ndarray, int):
        logging.info("___________________________________________________________________________________________")
        logging.info("Gauss - Seidel Simulation Started")

        # Create container
        grid_history = []
        max_cell_diff_hist = []

        # Initialise grid
        initial_grid, grid_size = initialize_grid(delta)
        grid_history.append(initial_grid)
        sink_filter = sink(initial_grid, sink_width) if sink is not None else None
        logging.info("Initialisation DONE")

        iter_num = 0
        for i in range(max_iter):
            new_grid, max_cell_diff = apply_gauss_seidel_iter_step(grid_history[-1], sink_filter)
            grid_history.append(new_grid)
            max_cell_diff_hist.append(max_cell_diff)

            if i % 100 == 0:
                logging.info(f"Iteration: {i} DONE")

            if max_cell_diff < atol:
                logging.info(
                    f"The maximum difference between iteration steps is negligible; therefore, the iteration loop "
                    f"has stopped. The last iteration was {i}.")
                iter_num = i
                break

        logging.info("Diffusion simulation DONE")

        return np.array(grid_history), np.array(max_cell_diff_hist), iter_num if iter_num != 0 else max_iter

    @staticmethod
    def sor_iteration(
            delta: float,
            omega: float,
            sink: Optional[callable] = None,
            sink_width: Optional[float] = 0.5,
            max_iter: int = 100000,
            atol: float = 1e-5
    ) -> (np.ndarray, np.ndarray, int):
        logging.info("___________________________________________________________________________________________")
        logging.info("Successive Over Relaxation Simulation Started")

        # Create container
        grid_history = []
        max_cell_diff_hist = []

        # Initialise grid and sink(s) if applicable
        initial_grid, grid_size = initialize_grid(delta)
        grid_history.append(initial_grid)
        sink_filter = sink(initial_grid, sink_width) if sink is not None else None
        logging.info("Initialisation DONE")

        iter_num = 0
        for i in range(max_iter):
            new_grid, max_cell_diff = apply_sor_iter_step(grid_history[-1], omega, sink_filter)
            grid_history.append(new_grid)
            max_cell_diff_hist.append(max_cell_diff)

            if i % 100 == 0:
                logging.info(f"Iteration: {i} DONE")

            if max_cell_diff < atol:
                logging.info(
                    f"The maximum difference between iteration steps is negligible; therefore, the iteration loop "
                    f"has stopped. The last iteration was {i}.")
                iter_num = i
                break

        logging.info("Diffusion simulation DONE")

        return np.array(grid_history), np.array(max_cell_diff_hist), iter_num if iter_num != 0 else max_iter


class Plotting(IterationApproaches):
    def __init__(
            self,
            file_name: str,
            save_dir: str,
            delta_input: Union[float, np.ndarray],
            omega_input: Optional[Union[float, np.ndarray]],
            sink: Optional[callable] = None,
            sink_width: Optional[Union[float, np.ndarray, list]] = 0.5
    ) -> None:
        super().__init__()
        self.file_name = file_name
        self.save_dir = save_dir
        self.delta_input = delta_input
        self.omega_input = omega_input
        self.sink = sink
        self.sink_width = sink_width

    @classmethod
    def plot_diffusion(
            cls,
            plot_method: str,
            file_name: str,
            save_dir: str,
            delta_input: Union[float, np.ndarray, list],
            omega_input: Optional[Union[float, np.ndarray, list]],
            sink: Optional[callable] = None,
            sink_width: Optional[Union[float, np.ndarray, list]] = 0.5,
    ) -> None:
        plot_instance = cls(file_name, save_dir, delta_input, omega_input, sink, sink_width)
        match plot_method:
            case "convergence_speed":
                plot_instance.plot_convergence_speed()
            case "convergence_speed_with_sink":
                plot_instance.plot_convergence_speed_with_sink()
            case "omega_delta_combination":
                plot_instance.plot_omega_delta_comb()
            case "temp_convergence":
                plot_instance.plot_temperature_convergence()
            case "omega_delta_combination_with_sink":
                plot_instance.plot_delta_omega_connection_with_sink()
            case "animate_diffusion":
                plot_instance.animate_diffusion()
            case "convergence_speed_sinks_with_diff_width":
                plot_instance.plot_convergence_speed_sinks_with_diff_width()
            case _:
                raise ValueError(f"Plot method {plot_method} is not valid.")

    def plot_convergence_speed(self):
        self._check_float_type(self.delta_input)
        self._check_array_type(self.omega_input)

        _, jac_max_cell_diff, _ = self.jacobi_iteration(self.delta_input)
        _, gs_max_cell_diff, _ = self.gauss_seidel_iteration(self.delta_input)

        sor_results = {}
        for omega in self.omega_input:
            _, sor_max_cell_diff, _ = self.sor_iteration(self.delta_input, omega)
            sor_results[omega] = sor_max_cell_diff

        plot_convergence_speed(jac_max_cell_diff, gs_max_cell_diff, sor_results, self.file_name, self.save_dir)

    def plot_convergence_speed_with_sink(self):
        self._check_float_type(self.delta_input)
        self._check_array_type(self.omega_input)
        self._check_float_type(self.sink_width)

        _, jac_max_cell_diff, _ = self.jacobi_iteration(self.delta_input)
        _, gs_max_cell_diff, _ = self.gauss_seidel_iteration(self.delta_input)

        sor_results = {}
        for omega in self.omega_input:
            _, sor_max_cell_diff, _ = self.sor_iteration(self.delta_input, omega)
            sor_results[omega] = sor_max_cell_diff

        _, jac_max_cell_diff_sink, _ = self.jacobi_iteration(self.delta_input, self.sink, self.sink_width)
        _, gs_max_cell_diff_sink, _ = self.gauss_seidel_iteration(self.delta_input, self.sink, self.sink_width)

        sor_results_sink = {}
        for omega in self.omega_input:
            _, sor_max_cell_diff_sink, _ = self.sor_iteration(self.delta_input, omega, self.sink, self.sink_width)
            sor_results_sink[omega] = sor_max_cell_diff_sink

        plot_convergence_speed_with_sink(
            jac_max_cell_diff,
            gs_max_cell_diff,
            sor_results,
            jac_max_cell_diff_sink,
            gs_max_cell_diff_sink,
            sor_results_sink,
            self.file_name,
            self.save_dir,
        )

    def plot_omega_delta_comb(self):
        self._check_array_type(self.delta_input)
        self._check_array_type(self.omega_input)

        sor_results = {}
        for delta in self.delta_input:
            temp_results = {}
            for omega in self.omega_input:
                _, _, max_iter = self.sor_iteration(delta, omega)
                temp_results[omega] = max_iter
            sor_results[delta] = temp_results

        for delta, omega_dict in sor_results.items():
            min_omega = min(omega_dict, key=omega_dict.get)
            print(f"For {delta}, the omega with the smallest number is {min_omega} with value {omega_dict[min_omega]}")

        plot_delta_omega_connection(
            sor_results,
            self.file_name,
            self.save_dir
        )

    def plot_temperature_convergence(self):
        self._check_float_type(self.delta_input)
        self._check_float_type(self.omega_input)

        jac_grid_hist, jac_max_cell_diff, _ = self.jacobi_iteration(self.delta_input)
        gs_grid_hist, gs_max_cell_diff, _ = self.gauss_seidel_iteration(self.delta_input)
        sor_grid_hist, sor_max_cell_diff, _ = self.sor_iteration(self.delta_input, self.omega_input)

        plot_convergence(
            np.mean(jac_grid_hist, axis=2),
            np.mean(gs_grid_hist, axis=2),
            np.mean(sor_grid_hist, axis=2),
            self.file_name,
            self.save_dir
        )

    def plot_delta_omega_connection_with_sink(self):
        self._check_array_type(self.delta_input)
        self._check_array_type(self.omega_input)
        self._check_float_type(self.sink_width)

        sor_results = {}
        for delta in self.delta_input:
            temp_results = {}
            for omega in self.omega_input:
                _, _, max_iter = self.sor_iteration(delta, omega)
                temp_results[omega] = max_iter
            sor_results[delta] = temp_results

        sor_results_sink = {}
        for delta in self.delta_input:
            temp_results = {}
            for omega in self.omega_input:
                _, _, max_iter = self.sor_iteration(delta, omega, self.sink, self.sink_width)
                temp_results[omega] = max_iter
            sor_results_sink[delta] = temp_results

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
            self.file_name,
            self.save_dir
        )

    def animate_diffusion(self):
        self._check_float_type(self.delta_input)
        self._check_float_type(self.omega_input)

        jac_grid_hist, jac_max_cell_diff, _ = self.jacobi_iteration(self.delta_input, self.sink)
        gs_grid_hist, gs_max_cell_diff, _ = self.gauss_seidel_iteration(self.delta_input, self.sink)
        sor_grid_hist, sor_max_cell_diff, _ = self.sor_iteration(self.delta_input, self.omega_input, self.sink)

        create_stat_diff_animation_for_all_iter(
            max(jac_grid_hist.shape[0], gs_grid_hist.shape[0], sor_grid_hist.shape[0]),
            self.save_dir,
            self.file_name,
            jac_grid_hist,
            gs_grid_hist,
            sor_grid_hist
        )

    def plot_convergence_speed_sinks_with_diff_width(self):
        self._check_float_type(self.delta_input)
        self._check_array_type(self.sink_width)

        results = {}
        for width in self.sink_width:
            _, max_cell_diff, _ = self.gauss_seidel_iteration(self.delta_input, self.sink, width)
            results[width] = max_cell_diff

        _, max_cell_diff, _ = self.gauss_seidel_iteration(self.delta_input)
        results[0] = max_cell_diff

        plot_convergence_diff_sinks(
            results,
            self.file_name,
            self.save_dir
        )

    @staticmethod
    def _check_array_type(input) -> None:
        if not (isinstance(input, np.ndarray) or isinstance(input, list)):
            raise ValueError(f"Input: {input} must be a numpy array or list.")

    @staticmethod
    def _check_float_type(input) -> None:
        if not isinstance(input, float):
            raise ValueError(f"Input: {input} must be a float.")


if __name__ == "__main__":
    #############################################################################
    # Question H

    delta = 0.02
    omega = (1.7 + 2) / 2
    Plotting.plot_diffusion("temp_convergence", "_temperature_linear_convergence.png", "figures", delta, omega)

    #############################################################################
    # Question I

    delta = 0.01
    omega = np.linspace(1.7, 1.99, 5)
    Plotting.plot_diffusion("convergence_speed", "_convergence_speed.png", "figures", delta, omega)

    #############################################################################
    # Question J

    delta = [1 / 200, 1 / 150, 1 / 100, 1 / 50, 1 / 25]
    omega = np.linspace(1, 1.99, 100)
    Plotting.plot_diffusion("omega_delta_combination", "_omega_delta_combinations.png", "figures", delta, omega)

    #############################################################################
    # Question K.a

    delta = 0.01
    omega = np.linspace(1.7, 1.99, 3)
    Plotting.plot_diffusion("convergence_speed_with_sink", "_convergence_with_sink.png", "figures", delta, omega,
                            create_hor_single_sink_filter)

    #############################################################################
    # Question K.b

    delta = [1 / 200, 1 / 150, 1 / 100, 1 / 50, 1 / 25]
    omega = np.linspace(1, 1.99, 100)
    Plotting.plot_diffusion("omega_delta_combination_with_sink", "_omega_delta_combinations_w_sink.png", "figures",
                            delta, omega, create_hor_single_sink_filter)

    #############################################################################
    # Question K.c

    delta = 0.01
    omega = 1.96
    sink_width = np.linspace(0.1, 0.9, 7)
    Plotting.plot_diffusion("convergence_speed_sinks_with_diff_width", "_convergence_sinks_with_diff_width.png",
                            "figures", delta, omega, create_hor_single_sink_filter, sink_width)

    #############################################################################
    # Question K.c

    delta = 0.02
    omega = 1.9
    Plotting.plot_diffusion("animate_diffusion", "_sink_animation.html", "figures", delta, omega, create_hor_single_sink_filter)

    #############################################################################
    # Question K.c

    delta = 0.02
    omega = 1.9
    Plotting.plot_diffusion("animate_diffusion", "_double_sink_animation.html", "figures", delta, omega,
                            create_hor_double_sink_filter)
