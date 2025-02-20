from typing import Dict
import os

import numpy as np
import matplotlib.pyplot as plt

from scientific_computing.stationary_diffusion.utils.grid_initialisation import calc_grid_size

TOLERANCE = 1e-5


def plot_max_cell_diff_measure(
        jacobi_result: np.ndarray,
        gs_result: np.ndarray,
        sor_results: Dict[float, np.ndarray],
        file_name: str,
        save_dir: str
) -> None:
    max_iter_len = max([len(jacobi_result), len(gs_result)] + [len(sor_result) for sor_result in sor_results.values()])
    plt.figure(figsize=(10, 10))

    plt.plot(_calc_x_axis(jacobi_result), jacobi_result, label="Jacobi")
    plt.plot(_calc_x_axis(gs_result), gs_result, label="Gauss - Seidel")

    for omega, sor_result in sor_results.items():
        plt.plot(_calc_x_axis(sor_result), sor_result, label=f"SOR ω: {omega:.2f}")

    plt.plot(np.arange(1, int((max_iter_len) * 1.1) + 1),
             np.ones_like(np.arange(1, int((max_iter_len) * 1.1) + 1)) * TOLERANCE,
             linestyle="--",
             color="darkred",
             linewidth=4,
             alpha=0.5
             )

    plt.title("Maximum Difference Between Iteration Steps\nfor Different Iterative Methods",
              fontsize=16,
              fontweight="bold")
    plt.xlabel("Iteration Steps")
    plt.ylabel("Maximum Difference Between Iterations")
    plt.legend(fontsize=12)

    plt.yscale("log")

    plt.savefig(os.path.join(save_dir, file_name), dpi=500)
    plt.show()


def plot_delta_omega_connection(
        sor_results: Dict[float, Dict[float, float]],
        file_name: str,
        save_dir: str
) -> None:
    plt.figure(figsize=(10, 10))
    for delta, sor_result_collection in sor_results.items():
        omega = list(sor_result_collection.keys())
        results = list(sor_result_collection.values())
        grid_size = calc_grid_size(delta)
        plt.plot(omega, results, label=f"{grid_size}")

    plt.title("Iteration Steps to Convergence\nas a Function of Grid Size and SOR's 'ω' Parameter",
              fontsize=16,
              fontweight="bold")
    plt.xlabel("Omega Parameter")
    plt.ylabel("Number of Iterations Needed to Finish")
    plt.grid(axis='y', linestyle='--', color='gray', alpha=0.7)

    plt.legend(fontsize=12, title="Grid Size")

    plt.savefig(os.path.join(save_dir, file_name), dpi=500)
    plt.show()


def plot_convergence(
        jacobi_result: np.ndarray,
        gs_result: np.ndarray,
        sor_result: np.ndarray,
        file_name: str,
        save_dir: str
) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(30, 10))

    # Jacobi
    jacobi_lines = []
    for i, iter_grid in enumerate(jacobi_result):
        if _select_iter_steps_to_plot(i, jacobi_result):
            line, = ax[0].plot(np.arange(iter_grid.shape[0]), iter_grid[::-1], label=f"{int((i / jacobi_result.shape[0]) * 100)} %")
            jacobi_lines.append(line)
    for i, line in enumerate(jacobi_lines):
        _add_label_above_line(ax[0], line, i, len(jacobi_lines))
    ax[0].plot(np.linspace(0, jacobi_result.shape[1] - 1, 10), np.linspace(0, 1, 10), linestyle="--", color="black", linewidth=3)

    # Gauss-Seidel
    gs_lines = []
    for i, iter_grid in enumerate(gs_result):
        if _select_iter_steps_to_plot(i, gs_result):
            line, = ax[1].plot(np.arange(iter_grid.shape[0]), iter_grid[::-1], label=f"{int((i / gs_result.shape[0]) * 100)} %")
            gs_lines.append(line)
    for i, line in enumerate(gs_lines):
        _add_label_above_line(ax[1], line, i, len(gs_lines))
    ax[1].plot(np.linspace(0, gs_result.shape[1] - 1, 10), np.linspace(0, 1, 10), linestyle="--", color="black", linewidth=3)

    # SOR
    sor_lines = []
    for i, iter_grid in enumerate(sor_result):
        if _select_iter_steps_to_plot(i, sor_result):
            line, = ax[2].plot(np.arange(iter_grid.shape[0]), iter_grid[::-1], label=f"{int((i / sor_result.shape[0]) * 100)} %")
            sor_lines.append(line)
    for i, line in enumerate(sor_lines):
        _add_label_above_line(ax[2], line, i, len(sor_lines))
    ax[2].plot(np.linspace(0, sor_result.shape[1] - 1, 10), np.linspace(0, 1, 10), linestyle="--", color="black", linewidth=3)

    # Titles and labels
    fig.suptitle(f"Heat Distribution along the Object for Various Iterations", fontsize=18, fontweight="bold")
    ax[0].set_title("Jacobi Iteration", fontsize=12)
    ax[1].set_title("Gauss-Seidel Iteration", fontsize=12)
    ax[2].set_title("Successive Over Relaxation", fontsize=12)
    for i in range(3):
        ax[i].set_xlabel("Object")
        ax[i].set_ylabel("Temperature")

    # Save and show
    plt.savefig(os.path.join(save_dir, file_name), dpi=500)
    plt.show()


def _calc_x_axis(result: np.ndarray) -> np.ndarray:
    return np.arange(1, len(result) + 1)


def _select_iter_steps_to_plot(i: int, iteration_ts: np.ndarray) -> bool:
    steps_to_plot = np.array([0.0001, 0.1, 0.25, 0.99])
    steps_to_plot = steps_to_plot * iteration_ts.shape[0]
    steps_to_plot = steps_to_plot.astype(int) + 1
    if i in steps_to_plot:
        return True
    return False


def _add_label_above_line(axis, line, index, total_lines):
    """Places a bold label above the line, shifting left as more labels are added, and further from the line."""
    x_data = line.get_xdata()
    y_data = line.get_ydata()

    if len(x_data) == 0 or len(y_data) == 0:
        return  # Avoid empty data issues

    # Compute label position: shift left as index increases
    fraction = 4/5 - (index / total_lines) * (3/5)  # Moves from 4/5 to 1/5 of x-range
    idx = int(len(x_data) * fraction)

    x_pos = x_data[idx]
    y_pos = y_data[idx] * 1.1  # Slightly above the line and **further away**

    # Angle adjustment: use the slope near the label position
    if idx < len(x_data) - 1:
        dx = x_data[idx + 1] - x_data[idx]
        dy = y_data[idx + 1] - y_data[idx]
        slope_angle = np.rad2deg(np.arctan2(dy, dx))
    else:
        slope_angle = 0  # Flat if at the end

    # Add the label at the computed position, rotated to follow the line
    axis.text(int(x_pos * 0.97), y_pos, line.get_label(),
              fontsize=12, fontweight="bold", color=line.get_color(),
              ha="center", va="bottom", rotation=slope_angle)
