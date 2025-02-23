from typing import Dict
import os

import numpy as np
import matplotlib.pyplot as plt

from scientific_computing.stationary_diffusion.utils.grid_initialisation import calc_grid_size

TOLERANCE = 1e-5


def plot_convergence_speed(
        jacobi_result: np.ndarray,
        gs_result: np.ndarray,
        sor_results: Dict[float, np.ndarray],
        file_name: str,
        save_dir: str
) -> None:
    max_iter_len = max([len(jacobi_result), len(gs_result)] + [len(sor_result) for sor_result in sor_results.values()])
    plt.figure(figsize=(10, 5))

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

    plt.title("Maximum Stepwise Difference for Different Iterative Methods",
              fontsize=16,
              fontweight="bold")
    plt.xlabel("Iteration Steps", fontsize=14)
    plt.ylabel("Maximum Difference Between Iterations", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=13, title="Iteration Method", title_fontsize=13)

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

    plt.title("Iteration Steps to Convergence\nas a Function of Grid Size\nand SOR's 'ω' Parameter",
              fontsize=25,
              fontweight="bold")
    plt.xlabel("ω Parameter", fontsize=18)
    plt.ylabel("Number of Iterations Needed to Finish", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis='y', linestyle='--', color='gray', alpha=0.7)

    plt.legend(fontsize=15, title="Grid Size", title_fontsize=15)

    plt.yscale("log")

    plt.savefig(os.path.join(save_dir, file_name), dpi=500)
    plt.show()


def plot_convergence(
        jacobi_result: np.ndarray,
        gs_result: np.ndarray,
        sor_result: np.ndarray,
        file_name: str,
        save_dir: str
) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(30, 6))

    # Jacobi
    jacobi_lines = []
    for i, iter_grid in enumerate(jacobi_result):
        if _select_iter_steps_to_plot(i, jacobi_result):
            label_name = int((i / jacobi_result.shape[0]) * 100) if int((i / jacobi_result.shape[0]) * 100) != 0 else 1
            line, = ax[0].plot(np.arange(iter_grid.shape[0]), iter_grid[::-1], label=f"{label_name} %")
            jacobi_lines.append(line)
    for i, line in enumerate(jacobi_lines):
        _add_label_above_line(ax[0], line, i, len(jacobi_lines))
    ax[0].plot(np.linspace(0, jacobi_result.shape[1] - 1, 10), np.linspace(0, 1, 10), linestyle="--", color="black", linewidth=3, alpha=0.5)

    # Gauss-Seidel
    gs_lines = []
    for i, iter_grid in enumerate(gs_result):
        if _select_iter_steps_to_plot(i, gs_result):
            label_name = int((i / gs_result.shape[0]) * 100) if int((i / gs_result.shape[0]) * 100) != 0 else 1
            line, = ax[1].plot(np.arange(iter_grid.shape[0]), iter_grid[::-1], label=f"{label_name} %")
            gs_lines.append(line)
    for i, line in enumerate(gs_lines):
        _add_label_above_line(ax[1], line, i, len(gs_lines))
    ax[1].plot(np.linspace(0, gs_result.shape[1] - 1, 10), np.linspace(0, 1, 10), linestyle="--", color="black", linewidth=3, alpha=0.5)

    # SOR
    sor_lines = []
    for i, iter_grid in enumerate(sor_result):
        if _select_iter_steps_to_plot(i, sor_result):
            label_name = int((i / sor_result.shape[0]) * 100) if int((i / sor_result.shape[0]) * 100) != 0 else 1
            line, = ax[2].plot(np.arange(iter_grid.shape[0]), iter_grid[::-1], label=f"{label_name} %")
            sor_lines.append(line)
    for i, line in enumerate(sor_lines):
        _add_label_above_line(ax[2], line, i, len(sor_lines))
    ax[2].plot(np.linspace(0, sor_result.shape[1] - 1, 10), np.linspace(0, 1, 10), linestyle="--", color="black", linewidth=3, alpha=0.5)

    # Titles and labels
    fig.suptitle(f"Heat Distribution along the Object for Various Iterations", fontsize=18, fontweight="bold")
    ax[0].set_title("Jacobi Iteration", fontsize=16)
    ax[1].set_title("Gauss-Seidel Iteration", fontsize=16)
    ax[2].set_title("Successive Over Relaxation Iteration", fontsize=16)
    for i in range(3):
        ax[i].set_xlabel("Object Cut Along 'y' Axis (Cell Coordinate)", fontsize=14)
        ax[i].set_ylabel("Temperature", fontsize=14)
        ax[i].tick_params(axis="both", labelsize=14)

    # Save and show
    plt.savefig(os.path.join(save_dir, file_name), dpi=500)
    plt.show()


def plot_delta_omega_connection_with_sink(
        sor_results: Dict[float, Dict[float, float]],
        sor_results_sink: Dict[float, Dict[float, float]],
        file_name: str,
        save_dir: str
) -> None:

    colors = ["red", "green", "blue", "purple", "yellow"]
    plt.figure(figsize=(10, 6))
    # Plot results without sink
    for idx, (delta, sor_result_collection) in enumerate(sor_results.items()):
        omega = list(sor_result_collection.keys())
        results = list(sor_result_collection.values())
        grid_size = calc_grid_size(delta)
        plt.plot(omega, results, label=f"{grid_size}", alpha=0.7, linestyle="--", color=colors[idx])

    # Plot results with sink
    for idx, (delta, sor_result_collection) in enumerate(sor_results_sink.items()):
        omega = list(sor_result_collection.keys())
        results = list(sor_result_collection.values())
        grid_size = calc_grid_size(delta)
        plt.plot(omega, results, label=f"{grid_size}", color=colors[idx])

    plt.title("Iteration Steps to Convergence\nas a Function of Grid Size and SOR's ω Parameter",
              fontsize=16,
              fontweight="bold")
    plt.xlabel("ω Parameter", fontsize=14)
    plt.ylabel("Number of Iterations Needed to Finish", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', color='gray', alpha=0.7)

    plt.legend(fontsize=13, title="Grid Size", title_fontsize=13, ncol=2)

    plt.yscale("log")

    plt.savefig(os.path.join(save_dir, file_name), dpi=500)
    plt.show()


def plot_convergence_speed_with_sink(
        jac_res: np.ndarray,
        gs_res: np.ndarray,
        sor_res: Dict[float, np.ndarray],
        jac_res_sink: np.ndarray,
        gs_res_sink: np.ndarray,
        sor_res_sink: Dict[float, np.ndarray],
        file_name: str,
        save_dir: str
) -> None:
    max_iter_len = max([len(jac_res), len(gs_res), len(jac_res_sink), len(gs_res_sink)]
                       + [len(sor) for sor in sor_res.values()]
                       + [len(sor) for sor in sor_res_sink.values()])
    plt.figure(figsize=(10, 6))

    sor_colors = ["green", "yellow", "magenta"]

    plt.plot(_calc_x_axis(jac_res), jac_res, label="Jacobi", color="purple", linestyle="--")
    plt.plot(_calc_x_axis(gs_res), gs_res, label="Gauss-Seidel", color="blue", linestyle="--")
    plt.plot(_calc_x_axis(jac_res_sink), jac_res_sink, label="Jacobi (with Sink)", color="purple")
    plt.plot(_calc_x_axis(gs_res_sink), gs_res_sink, label="Gauss-Seidel (with Sink)", color="blue")

    for idx, (omega, sor) in enumerate(sor_res.items()):
        plt.plot(_calc_x_axis(sor), sor, label=f"SOR ω: {omega:.2f}", color=sor_colors[idx], linestyle="--")

    for idx, (omega, sor) in enumerate(sor_res_sink.items()):
        plt.plot(_calc_x_axis(sor), sor, label=f"SOR ω: {omega:.2f} (with Sink)", color=sor_colors[idx])

    plt.plot(np.arange(1, int((max_iter_len) * 1.1) + 1),
             np.ones_like(np.arange(1, int((max_iter_len) * 1.1) + 1)) * TOLERANCE,
             linestyle="--",
             color="darkred",
             linewidth=4,
             alpha=0.5
             )

    plt.title("Maximum Stepwise Difference for Different Iterative Methods\nWith and Without Sink",
              fontsize=16,
              fontweight="bold")
    plt.xlabel("Iteration Steps", fontsize=14)
    plt.ylabel("Maximum Difference Between Iterations", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=13, title="Iteration Method", title_fontsize=13, ncol=2)

    plt.yscale("log")

    plt.savefig(os.path.join(save_dir, file_name), dpi=500)
    plt.show()


def plot_convergence_diff_sinks(
        results: Dict[float, np.ndarray],
        file_name: str,
        save_dir: str
) -> None:
    max_iter_len = max([len(sor_result) for sor_result in results.values()])
    plt.figure(figsize=(10, 5))

    plt.plot(_calc_x_axis(results[0]), results[0], label="No Sink")
    del results[0]

    for width, result in results.items():
        plt.plot(_calc_x_axis(result), result, label=f"{int(width * 100)} %")

    plt.plot(np.arange(1, int((max_iter_len) * 1.1) + 1),
             np.ones_like(np.arange(1, int((max_iter_len) * 1.1) + 1)) * TOLERANCE,
             linestyle="--",
             color="darkred",
             linewidth=4,
             alpha=0.5
             )

    plt.title("Maximum Stepwise Difference for Different Sink Widths\nUsing Gauss - Seidel approach with N=100",
              fontsize=16,
              fontweight="bold")
    plt.xlabel("Iteration Steps", fontsize=14)
    plt.ylabel("Maximum Difference Between Iterations", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=13, title="Sink Width", title_fontsize=13)

    plt.yscale("log")

    plt.savefig(os.path.join(save_dir, file_name), dpi=500)
    plt.show()


def _calc_x_axis(result: np.ndarray) -> np.ndarray:
    return np.arange(1, len(result) + 1)


def _select_iter_steps_to_plot(i: int, iteration_ts: np.ndarray) -> bool:
    steps_to_plot = np.array([0, 0.1, 0.25, 0.9999999])
    steps_to_plot = steps_to_plot * iteration_ts.shape[0]
    steps_to_plot = steps_to_plot.astype(int) + 1
    steps_to_plot[steps_to_plot > iteration_ts.shape[0] - 1] = iteration_ts.shape[0] - 1
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
              fontsize=15, fontweight="bold", color=line.get_color(),
              ha="center", va="bottom", rotation=slope_angle)
