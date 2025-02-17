from enum import StrEnum

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numba import njit
from scipy.special import erfc

from scientific_computing._core import td_diffusion_cylinder


class RunMode(StrEnum):
    Python = "python"
    Numba = "numba"
    Rust = "rust"


class Cylinder:
    def __init__(self, spatial_intervals: int, diffusivity: float):
        self.grid = np.zeros((spatial_intervals, spatial_intervals), dtype=np.float64)
        self.grid[0] = 1.0
        self.dx = 1 / spatial_intervals
        self.diffusivity = diffusivity

    def run(self, n_iters: int, dt: float, mode: RunMode = RunMode.Python):
        if n_iters < 0:
            raise ValueError("n_iters must be positive.")

        match mode:
            case RunMode.Python:
                update = one_step_diffusion
            case RunMode.Numba:
                update = one_step_diffusion_numba
            case RunMode.Rust:
                self.grid = td_diffusion_cylinder(
                    [n_iters],
                    intervals=self.grid.shape[0],
                    dt=dt,
                    diffusivity=self.diffusivity,
                )[-1]
                return

        buffer = np.zeros_like(self.grid)
        for _ in range(n_iters):
            self.grid = update(self.grid, buffer, dt, self.dx, self.diffusivity)

    def measure(
        self, measurement_times: list[float], dt: float, mode: RunMode = RunMode.Python
    ):
        if len(measurement_times) == 0:
            raise ValueError("Must specify at least one measurement time.")
        elif min(measurement_times) < 0.0 or max(measurement_times) > 1.0:
            raise ValueError("Measurement times must be in interval [0,1]")
        measurement_idxes = set(int(t / dt) for t in measurement_times)

        match mode:
            case RunMode.Python:
                update = one_step_diffusion
            case RunMode.Numba:
                update = one_step_diffusion_numba
            case RunMode.Rust:
                measurements = td_diffusion_cylinder(
                    list(measurement_idxes),
                    intervals=self.grid.shape[0],
                    dt=dt,
                    diffusivity=self.diffusivity,
                )
                self.grid = measurements[-1].copy()
                return measurements

        measurements = []
        time_steps = max(measurement_idxes)

        buffer = np.zeros_like(self.grid)
        for t in range(time_steps + 1):
            if t in measurement_idxes:
                measurements.append(self.grid.copy())
            self.grid = update(self.grid, buffer, dt, self.dx, self.diffusivity)

        return measurements


@njit
def one_step_diffusion_numba(
    grid: npt.NDArray[np.float64],
    buffer: npt.NDArray[np.float64],
    dt: float,
    dx: float,
    D: float,
):
    diffusion_coeff = (dt * D) / (dx**2)
    for i in range(1, grid.shape[0] - 1):
        for j in range(0, grid.shape[1]):
            neighbor_concentration_diff = (
                grid[i + 1, j]  # Lower neighbor
                + grid[i - 1, j]  # Upper neighbor
                + grid[i, (j + 1) % grid.shape[1]]  # Right neighbor
                + grid[i, (j - 1) % grid.shape[1]]  # Left neighbor
                - 4 * grid[i, j]  # Self
            )

            buffer[i, j] = grid[i, j] + diffusion_coeff * neighbor_concentration_diff

    for i in range(1, grid.shape[0] - 1):
        for j in range(0, grid.shape[1]):
            grid[i, j] = buffer[i, j]

    return grid


def one_step_diffusion(
    grid: npt.NDArray[np.float64],
    buffer: npt.NDArray[np.float64],
    dt: float,
    dx: float,
    D: float,
):
    diffusion_coeff = (dt * D) / (dx**2)
    for i in range(1, grid.shape[0] - 1):
        for j in range(0, grid.shape[1]):
            neighbor_concentration_diff = (
                grid[i + 1, j]  # Lower neighbor
                + grid[i - 1, j]  # Upper neighbor
                + grid[i, (j + 1) % grid.shape[1]]  # Right neighbor
                + grid[i, (j - 1) % grid.shape[1]]  # Left neighbor
                - 4 * grid[i, j]  # Self
            )
            buffer[i, j] = grid[i, j] + diffusion_coeff * neighbor_concentration_diff

    grid[1:-1, ...] = buffer[1:-1, ...]
    return grid


def analytical_solution(y: float, D: float, t: float, terms: int):
    solution = 0
    for i in range(terms):
        arg1 = (1 - y + 2 * i) / (2 * np.sqrt(D * t))
        arg2 = (1 + y + 2 * i) / (2 * np.sqrt(D * t))
        solution += erfc(arg1) - erfc(arg2)
    return solution


def plot_solution_comparison(
    dt: float,
    measurement_times: list[float],
    intervals: int,
    terms: int,
    mode: RunMode = RunMode.Python,
):
    y_range = np.linspace(0, 1, intervals)
    D = 1

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    cylinder = Cylinder(spatial_intervals=intervals, diffusivity=D)
    measurements = cylinder.measure(
        measurement_times=measurement_times, dt=dt, mode=mode
    )

    for idx, t in enumerate(measurement_times):
        grid_at_t = measurements[idx]
        simulation_result_y = grid_at_t[::-1, 0]
        analytical_result_y = [analytical_solution(y, D, t, terms) for y in y_range]

        ax = axes[idx]
        ax.plot(
            y_range,
            analytical_result_y,
            label=f"Analytical (t={t})",
            color="grey",
            linestyle="dashed",
        )
        ax.plot(y_range, simulation_result_y, label=f"Simulation (t={t})", color="blue")
        ax.set_title(f"Diffusion Profile at t={t}")
        ax.set_xlabel("y")
        ax.set_ylabel("Concentration (c)")
        ax.legend()
    plt.show()


def is_stable_scheme(dt: float, dx: float, diffusivity: float) -> bool:
    """Test if discretized diffusion scheme is stable."""
    return 4 * dt * diffusivity / dx**2 <= 1
