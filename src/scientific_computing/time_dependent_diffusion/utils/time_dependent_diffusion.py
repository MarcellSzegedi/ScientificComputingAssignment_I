from collections.abc import Sequence
from enum import StrEnum

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numba import njit
from scipy.special import erfc

from scientific_computing._core import td_diffusion_cylinder

type Rectangle = tuple[float, float, float, float]
type Circle = tuple[float, float, float]
type DomainObjects = Sequence[Rectangle | Circle]


class RunMode(StrEnum):
    Python = "python"
    Numba = "numba"
    Rust = "rust"


class Cylinder:
    rectangle_sinks: list[tuple[int, int, int, int]]

    def __init__(
        self,
        spatial_intervals: int,
        diffusivity: float,
        sinks: DomainObjects | None = None,
        insulators: DomainObjects | None = None,
    ):
        self.grid = np.zeros(
            (spatial_intervals + 1, spatial_intervals), dtype=np.float64
        )
        self.grid[0] = 1.0
        self.dx = 1 / spatial_intervals
        self.diffusivity = diffusivity
        self.init_objects(sinks=sinks, insulators=insulators, dx=self.dx)

    def discretise_coord(self, coord: float) -> int:
        return int(coord / self.dx)

    def init_objects(
        self, sinks: DomainObjects | None, insulators: DomainObjects | None, dx: float
    ):
        def discretise_rect(rect: Rectangle) -> tuple[int, int, int, int]:
            return (
                self.discretise_coord(rect[0]),
                self.discretise_coord(rect[1]),
                self.discretise_coord(rect[2]),
                self.discretise_coord(rect[3]),
            )

        if sinks is None:
            self.rectangle_sinks = [(0, 0, 0, 0)]
            self.circle_sinks = []
        else:
            rectangle_sinks = [(0, 0, 0, 0)]
            circle_sinks = []
            for obj in sinks:
                match obj:
                    case (_, _, _, _):
                        if not all(0 <= v <= 1 for v in obj):
                            raise ValueError(
                                f"Rectangle must fit inside cylinder with dimensions "
                                f"1x1. Found rectangle of shape: {obj}."
                            )
                        rectangle_sinks.append(discretise_rect(obj))
                    case (_, _, _):
                        raise NotImplementedError
                        # circle_sinks.append(obj)
            self.rectangle_sinks = rectangle_sinks
            self.circle_sinks = circle_sinks

        if insulators is None:
            self.rectangle_ins = [(0, 0, 0, 0)]
        else:
            rectangle_ins = [(0, 0, 0, 0)]
            for obj in insulators:
                match obj:
                    case (_, _, _, _):
                        if not all(0 <= v <= 1 for v in obj):
                            raise ValueError(
                                f"Rectangle must fit inside cylinder with dimensions "
                                f"1x1. Found rectangle of shape: {obj}."
                            )
                        rectangle_ins.append(discretise_rect(obj))
                    case (_, _, _):
                        raise NotImplementedError
            self.rectangle_ins = rectangle_ins

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
                    intervals=self.grid.shape[1],
                    dt=dt,
                    diffusivity=self.diffusivity,
                    rect_sinks=self.rectangle_sinks[1:],
                )[-1]
                return

        buffer = np.zeros_like(self.grid)
        for _ in range(n_iters):
            self.grid = update(
                self.grid,
                buffer,
                dt,
                self.dx,
                self.diffusivity,
                self.rectangle_sinks,
                self.rectangle_ins,
            )

    def measure(
        self, measurement_times: list[float], dt: float, mode: RunMode = RunMode.Python
    ):
        if len(measurement_times) == 0:
            raise ValueError("Must specify at least one measurement time.")
        elif min(measurement_times) < 0.0:  # or max(measurement_times) > 1.0:
            raise ValueError("Measurement times must be greater than zero.")
        measurement_idxes = set(int(t / dt) for t in measurement_times)

        match mode:
            case RunMode.Python:
                update = one_step_diffusion
            case RunMode.Numba:
                update = one_step_diffusion_numba
            case RunMode.Rust:
                measurements = td_diffusion_cylinder(
                    list(measurement_idxes),
                    intervals=self.grid.shape[1],
                    dt=dt,
                    diffusivity=self.diffusivity,
                    rect_sinks=self.rectangle_sinks[1:],
                )
                self.grid = measurements[-1].copy()
                return measurements

        measurements = []
        time_steps = max(measurement_idxes)

        buffer = np.zeros_like(self.grid)
        for t in range(time_steps + 1):
            if t in measurement_idxes:
                measurements.append(self.grid.copy())
            self.grid = update(
                self.grid,
                buffer,
                dt,
                self.dx,
                self.diffusivity,
                self.rectangle_sinks,
                self.rectangle_ins,
            )

        return measurements

    def measure_all(
        self,
        run_time: int,
        measure_every: int,
        dt: float,
        mode: RunMode = RunMode.Python,
    ):
        if run_time < 0.0:  # or max(measurement_times) > 1.0:
            raise ValueError("Measurement times must be greater than zero.")
        elif run_time / measure_every > 1000:
            raise ValueError("Measurements must be less than 1000.")

        match mode:
            case RunMode.Python:
                update = one_step_diffusion
            case RunMode.Numba:
                update = one_step_diffusion_numba
            case RunMode.Rust:
                measurements = td_diffusion_cylinder(
                    list(np.arange(0, run_time + 1, measure_every)),
                    intervals=self.grid.shape[1],
                    dt=dt,
                    diffusivity=self.diffusivity,
                    rect_sinks=self.rectangle_sinks[1:],
                )
                self.grid = measurements[-1].copy()
                return measurements

        measurements = []

        buffer = np.zeros_like(self.grid)
        for t in range(run_time + 1):
            if t % measure_every == 0:
                measurements.append(self.grid.copy())
            self.grid = update(
                self.grid,
                buffer,
                dt,
                self.dx,
                self.diffusivity,
                self.rectangle_sinks,
                self.rectangle_ins,
            )

        return measurements

    def solve_jacobi(
        self,
        epsilon: float = 1e-2,
        max_iters: int = 100000,
        mode: RunMode = RunMode.Numba,
    ) -> int | None:
        match mode:
            case RunMode.Numba:
                update = jacobi_update_numba
            case _:
                raise NotImplementedError(f"Unsupported run mode for Jacobi: {mode}")

        old_grid = self.grid
        buffer = np.empty_like(old_grid)
        new_grid, diff = update(old_grid, buffer, self.rectangle_sinks)
        iters = 1
        while diff > epsilon and iters < max_iters:
            new_grid, diff = update(new_grid, buffer, self.rectangle_sinks)
            iters += 1

        if diff > epsilon:
            iters += 1

        self.grid = new_grid
        return iters if iters <= max_iters else None

    def solve_gauss_seidel(
        self,
        epsilon: float = 1e-2,
        max_iters: int = 100000,
        mode: RunMode = RunMode.Numba,
    ) -> int | None:
        match mode:
            case RunMode.Numba:
                update = gauss_seidel_update_numba
            case _:
                raise NotImplementedError(f"Unsupported run mode for Jacobi: {mode}")

        old_grid = self.grid
        new_grid, diff = update(old_grid, self.rectangle_sinks)
        iters = 1
        while diff > epsilon and iters < max_iters:
            new_grid, diff = update(new_grid, self.rectangle_sinks)
            iters += 1

        if diff > epsilon:
            iters += 1

        self.grid = new_grid
        return iters if iters <= max_iters else None

    def solve_sor(
        self,
        omega: float = 1.0,
        epsilon: float = 1e-2,
        max_iters: int = 100000,
        mode: RunMode = RunMode.Numba,
    ) -> int | None:
        match mode:
            case RunMode.Numba:
                update = sor_update_numba
            case _:
                raise NotImplementedError(f"Unsupported run mode for Jacobi: {mode}")

        old_grid = self.grid
        new_grid, diff = update(old_grid, omega, self.rectangle_sinks)
        iters = 1
        while diff > epsilon and iters < max_iters:
            new_grid, diff = update(new_grid, omega, self.rectangle_sinks)
            iters += 1

        if diff > epsilon:
            iters += 1

        self.grid = new_grid
        return iters if iters <= max_iters else None


@njit
def jacobi_update_numba(
    grid: npt.NDArray[np.float64],
    buffer: npt.NDArray[np.float64],
    rectangle_sinks: list[tuple[int, int, int, int]],
):
    for i in range(1, grid.shape[0] - 1):
        for j in range(0, grid.shape[1]):
            buffer[i, j] = 0.25 * (
                grid[i - 1, j]  # North
                + grid[i + 1, j]  # South
                + grid[i, (j + 1) % grid.shape[1]]  # East
                + grid[i, (j - 1) % grid.shape[1]]  # West
            )

    for x, y, w, h in rectangle_sinks[1:]:
        for i in range(y, y + h):
            for j in range(x, x + w):
                buffer[i, j % grid.shape[1]] = 0.0

    max_diff = 0.0
    for i in range(1, grid.shape[0] - 1):
        for j in range(0, grid.shape[1]):
            diff = abs(buffer[i, j] - grid[i, j])
            if diff > max_diff:
                max_diff = diff
            grid[i, j] = buffer[i, j]

    return grid, max_diff


@njit
def contained_in_some_rect(
    row: int, col: int, rectangles: list[tuple[int, int, int, int]], ncol: int
) -> bool:
    for x, y, w, h in rectangles:
        if row < y or col < x:
            continue
        if row >= y + h or col >= ((x + w) % ncol):
            continue
        return True
    return False


@njit
def gauss_seidel_update_numba(
    grid: npt.NDArray[np.float64],
    rectangle_sinks: list[tuple[int, int, int, int]],
):
    rectangle_sinks = rectangle_sinks[1:]
    max_diff = 0.0
    for i in range(1, grid.shape[0] - 1):
        for j in range(0, grid.shape[1]):
            if contained_in_some_rect(i, j, rectangle_sinks, ncol=grid.shape[1]):
                grid[i, j] = 0.0
                continue
            new_val = 0.25 * (
                grid[i - 1, j]  # North
                + grid[i + 1, j]  # South
                + grid[i, (j + 1) % grid.shape[1]]  # East
                + grid[i, (j - 1) % grid.shape[1]]  # West
            )
            diff = abs(new_val - grid[i, j])
            if diff > max_diff:
                max_diff = diff
            grid[i, j] = new_val

    return grid, max_diff


@njit
def sor_update_numba(
    grid: npt.NDArray[np.float64],
    omega: float,
    rectangle_sinks: list[tuple[int, int, int, int]],
):
    rectangle_sinks = rectangle_sinks[1:]
    max_diff = 0.0
    for i in range(1, grid.shape[0] - 1):
        for j in range(0, grid.shape[1]):
            if contained_in_some_rect(i, j, rectangle_sinks, ncol=grid.shape[1]):
                grid[i, j] = 0.0
                continue
            new_val = (
                omega
                * 0.25
                * (
                    grid[i - 1, j]  # North
                    + grid[i + 1, j]  # South
                    + grid[i, (j + 1) % grid.shape[1]]  # East
                    + grid[i, (j - 1) % grid.shape[1]]  # West
                )
                + (1 - omega) * grid[i, j]
            )
            diff = abs(new_val - grid[i, j])
            if diff > max_diff:
                max_diff = diff
            grid[i, j] = new_val

    return grid, max_diff


@njit
def one_step_diffusion_numba(
    grid: npt.NDArray[np.float64],
    buffer: npt.NDArray[np.float64],
    dt: float,
    dx: float,
    D: float,
    rectangle_sinks,
    rectangle_ins,
):
    """
    Performs one step of diffusion on the grid using Numba for parallelising.

    :params grid: 2D array of concentrations at the current time step.
    :params buffer: 2D array to store updated concentrations during diffusion.
    :params dt: Time step for the diffusion process.
    :params dx: Spatial step size in each direction.
    :params D: Diffusion coefficient.
    :params rectangle_sinks: List of rectangular regions where concentration
                            is set to 0 (sinks).
    :params rectangle_ins: List of rectangular regions where flux
                            of diffusion is 0 (insulators).

    :returns: Updated grid after performing one diffusion step.
    """
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

    for x, y, w, h in rectangle_sinks[1:]:
        for i in range(y, y + h):
            for j in range(x, x + w):
                buffer[i, j % grid.shape[1]] = 0.0

    for x, y, w, h in rectangle_ins[1:]:
        for i in range(y, y + h):
            for j in range(x, x + w):
                buffer[i, j % grid.shape[1]] = 0.0
        buffer[y - 1, x : x + w] += diffusion_coeff * grid[y - 1, x : x + w]
        buffer[y + h, x : x + w] += diffusion_coeff * grid[y + h, x : x + w]
        buffer[y : y + h, x - 1] += diffusion_coeff * grid[y : y + h, x - 1]
        buffer[y : y + h, x + w] += diffusion_coeff * grid[y : y + h, x + w]

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
    rectangle_sinks: list[tuple[int, int, int, int]],
    rectangle_ins: list[tuple[int, int, int, int]],
):
    """
    Performs one step of the diffusion process on a discretised grid,
    updating the grid based on the diffusion equation.

    :returns: Updated grid after performing one diffusion step.
    """
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

    for x, y, w, h in rectangle_sinks[1:]:
        for i in range(y, y + h):
            for j in range(x, x + w):
                buffer[i, j % grid.shape[1]] = 0

    grid[1:-1, ...] = buffer[1:-1, ...]
    return grid


def analytical_solution(y: float, D: float, t: float, terms: int):
    """
    Computes the given analytical solution for the time-dependent diffusion equation.

    :param y: The x axis of the analytical solution.
    :param D: The diffusion coefficient.
    :param t: The time at which the measurements are taken (0.001, 0.01, 0.1, 1).
    :param terms: The number of terms used to compute the solution.

    :return: The analytical solution of the concentration profile as a function of y.
    """
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
    """
    Plots the concentration profile comparison between the simulated
    and analytical solutions of the diffusion equation.

    :return: Four plots at each time instance (t = 0.001, 0.01, 0.1, and 1)
    that compares the analytical solution of the concentration profile and
    the simulated solution of the concentration profile.
    """
    y_range = np.linspace(0, 1, intervals + 1)
    D = 1

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8, 5),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
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
        # ax.set_xlabel("y")
        # ax.set_ylabel("Concentration (c)")
        ax.legend()
    fig.supxlabel(r"y")
    fig.supylabel("Concentration")

    return fig


def is_stable_scheme(dt: float, dx: float, diffusivity: float) -> bool:
    """Test if discretized diffusion scheme is stable."""
    return 4 * dt * diffusivity / dx**2 <= 1
