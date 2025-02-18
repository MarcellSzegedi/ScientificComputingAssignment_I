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
    ):
        self.grid = np.zeros((spatial_intervals, spatial_intervals), dtype=np.float64)
        self.grid[0] = 1.0
        self.dx = 1 / spatial_intervals
        self.diffusivity = diffusivity
        self.init_objects(sinks=sinks, dx=self.dx)

    def discretise_coord(self, coord: float) -> int:
        return int(coord / self.dx)

    def init_objects(self, sinks: DomainObjects | None, dx: float):
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
                    rect_sinks=self.rectangle_sinks[1:],
                )[-1]
                return

        buffer = np.zeros_like(self.grid)
        for _ in range(n_iters):
            self.grid = update(
                self.grid, buffer, dt, self.dx, self.diffusivity, self.rectangle_sinks
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
                    intervals=self.grid.shape[0],
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
                self.grid, buffer, dt, self.dx, self.diffusivity, self.rectangle_sinks
            )

        return measurements

    def solve_jacobi(
        self,
        epsilon: float = 1e-2,
        max_iters: int = 100000,
        mode: RunMode = RunMode.Numba,
    ):
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
        if iters < max_iters:
            print(f"Converged after {iters} iterations.")
        else:
            print(f"Terminated early after {iters} iterations.")
        self.grid = new_grid

    def solve_gauss_seidel(
        self,
        epsilon: float = 1e-2,
        max_iters: int = 100000,
        mode: RunMode = RunMode.Numba,
    ):
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
        if iters < max_iters:
            print(f"Converged after {iters} iterations.")
        else:
            print(f"Terminated early after {iters} iterations.")
        self.grid = new_grid

    def solve_sor(
        self,
        omega: float = 1.0,
        epsilon: float = 1e-2,
        max_iters: int = 100000,
        mode: RunMode = RunMode.Numba,
    ):
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
        if iters < max_iters:
            print(f"Converged after {iters} iterations.")
        else:
            print(f"Terminated early after {iters} iterations.")
        self.grid = new_grid


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
    rectangle_sinks: list[tuple[int, int, int, int]],
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

    for x, y, w, h in rectangle_sinks[1:]:
        for i in range(y, y + h):
            for j in range(x, x + w):
                buffer[i, j % grid.shape[1]] = 0.0

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

    for x, y, w, h in rectangle_sinks[1:]:
        for i in range(y, y + h):
            for j in range(x, x + w):
                buffer[i, j % grid.shape[1]] = 0

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
