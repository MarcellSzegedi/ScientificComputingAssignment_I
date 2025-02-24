from enum import StrEnum

import numpy as np


class Initialisation(StrEnum):
    LowFreq = "low-freq"
    HighFreq = "high-freq"
    BoundedHighFreq = "bounded-high-freq"

    def as_equation_str(self) -> str:
        match self:
            case Initialisation.LowFreq:
                return r"$\Psi(x, t=0) = \sin(2\pi x)$"
            case Initialisation.HighFreq:
                return r"$\Psi(x, t=0) = \sin(5\pi x)$"
            case Initialisation.BoundedHighFreq:
                return r"$\Psi(1/5 < x < 2/5) = \sin(5\pi x)$"


def initialize_string(
    spatial_intervals: int, case: Initialisation, string_length: int = 1
):
    """
    Initializes the starting wave of the string with the following three cases.
        i. Ψ(x, t = 0) = sin(2πx).
        ii. Ψ(x, t = 0) = sin(5πx).
        iii. Ψ(x, t = 0) = sin(5πx) if 1/5 < x < 2/5, else Ψ = 0.
    :return: Array representing the initial wave shape.
    """
    x = np.linspace(0, string_length, spatial_intervals + 1)
    match case:
        case Initialisation.LowFreq:
            y = np.sin(2 * np.pi * x)
        case Initialisation.HighFreq:
            y = np.sin(5 * np.pi * x)
        case Initialisation.BoundedHighFreq:
            y = np.where((x > 1 / 5) & (x < 2 / 5), np.sin(5 * np.pi * x), 0)

    y[[0, spatial_intervals]] = 0.0
    return y


def initialize_grid(spatial_intervals: int, time_steps: int, case: Initialisation):
    """
    Initializes grid for discretizing the second order PDE.
    :return: Initialized grid
    """
    if time_steps < 2:
        raise ValueError("Time steps must be greater than 2.")

    grid = np.zeros((time_steps, spatial_intervals + 1))
    grid[0] = initialize_string(spatial_intervals, case)
    return grid
