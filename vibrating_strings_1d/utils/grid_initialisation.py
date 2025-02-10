import numpy as np
from constants import LATTICE_LENGTH, SPATIAL_POINTS, TIME_STEPS

def initialize_string(case: int):
    """
    Initializes the starting wave of the string with the following three cases.
        i. Ψ(x, t = 0) = sin(2πx).
        ii. Ψ(x, t = 0) = sin(5πx).
        iii. Ψ(x, t = 0) = sin(5πx) if 1/5 < x < 2/5, else Ψ = 0.
    :return: Array representing the initial wave shape.
    """
    x = np.linspace(0, LATTICE_LENGTH, SPATIAL_POINTS + 1)
    if case == 1:
        return np.sin(2 * np.pi * x)
    elif case == 2:
        return np.sin(5 * np.pi * x)
    elif case == 3:
        return np.where((x > 1 / 5) & (x < 2 / 5), np.sin(5 * np.pi * x), 0)

def initialize_grid(case: int):
    """
    Initializes grid for discretizing the second order PDE.
    :return: Initialized grid
    """
    grid = np.zeros((TIME_STEPS, SPATIAL_POINTS + 1))
    grid[:, 0] = initialize_string(case)
    grid[:, 1] = grid[:, 0]
    return grid
