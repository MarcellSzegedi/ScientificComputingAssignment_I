from .grid_initialisation import Initialisation, initialize_grid


def discretize_pde(
    spatial_intervals: int,
    temporal_intervals: int,
    string_length: float,
    runtime: float,
    c: float,
    case: Initialisation,
):
    """
    Applies the finite difference method to discretize the
    one-dimensional wave equation PDE over a given spatial and temporal grid.

    :param spatial_intervals: Number of discrete spatial points.
    :param temporal_intervals: Number of discrete time steps.
    :param string_length: Total length of the string.
    :param runtime: Total simulation time. Must be in the range [1, 1e10].
    :param c: Wave propagation velocity. Must not exceed 1e10.
    :param case: Initial conditions for the wave, defined by the `Initialisation` class.

    :return: A 2D NumPy array representing the wave displacement over time.

    :raises ValueError: If input parameters violate constraints (negative intervals,
                        excessive runtime, or invalid Courant number).
    """
    if spatial_intervals < 1:
        raise ValueError(
            f"Spatial intervals must be at least 1, found {spatial_intervals}"
        )
    if runtime < 1 or runtime > 1e10:
        raise ValueError(f"Runtime must be in range [1,1e10], found: {runtime}")
    if c > 1e10:
        raise ValueError(f"Propagation velocity exceeds allowable range: 1e10 < {c}")

    dt = runtime / temporal_intervals
    dx = string_length / spatial_intervals
    if (courant := c * (dt / dx)) > 1.0:
        raise ValueError(
            f"Courant number exceeds 1.0, results may be inaccurate: "
            f"c*(dt/dx) = {courant}."
        )

    r = (c * dt / dx) ** 2

    grid = initialize_grid(spatial_intervals, temporal_intervals, case)

    # Boundary (t=1), computed using symmetry at t=0 due to string at rest
    for i in range(1, spatial_intervals):
        grid[1, i] = (r / 2) * (
            grid[0, i + 1] - 2 * grid[0, i] + grid[0, i - 1]
        ) + grid[0, i]

    for t in range(1, temporal_intervals - 1):
        for i in range(1, spatial_intervals):
            grid[t + 1, i] = (
                r * (grid[t, i + 1] - 2 * grid[t, i] + grid[t, i - 1])
                + 2 * grid[t, i]
                - grid[t - 1, i]
            )

    return grid
