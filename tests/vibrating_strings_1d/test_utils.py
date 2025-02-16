import pytest
from hypothesis import given, settings
from hypothesis.strategies import floats, integers, sampled_from

from scientific_computing.vibrating_strings_1d.utils.discretize_pde import (
    discretize_pde,
)
from scientific_computing.vibrating_strings_1d.utils.grid_initialisation import (
    Initialisation,
    initialize_grid,
)


@given(
    grid_intervals=integers(min_value=0, max_value=1000),
    time_points=integers(min_value=2, max_value=1000),
    case=sampled_from(Initialisation),
)
def test_grid_shape(grid_intervals, time_points, case):
    grid = initialize_grid(grid_intervals, time_points, case)
    assert grid.shape == (time_points, grid_intervals + 1)


@settings(deadline=None)
@given(
    spatial_intervals=integers(min_value=1, max_value=1000),
    temporal_intervals=integers(min_value=2, max_value=1000),
    runtime=floats(min_value=1, max_value=1e10),
    propagation_velocity=floats(min_value=0, max_value=1e10),
)
def test_max_amplitude_always_at_most_initial_max(
    spatial_intervals, temporal_intervals, runtime, propagation_velocity
):
    states = discretize_pde(
        spatial_intervals,
        temporal_intervals,
        string_length=1,
        runtime=runtime,
        c=propagation_velocity,
        case=Initialisation.LowFreq,
    )
    assert states[0].max() >= states.max()


def test_string_simulation_raises_on_short_runtime():
    with pytest.raises(ValueError):
        discretize_pde(50, 100, 1, 0.99, c=1.0, case=Initialisation.LowFreq)


def test_string_simulation_raises_on_zero_intervals():
    with pytest.raises(ValueError):
        discretize_pde(0, 100, 1, 100, c=1.0, case=Initialisation.LowFreq)


def test_string_simulation_raises_on_large_velocity():
    with pytest.raises(ValueError):
        discretize_pde(1, 100, 1, 100, c=1e100, case=Initialisation.LowFreq)


def test_string_simulation_raises_on_large_runtime():
    with pytest.raises(ValueError):
        discretize_pde(1, 100, 1, runtime=1e100, c=1, case=Initialisation.LowFreq)


def test_initialize_grid_below_two_time_steps_fails():
    with pytest.raises(ValueError):
        initialize_grid(spatial_intervals=50, time_steps=1, case=Initialisation.LowFreq)
