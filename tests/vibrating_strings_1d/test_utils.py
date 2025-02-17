import pytest
from hypothesis import assume, given, settings
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
    spatial_intervals=integers(min_value=1, max_value=500),
    temporal_intervals=integers(min_value=2, max_value=500),
    runtime=floats(min_value=1, max_value=1e10),
    propagation_velocity=floats(min_value=0, max_value=1e10),
)
def test_max_amplitude_always_at_most_initial_max(
    spatial_intervals,
    temporal_intervals,
    runtime,
    propagation_velocity,
):
    # Only test if CFL condition is met
    dt = runtime / temporal_intervals
    dx = 1 / spatial_intervals
    assume(propagation_velocity * (dt / dx) <= 1.0)
    states = discretize_pde(
        spatial_intervals,
        temporal_intervals,
        string_length=1,
        runtime=runtime,
        c=propagation_velocity,
        case=Initialisation.LowFreq,
    )
    assert states[0].max() == pytest.approx(states.max(), abs=1e-12)


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


def test_string_simulation_raises_on_courant_gt_1():
    with pytest.raises(ValueError):
        discretize_pde(
            spatial_intervals=3,
            temporal_intervals=2,
            runtime=1,
            string_length=1,
            c=1,
            case=Initialisation.LowFreq,
        )


def test_initialize_grid_below_two_time_steps_fails():
    with pytest.raises(ValueError):
        initialize_grid(spatial_intervals=50, time_steps=1, case=Initialisation.LowFreq)
