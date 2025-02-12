import pytest

from hypothesis import given
from hypothesis.strategies import integers

from vibrating_strings_1d.utils.grid_initialisation import initialize_grid

@given(grid_intervals = integers(min_value=0, max_value=1000), time_points = integers(min_value=2, max_value=1000), case = integers(min_value=1, max_value=3))
def test_grid_shape(grid_intervals, time_points, case):
    grid = initialize_grid(grid_intervals, time_points, case)
    assert grid.shape == (time_points, grid_intervals + 1)

def test_initialize_grid_below_two_time_steps_fails():
    with pytest.raises(ValueError):
        initialize_grid(spatial_intervals=50, time_steps=1, case=1)
