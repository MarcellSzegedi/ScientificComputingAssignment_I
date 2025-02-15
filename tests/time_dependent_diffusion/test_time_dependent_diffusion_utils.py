import pytest
from hypothesis import given
from hypothesis.strategies import floats, integers

from scientific_computing.time_dependent_diffusion.utils import (
    time_dependent_diffusion,
)


@given(
    intervals=integers(min_value=1, max_value=10),
    time_steps=integers(min_value=1, max_value=1000),
    dt=floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
    D=floats(min_value=0, allow_nan=False, allow_infinity=False),
)
def test_grid_shape(time_steps, intervals, dt, D):
    grid, _ = time_dependent_diffusion(time_steps, intervals, dt, D)
    assert grid.shape == (intervals, intervals)


def test_diffusion_time_steps_and_intervals_less_than_one():
    with pytest.raises(ValueError):
        time_dependent_diffusion(time_steps=0, intervals=0, dt=0.01, D=5)
