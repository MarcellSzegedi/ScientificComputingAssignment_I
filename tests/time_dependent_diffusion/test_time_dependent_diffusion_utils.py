import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis.strategies import floats, integers

from scientific_computing.time_dependent_diffusion.utils import (
    is_stable_scheme,
    time_dependent_diffusion,
    time_dependent_diffusion_numba,
)


@settings(deadline=None)
@given(
    intervals=integers(min_value=1, max_value=10),
    time_steps=integers(min_value=1, max_value=1000),
    dt=floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
    D=floats(min_value=0, allow_nan=False, allow_infinity=False),
)
def test_grid_shape(time_steps, intervals, dt, D):
    assume(is_stable_scheme(dt, 1 / intervals, D))
    grid, _ = time_dependent_diffusion(time_steps, intervals, dt, D)
    assert grid.shape == (intervals, intervals)


@given(
    intervals=integers(min_value=1),
    dt=floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
    D=floats(min_value=0, allow_nan=False, allow_infinity=False),
)
def test_is_stable_scheme(intervals, dt, D):
    is_stable_scheme(dt, 1 / intervals, D)


@settings(deadline=None)
@given(
    intervals=integers(min_value=1, max_value=10),
    time_steps=integers(min_value=1, max_value=1000),
    dt=floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
    D=floats(min_value=0, allow_nan=False, allow_infinity=False),
)
def test_numba_version_gives_same_results(time_steps, intervals, dt, D):
    grid_python, _ = time_dependent_diffusion(time_steps, intervals, dt, D)
    grid_numba, _ = time_dependent_diffusion_numba(time_steps, intervals, dt, D)

    np.testing.assert_allclose(grid_python, grid_numba)


def test_diffusion_time_steps_and_intervals_less_than_one():
    with pytest.raises(ValueError):
        time_dependent_diffusion(time_steps=0, intervals=0, dt=0.01, D=5)
