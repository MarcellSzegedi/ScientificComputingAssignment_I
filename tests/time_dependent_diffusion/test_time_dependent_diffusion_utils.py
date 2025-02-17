import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis.strategies import floats, integers

from scientific_computing.time_dependent_diffusion import (
    Cylinder,
    RunMode,
    is_stable_scheme,
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
    cylinder = Cylinder(spatial_intervals=intervals, diffusivity=D)
    assert cylinder.grid.shape == (intervals, intervals)


@given(
    intervals=integers(min_value=1),
    dt=floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
    D=floats(min_value=0, allow_nan=False, allow_infinity=False),
)
def test_is_stable_scheme(intervals, dt, D):
    is_stable_scheme(dt, 1 / intervals, D)


@settings(deadline=None)
@given(
    intervals=integers(min_value=1, max_value=25),
    time_steps=integers(min_value=1, max_value=1000),
    dt=floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
    D=floats(min_value=0, allow_nan=False, allow_infinity=False),
)
def test_numba_version_gives_same_results(time_steps, intervals, dt, D):
    assume(is_stable_scheme(dt, 1 / intervals, D))
    cylinder_python = Cylinder(intervals, diffusivity=D)
    cylinder_python.run(time_steps, dt, mode=RunMode.Python)
    cylinder_numba = Cylinder(intervals, diffusivity=D)
    cylinder_numba.run(time_steps, dt, mode=RunMode.Numba)

    np.testing.assert_allclose(cylinder_python.grid, cylinder_numba.grid)


@settings(deadline=None)
@given(
    intervals=integers(min_value=1, max_value=25),
    time_steps=integers(min_value=1, max_value=1000),
    dt=floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
    D=floats(min_value=0, allow_nan=False, allow_infinity=False),
)
def test_rust_version_gives_same_results(time_steps, intervals, dt, D):
    assume(is_stable_scheme(dt, 1 / intervals, D))

    cylinder_python = Cylinder(intervals, diffusivity=D)
    cylinder_python.run(time_steps, dt, mode=RunMode.Python)
    cylinder_rust = Cylinder(intervals, diffusivity=D)
    cylinder_rust.run(time_steps, dt, mode=RunMode.Rust)

    np.testing.assert_allclose(cylinder_python.grid, cylinder_rust.grid)


def test_diffusion_run_raises_on_neg_iters():
    cylinder = Cylinder(spatial_intervals=4, diffusivity=5)
    with pytest.raises(ValueError):
        cylinder.run(n_iters=-1, dt=0.01)


def test_diffusion_measure_raises_on_no_measure_times():
    cylinder = Cylinder(spatial_intervals=4, diffusivity=5)
    with pytest.raises(ValueError):
        cylinder.measure(measurement_times=[], dt=0.01)


def test_diffusion_measure_raises_on_measure_time_lt_0():
    cylinder = Cylinder(spatial_intervals=4, diffusivity=5)
    with pytest.raises(ValueError):
        cylinder.measure(measurement_times=[-0.0001, 0.1, 0.5], dt=0.01)


def test_diffusion_measure_raises_on_measure_time_gt_1():
    cylinder = Cylinder(spatial_intervals=4, diffusivity=5)
    with pytest.raises(ValueError):
        cylinder.measure(measurement_times=[0.1, 1.0001, 0.5], dt=0.01)


def test_diffusion_measure_okay_ok_valid_measure_times():
    cylinder = Cylinder(spatial_intervals=4, diffusivity=5)
    cylinder.measure(measurement_times=[0.0, 0.1, 0.5, 1.0], dt=0.01)
