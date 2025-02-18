use std::{collections::HashSet, ops::Not};

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

pub mod cylinder;

use cylinder::{morton, Cylinder, CylinderZOrder};

#[pyfunction]
fn td_diffusion_cylinder(
    py: Python<'_>,
    measurement_timesteps: Vec<u32>,
    intervals: u32,
    dt: f64,
    diffusivity: f64,
    rect_sinks: Vec<Vec<usize>>,
) -> Vec<Bound<'_, PyArray2<f64>>> {
    let dx = 1f64 / (intervals as f64);
    let mut cylinder = Cylinder::new(intervals as usize, rect_sinks);
    let measurement_timesteps = measurement_timesteps.into_iter().collect::<HashSet<_>>();

    let n_iters = measurement_timesteps.clone().into_iter().max().unwrap_or(0);

    let mut measurements = vec![];
    for i in 0..=n_iters {
        if measurement_timesteps.contains(&i) {
            measurements.push(cylinder.grid.clone().into_pyarray(py))
        }
        cylinder.update(dx, dt, diffusivity);
    }

    measurements
}

#[pyfunction]
fn td_diffusion_cylinder_z_order(
    py: Python<'_>,
    measurement_timesteps: Vec<u32>,
    intervals: u32,
    dt: f64,
    diffusivity: f64,
    rect_sinks: Vec<Vec<usize>>,
) -> Vec<Bound<'_, PyArray2<f64>>> {
    let dx = 1f64 / (intervals as f64);
    let mut cylinder = CylinderZOrder::new(intervals as usize);
    let measurement_timesteps = measurement_timesteps.into_iter().collect::<HashSet<_>>();

    let n_iters = measurement_timesteps.clone().into_iter().max().unwrap_or(0);

    let mut measurements: Vec<Vec<f64>> = Vec::with_capacity(measurement_timesteps.len());
    for i in 0..=n_iters {
        if measurement_timesteps.contains(&i) {
            //measurements.push(cylinder.grid.clone().into_pyarray(py))
            measurements.push(cylinder.grid.clone())
        }
        cylinder.update(dx, dt, diffusivity);
    }

    let mut grids = Vec::with_capacity(measurement_timesteps.len());
    for measurement in measurements {
        let mut grid = Array2::zeros((intervals as usize, intervals as usize));
        for row in 0..(intervals as usize) {
            for col in 0..(intervals as usize) {
                grid[[row, col]] = measurement[morton(row, col)];
            }
        }
        grids.push(grid.into_pyarray(py));
    }

    grids
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(td_diffusion_cylinder, m)?)?;
    m.add_function(wrap_pyfunction!(td_diffusion_cylinder_z_order, m)?)?;
    Ok(())
}
