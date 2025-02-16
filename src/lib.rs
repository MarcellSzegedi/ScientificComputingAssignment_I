use std::collections::HashSet;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

pub struct Cylinder {
    pub grid: Array2<f64>,
    buffer: Array2<f64>,
    intervals: usize,
}

impl Cylinder {
    fn new(intervals: usize) -> Self {
        let mut grid = Array2::zeros((intervals, intervals));
        let buffer = Array2::zeros((intervals, intervals));
        grid.row_mut(0).fill(1.0);
        Self {
            grid,
            buffer,
            intervals,
        }
    }

    fn update(&mut self, dx: f64, dt: f64, diffusivity: f64) {
        let diffusion_coeff = dt * diffusivity / (dx * dx);
        for row in 1..(self.intervals - 1) {
            for col in 0..self.intervals {
                let neighbor_concentration_diff = {
                    self.grid[(row + 1, col)]
                        + self.grid[(row - 1, col)]
                        + self.grid[(row, (col + 1) % self.intervals)]
                        + self.grid[(row, ((col + self.intervals) - 1) % self.intervals)]
                        - 4.0 * self.grid[(row, col)]
                };

                self.buffer[(row, col)] =
                    self.grid[(row, col)] + diffusion_coeff * neighbor_concentration_diff;
            }
        }

        for row in 1..(self.intervals - 1) {
            for col in 0..self.intervals {
                self.grid[(row, col)] = self.buffer[(row, col)]
            }
        }
    }
}

#[pyfunction]
fn td_diffusion_cylinder(
    py: Python<'_>,
    measurement_timesteps: Vec<u32>,
    intervals: u32,
    dt: f64,
    diffusivity: f64,
) -> Vec<Bound<'_, PyArray2<f64>>> {
    let dx = 1f64 / (intervals as f64);
    let mut cylinder = Cylinder::new(intervals as usize);
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
fn hello_from_bin() -> String {
    "Hello from scientific-computing!".to_string()
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_from_bin, m)?)?;
    m.add_function(wrap_pyfunction!(td_diffusion_cylinder, m)?)?;
    Ok(())
}
