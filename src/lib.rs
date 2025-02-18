use std::{collections::HashSet, ops::Not};

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

struct Rectangle([usize; 4]);

impl Rectangle {
    fn min_row(&self) -> usize {
        self.0[1]
    }

    fn max_row(&self) -> usize {
        (self.0[1] + self.0[3]) - 1
    }

    fn min_col(&self) -> usize {
        self.0[0]
    }

    fn max_col(&self) -> usize {
        (self.0[0] + self.0[2]) - 1
    }
}

impl TryFrom<Vec<usize>> for Rectangle {
    type Error = ();

    fn try_from(value: Vec<usize>) -> Result<Self, Self::Error> {
        if value.len() != 4 {
            Err(())
        } else {
            Ok(Rectangle([value[0], value[1], value[2], value[3]]))
        }
    }
}

pub struct Cylinder {
    pub grid: Array2<f64>,
    buffer: Array2<f64>,
    intervals: usize,
    rect_sinks: Vec<Rectangle>,
}

impl Cylinder {
    fn new(intervals: usize, rect_sinks: Vec<Vec<usize>>) -> Self {
        let mut grid = Array2::zeros((intervals, intervals));
        let buffer = Array2::zeros((intervals, intervals));
        grid.row_mut(0).fill(1.0);
        let rect_sinks = {
            let parsed_rect_sinks: Result<Vec<Rectangle>, _> =
                rect_sinks.into_iter().map(|r| r.try_into()).collect();
            let valid_structure_sinks = parsed_rect_sinks
                .expect("Some rectangles had incorrect structure. Expected (x, y, w, h).");
            valid_structure_sinks
                .iter()
                .any(|rect| rect.0.iter().any(|v| v < &0 || v > &(intervals + 1)))
                .not()
                .then_some(valid_structure_sinks)
                .expect("Some rectangles had invalid definition.")
        };
        //let rect_sinks: Result<Vec<Rectangle>, _> =
        //    rect_sinks.into_iter().map(|r| r.try_into()).collect();

        Self {
            grid,
            buffer,
            intervals,
            rect_sinks,
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

        for sink in &self.rect_sinks {
            for row in sink.min_row()..=sink.max_row() {
                for col in sink.min_col()..=sink.max_col() {
                    self.buffer[(row, col)] = 0.0;
                }
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
