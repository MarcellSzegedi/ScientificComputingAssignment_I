use std::ops::Not;

use ndarray::Array2;

pub const fn morton(y: usize, x: usize) -> usize {
    const B: [usize; 4] = [0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF];
    const S: [usize; 4] = [1, 2, 4, 8];

    debug_assert!((x <= 65536) & (y <= 65536), "Intervals should be <= 65536");

    let mut x = x;
    let mut y = y;
    x = (x | (x << S[3])) & B[3];
    x = (x | (x << S[2])) & B[2];
    x = (x | (x << S[1])) & B[1];
    x = (x | (x << S[0])) & B[0];

    y = (y | (y << S[3])) & B[3];
    y = (y | (y << S[2])) & B[2];
    y = (y | (y << S[1])) & B[1];
    y = (y | (y << S[0])) & B[0];

    x | (y << 1)
}

pub const fn morton_inv(z: usize) -> (usize, usize) {
    const B: [usize; 4] = [0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF];
    const S: [usize; 4] = [1, 2, 4, 8];

    let mut x = z & B[0];
    x = (x | (x >> S[0])) & B[1];
    x = (x | (x >> S[1])) & B[2];
    x = (x | (x >> S[2])) & B[3];
    x = (x | (x >> S[3])) & 0x0000ffff;

    let mut y = (z >> 1) & B[0];
    y = (y | (y >> S[0])) & B[1];
    y = (y | (y >> S[1])) & B[2];
    y = (y | (y >> S[2])) & B[3];
    y = (y | (y >> S[3])) & 0x0000ffff;

    (y, x)
}

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
    pub fn new(intervals: usize, rect_sinks: Vec<Vec<usize>>) -> Self {
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

    pub fn update(&mut self, dx: f64, dt: f64, diffusivity: f64) {
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

pub struct CylinderZOrder {
    pub grid: Vec<f64>,
    buffer: Vec<f64>,
    intervals: usize,
    lookup: Vec<(usize, usize)>,
}

impl CylinderZOrder {
    pub fn new(intervals: usize) -> Self {
        let m = intervals.next_power_of_two();
        let mut grid = vec![0.0f64; m * m];
        let buffer = grid.clone();
        let mut z: usize;
        for x in 0..intervals {
            z = morton(0, x);
            grid[z] = 1.0;
        }
        let lookup: Vec<(usize, usize)> = (0..buffer.len()).into_iter().map(morton_inv).collect();

        Self {
            grid,
            buffer,
            intervals,
            lookup,
        }
    }

    pub fn update(&mut self, dx: f64, dt: f64, diffusivity: f64) {
        let diffusion_coeff = dt * diffusivity / (dx * dx);
        for (i, concentration) in self.grid.iter().enumerate() {
            let (row, col) = self.lookup[i];
            //let (row, col) = morton_inv(i);
            if (row == 0) | (row >= self.intervals - 1) {
                continue;
            }
            let north = morton(row - 1, col);
            let east = morton(row, (col + 1) % self.intervals);
            let west = morton(row, (col + self.intervals + 1) % self.intervals);
            let south = morton(row + 1, col);

            let neighbor_concentration_diff = {
                self.grid[south] + self.grid[north] + self.grid[east] + self.grid[west]
                    - 4.0 * concentration
            };

            self.buffer[i] = self.grid[i] + diffusion_coeff * neighbor_concentration_diff;
        }

        for (i, concentration) in self.buffer.iter().enumerate() {
            let (row, _) = morton_inv(i);
            if (row == 0) | (row == self.intervals) {
                continue;
            }
            self.grid[i] = *concentration;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_z_order_init() {
        let _cylinder = CylinderZOrder::new(16);
    }

    #[test]
    fn morton_roundtrip() {
        let x = 10;
        let y = 10;
        let z = morton(y, x);
        assert_eq!(morton_inv(z), (y, x));

        let x = 0;
        let y = 10;
        let z = morton(y, x);
        assert_eq!(morton_inv(z), (y, x));

        let x = 10;
        let y = 0;
        let z = morton(y, x);
        assert_eq!(morton_inv(z), (y, x));

        let x = 1275;
        let y = 3977;
        let z = morton(y, x);
        assert_eq!(morton_inv(z), (y, x))
    }
}
