import unittest

from scientific_computing.stationary_diffusion.iteration_techniques import (
    jacobi_iteration,
)
from scientific_computing.stationary_diffusion.utils import (
    initialize_grid,
)


class TestJacobiIteration(unittest.TestCase):
    def test_output_shape(self):
        """Testing that the output grid has the correct shape."""
        delta = 0.1
        iters = 1000
        grid = initialize_grid(delta)
        result, _ = jacobi_iteration(grid, iters)
        self.assertEqual(result.shape, grid.shape)

    def test_convergence(self):
        """Testing that Jacobi iteration converges properly."""
        delta = 0.1
        iters = 1000
        grid = initialize_grid(delta)
        _, max_diff = jacobi_iteration(grid, iters)
        self.assertLess(max_diff, 1e-5, "Jacobi iteration did not converge properly.")

    def test_boundary_conditions(self):
        """Testing that boundary conditions remain unchanged."""
