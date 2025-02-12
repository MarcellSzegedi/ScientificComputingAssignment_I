import unittest

import numpy as np

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
        grid = initialize_grid(delta)
        result = jacobi_iteration(delta, iters=100, tol=1e-5)
        self.assertEqual(result.shape, grid.shape)

    def test_convergence(self):
        """Testing that Jacobi iteration converges properly."""
        delta = 0.1
        result = jacobi_iteration(delta, iters=100, tol=1e-5)
        max_change = np.max(np.abs(result - initialize_grid(delta)))
        self.assertLess(max_change, 1e-5, "Jacobi iteration did not converge properly.")

    def test_boundary_conditions(self):
        """Testing that boundary conditions remain unchanged."""
