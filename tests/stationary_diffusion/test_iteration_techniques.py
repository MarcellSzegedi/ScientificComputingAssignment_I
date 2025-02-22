import unittest

import numpy as np

from scientific_computing.stationary_diffusion.main import IterationApproaches


class TestJacobiIteration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.delta = 0.02

    def setUp(self):
        self.iteration_instance = IterationApproaches()

    def test_output_shape(self):
        """Testing that the output grid has the correct shape."""
        expected_shape = (1 / self.delta, 1 / self.delta)
        grid_hist, _, _ = self.iteration_instance.jacobi_iteration(self.delta)
        self.assertEqual(grid_hist.shape[1], expected_shape[0])
        self.assertEqual(grid_hist.shape[2], expected_shape[1])

    def test_convergence(self):
        """Testing that Jacobi iteration converges properly."""
        a, max_diff_hist, _ = self.iteration_instance.jacobi_iteration(self.delta)
        self.assertLess(
            max_diff_hist[-1], 1e-5, "Jacobi iteration did not converge properly."
        )

    def test_boundary_conditions(self):
        """Testing that boundary conditions remain unchanged."""
        grid_hist, _, _ = self.iteration_instance.jacobi_iteration(self.delta)
        self.assertTrue(
            np.all(np.isin(grid_hist[:, 0, :], [1])),
            "Jacobi iteration changed the top boundary condition.",
        )
        self.assertTrue(
            np.all(np.isin(grid_hist[:, -1, :], [0])),
            "Jacobi iteration changed the bottom boundary condition.",
        )


class TestGaussSeidel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.delta = 0.02

    def setUp(self):
        self.iteration_instance = IterationApproaches()

    def test_output_shape(self):
        """Testing that the output grid has the correct shape."""
        expected_shape = (1 / self.delta, 1 / self.delta)
        grid_hist, _, _ = self.iteration_instance.gauss_seidel_iteration(self.delta)
        self.assertEqual(grid_hist.shape[1], expected_shape[0])
        self.assertEqual(grid_hist.shape[2], expected_shape[1])

    def test_convergence(self):
        """Testing that Jacobi iteration converges properly."""
        a, max_diff_hist, _ = self.iteration_instance.gauss_seidel_iteration(self.delta)
        self.assertLess(
            max_diff_hist[-1], 1e-5, "GS iteration did not converge properly."
        )

    def test_boundary_conditions(self):
        """Testing that boundary conditions remain unchanged."""
        grid_hist, _, _ = self.iteration_instance.gauss_seidel_iteration(self.delta)
        self.assertTrue(
            np.all(np.isin(grid_hist[:, 0, :], [1])),
            "GS iteration changed the top boundary condition.",
        )
        self.assertTrue(
            np.all(np.isin(grid_hist[:, -1, :], [0])),
            "GS iteration changed the bottom boundary condition.",
        )


class TestSOR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.delta = 0.02
        cls.omega = 1.9

    def setUp(self):
        self.iteration_instance = IterationApproaches()

    def test_output_shape(self):
        """Testing that the output grid has the correct shape."""
        expected_shape = (1 / self.delta, 1 / self.delta)
        grid_hist, _, _ = self.iteration_instance.sor_iteration(self.delta, self.omega)
        self.assertEqual(grid_hist.shape[1], expected_shape[0])
        self.assertEqual(grid_hist.shape[2], expected_shape[1])

    def test_convergence(self):
        """Testing that Jacobi iteration converges properly."""
        a, max_diff_hist, _ = self.iteration_instance.sor_iteration(
            self.delta, self.omega
        )
        self.assertLess(
            max_diff_hist[-1], 1e-5, "SOR iteration did not converge properly."
        )

    def test_boundary_conditions(self):
        """Testing that boundary conditions remain unchanged."""
        grid_hist, _, _ = self.iteration_instance.sor_iteration(self.delta, self.omega)
        self.assertTrue(
            np.all(np.isin(grid_hist[:, 0, :], [1])),
            "SOR iteration changed the top boundary condition.",
        )
        self.assertTrue(
            np.all(np.isin(grid_hist[:, -1, :], [0])),
            "SOR iteration changed the bottom boundary condition.",
        )
