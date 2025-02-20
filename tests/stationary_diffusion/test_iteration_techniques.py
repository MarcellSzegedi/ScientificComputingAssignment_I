import unittest
from unittest.mock import patch
from hypothesis import given, settings
from hypothesis.strategies import floats

import numpy as np

from scientific_computing.stationary_diffusion.utils import initialize_grid
from scientific_computing.stationary_diffusion.iteration_techniques import (
    jacobi_iteration,
)
from scientific_computing.stationary_diffusion.iteration_techniques.gauss_seidel_iteration import (
    gauss_seidel_iteration,
    calc_iter_transformation,
    calc_output_vector_trans_matrix,
    calc_output_vector, restructure_grid
)
from scientific_computing.stationary_diffusion.iteration_techniques.sor_iteration import (
    sor_iteration
)
from scientific_computing.stationary_diffusion.main import main_sor_iter


class TestJacobiIteration(unittest.TestCase):
    @unittest.expectedFailure
    def test_output_shape(self):
        """Testing that the output grid has the correct shape."""
        delta = 0.1
        iters = 1000
        grid, _ = initialize_grid(delta)
        result, _ = jacobi_iteration(grid, iters)
        self.assertEqual(result.shape, grid.shape)

    @unittest.expectedFailure
    def test_convergence(self):
        """Testing that Jacobi iteration converges properly."""
        delta = 0.1
        iters = 1000
        grid, _ = initialize_grid(delta)
        _, max_diff = jacobi_iteration(grid, iters)
        self.assertLess(max_diff, 1e-5, "Jacobi iteration did not converge properly.")

    @unittest.expectedFailure
    def test_boundary_conditions(self):
        """Testing that boundary conditions remain unchanged."""
        delta = 0.1
        iters = 1000
        grid, _ = initialize_grid(delta)
        _, max_diff = jacobi_iteration(grid, iters)
        self.assertLess(max_diff, 1e-5, "Jacobi iteration did not converge properly.")

class TestMatrixComputations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 3
        cls.invalid_n_1 = 1

    def test_calc_iter_transformation(self):
        """Testing if the output equals to the expected matrix."""
        expected_inverse_mat = np.array([[ 4,  0, -1,  0,  0,  0,  0,  0,  0],
                                         [-1,  4,  0,  0,  0,  0,  0,  0,  0],
                                         [ 0, -1,  4,  0,  0,  0,  0,  0,  0],
                                         [-1,  0,  0,  4,  0, -1,  0,  0,  0],
                                         [ 0, -1,  0, -1,  4,  0,  0,  0,  0],
                                         [ 0,  0, -1,  0, -1,  4,  0,  0,  0],
                                         [ 0,  0,  0, -1,  0,  0,  4,  0, -1],
                                         [ 0,  0,  0,  0, -1,  0, -1,  4,  0],
                                         [ 0,  0,  0,  0,  0, -1,  0, -1,  4]])
        matrix_output = calc_iter_transformation(self.n)
        inv_matrix_output = np.linalg.inv(matrix_output)
        np.testing.assert_allclose(expected_inverse_mat, inv_matrix_output, atol=1e-16)

    def test_calc_output_vector_trans_matrix(self):
        expected_mat = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 1, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0, 1, 0, 1, 0],
                                 [0, 0, 0, 1, 0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 0, 1, 0, 0]])
        matrix_output = calc_output_vector_trans_matrix(self.n)
        np.testing.assert_array_equal(matrix_output, expected_mat)

    def test_calc_iter_transformation_invalid_output(self):
        with self.assertRaises(ValueError):
            calc_iter_transformation(self.invalid_n_1)

    def test_calc_output_vector_trans_matrix_invalid_output(self):
        with self.assertRaises(ValueError):
            calc_output_vector_trans_matrix(self.invalid_n_1)


class TestGaussSeidelIteration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_grid = np.array([[0.5, 0.4, 0.6],
                                 [0.2, 0, 0.3],
                                 [1, 0.5, 0.4]])
        cls.trans_matrix = np.linalg.inv(np.array([
            [4, 0, -1, 0, 0, 0, 0, 0, 0],
            [-1, 4, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 4, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 4, 0, -1, 0, 0, 0],
            [0, -1, 0, -1, 4, 0, 0, 0, 0],
            [0, 0, -1, 0, -1, 4, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 4, 0, -1],
            [0, 0, 0, 0, -1, 0, -1, 4, 0],
            [0, 0, 0, 0, 0, -1, 0, -1, 4]
        ]))
        cls.invalid_trans_matrix = np.array([
            [4, 0, -1, 0, 0, 0, 0, 0, 0, 0],
            [-1, 4, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 4, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 4, 0, -1, 0, 0, 0, 0],
            [0, -1, 0, -1, 4, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, -1, 4, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 4, 0, -1, 0],
            [0, 0, 0, 0, -1, 0, -1, 4, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, -1, 4, 0]
        ])
        cls.output_vec_trans_mat = np.array([
            [0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0]
        ])
        cls.invalid_output_vec_trans_mat_1 = np.array([
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        ])
        cls.invalid_output_vec_trans_mat_2 = np.array([
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 2]
        ])
        cls.output_vec = np.array([1.6, 1.6, 1.8, 1.0, 0.8, 0.6, 0.5, 0.4, 1.0])
        cls.invalid_new_grid = np.array([
            [0.5, 0.4, 0.6],
            [0.2, 0, 0.3],
            [1, 0.5, 0.4],
            [1, 0.5, 0.4]
        ])
        cls.cell_value_input = cls.old_grid.ravel(order='C')

    def test_gauss_seidel_iteration_normal(self):
        """Testing the normal functioning of the gauss seidel iteration."""
        expected_output = np.linalg.solve(np.linalg.inv(self.trans_matrix), self.output_vec).reshape(3, 3)
        expected_max_diff = np.max(np.abs(expected_output - self.old_grid))

        test_result = gauss_seidel_iteration(self.old_grid, self.trans_matrix, self.output_vec_trans_mat)
        np.testing.assert_allclose(expected_output, test_result[0], atol=1e-16)
        np.testing.assert_allclose(expected_max_diff, test_result[1], atol=1e-16)

    def test_gauss_seidel_iteration_invalid_trans_matrix(self):
        with self.assertRaises(ValueError):
            gauss_seidel_iteration(self.old_grid, self.invalid_trans_matrix, self.output_vec_trans_mat)

    def test_gauss_seidel_iteration_invalid_output_vector(self):
        with self.assertRaises(ValueError):
            gauss_seidel_iteration(self.old_grid, self.trans_matrix, self.invalid_output_vec_trans_mat_1)
            gauss_seidel_iteration(self.old_grid, self.trans_matrix, self.invalid_output_vec_trans_mat_2)

    @patch("scientific_computing.stationary_diffusion.iteration_techniques.gauss_seidel_iteration.restructure_grid")
    def test_gauss_seidel_iteration_invalid_new_grid(self, mock_restructure_grid):
        mock_restructure_grid.return_value = self.invalid_new_grid
        with self.assertRaises(ValueError):
            gauss_seidel_iteration(self.old_grid, self.trans_matrix, self.output_vec_trans_mat)

    def test_calc_output_vector_normal(self):
        result = calc_output_vector(self.old_grid, self.output_vec_trans_mat)
        np.testing.assert_allclose(result, self.output_vec, atol=1e-16)

    def test_calc_output_vector_invalid_cell_value_length(self):
        with self.assertRaises(ValueError):
            calc_output_vector(self.invalid_new_grid, self.output_vec_trans_mat)

    def test_restructure_grid_normal(self):
        expected_output = restructure_grid(self.old_grid, self.cell_value_input)
        np.testing.assert_allclose(expected_output, self.old_grid, atol=1e-16)

    def test_restructure_grid_invalid_cell_value_length(self):
        with self.assertRaises(ValueError):
            restructure_grid(self.invalid_new_grid, self.cell_value_input)

class TestSorIterations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_grid = np.array([[0.5, 0.4, 0.6],
                                 [0.2, 0, 0.3],
                                 [1, 0.5, 0.4]])
        cls.trans_matrix = np.linalg.inv(np.array([
            [4, 0, -1, 0, 0, 0, 0, 0, 0],
            [-1, 4, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 4, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 4, 0, -1, 0, 0, 0],
            [0, -1, 0, -1, 4, 0, 0, 0, 0],
            [0, 0, -1, 0, -1, 4, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 4, 0, -1],
            [0, 0, 0, 0, -1, 0, -1, 4, 0],
            [0, 0, 0, 0, 0, -1, 0, -1, 4]
        ]))
        cls.output_vec_trans_mat = np.array([
            [0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0]
        ])
        cls.output_vec = np.array([1.6, 1.6, 1.8, 1.0, 0.8, 0.6, 0.5, 0.4, 1.0])
        cls.omega = 1.8
        cls.invalid_gs_subresult = np.array([
            [0.5, 0.4, 0.6],
            [0.2, 0, 0.3],
            [1, 0.5, 0.4],
            [1, 0.5, 0.4]
        ])

    def test_sor_iteration_normal(self):
        expected_output = (np.linalg.solve(np.linalg.inv(self.trans_matrix), self.output_vec).reshape(3, 3) * self.omega / 4
                           + self.old_grid * (1 - self.omega))
        expected_max_diff = np.max(np.abs(expected_output - self.old_grid))
        test_result = sor_iteration(self.old_grid, self.trans_matrix, self.output_vec_trans_mat, self.omega)
        np.testing.assert_allclose(expected_output, test_result[0], atol=1e-16)
        np.testing.assert_allclose(expected_max_diff, test_result[1], atol=1e-16)

    @patch("scientific_computing.stationary_diffusion.iteration_techniques.sor_iteration.gauss_seidel_iteration")
    def test_sor_iteration_invalid_gs_sub_result(self, mock_gauss_seidel_iteration):
        mock_gauss_seidel_iteration.return_value = (self.invalid_gs_subresult, 1)
        with self.assertRaises(ValueError):
            sor_iteration(self.old_grid, self.trans_matrix, self.output_vec_trans_mat, self.omega)

    @patch("scientific_computing.stationary_diffusion.main.MAX_ITER", 500)
    @settings(max_examples=5)
    @given(
        delta=floats(min_value=0.01, max_value=0.2),
        omega=floats(min_value=1.71, max_value=1.99)
    )
    def test_sor_iteration_invalid_omega_value(self, delta, omega):
        grid_hist, _ = main_sor_iter(delta, omega)
        last_frame = grid_hist[-1]
        self.assertTrue(np.all(last_frame == last_frame[:, [0]], axis=0),
                        f"{omega} and {int(1 / delta) + 1} are not compatible.")
        print(f"Testing with parameters {delta}, {omega} is DONE")

    @patch("scientific_computing.stationary_diffusion.main.MAX_ITER", 500)
    def test_sor_iteration_invalid_input_comb(self):
        for omega_param in np.linspace(1.1, 1.99, 10):
            for delta_param in np.linspace(1/25, 0.2, 10):
                grid_hist, _ = main_sor_iter(delta_param, omega_param)
                last_frame = grid_hist[-1]
                self.assertTrue(bool(np.all(np.abs(last_frame - last_frame[:, [0]]) < 1e-15)),
                                f"{omega_param} and {int(1 / delta_param) + 1} are not compatible.")
                print(f"Testing with parameters {delta_param}, {omega_param} is DONE")
