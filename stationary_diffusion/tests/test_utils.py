import unittest

import numpy as np
import numpy.testing as npt

from stationary_diffusion.utils.common_functions import reset_grid_wrapping
from stationary_diffusion.utils.constants import LATTICE_LENGTH
from stationary_diffusion.utils.grid_initialisation import initialize_grid


class TestGridInitialisation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.delta_1 = LATTICE_LENGTH / 3
        cls.delta_2 = LATTICE_LENGTH / 3.5
        cls.delta_3 = 1.5 * LATTICE_LENGTH

    def test_initialize_grid_whole_grid_size(self):
        """Testing the output if the given delta produces a whole number as the
        grid size."""
        expected_grid = np.array(
            [
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=float,
        )
        grid_result = initialize_grid(delta=self.delta_1)
        npt.assert_array_equal(grid_result, expected_grid)

    def test_initialize_grid_fraction_grid_size(self):
        """Testing the output if the given delta produces a fraction number as the grid
        size."""
        expected_grid = np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        )
        grid_result = initialize_grid(delta=self.delta_2)
        npt.assert_array_equal(grid_result, expected_grid)

    def test_initialize_grid_big_delta(self):
        """Testing whether the function raises error if the delta input is too large."""
        with self.assertRaises(ValueError):
            _ = initialize_grid(delta=self.delta_3)


class TestCommonFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.existing_grid = np.array(
            [
                [1, 0.8, 0.6, 1, 0.7],
                [0.5, 0.3, 0.2, 0.1, 0.4],
                [0, 0, 0.1, 0.2, 0.3],
                [0, 0.15, 0, 0.05, 0.1],
                [0.2, 0.1, 0.005, 0.1, 0.15],
            ]
        )
        cls.rectangular_grid = np.array(
            [
                [1, 0.8, 0.6, 1, 0.7],
                [0.5, 0.3, 0.2, 0.1, 0.4],
                [0, 0, 0.1, 0.2, 0.3],
                [0, 0.15, 0, 0.05, 0.1],
            ]
        )
        cls.one_d_grid = np.array(
            [
                [1, 0.8, 0.6, 1, 0.7],
            ]
        )

    def test_reset_grid_wrapping(self):
        """Testing the new wrapping (outer layer of the 2D numpy array)"""
        expected_grid = np.array(
            [
                [1, 1, 1, 1, 1],
                [0.1, 0.3, 0.2, 0.1, 0.3],
                [0.2, 0, 0.1, 0.2, 0],
                [0.05, 0.15, 0, 0.05, 0.15],
                [0, 0, 0, 0, 0],
            ]
        )
        grid_result = reset_grid_wrapping(self.existing_grid)
        npt.assert_array_equal(grid_result, expected_grid)

    def test_reset_grid_wrapping_incorrect_shapes_1(self):
        """Testing whether the function raises error due to the rectangular grid
        shape."""
        with self.assertRaises(ValueError):
            _ = reset_grid_wrapping(self.rectangular_grid)

    def test_reset_grid_wrapping_incorrect_shapes_2(self):
        """Testing whether the function raises error due to the 1D grid."""
        with self.assertRaises(ValueError):
            _ = reset_grid_wrapping(self.one_d_grid)
