import unittest

import numpy as np
import numpy.testing as npt

from scientific_computing.stationary_diffusion.utils.grid_initialisation import (
    initialize_grid,
)

LATTICE_LENGTH = 1


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
                [1, 1, 1],
                [0, 0, 0],
                [0, 0, 0]
            ],
            dtype=float,
        )
        grid_result, _ = initialize_grid(delta=self.delta_1)
        npt.assert_array_equal(grid_result, expected_grid)

    def test_initialize_grid_fraction_grid_size(self):
        """Testing the output if the given delta produces a fraction number as the grid
        size."""
        expected_grid = np.array(
            [
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ],
            dtype=float,
        )
        grid_result, _ = initialize_grid(delta=self.delta_2)
        npt.assert_array_equal(grid_result, expected_grid)

    def test_initialize_grid_big_delta(self):
        """Testing whether the function raises error if the delta input is too large."""
        with self.assertRaises(ValueError):
            initialize_grid(delta=self.delta_3)
