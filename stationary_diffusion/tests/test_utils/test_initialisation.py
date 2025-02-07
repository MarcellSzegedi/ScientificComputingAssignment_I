import unittest

import numpy as np

from stationary_diffusion.utils.grid_initialisation import initialize_grid

class TestGridInitialisation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.delta_1 = 1 / 3
        cls.delta_2 = 1 / 3.5

    def test_initialize_grid_whole_grid_size(self):
        """Testing the output if the given delta produces a whole number as the grid size."""
        expected_grid = np.array([[1, 1, 1],
                                  [0, 0, 0],
                                  [0, 0, 0]])
        grid_result = initialize_grid(delta=self.delta_1)
        self.assertEqual(grid_result, expected_grid)
