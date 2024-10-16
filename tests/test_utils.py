# tests/test_rn.py
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
from mesa_helper import utils


class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Check for verbose command line arguments
        cls.verbose = "--verbose" in sys.argv or "-v" in sys.argv

        # define the path to this file
        cls.path = os.path.dirname(os.path.abspath(__file__))

    def setUp(self):
        pass

    def test_single_mask(self):
        """Tests whether the mask function works."""

        # if we supply no mask function, it should return all True
        x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        mask = utils.single_data_mask(x)
        self.assertTrue(np.all(mask))

        # if we supply a mask function, it should return the correct mask
        mask = utils.single_data_mask(x, lambda x: x < 0.5)
        x_comparison = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        self.assertTrue(np.allclose(x[mask], x_comparison))

    def test_multiple_mask(self):
        """Tests whether the multiple mask function works."""

        x_1 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        x_2 = -x_1

        # if we supply no mask function, it should return all True
        mask = utils.multiple_data_mask([x_1, x_2])
        self.assertTrue(np.all(mask))

        # if we supply only one mask function, it should return the correct mask
        mask = utils.multiple_data_mask([x_1, x_2], [lambda x: x < 0.5, None])
        x_1_comparison = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        self.assertTrue(np.allclose(x_1[mask], x_1_comparison))
        
        # if we supply a mask function, it should return the correct mask
        mask = utils.multiple_data_mask([x_1, x_2], [lambda x: x < 0.5, lambda x: x < -0.1])
        x_1_comparison = np.array([0.2, 0.3, 0.4])
        self.assertTrue(np.allclose(x_1[mask], x_1_comparison))


if __name__ == "__main__":
    unittest.main()
