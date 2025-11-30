import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mesa_helper import Simulation


class TestInterpolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verbose = "--verbose" in sys.argv or "-v" in sys.argv
        cls.tests_path = os.path.dirname(os.path.abspath(__file__))

    def setUp(self):
        self.sim = Simulation(parent_dir=self.tests_path + "/LOGS", simulation_dir="test_planet_1", verbose=self.verbose)

    def test_interpolate_profile_data(self):
        """Test that interpolating profile data reproduces original profile points closely."""
        # choose a profile number that exists in the test logs (1 is used throughout tests)
        profile_number = 1

        # pick keys present in profile tests
        x_key = "zone"
        y_key = "mass_Jup"

        # get the raw profile data
        profile = self.sim.log.profile_data(profile_number=profile_number)
        x = profile.data(x_key)
        y = profile.data(y_key)

        # build interpolation callable using the public wrapper
        interp = self.sim.interpolate_profile_data(x=x_key, y=y_key, profile_number=profile_number)
        
        # evaluate interpolation at original x points
        y_interp = interp(x)

        # compute relative error (ignore zeros)
        rel_err = np.abs(y - y_interp) / np.abs(y)

        # assert median relative error is small
        # profile interpolation may introduce small relative differences depending on
        self.assertLess(np.median(rel_err), 1e-6)


    def test_interpolate_history_data(self):
        """Test that interpolating history data reproduces original history points closely."""
        x_key = "star_age"
        y_key = "star_mass"

        # get history arrays
        history = self.sim.log.history
        x = history.data(x_key)
        y = history.data(y_key)

        interp = self.sim._interpolate_mesa_data(x=x_key, y=y_key, kind="history")

        y_interp = interp(x)

        rel_err = np.abs(y - y_interp) / np.abs(y)

        # history may be more coarsely sampled; allow a slightly larger tolerance
        self.assertLess(np.median(rel_err), 1e-6)


if __name__ == "__main__":
    unittest.main()
