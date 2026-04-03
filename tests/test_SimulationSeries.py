import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mesa_helper import SimulationSeries


class TestSimulationSeries(unittest.TestCase):
    def test_remove_updates_simulation_count(self):
        series = SimulationSeries(series_dir="tests/LOGS")

        self.assertEqual(series.n_simulations, 2)
        self.assertEqual(series.n_sims, 2)

        series.remove("test_planet_2")

        self.assertEqual(series.n_simulations, 1)
        self.assertEqual(series.n_sims, 1)
        self.assertEqual(len(series.log_dirs), 1)
        self.assertEqual(len(series.simulations), 1)


if __name__ == "__main__":
    unittest.main()
