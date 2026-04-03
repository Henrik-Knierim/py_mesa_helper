import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mesa_helper import SimulationSeries


class TestSimulationSeries(unittest.TestCase):
    def test_empty_series_can_add_simulation(self):
        series = SimulationSeries()

        self.assertEqual(series.n_simulations, 0)
        self.assertEqual(series.n_sims, 0)
        self.assertEqual(series.log_dirs, [])
        self.assertEqual(series.simulations, {})

        series.add_simulation("tests/LOGS/test_planet_1")

        self.assertEqual(series.n_simulations, 1)
        self.assertEqual(series.n_sims, 1)
        self.assertEqual(series.log_dirs, ["test_planet_1"])
        self.assertIn("test_planet_1", series.simulations)
        self.assertEqual(series.results["log_dir"].tolist(), ["test_planet_1"])

    def test_add_simulation_accepts_lists(self):
        series = SimulationSeries()
        series.add_simulation(["tests/LOGS/test_planet_1", "tests/LOGS/test_planet_2"])

        self.assertEqual(series.n_simulations, 2)
        self.assertEqual(series.n_sims, 2)
        self.assertEqual(sorted(series.log_dirs), ["test_planet_1", "test_planet_2"])
        self.assertEqual(
            sorted(series.results["log_dir"].tolist()),
            ["test_planet_1", "test_planet_2"],
        )

    def test_remove_updates_simulation_count(self):
        series = SimulationSeries(series_dir="tests/LOGS")

        self.assertEqual(series.n_simulations, 2)
        self.assertEqual(series.n_sims, 2)

        series.remove_simulation("test_planet_2")

        self.assertEqual(series.n_simulations, 1)
        self.assertEqual(series.n_sims, 1)
        self.assertEqual(len(series.log_dirs), 1)
        self.assertEqual(len(series.simulations), 1)

    def test_remove_simulation_accepts_lists(self):
        series = SimulationSeries(series_dir="tests/LOGS")
        series.remove_simulation(["test_planet_1", "test_planet_2"])

        self.assertEqual(series.n_simulations, 0)
        self.assertEqual(series.n_sims, 0)
        self.assertEqual(series.log_dirs, [])
        self.assertEqual(series.simulations, {})
        self.assertTrue(series.results.empty)

    def test_add_history_data_with_multiple_keys(self):
        series = SimulationSeries(series_dir="tests/LOGS")
        series.add_history_data(["num_zones", "star_age"])

        self.assertIn("num_zones", series.results.columns)
        self.assertIn("star_age", series.results.columns)
        self.assertEqual(len(series.results), series.n_simulations)


if __name__ == "__main__":
    unittest.main()
