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

    def test_kwargs_cycling_single_value(self):
        series = SimulationSeries(series_dir="tests/LOGS")
        fig, ax = series.history_plot(
            x="star_age",
            y="num_zones",
            color="red",
            linewidth=2.5,
        )
        lines = ax.get_lines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].get_color(), "red")
        self.assertEqual(lines[1].get_color(), "red")
        self.assertEqual(lines[0].get_linewidth(), 2.5)
        self.assertEqual(lines[1].get_linewidth(), 2.5)

    def test_kwargs_cycling_list_values(self):
        series = SimulationSeries(series_dir="tests/LOGS")
        fig, ax = series.history_plot(
            x="star_age",
            y="num_zones",
            color=["blue", "green"],
            linewidth=[1.5, 3.5],
        )
        lines = ax.get_lines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].get_color(), "blue")
        self.assertEqual(lines[1].get_color(), "green")
        self.assertEqual(lines[0].get_linewidth(), 1.5)
        self.assertEqual(lines[1].get_linewidth(), 3.5)

    def test_kwargs_cycling_list_values_cycle(self):
        series = SimulationSeries(series_dir="tests/LOGS")
        fig, ax = series.history_plot(
            x="star_age",
            y="num_zones",
            color=["orange"],
        )
        lines = ax.get_lines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].get_color(), "orange")
        self.assertEqual(lines[1].get_color(), "orange")

    def test_kwargs_rgb_dashes_single_values(self):
        series = SimulationSeries(series_dir="tests/LOGS")
        fig, ax = series.history_plot(
            x="star_age",
            y="num_zones",
            color=[1.0, 0.0, 0.0],
            dashes=[2, 2],
        )
        lines = ax.get_lines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].get_color(), [1.0, 0.0, 0.0])
        self.assertEqual(lines[1].get_color(), [1.0, 0.0, 0.0])
        self.assertEqual(lines[0].get_linestyle(), lines[1].get_linestyle())

    def test_add_profile_data_at_condition(self):
        """Tests SimulationSeries.add_profile_data_at_condition (both Case A and Case B)."""
        series = SimulationSeries(series_dir="tests/LOGS")
        
        # 1. Test grid point selection (Case A)
        series.add_profile_data_at_condition(
            quantity="mass",
            condition="zone",
            value=1,
            profile_number=1,
            name="mass_zone_1"
        )
        self.assertIn("mass_zone_1", series.results.columns)
        self.assertEqual(len(series.results), 2)
        
        # 2. Test profile selection + reduction (Case B)
        series.add_profile_data_at_condition(
            quantity="mass_Jup",
            condition="star_age",
            value=1e4,
            kind="integrate",
            unit="M_Jup",
            name="integrated_mass_Jup_1e4"
        )
        self.assertIn("integrated_mass_Jup_1e4", series.results.columns)
        
        # 3. Test profile selection + mean + filter
        import numpy as np
        series.add_profile_data_at_condition(
            quantity="entropy",
            condition="star_age",
            value=1e4,
            kind="mean",
            filter_x=lambda dm: np.cumsum(dm) > 0.0,
            name="mean_entropy_filtered_1e4"
        )
        self.assertIn("mean_entropy_filtered_1e4", series.results.columns)


if __name__ == "__main__":
    unittest.main()
