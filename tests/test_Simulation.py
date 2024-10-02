# tests/test_rn.py
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
from mesa_helper import Simulation


class TestSimulation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Check for verbose command line arguments
        cls.verbose = "--verbose" in sys.argv or "-v" in sys.argv

        # define the path to this file
        cls.path = os.path.dirname(os.path.abspath(__file__))

    def setUp(self):

        # Create the inlist instance that we want to change and compare
        self.path = "tests/LOGS"
        self.sim = Simulation(
            parent_dir=self.path, simulation_dir="test_planet_1", verbose=self.verbose
        )

    def test_conservation_check(self):
        """Tests whether the convergence testing functions work."""

        quantities = {"star_mass": True, "model_number": False, "star_age": False}
        for quant, is_conserved in quantities.items():
            convergence_check = self.sim.check_if_conserved(quant)
            self.assertEqual(convergence_check, is_conserved)

    def test_value_check(self):
        """Tests whether the value checking functions work."""

        quantities = {
            "star_age": {"value": 5e9, "is_reached": True},
            "model_number": {"value": 131, "is_reached": True},
            "Teff": {"value": 1e3, "is_reached": False},
            "num_zones": {"value": 100, "is_reached": False},
        }
        for quant, item in quantities.items():
            value_check = self.sim.check_value(quant, item["value"])
            self.assertEqual(value_check, item["is_reached"])

    def test_add_history_data(self):
        """Tests whether the history file is added to the simulation object."""

        self.sim.add_history_data("num_zones")
        self.sim.add_history_data("center_entropy", "star_age", 1e7)
        results = pd.DataFrame(
            {
                "log_dir": [self.sim.sim],
                "num_zones": 560,
                "center_entropy": 8.0942554212152853,
            }
        )

        # star age clostest to 1e7
        # compare if results is equal to sim.results
        self.assertTrue(results.equals(self.sim.results))

    def test_profile_integration(self):
        """Tests whether the integration methods work."""

        # the planet's mass is 1 M_Jup, so the integrated mass should be 0.5 M_Jup
        m_int_comparison: float = 0.5

        m_int: np.float_ = self.sim.integrate("mass_Jup", mass_unit="M_Jup")
        m_mean = self.sim.mean("mass_Jup")

        if self.verbose:
            print(f"Integrated mass: {m_int}")
            print(f"Mean mass: {m_mean}")
            print(f"Comparison mass: {m_int_comparison}")

        self.assertAlmostEqual(m_int, m_int_comparison, places=2)
        self.assertAlmostEqual(m_mean, m_int_comparison, places=2)

    def test_get_profile_data_at_condition(self):
        """Tests whether the get_profile_at_condition method works."""

        # get the profile at the point where the temperature is 1e6 K
        mass = self.sim.get_profile_data_at_condition(
            quantity="mass", condition="zone", value=1, profile_number=1
        )
        mass_comparison = 9.5459960393944109e-004
        self.assertEqual(mass, mass_comparison)

    def test_profile_header_functions(self):
        """Tests whether the profile header functions work."""

        # Test if the star_age of the reference profile is correct
        self.sim._create_profile_header_df('star_age')
        star_age_df = self.sim.profile_header_df
        star_age = star_age_df['star_age'].values
        star_age_comparison = np.array(
            [
                1.11463044e02,
                1.09040003e03,
                1.13391014e04,
                1.17860522e05,
                1.08150539e06,
                1.03526007e07,
                1.07114530e08,
                2.06146427e08,
                1.03336245e09,
                2.13885379e09,
                3.04759198e09,
                4.61789159e09,
                5.00000000e09,
            ]
        )

        self.assertTrue(np.allclose(star_age, star_age_comparison))

    def test_get_profile_data_at_header_condition(self):
        """Tests whether the get_profile_data_at_header_condition method works."""
        t = 1e3
        zone = self.sim.get_profile_data_at_header_condition('zone','star_age', t)[-1]
        zone_comparison = 489
        self.assertEqual(zone, zone_comparison)

        logRho = self.sim.get_profile_data_at_header_condition('logRho','star_age', t)[0]
        logRho_comparison = -6.4074845953528037E+000
        self.assertEqual(logRho, logRho_comparison)

    def test_get_mean_profile_data_sequence(self):
        """Tests whether the get_mean_profile_data_sequence method works."""
        entropy_profile = self.sim.get_mean_profile_data_sequence("entropy", profile_numbers = [1, -1])
        entropy_model = self.sim.get_mean_profile_data_sequence("entropy", model_numbers = [1, -1])

        entropy_comparison = [10.977800300599148, 5.763208925874653]

        self.assertTrue(np.allclose(entropy_profile, entropy_comparison))
        self.assertTrue(np.allclose(entropy_model, entropy_comparison))

    def test_export_history(self):
        """Tests whether the export_history method works."""
        self.sim.export_history_data('tests/export_history.csv', ['star_age', 'm_RCB', 's_env'])

        # read the exported file and the comparison and assess if they are equal
        comparison = pd.read_csv('tests/export_history_comparison.csv')
        exported = pd.read_csv('tests/export_history.csv')

        # remove the exported file
        os.remove('tests/export_history.csv')

        self.assertTrue(comparison.equals(exported))

    def test_export_profile(self):
        """Tests whether the export_profile method works."""
        self.sim.export_profile_data('tests/export_profile.csv', ['zone', 'entropy'])

        # read the exported file and the comparison and assess if they are equal
        comparison = pd.read_csv('tests/export_profile_comparison.csv')
        exported = pd.read_csv('tests/export_profile.csv')

        # remove the exported file
        os.remove('tests/export_profile.csv')

        self.assertTrue(comparison.equals(exported))


if __name__ == "__main__":
    unittest.main()
