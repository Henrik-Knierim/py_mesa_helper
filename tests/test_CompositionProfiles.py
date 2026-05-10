"""Unit tests for CompositionProfiles class.

Tests helper functions that translate mathematical profiles into
composition-specific profiles.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mesa_helper import CompositionProfiles, MathProfile


class TestLinearCompositionProfiles(unittest.TestCase):
    """Tests for linear composition helper profiles."""

    def test_lin_slope_fixed(self):
        """Test linear profile with fixed slope."""
        m = np.array([0.0, 0.5, 1.0])
        result = CompositionProfiles.lin_slope_fixed(m, m_core=0.5, f_0=1.0, f_atm=0.5)
        expected = np.array([1.0, 0.5, 0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_lin_M_z(self):
        """Test linear profile with fixed integral."""
        m = np.array([0.0, 0.5, 1.0])
        result = CompositionProfiles.lin_M_z(m, m_1=0.2, m_2=0.8, M_z=0.2, f_atm=0.1)
        self.assertEqual(len(result), 3)
        self.assertTrue(np.all(result >= 0.1))
        self.assertTrue(np.all(result <= 1.0))


class TestPiecewiseCompositionProfiles(unittest.TestCase):
    """Tests for piecewise composition profiles."""

    def test_piecewise_exponential_transition(self):
        """Test piecewise profile with exponential transition."""
        m = np.linspace(0, 1, 101)
        result = CompositionProfiles.piecewise_with_smoothed_exponential_transition(
            m,
            m_core=0.2,
            dm_core=0.05,
            m_dilute=0.7,
            dm_dilute=0.05,
            f_core=1.0,
            f_env=0.0,
            alpha=-1.0,
        )

        self.assertAlmostEqual(result[0], 1.0, places=1)
        self.assertAlmostEqual(result[-1], 0.0, places=1)

        diffs = np.diff(result)
        self.assertTrue(np.all(diffs <= 0.01), "Should be approximately decreasing")

    def test_piecewise_two_transitions(self):
        """Test piecewise profile with two exponential transitions."""
        m = np.linspace(0, 1, 101)
        result = (
            CompositionProfiles.piecewise_with_two_smoothed_exponential_transitions(
                m,
                alphas=[-1.0, -1.0],
                m_cores=[0.2, 0.5, 0.8],
                dm_cores=[0.05, 0.05, 0.05],
                f_values=[1.0, 0.5, 0.0],
            )
        )

        self.assertEqual(len(result), len(m))

        diffs = np.diff(result)
        violations = np.sum(diffs > 0.01)
        self.assertLess(violations, 5, "Should be mostly decreasing")


class TestJoinCompositions(unittest.TestCase):
    """Tests for joining compositional gradients."""

    def test_join_simple(self):
        """Test joining two simple functions."""
        m = np.linspace(0, 1, 101)

        f_1 = lambda m: np.ones_like(m)
        f_2 = lambda m: 0.5 * np.ones_like(m)

        joined = CompositionProfiles.join_compositional_gradients(
            profile_functions=[f_1, f_2],
            transition_functions=[MathProfile.cubic_transition],
            transition_masses=[0.5],
            dms=[0.1],
        )

        result = joined(m)

        self.assertAlmostEqual(result[0], 1.0, places=1)
        self.assertAlmostEqual(result[-1], 0.5, places=1)

    def test_join_multiple(self):
        """Test joining multiple profile functions."""
        m = np.linspace(0, 1, 101)

        f_1 = lambda m: np.ones_like(m)
        f_2 = lambda m: 0.7 * np.ones_like(m)
        f_3 = lambda m: 0.3 * np.ones_like(m)

        joined = CompositionProfiles.join_compositional_gradients(
            profile_functions=[f_1, f_2, f_3],
            transition_functions=[
                MathProfile.cubic_transition,
                MathProfile.cubic_transition,
            ],
            transition_masses=[0.33, 0.67],
            dms=[0.05, 0.05],
        )

        result = joined(m)

        self.assertEqual(len(result), len(m))
        self.assertGreater(result[0], result[-1])


if __name__ == "__main__":
    unittest.main()
