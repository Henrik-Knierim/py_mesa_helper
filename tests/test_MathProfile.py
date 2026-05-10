"""Unit tests for MathProfile class.

Tests all mathematical profile and transition functions provided by MathProfile.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mesa_helper import MathProfile


class TestLinearProfile(unittest.TestCase):
    """Tests for linear profile functions."""

    def test_lin_basic(self):
        """Test basic linear profile between two points."""
        m = np.array([0.0, 0.5, 1.0])
        result = MathProfile.lin(m, m_1=0.0, m_2=1.0, f_1=1.0, f_2=0.0)
        expected = np.array([1.0, 0.5, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_lin_invalid_bounds(self):
        """Test that invalid bounds raise errors."""
        m = np.array([0.5])

        # m_2 < m_1
        with self.assertRaises(ValueError):
            MathProfile.lin(m, m_1=1.0, m_2=0.0, f_1=1.0, f_2=0.0)

        # m_1 < 0
        with self.assertRaises(ValueError):
            MathProfile.lin(m, m_1=-0.5, m_2=1.0, f_1=1.0, f_2=0.0)


class TestStepwiseProfile(unittest.TestCase):
    """Tests for stepwise profile."""

    def test_stepwise_basic(self):
        """Test basic stepwise profile."""
        m = np.array([0.0, 0.5, 1.0])
        result = MathProfile.stepwise(m, m_transition=0.5, f_1=1.0, f_2=0.0)
        expected = np.array([1.0, 1.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_stepwise_transition_at_boundary(self):
        """Test stepwise at transition boundary."""
        m = np.array([0.5])
        result = MathProfile.stepwise(m, m_transition=0.5, f_1=1.0, f_2=0.0)
        # At transition point, should use f_1 (<=)
        np.testing.assert_array_equal(result, np.array([1.0]))


class TestExponentialProfile(unittest.TestCase):
    """Tests for exponential profile."""

    def test_exponential_basic(self):
        """Test basic exponential profile."""
        m = np.linspace(0, 1, 11)
        result = MathProfile.exponential(
            m, alpha=-1.0, m_start=0.0, m_end=1.0, f_start=1.0, f_end=0.0
        )
        # Check boundary values
        self.assertAlmostEqual(result[0], 1.0, places=5)
        self.assertAlmostEqual(result[-1], 0.0, places=5)
        # Check monotonicity
        diffs = np.diff(result)
        self.assertTrue(np.all(diffs <= 0), "Should be monotonically decreasing")

    def test_exponential_alpha_zero(self):
        """Test exponential with alpha=0 (becomes linear)."""
        m = np.array([0.0, 0.5, 1.0])
        result = MathProfile.exponential(
            m, alpha=0.0, m_start=0.0, m_end=1.0, f_start=1.0, f_end=0.0
        )
        expected = np.array([1.0, 0.5, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_exponential_invalid_bounds(self):
        """Test invalid bounds for exponential."""
        m = np.array([0.5])

        # m_end < m_start
        with self.assertRaises(ValueError):
            MathProfile.exponential(
                m, alpha=-1.0, m_start=1.0, m_end=0.0, f_start=1.0, f_end=0.0
            )


class TestGaussianProfile(unittest.TestCase):
    """Tests for Gaussian profile."""

    def test_gaussian_basic(self):
        """Test basic Gaussian profile."""
        m = np.linspace(0, 1, 11)
        f_atm = 0.0
        f_core = 1.0
        result = MathProfile.gaussian(m, M_z=0.5, f_core=f_core, f_atm=f_atm)

        # Check that it's between f_atm and f_core
        self.assertTrue(np.all(result >= f_atm))
        self.assertTrue(np.all(result <= f_core))

        # Check monotonicity (should decrease)
        diffs = np.diff(result)
        self.assertTrue(np.all(diffs <= 0), "Should be monotonically decreasing")

    def test_gaussian_at_center(self):
        """Test Gaussian at m=0."""
        m = np.array([0.0])
        result = MathProfile.gaussian(m, M_z=0.5, f_core=1.0, f_atm=0.1)
        # At m=0, should have maximum value
        self.assertAlmostEqual(result[0], 1.0, places=5)


class TestReverseSigmoidProfile(unittest.TestCase):
    """Tests for reverse sigmoid profile."""

    def test_reverse_sigmoid_basic(self):
        """Test basic reverse sigmoid profile."""
        m = np.linspace(0, 2, 11)
        f_core = 1.0
        f_env = 0.0
        result = MathProfile.reverse_sigmoid(
            m, m_b=1.0, steepness=10, f_core=f_core, f_env=f_env
        )

        # Check that values are between f_env and f_core
        self.assertTrue(np.all(result >= f_env))
        self.assertTrue(np.all(result <= f_core))

        # At m_b, should be close to midpoint
        idx_mid = np.argmin(np.abs(m - 1.0))
        self.assertAlmostEqual(result[idx_mid], 0.5, places=1)

    def test_reverse_sigmoid_monotonicity(self):
        """Test that reverse sigmoid is monotonic."""
        m = np.linspace(0, 2, 101)
        result = MathProfile.reverse_sigmoid(
            m, m_b=1.0, steepness=10, f_core=1.0, f_env=0.0
        )
        diffs = np.diff(result)
        self.assertTrue(np.all(diffs <= 0), "Should be monotonically decreasing")


class TestTransitionFunctions(unittest.TestCase):
    """Tests for transition functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Define simple test functions
        self.f_1 = lambda m: np.ones_like(m)  # constant 1
        self.f_2 = lambda m: np.zeros_like(m)  # constant 0

    def test_linear_transition(self):
        """Test linear transition between two functions."""
        m = np.linspace(0, 1, 11)
        result = MathProfile.linear_transition(m, self.f_1, self.f_2, m_1=0.3, m_2=0.7)

        # Check boundary values
        self.assertAlmostEqual(result[0], 1.0, places=5)  # Before transition
        self.assertAlmostEqual(result[-1], 0.0, places=5)  # After transition

        # Check transition range has intermediate values
        mid_idx = len(result) // 2
        self.assertTrue(0 < result[mid_idx] < 1)

    def test_cosine_transition(self):
        """Test cosine transition."""
        m = np.linspace(0, 1, 11)
        result = MathProfile.cosine_transition(m, self.f_1, self.f_2, m_1=0.3, m_2=0.7)

        # Check boundary values
        self.assertAlmostEqual(result[0], 1.0, places=5)
        self.assertAlmostEqual(result[-1], 0.0, places=5)

    def test_cubic_transition(self):
        """Test cubic transition."""
        m = np.linspace(0, 1, 11)
        result = MathProfile.cubic_transition(m, self.f_1, self.f_2, m_1=0.3, m_2=0.7)

        # Check boundary values
        self.assertAlmostEqual(result[0], 1.0, places=5)
        self.assertAlmostEqual(result[-1], 0.0, places=5)

        # Cubic should be smoother than linear in transition region
        self.assertTrue(np.all(result >= 0) and np.all(result <= 1))

    def test_cubic_transition_fast_decrease(self):
        """Test cubic transition with fast decrease."""
        m = np.linspace(0, 1, 11)
        result = MathProfile.cubic_transition_fast_decrease(
            m, self.f_1, self.f_2, m_1=0.3, m_2=0.7
        )

        self.assertAlmostEqual(result[0], 1.0, places=5)
        self.assertAlmostEqual(result[-1], 0.0, places=5)

    def test_exponential_transition(self):
        """Test exponential transition."""
        m = np.linspace(0, 1, 11)
        result = MathProfile.exponential_transition(
            m, self.f_1, self.f_2, m_1=0.3, m_2=0.7, alpha=-1.0
        )

        self.assertAlmostEqual(result[0], 1.0, places=5)
        self.assertAlmostEqual(result[-1], 0.0, places=5)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""

    def test_single_point(self):
        """Test with single point."""
        m = np.array([0.5])
        result = MathProfile.lin(m, m_1=0.0, m_2=1.0, f_1=1.0, f_2=0.0)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], 0.5)

    def test_scalar_input(self):
        """Test with scalar input (should be converted to array)."""
        result = MathProfile.lin(np.array(0.5), m_1=0.0, m_2=1.0, f_1=1.0, f_2=0.0)
        self.assertAlmostEqual(float(result), 0.5)

    def test_large_array(self):
        """Test with large array."""
        m = np.linspace(0, 1, 10000)
        result = MathProfile.lin(m, m_1=0.0, m_2=1.0, f_1=1.0, f_2=0.0)
        self.assertEqual(len(result), 10000)
        # Check endpoints
        self.assertAlmostEqual(result[0], 1.0, places=5)
        self.assertAlmostEqual(result[-1], 0.0, places=5)

    def test_identical_endpoints(self):
        """Test profiles when endpoints are identical."""
        m = np.array([0.0, 0.5, 1.0])
        result = MathProfile.lin(m, m_1=0.0, m_2=1.0, f_1=1.0, f_2=1.0)
        expected = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
