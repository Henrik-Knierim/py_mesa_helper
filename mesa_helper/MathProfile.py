# Pure mathematical functions for creating composition gradients
# These functions are domain-agnostic and can be reused in any context
# Validation is minimal and left to domain-specific classes (e.g., CompositionGradient)

import numpy as np
from scipy.special import erf
from typing import Callable


class ProfileFunction:
    """Callable profile wrapper with optional analytic integration."""

    def __init__(
        self,
        profile_func: Callable,
        analytic_func: Callable | None = None,
        default_kwargs: dict | None = None,
        analytic_kwargs_builder: Callable | None = None,
    ) -> None:
        self._profile_func = profile_func
        self._analytic_func = analytic_func
        self._default_kwargs = default_kwargs or {}
        self._analytic_kwargs_builder = analytic_kwargs_builder

    def __call__(self, m: np.ndarray, **kwargs) -> np.ndarray:
        call_kwargs = {**self._default_kwargs, **kwargs}
        return self._profile_func(m, **call_kwargs)

    def integrate(
        self,
        m_min: float,
        m_max: float,
        n: int = 10_000,
        prefer_analytic: bool = True,
        validate_analytic: bool = False,
        return_both: bool = False,
        **kwargs,
    ) -> float | tuple[float | None, float]:
        """Integrate the profile on [m_min, m_max]."""
        call_kwargs = {**self._default_kwargs, **kwargs}

        numeric = None
        if (
            self._analytic_func is None
            or validate_analytic
            or return_both
            or not prefer_analytic
        ):
            m = np.linspace(m_min, m_max, n)
            numeric = np.trapz(self._profile_func(m, **call_kwargs), m)

        analytic = None
        if self._analytic_func is not None:
            if self._analytic_kwargs_builder is None:
                analytic_kwargs = call_kwargs
            else:
                analytic_kwargs = self._analytic_kwargs_builder(
                    m_min, m_max, call_kwargs
                )
            analytic = self._analytic_func(**analytic_kwargs)

        if return_both:
            return analytic, numeric

        if self._analytic_func is not None and prefer_analytic:
            return analytic

        return numeric


class MathProfile:
    """Mathematical profile and transition functions.

    This class provides pure mathematical functions without domain-specific validation.
    It can be reused in different contexts where the valid ranges may differ.
    """

    # ----------------------------------------- #
    # ----------- Profile Functions ----------- #
    # ----------------------------------------- #

    @staticmethod
    def profile(
        profile_func: Callable,
        analytic_func: Callable | None = None,
        default_kwargs: dict | None = None,
        analytic_kwargs_builder: Callable | None = None,
    ) -> ProfileFunction:
        """Create a callable profile wrapper with optional analytic integration."""
        return ProfileFunction(
            profile_func=profile_func,
            analytic_func=analytic_func,
            default_kwargs=default_kwargs,
            analytic_kwargs_builder=analytic_kwargs_builder,
        )

    @staticmethod
    def lin(
        m: np.ndarray, m_1: float, m_2: float, f_1: float, f_2: float, **kwargs
    ) -> np.ndarray:
        """Linear profile with a constant value outside [m_1, m_2].

        Creates a piecewise-linear profile: constant at f_1 for m < m_1,
        linear interpolation between f_1 and f_2 for m_1 ≤ m ≤ m_2,
        and constant at f_2 for m > m_2.

        Parameters
        ----------
        m : np.ndarray
            Independent variable (e.g., mass coordinate).
        m_1 : float
            Start of linear region.
        m_2 : float
            End of linear region.
        f_1 : float
            Function value at m_1.
        f_2 : float
            Function value at m_2.

        Returns
        -------
        np.ndarray
            Profile values at each point in m.

        Examples
        --------
        >>> m = np.array([0, 0.5, 1.0])
        >>> MathProfile.lin(m, 0, 1, 1.0, 0.0)
        array([1. , 0.5, 0. ])
        """
        m = np.asarray(m, dtype=float)

        if m_2 < m_1:
            raise ValueError("m_2 must be larger than m_1")
        elif m_1 < 0:
            raise ValueError("m_1 needs to be >= 0")

        # linear function f = a m + b
        a = -(f_2 - f_1) / (m_1 - m_2)
        b = -(m_2 * f_1 - m_1 * f_2) / (m_1 - m_2)

        return np.piecewise(
            m,
            [m < m_1, ((m_1 <= m) & (m <= m_2)), m > m_2],
            [lambda m: f_1, lambda m: a * m + b, f_2],
        )

    @staticmethod
    def stepwise(m, m_transition, f_1, f_2, **kwargs) -> np.ndarray:
        """Stepwise profile with abrupt transition at a single mass point.

        Creates a discontinuous jump from constant value f_1 to f_2 at m_transition.
        Useful for modeling sharp compositional boundaries.

        Parameters
        ----------
        m : np.ndarray
            Independent variable (mass).
        m_transition : float
            Mass coordinate at which the step occurs.
        f_1 : float
            Function value for m ≤ m_transition.
        f_2 : float
            Function value for m > m_transition.

        Returns
        -------
        np.ndarray
            Stepwise profile values.

        Examples
        --------
        >>> m = np.array([0.0, 0.5, 1.0])
        >>> MathProfile.stepwise(m, 0.5, 1.0, 0.0)
        array([1., 1., 0.])
        """
        m = np.asarray(m, dtype=float)

        return np.piecewise(m, [m <= m_transition, m > m_transition], [f_1, f_2])

    @staticmethod
    def power_law(
        x: np.ndarray,
        power_law_exponent: float = 1.0,
        power_law_scale: float = 1.0,
        offset: float = 0.0,
    ) -> np.ndarray:
        """Power-law profile.

        Parameters
        ----------
        x : np.ndarray
            Independent variable.
        power_law_exponent : float, optional
            Power-law exponent. Default is 1.0 (linear).
        power_law_scale : float, optional
            Scale factor for the power-law. Default is 1.0.
        offset : float, optional
            Offset for the power-law. Default is 0.0.

        Returns
        -------
        np.ndarray
            Power-law profile values.
        """
        x = np.asarray(x, dtype=float)
        return np.power(x, power_law_exponent) * power_law_scale + offset

    @staticmethod
    def power_law_integral(x_min, x_max, power_law_exponent, power_law_scale, offset):
        """Integral of the power-law profile.

        Parameters
        ----------
        x_min : float
            Lower limit of integration.
        x_max : float
            Upper limit of integration.
        power_law_exponent : float, optional
            Power-law exponent. Default is 1.0 (linear).
        power_law_scale : float, optional
            Scale factor for the power-law. Default is 1.0.
        offset : float, optional
            Offset for the power-law. Default is 0.0.

        Returns
        -------
        float
            Integral of the power-law profile values.
        """
        if power_law_exponent == -1:
            return power_law_scale * (np.log(x_max) - np.log(x_min)) + offset * (
                x_max - x_min
            )
        else:
            return power_law_scale * (
                np.power(x_max, power_law_exponent + 1)
                - np.power(x_min, power_law_exponent + 1)
            ) / (power_law_exponent + 1) + offset * (x_max - x_min)

    @staticmethod
    def exponential(
        m: np.ndarray,
        alpha: float,
        m_start: float,
        m_end: float,
        f_start: float,
        f_end: float,
    ) -> np.ndarray:
        """Exponential profile between two points.

        Connects f_start and f_end via exponential decay characterized by
        parameter alpha. For alpha < 0, stronger decay occurs near m_start.
        Special case: alpha = 0 reduces to linear interpolation.

        Parameters
        ----------
        m : np.ndarray
            Independent variable.
        alpha : float
            Exponential decay constant. Negative values give smooth decay.
        m_start : float
            Independent variable where profile starts.
        m_end : float
            Independent variable where profile ends.
        f_start : float
            Profile value at m_start.
        f_end : float
            Profile value at m_end.

        Returns
        -------
        np.ndarray
            Profile values at each point in m.

        Notes
        -----
        When alpha=0, the function reduces to linear interpolation.
        For |alpha| > 0, the profile shape depends on alpha's sign and magnitude.

        Examples
        --------
        >>> m = np.linspace(0, 1, 3)
        >>> MathProfile.exponential(m, -1, 0, 1, 1.0, 0.0)
        array([1.        , 0.62...  , 0.        ])
        """
        m = np.asarray(m, dtype=float)

        if m_end < m_start:
            raise ValueError(
                f"m_end must be larger than m_start, but m_end = {m_end} and m_start = {m_start}."
            )
        elif m_start < 0:
            raise ValueError("m_start needs to be >= 0")

        if alpha == 0:
            retval = (m * f_start - m_end * f_start - m * f_end + m_start * f_end) / (
                m_start - m_end
            )
        else:
            retval = (
                np.exp(alpha * m) * (f_start - f_end)
                + np.exp(alpha * m_start) * f_end
                - np.exp(alpha * m_end) * f_start
            ) / (np.exp(alpha * m_start) - np.exp(alpha * m_end))

        return retval

    @staticmethod
    def gaussian(
        m: np.ndarray, M_z: float, f_core: float, f_atm: float, **kwargs
    ) -> np.ndarray:
        """Gaussian profile centered at core (m=0).

        Models a smooth bell-shaped distribution with peak f_core at m=0,
        asymptotically approaching f_atm. Sigma is computed to match the
        specified integral M_z.

        Parameters
        ----------
        m : np.ndarray
            Independent variable (mass).
        M_z : float
            Integral of the profile above the atmosphere level.
            Determines the width (sigma) of the Gaussian.
        f_core : float
            Maximum value at core (m=0).
        f_atm : float
            Asymptotic value at large m (atmosphere level).

        Returns
        -------
        np.ndarray
            Gaussian profile values.

        Notes
        -----
        Sigma is chosen such that the integral from 0 to 3σ equals M_z.
        """
        m = np.asarray(m, dtype=float)

        if M_z < 0:
            raise ValueError("M_z needs to be >= 0")

        # fix sigma such that the integral of the Gaussian to 3 sigma is equal to M_z
        sigma = (2.0 * M_z) / (
            6.0 * f_atm
            + (np.sqrt(2.0 * np.pi) * erf(3.0 / np.sqrt(2.0)) * (f_core - f_atm))
        )

        return f_atm + (f_core - f_atm) * np.exp(-(m**2) / (2.0 * sigma**2))

    @staticmethod
    def reverse_sigmoid(
        m: np.ndarray,
        m_b: float,
        steepness: float = 10,
        f_core: float = 1,
        f_env: float = 0,
        M_p: float = 1.0,
    ) -> np.ndarray:
        """Sigmoid profile decreasing from f_core to f_env.

        Models smooth transition using a logistic curve with midpoint at m_b.
        The steepness parameter controls transition width: larger values give
        sharper transitions.

        Parameters
        ----------
        m : np.ndarray
            Independent variable (mass).
        m_b : float
            Mass coordinate at the midpoint (where f = (f_core + f_env)/2).
        steepness : float
            Shape parameter controlling transition steepness. Default is 100.
            Larger values → sharper transition.
        f_core : float
            Function value in the core (m → -∞). Default is 1.
        f_env : float
            Function value in the envelope (m → +∞). Default is 0.
        M_p : float
            Normalization range for the exponent. Default is 1.0.

        Returns
        -------
        np.ndarray
            Sigmoid profile values.

        Notes
        -----
        Formula: f = f_core - (f_core - f_env) / (1 + exp(-steepness*(m - m_b)))
        """
        m = np.asarray(m, dtype=float)

        if m_b < 0:
            raise ValueError("m_b needs to be >= 0")
        if M_p <= 0:
            raise ValueError("M_p needs to be > 0")

        return f_core - (f_core - f_env) / (1 + np.exp(-steepness * (m - m_b) / M_p))

    @staticmethod
    def reverse_sigmoid_integral(
        M_p: float,
        m_b: float,
        steepness: float = 10,
        f_core: float = 1,
        f_env: float = 0,
    ) -> float:
        """Returns the integral of the reverse sigmoid function from 0 to M_p."""

        if M_p < 0:
            raise ValueError("M_p needs to be >= 0")
        elif m_b < 0:
            raise ValueError("m_b needs to be >= 0")
        elif not 0 <= f_core <= 1:
            raise ValueError("f_core needs to be between 0 and 1")
        elif not 0 <= f_env <= 1:
            raise ValueError("f_env needs to be between 0 and 1")

        # Analytic edge case: if M_p is zero, the integral is zero.
        if M_p == 0.0:
            return 0.0

        # Analytic edge case: if steepness is zero, the profile is constant.
        if steepness == 0.0:
            return M_p * 0.5 * (f_core + f_env)

        pI = M_p * f_core
        s = steepness / M_p

        # Numerically stable log-term via logaddexp.
        log_term = np.logaddexp(0.0, s * (M_p - m_b)) - np.logaddexp(0.0, -s * m_b)

        pII = (f_env - f_core) * log_term / s

        return pI + pII

    @staticmethod
    def integrate_profile(
        profile_func: Callable,
        m_start: float,
        m_end: float,
        n: int = 10_000,
        profile_kwargs: dict | None = None,
        analytic_func: Callable | None = None,
        analytic_kwargs: dict | None = None,
        prefer_analytic: bool = True,
        validate_analytic: bool = False,
        return_both: bool = False,
    ) -> float | tuple[float | None, float]:
        """Integrate a profile numerically, with optional analytic fallback.

        If analytic_func is provided and prefer_analytic is True, the analytic
        result is returned. Set validate_analytic or return_both to also compute
        the numerical integral for comparison.
        """
        profile_kwargs = profile_kwargs or {}
        analytic_kwargs = analytic_kwargs or {}

        numeric = None
        if (
            analytic_func is None
            or validate_analytic
            or return_both
            or not prefer_analytic
        ):
            m = np.linspace(m_start, m_end, n)
            numeric = np.trapz(profile_func(m, **profile_kwargs), m)

        analytic = None
        if analytic_func is not None:
            analytic = analytic_func(**analytic_kwargs)

        if return_both:
            return analytic, numeric

        if analytic_func is not None and prefer_analytic:
            return analytic

        return numeric

    # ----------------------------------------- #
    # --------- Transition Functions ---------- #
    # ----------------------------------------- #

    @staticmethod
    def _transition_function(
        m: np.ndarray,
        f_transition: Callable,
        f_1: Callable,
        f_2: Callable,
        m_1: float,
        m_2: float,
    ) -> np.ndarray:
        """Parent function for transition functions.

        Parameters
        ----------
        m : np.ndarray
            array of independent variables (mass bins)
        f_transition : Callable
            transition function
        f_1 : Callable
            function for the first part of the transition
        f_2 : Callable
            function for the second part of the transition
        m_1 : float
            independent variable at which the transition starts
        m_2 : float
            independent variable at which the transition ends
        """
        m = np.asarray(m, dtype=float)

        if m_2 < m_1:
            raise ValueError("m_2 must be larger than m_1")
        elif m_1 < 0:
            raise ValueError(f"m_1 needs to be >= 0, but is {m_1}")

        if m_2 == m_1:
            return np.piecewise(m, [m < m_1, m >= m_1], [f_1, f_2])

        return np.piecewise(
            m, [m < m_1, ((m_1 <= m) & (m <= m_2)), m > m_2], [f_1, f_transition, f_2]
        )

    @staticmethod
    def linear_transition(
        m: np.ndarray, f_1: Callable, f_2: Callable, m_1: float, m_2: float
    ) -> np.ndarray:
        """Linear transition between two functions (uniform decay rate).

        Linearly interpolates between f_1 and f_2 over [m_1, m_2].
        f_1's contribution decreases uniformly: c₁ = 1 - x, where x = (m-m_1)/(m_2-m_1).

        Parameters
        ----------
        m : np.ndarray
            Independent variable.
        f_1 : Callable
            Function before transition.
        f_2 : Callable
            Function after transition.
        m_1 : float
            Transition start point.
        m_2 : float
            Transition end point.

        Returns
        -------
        np.ndarray
            Linearly transitioned profile values.
        """
        f_transition = lambda m: f_1(m) * (1 - (m - m_1) / (m_2 - m_1)) + f_2(m) * (
            m - m_1
        ) / (m_2 - m_1)

        return MathProfile._transition_function(m, f_transition, f_1, f_2, m_1, m_2)

    @staticmethod
    def cosine_transition(
        m: np.ndarray, f_1: Callable, f_2: Callable, m_1: float, m_2: float
    ) -> np.ndarray:
        """Cosine transition between two functions (smooth ease-in/ease-out).

        Connects f_1 and f_2 via cosine interpolation over [m_1, m_2].
        f_1's contribution: 0.5 * (1 + cos(π*x)), where x = (m-m_1)/(m_2-m_1).
        Provides smooth first derivatives at boundaries.

        Parameters
        ----------
        m : np.ndarray
            Independent variable.
        f_1 : Callable
            Function before transition.
        f_2 : Callable
            Function after transition.
        m_1 : float
            Transition start point.
        m_2 : float
            Transition end point.

        Returns
        -------
        np.ndarray
            Smoothly transitioned profile values with continuous first derivatives.
        """
        f_transition = lambda m: f_1(m) * 1 / 2 * (
            1 + np.cos(np.pi * (m - m_1) / (m_2 - m_1))
        ) + f_2(m) * (1 - 1 / 2 * (1 + np.cos(np.pi * (m - m_1) / (m_2 - m_1))))

        return MathProfile._transition_function(m, f_transition, f_1, f_2, m_1, m_2)

    @staticmethod
    def cubic_transition(
        m: np.ndarray, f_1: Callable, f_2: Callable, m_1: float, m_2: float
    ) -> np.ndarray:
        """Returns an array of values for a cubic transition between two functions.

        The first function's contribution decreases more slowly.

        Parameters
        ----------
        m : np.ndarray
            array of independent variables (mass bins)
        f_1 : Callable
            function for the first part of the transition
        f_2 : Callable
            function for the second part of the transition
        m_1 : float
            independent variable at which the transition starts
        m_2 : float
            independent variable at which the transition ends
        """
        f_transition = lambda m: f_1(m) * (
            1 - np.power(((m - m_1) / (m_2 - m_1)), 3)
        ) + f_2(m) * np.power((m - m_1) / (m_2 - m_1), 3)

        return MathProfile._transition_function(m, f_transition, f_1, f_2, m_1, m_2)

    @staticmethod
    def cubic_transition_fast_decrease(
        m: np.ndarray, f_1: Callable, f_2: Callable, m_1: float, m_2: float
    ) -> np.ndarray:
        """Returns an array of values for a cubic transition between two functions.

        The first function's contribution decreases more rapidly.

        Parameters
        ----------
        m : np.ndarray
            array of independent variables (mass bins)
        f_1 : Callable
            function for the first part of the transition
        f_2 : Callable
            function for the second part of the transition
        m_1 : float
            independent variable at which the transition starts
        m_2 : float
            independent variable at which the transition ends
        """
        f_transition = lambda m: f_1(m) * np.power(
            1 - ((m - m_1) / (m_2 - m_1)), 3
        ) + f_2(m) * (1 - np.power(1 - ((m - m_1) / (m_2 - m_1)), 3))

        return MathProfile._transition_function(m, f_transition, f_1, f_2, m_1, m_2)

    @staticmethod
    def exponential_transition(
        m: np.ndarray,
        f_1: Callable,
        f_2: Callable,
        m_1: float,
        m_2: float,
        alpha: float = -1,
    ) -> np.ndarray:
        """Returns an array of values for an exponential transition between two functions.

        Parameters
        ----------
        m : np.ndarray
            array of independent variables (mass bins)
        f_1 : Callable
            function for the first part of the transition
        f_2 : Callable
            function for the second part of the transition
        m_1 : float
            independent variable at which the transition starts
        m_2 : float
            independent variable at which the transition ends
        alpha : float
            exponential transition rate. Default is -1.
        """
        scaling_factor = lambda x: (np.exp(alpha) - np.exp(alpha * x)) / (
            np.exp(alpha) - 1
        )

        f_transition = lambda m: f_1(m) * scaling_factor((m - m_1) / (m_2 - m_1)) + f_2(
            m
        ) * (1 - scaling_factor((m - m_1) / (m_2 - m_1)))

        return MathProfile._transition_function(m, f_transition, f_1, f_2, m_1, m_2)
