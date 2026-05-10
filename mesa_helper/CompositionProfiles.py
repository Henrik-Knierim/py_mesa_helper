import numpy as np
from typing import Callable

from mesa_helper.astrophys import Z_Sol
from mesa_helper.MathProfile import MathProfile


class CompositionProfiles:
    """Helper functions for building composition profiles.

    These functions translate domain-specific concepts (e.g., core mass or
    envelope abundance) into numerical profiles using MathProfile primitives.
    """

    @staticmethod
    def lin_slope_fixed(
        m: np.ndarray, m_core: float, f_0: float, f_atm: float, **kwargs
    ) -> np.ndarray:
        """Linear profile with core-centered slope.

        Constant value f_0 at core (m <= m_core), then linear increase
        to f_atm. Useful for profiles with a flat core region.
        """
        m = np.asarray(m, dtype=float)

        if m_core < 0:
            raise ValueError("m_core needs to be >= 0")

        return np.piecewise(
            m,
            [m <= m_core, m > m_core],
            [lambda m: f_0 + m * (-f_0 + f_atm) / m_core, f_atm],
        )

    @staticmethod
    def lin_M_z(
        m: np.ndarray, m_1: float, m_2: float, M_z: float, f_atm: float, **kwargs
    ) -> np.ndarray:
        """Linear profile with constrained integral.

        Creates a linear profile over [m_1, m_2] such that the integral of
        (f - f_atm) equals M_z. Outside this range, f = f_atm.
        """
        m = np.asarray(m, dtype=float)

        if m_2 < m_1:
            raise ValueError("m_2 must be larger than m_1")
        elif m_1 < 0:
            raise ValueError("m_1 needs to be >= 0")
        elif M_z < 0:
            raise ValueError("M_z needs to be >= 0")

        return np.piecewise(
            m,
            [m < m_1, ((m_1 <= m) & (m <= m_2)), m > m_2],
            [
                lambda m: (-2 * M_z + (-m_1 + m_2) * f_atm) / (m_1 - m_2),
                lambda m: (
                    2 * (-m + m_2) * M_z + (m_1 - m_2) * (-2 * m + m_1 + m_2) * f_atm
                )
                / (m_1 - m_2) ** 2,
                f_atm,
            ],
        )

    @staticmethod
    def piecewise_with_smoothed_exponential_transition(
        m: np.ndarray,
        m_core: float,
        dm_core: float,
        m_dilute: float,
        dm_dilute: float,
        f_core: float | None = None,
        f_env: float | None = None,
        alpha: float = -1.0,
        **kwargs,
    ) -> np.ndarray:
        """Piecewise profile with smoothed exponential transitions.

        Builds a core -> dilute -> envelope profile with cubic transitions,
        using an exponential decay in the dilute region.
        """
        m = np.asarray(m, dtype=float)

        if f_core is None:
            f_core = kwargs.get("Z_core")
        if f_env is None:
            f_env = kwargs.get("Z_env")

        if f_core is None or f_env is None:
            raise ValueError("f_core and f_env must be provided (or Z_core/Z_env).")
        if m_core < 0:
            raise ValueError("m_core needs to be >= 0")
        elif dm_core < 0:
            raise ValueError("dm_core needs to be >= 0")
        elif m_dilute < 0:
            raise ValueError("m_dilute needs to be >= 0")
        elif dm_dilute < 0:
            raise ValueError("dm_dilute needs to be >= 0")

        # profile functions
        f_core_func = lambda m: np.full_like(m, f_core)
        f_dilute_func = lambda m: MathProfile.exponential(
            m, alpha=alpha, m_start=m_core, m_end=m_dilute, f_start=f_core, f_end=f_env
        )
        f_env_func = lambda m: np.full_like(m, f_env)

        # transition functions for the cubic transition
        f_core_dilute = lambda m: MathProfile.cubic_transition_fast_decrease(
            m, f_1=f_core_func, f_2=f_dilute_func, m_1=m_core, m_2=m_core + dm_core
        )
        f_complete = MathProfile.cubic_transition(
            m, f_1=f_core_dilute, f_2=f_env_func, m_1=m_dilute - dm_dilute, m_2=m_dilute
        )

        return f_complete

    @staticmethod
    def piecewise_with_two_smoothed_exponential_transitions(
        m: np.ndarray,
        alphas: list[float],
        m_cores: list[float],
        dm_cores: list[float],
        f_values: list[float],
    ) -> np.ndarray:
        """Profile with two smoothed exponential transitions."""
        if len(alphas) != 2:
            raise ValueError("alphas should contain exactly two values")
        elif len(m_cores) != 3:
            raise ValueError("m_cores should contain exactly three values")
        elif len(dm_cores) != 3:
            raise ValueError("dm_cores should contain exactly three values")
        elif len(f_values) != 3:
            raise ValueError("f_values should contain exactly three values")

        m = np.asarray(m, dtype=float)

        # profile functions
        f_const_0 = lambda m: np.full_like(m, f_values[0])
        f_exp_0_1 = lambda m: MathProfile.exponential(
            m,
            alpha=alphas[0],
            m_start=m_cores[0],
            m_end=m_cores[1],
            f_start=f_values[0],
            f_end=f_values[1],
        )
        f_exp_1_2 = lambda m: MathProfile.exponential(
            m,
            alpha=alphas[1],
            m_start=m_cores[1],
            m_end=m_cores[2],
            f_start=f_values[1],
            f_end=f_values[2],
        )
        f_const_2 = lambda m: np.full_like(m, f_values[2])

        profile_functions = [f_const_0, f_exp_0_1, f_exp_1_2, f_const_2]

        # transition functions
        transition_functions = [
            MathProfile.cubic_transition_fast_decrease,
            MathProfile.cubic_transition,
            MathProfile.cubic_transition,
        ]

        return CompositionProfiles.join_compositional_gradients(
            profile_functions, transition_functions, m_cores, dm_cores
        )(m)

    @staticmethod
    def reverse_sigmoid_integral(
        M_p: float,
        m_b: float,
        steepness: float = 100,
        Z_core: float = 1,
        Z_env: float = Z_Sol,
    ) -> float:
        """Returns the integral of the reverse sigmoid profile from 0 to M_p."""
        return MathProfile.reverse_sigmoid_integral(
            M_p=M_p,
            m_b=m_b,
            steepness=steepness,
            f_core=Z_core,
            f_env=Z_env,
        )

    @staticmethod
    def join_compositional_gradients(
        profile_functions: list[Callable],
        transition_functions: list[Callable],
        transition_masses: list[float],
        dms: list[float],
        verbose: bool = False,
    ) -> Callable[..., np.ndarray]:
        """Composes N profile functions using N-1 transition functions."""
        if len(profile_functions) != len(transition_functions) + 1:
            raise ValueError(
                "The number of profile functions must be one more than the number of transition functions."
            )
        elif len(transition_functions) != len(transition_masses):
            raise ValueError(
                "The number of transition functions must be equal to the number of transition masses."
            )
        elif len(transition_masses) != len(dms):
            raise ValueError(
                "The number of transition masses must be equal to the number of transition widths."
            )

        interpolation_ranges = np.array(
            [
                [m + dm, m] if dm < 0.0 else [m, m + dm]
                for m, dm in zip(transition_masses, dms)
            ]
        )
        if verbose:
            print("Interpolation Ranges:")
            for i, range in enumerate(interpolation_ranges):
                print(f"Range {i+1}: {range[0]} - {range[1]}")

        c: list[Callable[..., np.ndarray] | None] = [None] * len(transition_functions)

        def create_function(
            i, prev_func, transition_functions, profile_functions, interpolation_ranges
        ):
            if i == 0:
                return lambda m: transition_functions[i](
                    m,
                    f_1=profile_functions[i],
                    f_2=profile_functions[i + 1],
                    m_1=interpolation_ranges[i, 0],
                    m_2=interpolation_ranges[i, 1],
                )
            else:
                return lambda m: transition_functions[i](
                    m,
                    f_1=prev_func,
                    f_2=profile_functions[i + 1],
                    m_1=interpolation_ranges[i, 0],
                    m_2=interpolation_ranges[i, 1],
                )

        for i, t in enumerate(transition_functions):
            c[i] = create_function(
                i,
                c[i - 1],
                transition_functions,
                profile_functions,
                interpolation_ranges,
            )

        if c[-1] is None:
            raise ValueError(
                "Something went wrong with the composition of the functions."
            )

        return c[-1]
