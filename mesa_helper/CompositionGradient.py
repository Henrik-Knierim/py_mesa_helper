# class for creating compositional gradients
# packages
import numpy as np
from scipy.special import erf
from typing import Callable
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from mesa_helper.astrophys import (
    scaled_solar_ratio_mass_fractions,
    X_Sol,
    Y_Sol,
    Z_Sol,
    M_Earth_in_Jup,
    M_Jup_in_Earth,
    X_el_basic,
)


class CompositionGradient:

    def __init__(
        self, gradient: str = "Z", M_p: float = 1.0, iso_net: str = "basic", verbose: bool = False
    ) -> None:
        """Creates a compositional gradient for a planet.

        Parameters
        ----------
        gradient : str
            gradient type. Default is 'Z'. Options are:
            - 'Y' : gradient in Helium mass fraction (assumes pure H-He)
            - 'Z' : gradient in metal mass fraction (assumes metal abundances from Lodders+2020)
        M_p : float
            planet mass in Jupiter masses. Default is 1; if your profile is given in relative mass fraction, you can leave this as 1.
        iso_net : str
            isotope network. Options are:
            - 'basic' : basic.net from MESA
            - 'planets' : planets.net, custom network for planets with only h (X), he4 (Y), and o16 (Z)

        """

        self.verbose = verbose

        # planet mass as input parameter
        self.M_p = M_p

        # set MESA reaction network
        # test
        if not iso_net in ["planets", "basic"]:
            raise Exception(f"iso_net={iso_net} not supported.")

        self.iso_net = iso_net

        if gradient in ["Y", "Z"]:
            self.gradient = gradient
        else:
            raise Exception(f"gradient={gradient} not supported.")

        # set scaled abundances
        self._scaled_abundances()

    def _Z_is_defined(self) -> bool:
        """Returns True if `self.abu_profile` is defined."""
        is_true = hasattr(self, "abu_profile")
        if not is_true:
            raise Exception("self.abu_profile is not defined.")
        return is_true

    def _scaled_abundances(self) -> None:
        """Defines `self.scaled_abundances` depending on `self.gradient`."""
        if self.gradient == "Z":
            self.scaled_abundances = self._scaled_abundances_Z
        elif self.gradient == "Y":
            self.scaled_abundances = self._scaled_abundances_H_He
        else:
            raise Exception(f"gradient={self.gradient} not supported.")

    # scaled abundances
    def _scaled_abundances_Z(self, Z: float) -> dict:
        [X, Y, Z] = scaled_solar_ratio_mass_fractions(Z)
        if self.iso_net == "basic":
            f = lambda el: (
                X / X_Sol
                if el == "H"
                else (Y / Y_Sol if el in ["He3", "He4"] else Z / Z_Sol)
            )
            abu = {}
            abu.update((el, X_el * f(el)) for el, X_el in X_el_basic.items())
        elif self.iso_net == "planets":
            abu = {"H": X, "He4": Y, "O16": Z}
        else:
            raise Exception(f"iso_net={self.iso_net} not supported.")

        return abu

    # scaled abundances for pure H-He
    def _scaled_abundances_H_He(self, Y: float) -> dict:
        X = 1 - Y
        if self.iso_net == "basic":
            f = lambda el: (
                X / X_Sol if el == "H" else (Y / Y_Sol if el in ["He3", "He4"] else 0.0)
            )
            abu = {}
            abu.update((el, X_el * f(el)) for el, X_el in X_el_basic.items())
        elif self.iso_net == "planets":
            abu = {"H": X, "He4": Y, "O16": 0.0}
        else:
            raise Exception(f"iso_net={self.iso_net} not supported.")
        return abu

    # create file for relax_initial_composition

    def _mass_points(
        self, m_start: float = 0.0, m_end: float | None = None, n_bins=20_000, **kwargs
    ) -> np.ndarray:
        """Generates an array of mass points for the compositional gradient."""

        if m_end is None:
            m_end = self.M_p

        return np.linspace(m_start, m_end, n_bins)

    # TODO: this function could be generalized. Scounting for individual key words is not very elegant.
    def _create_composition_list(self, *args, **kwargs):

        # TODO: you could do these three checks in one with an any() function
        # we don't need points in a constant regime (i.e., outside of m_2)
        if "m_2" in kwargs:
            # make m create points only between 0 and m_2
            kwargs["m_end"] = kwargs["m_2"]

        # check m_dilute first to make sure m_end isn't set to m_core
        elif "m_dilute" in kwargs:
            kwargs["m_end"] = kwargs["m_dilute"]
        elif "m_core" in kwargs:
            kwargs["m_end"] = kwargs["m_core"]

        # list of Z(m) (or Y(m))
        mass_bins = self._mass_points(**kwargs)
        abu_list = self._abu_profile(mass_bins, *args, **kwargs)

        l = []
        # first mass bin for m_2:
        # print(mass_bins)
        # if 'm_2' in kwargs:
        #         l.append([(self.M_p-kwargs['m_2'])/self.M_p, *self.scaled_abundances(kwargs["Z_atm"]).values()])
        #         #mass_bins = mass_bins[:-1]

        for i, m_bin in enumerate(mass_bins):
            # creates list [mass_bin, X_H(mass_bin), ..., X_Mg24(mass_bin)]
            l.append(
                [
                    (self.M_p - m_bin) / self.M_p,
                    *self.scaled_abundances(abu_list[i]).values(),
                ]
            )

        # reverse order for MESA's relax_inital_composition format
        return np.flip(l, 0)

    def create_relax_inital_composition_file(
        self, relax_composition_filename="relax_composition_file.dat", *args, **kwargs
    ):
        """
        Creates a file for `MESA`'s `relax_inital_composition functionality`. The `**kwargs` depend upon the self.method used.
        """

        # tests
        if not self._Z_is_defined():
            raise Exception("self.abu_profile is not defined.")

        # comp_list = [[mass_bin, spec_1, spec_2, ..., spec_N], ...]
        comp_list = self._create_composition_list(*args, **kwargs)
        num_points = len(comp_list)
        num_species = len(comp_list[0]) - 1

        with open(relax_composition_filename, "w") as file:
            file.write(f"{num_points}  {num_species}\n")
            for l in comp_list:
                str_version = [f"{el:.16e}" for el in l]
                file.write("  ".join(str_version) + "\n")

        print(f"{relax_composition_filename} was created successfully.") if self.verbose else None

    # ----------------------------------------- #
    # -------- Composition Gradients ---------- #
    # ----------------------------------------- #

    @property
    def abu_profile(self):
        return self._abu_profile

    @abu_profile.setter
    def abu_profile(self, func):

        if not callable(func):
            raise Exception("func must be a callable function.")

        self._abu_profile = func

    @staticmethod
    def lin(
        m: np.ndarray, m_1: float, m_2: float, f_1: float, f_2: float, **kwargs
    ) -> np.ndarray:
        """Returns an array of mass fractions for a linear compositional gradient.

        Parameters
        ----------
        m : np.ndarray
            mass bins
        m_1 : float
            start of the linear gradient
        m_2 : float
            end of the linear gradient
        f_1 : float
            function value at m_1
        f_2 : float
            function value at m_2

        Returns
        -------
        np.ndarray
            array of function values
        """

        # tests
        if m_2 < m_1:
            raise Exception("m_2 must be larger than m_1")
        elif m_1 < 0:
            raise Exception("m_1 needs to be >= 0")
        elif any(n < 0 for n in m):
            raise Exception("m should contain positive numbers only")

        # linear function f = a m + b
        a = -(f_2 - f_1) / (m_1 - m_2)
        b = -(m_2 * f_1 - m_1 * f_2) / (m_1 - m_2)

        return np.piecewise(
            m,
            [m < m_1, ((m_1 <= m) & (m <= m_2)), m > m_2],
            [lambda m: f_1, lambda m: a * m + b, f_2],
        )

    @staticmethod
    def Z_lin_slope_fixed(
        m: np.ndarray, m_core: float, Z_0: float, Z_atm: float, **kwargs
    ) -> np.ndarray:
        """Returns an array of mass fractions for a linear compositional gradient with a fixed slope.

        Parameters
        ----------
        m : np.ndarray
            mass bins
        m_core : float
            core mass in units of m
        Z_0 : float
            heavy element mass fraction at the core
        Z_atm : float
            heavy element mass fraction at the atmosphere

        Returns
        -------
        np.ndarray
            heavy element mass fractions
        """

        # tests

        if any(n < 0 for n in m):
            raise Exception("m should contain positive numbers only")
        elif m_core < 0:
            raise Exception("m_core needs to be >= 0")
        elif not 0 <= Z_atm <= 1:
            raise Exception("Z_atm needs to be between 0 and 1")
        elif not 0 <= Z_0 <= 1:
            raise Exception("Z_atm needs to be between 0 and 1")

        return np.piecewise(
            m,
            [m <= m_core, m > m_core],
            [lambda m: Z_0 + m * (-Z_0 + Z_atm) / m_core, Z_atm],
        )

    @staticmethod
    def Z_lin_M_z(
        m: np.ndarray, m_1: float, m_2: float, M_z: float, Z_atm: float, **kwargs
    ) -> np.ndarray:
        """Returns an array of mass fractions for a linear compositional gradient with a fixed slope."""

        M_z = M_z * M_Earth_in_Jup
        # tests
        if m_2 < m_1:
            raise Exception("m_2 must be larger than m_1")
        elif m_1 < 0:
            raise Exception("m_1 needs to be >= 0")
        elif any(n < 0 for n in m):
            raise Exception("m should contain positive numbers only")
        elif M_z < 0:
            raise Exception("M_z needs to be >= 0")
        elif not 0 <= Z_atm <= 1:
            raise Exception("Z_atm needs to be between 0 and 1")

        # for some reason, I need to pass the lambda functions directly without predefining them ...
        return np.piecewise(
            m,
            [m < m_1, ((m_1 <= m) & (m <= m_2)), m > m_2],
            [
                lambda m: (-2 * M_z + (-m_1 + m_2) * Z_atm) / (m_1 - m_2),
                lambda m: (
                    2 * (-m + m_2) * M_z + (m_1 - m_2) * (-2 * m + m_1 + m_2) * Z_atm
                )
                / (m_1 - m_2) ** 2,
                Z_atm,
            ],
        )

    @staticmethod
    def stepwise(m, m_transition, f_1, f_2, **kwargs) -> np.ndarray:

        # tests
        if any(n < 0 for n in m):
            raise Exception("m should contain positive numbers only")

        return np.piecewise(m, [m <= m_transition, m > m_transition], [f_1, f_2])

    @staticmethod
    def exponential(
        m: np.ndarray,
        alpha: float,
        m_start: float,
        m_end: float,
        Z_start: float,
        Z_end: float,
    ) -> np.ndarray:
        """Returns an array of mass fractions for an exponential function that is `Z_core` at `m_start` and `Z_env` at `m_end`.

        Parameters
        ----------
        m : np.ndarray
            array of mass bins
        alpha : float
            exponential decay constant
        m_start : float
            mass at which the exponential function starts
        m_end : float
            mass at which the exponential function ends
        Z_start : float
            metallicity at `m_start`
        Z_end : float
            metallicity at `m_end`

        Returns
        -------
        np.ndarray
            array of mass fractions

        """

        # tests
        if m_end < m_start:
            raise Exception(
                f"m_end must be larger than m_start, but m_end = {m_end} and m_start = {m_start}."
            )
        elif m_start < 0:
            raise Exception("m_start needs to be >= 0")
        elif any(n < 0 for n in m):
            raise Exception("m should contain positive numbers only")
        elif not 0 <= Z_start <= 1:
            raise Exception("Z_core needs to be between 0 and 1")
        elif not 0 <= Z_end <= 1:
            raise Exception("Z_env needs to be between 0 and 1")
        if alpha == 0:
            retval = (m * Z_start - m_end * Z_start - m * Z_end + m_start * Z_end) / (
                m_start - m_end
            )
        else:
            retval = (
                np.exp(alpha * m) * (Z_start - Z_end)
                + np.exp(alpha * m_start) * Z_end
                - np.exp(alpha * m_end) * Z_start
            ) / (np.exp(alpha * m_start) - np.exp(alpha * m_end))

        if min(retval) < 0 or max(retval) > 1:
            print("WARNING: exponential function is not between 0 and 1.")

        return retval

    @staticmethod
    def Gaussian(
        m: np.ndarray, M_z: float, Z_core: float, Z_atm: float, **kwargs
    ) -> np.ndarray:
        """Returns an array of mass fractions for a Gaussian compositional gradient.

        Parameters
        ----------
        m : np.ndarray
            array of mass bins
        M_z : float
            mass of the planet in Earth masses
        Z_core : float
            heavy element mass fraction at the core
        Z_atm : float
            heavy element mass fraction at the atmosphere

        Returns
        -------
        np.ndarray
            array of mass fractions
        """

        # tests
        if any(n < 0 for n in m):
            raise Exception("m should contain positive numbers only")
        elif M_z < 0:
            raise Exception("M_z needs to be >= 0")
        elif not 0 <= Z_core <= 1:
            raise Exception("Z_core needs to be between 0 and 1")
        elif not 0 <= Z_atm <= 1:
            raise Exception("Z_atm needs to be between 0 and 1")

        # fix sigma such that the integral of the Gaussian to 3 sigma is equal to M_z
        # additonally, use a conversion factor for M_z to convert it to Earth masses

        sigma = (2.0 * M_z) / (
            6.0 * M_Jup_in_Earth * Z_atm
            + (
                M_Jup_in_Earth
                * np.sqrt(2.0 * np.pi)
                * erf(3.0 / np.sqrt(2.0))
                * (Z_core - Z_atm)
            )
        )

        return Z_atm + (Z_core - Z_atm) * np.exp(-(m**2) / (2.0 * sigma**2))

    @staticmethod
    def reverse_sigmoid(
        m: np.ndarray,
        m_b: float,
        steepness: float = 100,
        Z_core: float = 1,
        Z_env: float = Z_Sol,
        **kwargs,
    ) -> np.ndarray:
        """Returns an array of mass fractions for a reverse sigmoid compositional gradient.

        Parameters
        ----------
        m : np.ndarray
            array of mass bins
        steepness : float
            steepness of the sigmoid slope. The larger the steeper. Default is 100.
        m_b : float
            mass of the core in Earth mass. Defined as the midpoint of the sigmoid.
        Z_core : float
            metallicity of the core. Default is 1.
        Z_env : float
            metallicity of the envelope. Default is solar metallicity.
        """

        # tests
        if any(n < 0 for n in m):
            raise Exception("m should contain positive numbers only")
        elif m_b < 0:
            raise Exception("m_core needs to be >= 0")
        elif not 0 <= Z_core <= 1:
            raise Exception("Z_core needs to be between 0 and 1")
        elif not 0 <= Z_env <= 1:
            raise Exception("Z_env needs to be between 0 and 1")

        m_core_M_Jup = m_b / M_Jup_in_Earth

        return Z_core - (Z_core - Z_env) / (1 + np.exp(-steepness * (m - m_core_M_Jup)))

    # TODO: This function is a special case of the piecewise_with_two_smoothed_exponential_transitions function.
    @staticmethod
    def piecewise_with_smoothed_exponential_transition(
        m: np.ndarray,
        m_core: float,
        dm_core: float,
        m_dilute: float,
        dm_dilute: float,
        Z_core: float,
        Z_env: float,
        alpha: float,
    ) -> np.ndarray:
        """Returns an array of mass fractions for a piecewise compositional gradient with a smoothed exponential transition.

        The smoothing is done by a cubic transition function, where the fast decreasing side is always the constant function.
        All mass units are relative to the planet mass.

        Parameters
        ----------
        m : np.ndarray
            array of mass bins

        m_core : float
            mass coordinate of the core
        dm_core : float
            width of the core - dilute transition
        m_dilute : float
            mass coordinate of the end of the dilute region
        dm_dilute : float
            width of the dilute - envelope transition
        Z_core : float
            metallicity of the core
        Z_env : float
            metallicity of the envelope
        alpha : float
            exponential decay constant
        """

        # tests
        if any(n < 0 for n in m):
            raise Exception("m should contain positive numbers only")
        elif m_core < 0:
            raise Exception("m_core needs to be >= 0")
        elif dm_core < 0:
            raise Exception("dm_core needs to be >= 0")
        elif m_dilute < 0:
            raise Exception("m_dilute needs to be >= 0")
        elif dm_dilute < 0:
            raise Exception("dm_dilute needs to be >= 0")
        elif not 0 <= Z_core <= 1:
            raise Exception("Z_core needs to be between 0 and 1")
        elif not 0 <= Z_env <= 1:
            raise Exception("Z_env needs to be between 0 and 1")

        # transition functions
        f_core = lambda m: np.full_like(m, Z_core)
        f_dilute = lambda m: CompositionGradient.exponential(
            m, alpha=alpha, m_start=m_core, m_end=m_dilute, Z_start=Z_core, Z_end=Z_env
        )
        f_env = lambda m: np.full_like(m, Z_env)

        # transition functions for the cubic transition
        f_core_dilute = lambda m: CompositionGradient.cubic_transition_fast_decrease(
            m, f_1=f_core, f_2=f_dilute, m_1=m_core, m_2=m_core + dm_core
        )
        f_complete = CompositionGradient.cubic_transition(
            m, f_1=f_core_dilute, f_2=f_env, m_1=m_dilute - dm_dilute, m_2=m_dilute
        )

        return f_complete

    @staticmethod
    def piecewise_with_two_smoothed_exponential_transitions(
        m: np.ndarray,
        alphas: list[float],
        m_cores: list[float],
        dm_cores: list[float],
        Z_values: list[float],
    ) -> np.ndarray:
        """Returns an array of mass fractions for a compositional gradient with two smoothed exponential transitions between constant regions.

        Parameters
        ----------
        m : np.ndarray
            mass bins
        alphas : list[float]
            decay constants for the two exponential transitions
        m_cores : list[float]
            mass coordinates of the transitions
        dm_cores : list[float]
            widths of the transitions, sign determines the direction of the transition
        Z_values : list[float]
            heavy element mass fractions at the transitions

        Returns
        -------
        np.ndarray
            array of mass fractions

        """

        # tests
        if any(n < 0 for n in m):
            raise Exception("m should contain positive numbers only")
        elif any(n < 0 for n in m_cores):
            raise Exception("m_cores should contain positive numbers only")
        elif any(n < 0 for n in Z_values):
            raise Exception("Z_values should contain positive numbers only")
        elif len(alphas) != 2:
            raise Exception("alphas should contain exactly two values")
        elif len(m_cores) != 3:
            raise Exception("m_cores should contain exactly three values")
        elif len(dm_cores) != 3:
            raise Exception("dm_cores should contain exactly three values")
        elif len(Z_values) != 3:
            raise Exception("Z_values should contain exactly three values")

        # profile functions
        f_core = lambda m: np.full_like(m, Z_values[0])
        f_exp_core_dilute = lambda m: CompositionGradient.exponential(
            m,
            alpha=alphas[0],
            m_start=m_cores[0],
            m_end=m_cores[1],
            Z_start=Z_values[0],
            Z_end=Z_values[1],
        )
        f_exp_dilute_env = lambda m: CompositionGradient.exponential(
            m,
            alpha=alphas[1],
            m_start=m_cores[1],
            m_end=m_cores[2],
            Z_start=Z_values[1],
            Z_end=Z_values[2],
        )
        f_env = lambda m: np.full_like(m, Z_values[2])

        profile_functions = [f_core, f_exp_core_dilute, f_exp_dilute_env, f_env]

        # transition functions
        transition_functions = [
            CompositionGradient.cubic_transition_fast_decrease,
            CompositionGradient.cubic_transition,
            CompositionGradient.cubic_transition,
        ]

        return CompositionGradient.join_compositional_gradients(
            profile_functions, transition_functions, m_cores, dm_cores
        )(m)

    # ----------------------------------------- #
    # --------- Transition Functions ---------- #
    # ----------------------------------------- #

    # Here, we define a number of functions that can be used to create a transition between two functions.
    # Many of these use the same parameters, so we define a parent function that takes care of the tests and then calls the transition function.

    @staticmethod
    def _transition_function(
        m: np.ndarray,
        f_transition: Callable,
        f_1: Callable,
        f_2: Callable,
        m_1: float,
        m_2: float,
    ) -> np.ndarray:
        """Parent function for transition functions. Tests the input parameters and then calls the transition function.

        Parameters
        ----------
        m : np.ndarray
            array of mass bins
        f_transition : Callable
            transition function
        f_1 : Callable
            function for the first part of the transition
        f_2 : Callable
            function for the second part of the transition
        m_1 : float
            mass at which the transition starts
        m_2 : float
            mass at which the transition ends
        """
        verbose = False

        # tests
        if m_2 < m_1:
            raise Exception("m_2 must be larger than m_1")
        elif m_1 < 0:
            raise Exception(f"m_1 needs to be >= 0, but is {m_1}")
        elif any(n < 0 for n in m):
            raise Exception("m should contain positive numbers only")
        elif m_2 == m_1:
            (
                print("Note: m_2 = m_1. A piecewise function is returned.")
                if verbose
                else None
            )
            return np.piecewise(m, [m < m_1, m >= m_1], [f_1, f_2])

        return np.piecewise(
            m, [m < m_1, ((m_1 <= m) & (m <= m_2)), m > m_2], [f_1, f_transition, f_2]
        )

    @staticmethod
    def linear_transition(
        m: np.ndarray, f_1: Callable, f_2: Callable, m_1: float, m_2: float
    ) -> np.ndarray:
        """Returns an array of mass fractions for a linear transition between two functions.

        Parameters
        ----------
        m : np.ndarray
            array of mass bins
        f_1 : Callable
            function for the first part of the transition
        f_2 : Callable
            function for the second part of the transition
        m_1 : float
            mass at which the transition starts
        m_2 : float
            mass at which the transition ends
        """
        f_transition = lambda m: f_1(m) * (1 - (m - m_1) / (m_2 - m_1)) + f_2(m) * (
            m - m_1
        ) / (m_2 - m_1)

        return CompositionGradient._transition_function(
            m, f_transition, f_1, f_2, m_1, m_2
        )

    @staticmethod
    def cosine_transition(
        m: np.ndarray, f_1: Callable, f_2: Callable, m_1: float, m_2: float
    ) -> np.ndarray:
        """Returns an array of mass fractions for a cosine transition between two functions.

        Parameters
        ----------
        m : np.ndarray
            array of mass bins
        f_1 : Callable
            function for the first part of the transition
        f_2 : Callable
            function for the second part of the transition
        m_1 : float
            mass at which the transition starts
        m_2 : float
            mass at which the transition ends
        """

        # transition function
        f_transition = lambda m: f_1(m) * 1 / 2 * (
            1 + np.cos(np.pi * (m - m_1) / (m_2 - m_1))
        ) + f_2(m) * (1 - 1 / 2 * (1 + np.cos(np.pi * (m - m_1) / (m_2 - m_1))))

        return CompositionGradient._transition_function(
            m, f_transition, f_1, f_2, m_1, m_2
        )

    @staticmethod
    def cubic_transition(
        m: np.ndarray, f_1: Callable, f_2: Callable, m_1: float, m_2: float
    ) -> np.ndarray:
        """Returns an array of mass fractions for a cubic transition between the two functions `f_1` and `f_2`, where `f_1`'s contribution decreases more slowly.

        Parameters
        ----------
        m : np.ndarray
            array of mass bins
        f_1 : Callable
            function for the first part of the transition
        f_2 : Callable
            function for the second part of the transition
        m_1 : float
            mass at which the transition starts
        m_2 : float
            mass at which the transition ends
        """

        # transition function
        f_transition = lambda m: f_1(m) * (
            1 - np.power(((m - m_1) / (m_2 - m_1)), 3)
        ) + f_2(m) * np.power((m - m_1) / (m_2 - m_1), 3)

        return CompositionGradient._transition_function(
            m, f_transition, f_1, f_2, m_1, m_2
        )

    @staticmethod
    def cubic_transition_fast_decrease(
        m: np.ndarray, f_1: Callable, f_2: Callable, m_1: float, m_2: float
    ) -> np.ndarray:
        """Returns an array of mass fractions for a cubic transition between the two functions `f_1` and `f_2`, where `f_1`'s contribution decreases more rapidly.

        Parameters
        ----------
        m : np.ndarray
            array of mass bins
        f_1 : Callable
            function for the first part of the transition
        f_2 : Callable
            function for the second part of the transition
        m_1 : float
            mass at which the transition starts
        m_2 : float
            mass at which the transition ends
        """

        # transition function
        f_transition = lambda m: f_1(m) * np.power(
            1 - ((m - m_1) / (m_2 - m_1)), 3
        ) + f_2(m) * (1 - np.power(1 - ((m - m_1) / (m_2 - m_1)), 3))

        return CompositionGradient._transition_function(
            m, f_transition, f_1, f_2, m_1, m_2
        )

    @staticmethod
    def exponential_transition(
        m: np.ndarray,
        f_1: Callable,
        f_2: Callable,
        m_1: float,
        m_2: float,
        alpha: float = -1,
    ) -> np.ndarray:
        """Returns an array of mass fractions for an exponential transition between the two functions `f_1` and `f_2`.

        Parameters
        ----------
        m : np.ndarray
            array of mass bins
        f_1 : Callable
            function for the first part of the transition
        f_2 : Callable
            function for the second part of the transition
        m_1 : float
            mass at which the transition starts
        m_2 : float
            mass at which the transition ends
        """

        # transition function
        # scaling from x = 0 to x = 1
        scaling_factor = lambda x: (np.exp(alpha) - np.exp(alpha * x)) / (
            np.exp(alpha) - 1
        )

        f_transition = lambda m: f_1(m) * scaling_factor((m - m_1) / (m_2 - m_1)) + f_2(
            m
        ) * (1 - scaling_factor((m - m_1) / (m_2 - m_1)))

        return CompositionGradient._transition_function(
            m, f_transition, f_1, f_2, m_1, m_2
        )

    # ----------------------------------------- #
    # --------- Gradient Compositions --------- #
    # ----------------------------------------- #

    @staticmethod
    def join_compositional_gradients(
        profile_functions: list[Callable],
        transition_functions: list[Callable],
        transition_masses: list[float],
        dms: list[float],
        verbose=False,
    ) -> Callable[..., np.ndarray]:
        """Uses N-1 transition functions to compose N profile functions at N-1 transition masses over N-1 widths.

        Parameters
        ----------
        profile_functions : list[Callable]
            list of profile functions
        transition_functions : list[Callable]
            list of transition functions.
        transition_masses : list[float]
            list of transition masses
        dms : list[float]
            list of transition widths
        """

        # tests
        if len(profile_functions) != len(transition_functions) + 1:
            raise Exception(
                "The number of profile functions must be one more than the number of transition functions."
            )
        elif len(transition_functions) != len(transition_masses):
            raise Exception(
                "The number of transition functions must be equal to the number of transition masses."
            )
        elif len(transition_masses) != len(dms):
            raise Exception(
                "The number of transition masses must be equal to the number of transition widths."
            )

        # create an array with the interpolation boundaries
        # if dms[i] < 0, the interpolation goes from transition_masses[i] + dms[i] to transition_masses[i]
        # if dms[i] > 0, the interpolation goes from transition_masses[i] to transition_masses[i] + dms[i]

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

        # if c[-1] is still None, we throw an error
        if c[-1] is None:
            raise Exception(
                "Something went wrong with the composition of the functions."
            )

        return c[-1]
    
    @staticmethod
    def plot_relax_composition_file(
        file: str,
        fig: plt.Figure | None = None,
        ax: Axes | None = None,
        ):
        """Plots the composition file."""
        if ax is None:
            fig, ax = plt.subplots()

        q, X, Y, Z = np.loadtxt(file, unpack=True, skiprows=1)
        m_over_M_p = 1.0 - q
        ax.plot(m_over_M_p, X, label="X")
        ax.plot(m_over_M_p, Y, label="Y")
        ax.plot(m_over_M_p, Z, label="Z")
        ax.set_xlabel(r"$m/M$")
        ax.set_ylabel("Mass Fraction")
        ax.legend()

        return fig, ax
    
    @staticmethod
    def compute_heavy_metal_mass(file: str):
        """Computes the heavy metal mass from the composition file."""
        q, X, Y, Z = np.loadtxt(file, unpack=True, skiprows=1)
        m_over_M_p = 1.0 - q
        return np.trapz(Z[::-1], m_over_M_p[::-1])

