# class for creating compositional gradients
# packages
import numpy as np
from typing import Callable
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from mesa_helper.astrophys import (
    scaled_solar_ratio_mass_fractions,
    X_Sol,
    Y_Sol,
    Z_Sol,
    X_el_basic,
)


class CompositionGradient:

    def __init__(
        self,
        gradient: str = "Z",
        M_p: float = 1.0,
        iso_net: str = "basic",
        verbose: bool = False,
        debug_skip_validation: bool = False,
    ) -> None:
        """Creates a compositional gradient for a planet.

        Parameters
        ----------
        gradient : str
            gradient type. Default is 'Z'. Options are:
            - 'Y' : gradient in Helium mass fraction (assumes pure H-He)
            - 'Z' : gradient in metal mass fraction (assumes metal abundances from Lodders+2020)
            - 'XZ' : gradient in metal mass fraction (assuming only H and Z)
        M_p : float
            planet mass in Jupiter masses. Default is 1; if your profile is given in relative mass fraction, you can leave this as 1.
        iso_net : str
            isotope network. Options are:
            - 'basic' : basic.net from MESA
            - 'planets' : planets.net, custom network for planets with only h (X), he4 (Y), and o16 (Z)
        verbose : bool
            print debug information. Default is False.
        debug_skip_validation : bool
            skip composition-specific validation checks. Default is False. Useful for testing arbitrary profiles.

        """

        self.verbose = verbose
        self.debug_skip_validation = debug_skip_validation

        # planet mass as input parameter
        self.M_p = M_p

        # set MESA reaction network
        # test
        if not iso_net in ["planets", "basic"]:
            raise Exception(f"iso_net={iso_net} not supported.")

        self.iso_net = iso_net

        if gradient in ["Y", "Z", "XZ"]:
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
        elif self.gradient == "XZ":
            self.scaled_abundances = self._scaled_abundances_XZ
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
    def _scaled_abundances_H_He(self, Y: float):
        X = 1.0 - Y
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

    # scaled abundances for H-Z
    def _scaled_abundances_XZ(self, Z: float):
        X = 1.0 - Z
        if self.iso_net == "basic":
            f = lambda el: (
                X / X_Sol if el == "H" else (0.0 if el in ["He3", "He4"] else Z / Z_Sol)
            )
            abu = {}
            abu.update((el, X_el * f(el)) for el, X_el in X_el_basic.items())
        elif self.iso_net == "planets":
            abu = {"H": X, "He4": 0.0, "O16": Z}
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
                line = "  ".join(str_version) + "\n"
                (
                    print(f"create_relax_initial_composition: line = {line}")
                    if self.verbose
                    else None
                )
                file.write(line)

        (
            print(f"{relax_composition_filename} was created successfully.")
            if self.verbose
            else None
        )

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

        # Validate the profile if not skipped for debugging
        if not self.debug_skip_validation:
            self._validate_abundance_profile()

    def _validate_abundance_profile(self) -> None:
        """Validates that the abundance profile meets composition-specific requirements.

        For composition profiles (Z, Y, or XZ), the following must be true:
        - All values are between 0 and 1 (mass fractions)
        - The profile is monotonically decreasing from core to envelope (m=0 to m=M)
        """
        # Test with a sample mass grid
        m_test = np.linspace(0, self.M_p, 1000)
        abu_test = self._abu_profile(m_test)

        # Check that all values are between 0 and 1
        if np.any(abu_test < 0) or np.any(abu_test > 1):
            raise ValueError(
                f"Abundance profile values must be between 0 and 1, "
                f"but got min={np.min(abu_test):.6f}, max={np.max(abu_test):.6f}. "
                f"Use debug_skip_validation=True to skip this check."
            )

        # Check monotonicity (should be decreasing from core at m=0 to envelope at m=M)
        diffs = np.diff(abu_test)
        if not np.all(diffs <= 1e-10):  # Allow small numerical errors
            n_violations = np.sum(diffs > 1e-10)
            raise ValueError(
                f"Abundance profile is not monotonically decreasing from core to envelope. "
                f"Found {n_violations} points with increasing values. "
                f"Use debug_skip_validation=True to skip this check."
            )

        if self.verbose:
            print(f"Abundance profile validated successfully.")
            print(f"  Profile range: [{np.min(abu_test):.6f}, {np.max(abu_test):.6f}]")
            print(f"  Monotonicity check: passed")

    # ----------------------------------------- #
    # --------- MESA-Specific Methods --------- #
    # ----------------------------------------- #

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
