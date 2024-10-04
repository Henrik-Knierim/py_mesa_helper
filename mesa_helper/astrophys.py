import numpy as np

# * constants *
# physics
sigma_SB: float = 5.670374e-5  # erg cm^-2 K^-4
NA: float = 6.02214e23  # per mol
kB: float = 1.38065e-16  # erg/K
yr_in_s: float = 3.1536e7  # seconds
s_in_yr: float = 1 / yr_in_s  # years

# Sun
R_Sol_in_cm: float = 6.957e10
M_Sol_in_g: float = 1.988435e33
M_Sol_in_Earth: float = 3.329e-5
L_Sol_in_erg_s: float = 3.828e33

# abundances
# proto-solar values accoding to Lodders+2020
# in the format of the basic.net from MESA
X_el_basic = {
    "H": 0.706071,
    "He3": 0.0000346097303534802,
    "He4": 0.276920110427544,
    "C12": 0.00301079787801854,
    "N14": 0.000848188700511441,
    "O16": 0.00737738790084843,
    "Ne20": 0.00226085557062686,
    "Mg24": 0.000539806489881855
}
X_Sol: float = X_el_basic["H"]
Y_Sol: float = X_el_basic["He3"] + X_el_basic["He4"]
Z_Sol: float = 1 - X_Sol - Y_Sol

# Jupiter
R_Jup_in_Sol: float = 0.10054
R_Jup_in_cm: float = 6.995e9
M_Jup_in_g: float = 1.89813e30
M_Jup_in_Sol: float = 0.000955
M_Jup_in_Earth: float = 317.8

# Earth
M_Earth_in_Sol: float = 3.0e-6
M_Earth_in_Jup: float = 0.00314636
M_Earth_in_g: float = 5.97e27

Teff_Jup: float = 124.0  # K

# * astro functions *


def from_flux_to_equilibrium_temperature(flux: float | np.ndarray) -> float | np.ndarray:
    """Return the equilibrium temperature for a given flux."""

    return (flux / 4 / sigma_SB) ** (1 / 4)


def scaled_solar_ratio_mass_fractions(Z: float | np.ndarray) -> np.ndarray:
    """Return the mass fractions of hydrogen, helium, and metals for a given metallicity, assuming solar scaled abundances."""
    a = (1 - Z) / (1 - Z_Sol)
    X = a * X_Sol
    Y = a * Y_Sol
    return np.array([X, Y, Z])


def diffusion_timescale(D: float, L: float) -> float:
    """Return the diffusion timescale.

    Parameters
    ----------
    D : float
        Diffusion coefficient (typically in cm^2/s)
    L : float
        The characteristic length scale (typically in cm)

    Returns
    -------
    float
        timescale (typically in s; depends on D and L)
    """
    return L**2 / D


# converts kerg per baryon into specific entropy (erg/g/K)


def specific_entropy(entropy_in_kerg):
    return NA * kB * entropy_in_kerg


def specific_entropy_in_kerg(specific_entropy):
    return specific_entropy / (NA * kB)


# visible opacity fit from Guillot 2010
def kappa_v_Guillot_2010(T_eq: float | np.ndarray) -> float | np.ndarray:
    "Return the visible opacity in cm^2/g for a given equilibrium temperature according to Guillot (2010)."
    kappa_0: float = 6e-3
    T_0: float = 2000.0
    T_irr: float = np.sqrt(2) * T_eq
    return kappa_0 * np.sqrt(T_irr / T_0)
