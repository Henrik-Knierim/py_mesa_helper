import numpy as np

# * constants *
# physics
sigma_SB: float = 5.670374e-5  # erg cm^-2 K^-4
NA: float = 6.02214e23  # per mol
kB: float = 1.38065e-16  # erg/K
yr_in_s: float = 3.1536e7  # seconds
s_in_yr: float = 1 / yr_in_s  # years
G: float = 6.67430e-8  # cm^3 g^-1 s^-2

# Sun
R_Sol_in_cm: float = 6.957e10
M_Sol_in_g: float = 1.988435e33
M_Sol_in_Earth: float = 3.329e-5
L_Sol_in_erg_s: float = 3.828e33

age_solar_system: float = 4.56e9  # age of the solar system in yr
log_age_solar_system: float = np.log10(age_solar_system)  # log of the age of the solar system in yr

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
# R_Jup : average radius of Jupiter
R_Jup_in_Sol: float = 0.10054
R_Jup_in_cm: float = 6.995e9
R_Jup_equatorial_in_cm: float = 7.1492e9
R_Jup_polar_in_cm: float = 6.6854e9 
M_Jup_in_g: float = 1.89813e30
M_Jup_in_Sol: float = 0.000955
M_Jup_in_Earth: float = 317.8

NMoI_Jupiter: dict = {
    "Helled_2011": 0.265,
    "Ni_2018": 0.276
}

P_Jup: float = 9.925 * 3600  # measured rotation period of Jupiter today in s
Omega_Jup: float = 2.0 * np.pi / P_Jup  # measured rotation rate of Jupiter today in s^-1

J2_Jup: float = 14696.572e-6    # measured J2 of Jupiter today
J4_Jup: float = -586.609e-6     # measured J4 of Jupiter today

Teff_Jup: float = 125.57        # effective temperature of Jupiter today in K
T_one_bar_Jup_0N:float = 167.3  # 1 bar temperature of Jupiter at the equator (0°N) according to Gupta et al. 2022
T_one_bar_Jup_12S:float = 170.3  # 1 bar temperature of Jupiter at 12°S according to Gupta et al. 2022

Y_Jup_div_X_p_Y: float = 0.238
Y_Jup_div_X_p_Y_err: float = 0.005

# Earth
M_Earth_in_Sol: float = 3.0e-6
M_Earth_in_Jup: float = 0.00314636
M_Earth_in_g: float = 5.97e27

Teff_Jup: float = 124.0  # K

# Saturn
R_Sat_in_cm: float = 5.830e9
M_Sat_in_g: float = 5.683e29


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

def rotational_parameter(Omega: float, R_mean: float, M_p: float) -> float:
    """Computes the rotational parameter m_rot."""
    return Omega**2 * R_mean**3 / (G * M_p)

def break_up_rotation_rate(M: float, R: float) -> float:
    """Return the break-up rotation rate for a given moment of inertia, mass, and radius."""
    return np.sqrt(G * M / R**3)

def flattenting(R_eq: float, R_polar: float) -> float:
    """Computes the flattening of a planet."""
    return (R_eq - R_polar) / R_eq

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


def _compute_mean(f: np.ndarray, dx: np.ndarray) -> np.float64:
    """Computes the mean of f(x) for a given dx."""
    return np.dot(dx, f) / np.sum(dx)


def _integrate(f: np.ndarray, dx: np.ndarray, unit: str | float | None = None) -> np.float64:
    """Integrates f(x) for a given dx. The unit can be specified to normalize the result. 
    
    If unit is None, the result is in the same unit as f(x).
    If unit is a float, the result is divided by this float.
    If unit is a string, the result is divided by the corresponding normalization.
    """

    if unit is None:
        return np.dot(f, dx)

    normalizations = {
        "M_Jup": M_Jup_in_g,
        "M_Sol": M_Sol_in_g,
        "M_Earth": M_Earth_in_g,
        "g": 1.0,
    }

    if isinstance(unit, float):
        normalization: float = unit
    else:
        normalization: float = normalizations[unit]

    
    return np.dot(f, dx) / normalization
