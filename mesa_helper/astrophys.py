import numpy as np
from scipy import integrate
from os import path

resources_dir = path.join(path.dirname(__file__), 'resources')

# >>> constants >>>
# physics
sigma_SB = 5.670374e-5  # erg cm^-2 K^-4
NA = 6.02214e23         # per mol
kB = 1.38065e-16        # erg/K
yr_in_s = 3.1536e7      # seconds
s_in_yr = 1/yr_in_s     # years

# Sun
R_Sol_in_cm = 6.957e10
M_Sol_in_g = 1.988435e33
M_Sol_in_Earth = 3.329e-5
L_Sol_in_erg_s = 3.828e33

# abundances
Z_Sol = 0.0174064       # proto-solar values accoding to Lodders21
Y_Sol = 0.276522        # proto-solar values according to Lodders21
X_Sol = 1 - Y_Sol - Z_Sol

# Jupiter
R_Jup_in_Sol = 0.10054
R_Jup_in_cm = 6.995e9
M_Jup_in_g = 1.89813e30
M_Jup_in_Sol = 0.000955
M_Jup_in_Earth = 317.8

# Earth
M_Earth_in_Sol = 3.e-6
M_Earth_in_Jup = 0.00314636
M_Earth_in_g = 5.97e27

Teff_Jup = 124  # K

# <<< constants <<<


# >>> astro functions >>>

def from_flux_to_equilibrium_temperature(flux):
    return (flux/4/sigma_SB)**(1/4)

def scaled_solar_ratio_mass_fractions(Z : float | np.ndarray ) -> np.ndarray:
    a = (1-Z)/(1-Z_Sol)
    X = a * X_Sol
    Y = a * Y_Sol
    return np.array([X, Y, Z])

def diffusion_timescale(D : float, L : float)->float:
    return L**2/D

# converts kerg per baryon into specific entropy (erg/g/K)

def specific_entropy(entropy_in_kerg):
    return NA*kB*entropy_in_kerg

def specific_entropy_in_kerg(specific_entropy):
    return specific_entropy/(NA*kB)

# visible opacity fit from Guillot 2010
def kappa_v_Guillot_2010(T_eq):
    "Return the visible opacity in cm^2/g for a given equilibrium temperature."
    kappa_0 = 6e-3
    T_0 = 2000
    T_irr = np.sqrt(2) * T_eq
    return kappa_0 * np.sqrt(T_irr/T_0)

# Compute the intial radius for an adiabatic model given the inital mass and center entropy

# <<< astro functions <<<
