import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy import integrate
from os import path

resources_dir = path.join(path.dirname(__file__), 'resources')

# >>> constants >>>
# physics
sigma_SB = 5.670374e-5  # erg cm^-2 K^-4
NA = 6.02214e23         # per mol
kB = 1.38065e-16        # erg/K
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

# >>> math functions >>>
def mean_functional_residual_from_list(f: list, f0: list, dx: list) -> float:
 
        # calculate the residual
        variance = np.sum(np.power(f - f0, 2) * dx) / np.sum(dx)

        return np.sqrt(variance)

def mean_functional_residual(f, f0, x_min: float, x_max: float):

    df2 = lambda x: np.power(f(x) - f0(x), 2)
    
    h2 = integrate.quad(df2, x_min, x_max)[0]/(x_max - x_min)

    return np.sqrt(h2)

def heterogeneity(f, x_min: float, x_max: float):
    
        f_mean = integrate.quad(f, x_min, x_max)[0]/(x_max - x_min)
    
        df2 = lambda x: np.power(f(x) - f_mean, 2)
        
        h2 = integrate.quad(df2, x_min, x_max)[0]/(x_max - x_min)
    
        return np.sqrt(h2)

# <<< math functions <<<

# >>> astro functions >>>

def from_Jupiter_to_Solar_mass(mass):
    return mass * M_Jup_in_Sol


def from_flux_to_equilibrium_temperature(flux):
    return (flux/4/sigma_SB)**(1/4)


def scaled_solar_ratio_mass_fractions(Z)->list:
    a = (1-Z)/(1-Z_Sol)
    X = a * X_Sol
    Y = a * Y_Sol
    return [X, Y, Z]

# converts kerg per baryon into specific entropy (erg/g/K)

def specific_entropy(entropy_in_kerg):
    return NA*kB*entropy_in_kerg

# visible opacity fit from Guillot 2010
def kappa_v_Guillot_2010(T_eq):
    "Return the visible opacity in cm^2/g for a given equilibrium temperature."
    kappa_0 = 6e-3
    T_0 = 2000
    T_irr = np.sqrt(2) * T_eq
    return kappa_0 * np.sqrt(T_irr/T_0)

# Compute the intial radius for an adiabatic model given the inital mass and center entropy

# Load the entropy data
entripy_interpolation_file = 'initial_entropy_interpolation_file.txt'
src_entropy_interpolation = path.join(resources_dir, entripy_interpolation_file)
mass_grid, inital_radius_grid, initial_entropy_grid = np.loadtxt(src_entropy_interpolation, unpack=True)

# Create a linear interpolation object
interp = LinearNDInterpolator(list(zip(mass_grid, initial_entropy_grid)), inital_radius_grid)

# Define the function to return interpolated y value for given x and z values
def initial_radius(M_p : float, s0 : float, **kwargs) -> float: # [M_J] and [kB/baryon]:
    return interp(M_p, s0)


# <<< astro functions <<<
