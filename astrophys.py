import numpy as np

# >>> constants >>>
# physics
sigma_SB = 5.670374e-5  # erg cm^-2 K^-4

# Sun
M_Sol_in_g = 1.988435e33
Z_Sol = 0.0174064  # proto-solar values accoding to Lodders21
Y_Sol = 0.276522  # proto-solar values according to Lodders21
X_Sol = 1 - Y_Sol - Z_Sol

# Jupiter
R_Jup_in_Sol = 0.10054
R_Jup_in_cm = 6.995e9
M_Jup_in_g = 1.89813e30
M_Jup_in_Sol = 0.000955

# Earth
M_Earth_in_Sol = 3.e-6
M_Earth_in_Jup = 0.00314636

Teff_Jup = 124  # K

# abundace data for basic.net
# src: Lodders+2020
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
# <<< constants <<<

# >>> astro functions >>>


def from_flux_to_equilibrium_temperature(flux):
    return (flux/4/sigma_SB)**(1/4)


def scaled_solar_ratio_mass_fractions(Z):
    a = (1-Z)/(1-Z_Sol)
    X = a * X_Sol
    Y = a * Y_Sol
    return [X, Y, Z]

# creates a linear compositional (Z) gradient between the mass bins m_1 and m_2

# linear gradient computed from user input Z_1 and Z_2
# same as grad_Y_lin, only that (Y_1, Y_2) = (Z_1, Z_2)
grad_Z_lin_Z_1_Z_2 = lambda m, m_1, m_2, Z_1, Z_2: grad_Y_lin(m, m_1, m_2, Z_1, Z_2)

grad_Z_log = lambda m, m_1, m_2, log_Z_1, log_Z_2: 10**grad_Z_lin_Z_1_Z_2(m, m_1, m_2, log_Z_1, log_Z_2)

# linear gradient computed from bulk heavy content (M_z)
def grad_Z_lin_M_z(m, m_1, m_2, M_z, Z_atm):

    # tests
    if m_2 < m_1:
        raise Exception("m_2 must be larger than m_1")
    elif m_1 < 0:
        raise Exception("m_1 needs to be >= 0")
    elif any(f < 0 for f in m):
        raise Exception("m should contain positive numbers only")

    # for some reason, I need to pass the lambda functions directly without predefining them ...
    return np.piecewise(m, [m < m_1, ((m_1 <= m) & (m <= m_2)), m > m_2], [lambda m:(-2*M_z+(-m_1+m_2)*Z_atm)/(m_1 - m_2), lambda m: (2*(-m+m_2)*M_z + (m_1-m_2)*(-2*m+m_1+m_2)*Z_atm)/(m_1-m_2)**2, Z_atm])

def grad_Z_stepwise(m, m_transition, Z_inner, Z_outer):

    # tests
    if any(f < 0 for f in m):
        raise Exception("m should contain positive numbers only")
    
    return np.piecewise(m, [m <= m_transition, m > m_transition], [Z_inner, Z_outer])

def grad_Y_lin(m, m_1, m_2, Y_1, Y_2):
    
    # tests
    if m_2 < m_1:
        raise Exception("m_2 must be larger than m_1")
    elif m_1 < 0:
        raise Exception("m_1 needs to be >= 0")
    elif any(f < 0 for f in m):
        raise Exception("m should contain positive numbers only")

    a = - (Y_2-Y_1)/(m_1-m_2)
    b = - (m_2*Y_1 - m_1*Y_2)/(m_1 - m_2)

    return np.piecewise(m, [m < m_1, ((m_1 <= m) & (m <= m_2)), m > m_2], [lambda m: Y_1, lambda m: a*m+b, Y_2])

def grad_Y_stepwise(m, m_transition, Y_inner, Y_outer):

    # tests
    if any(f < 0 for f in m):
        raise Exception("m should contain positive numbers only")
    
    return np.piecewise(m, [m <= m_transition, m > m_transition], [Y_inner, Y_outer])

def scaled_heavy_mass_abundaces(Z, iso_net):
    [X, Y, Z] = scaled_solar_ratio_mass_fractions(Z)
    if iso_net == 'basic':
        f = lambda el: X/X_Sol if el=="H" else (Y/Y_Sol if el in ["He3","He4"] else Z/Z_Sol) 
        abu = {}
        abu.update((el,X_el*f(el)) for el, X_el in X_el_basic.items())
    elif iso_net == 'planets':
        abu = {"H":X, "He4":Y, "O16":Z}

    return abu

# similar to scaled_heavy_mass_abundaces but just for H and He
def scaled_H_He_mass_abundances(Y, iso_net):
    X = 1-Y
    if iso_net == 'basic':
        f = lambda el: X/X_Sol if el=="H" else (Y/Y_Sol if el in ["He3","He4"] else 0.) 
        abu = {}
        abu.update((el,X_el*f(el)) for el, X_el in X_el_basic.items())
    elif iso_net == 'planets':
        abu = {"H":X, "He4":Y, "O16":0.}
    return abu

# <<< astro functions <<<
