# >>> constants >>>
# physics
sigma_SB = 5.670374e-5 # erg cm^-2 K^-4

# Sun
M_Sol_in_gm = 1.988435e33
Z_Sol = 0.0174064 # proto-solar values accoding to Lodders21
Y_Sol = 0.276522 # proto-solar values according to Lodders21
X_Sol = 1 - Y_Sol - Z_Sol

# Jupiter
R_Jup_in_Sol = 0.10054
M_Jup_in_gm = 1.89813e30
M_Jup_in_Sol = 0.000955

# Earth
M_Earth_in_Sol = 3.e-6 # M_Sol

Teff_Jup = 124 # K
# <<< constants <<<

# >>> astro functions >>>
def from_flux_to_equilibrium_temperature(flux):
    return (flux/4/sigma_SB)**(1/4)

def scaled_solar_ratio_mass_fractions(Z):
    a = (1-Z)/(1-Z_Sol)
    X = a * X_Sol
    Y = a * Y_Sol
    return [X, Y, Z]
# <<< astro functions <<<