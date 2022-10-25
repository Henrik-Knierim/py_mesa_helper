# >>> constants >>>
# physics
sigma_SB = 5.670374e-5 # erg cm^-2 K^-4
# Jupiter
R_Jup_in_Sol = 0.10054 # R_Sun

Teff_Jup = 124 # K
# <<< constants <<<

# >>> astro functions >>>
def from_flux_to_equilibrium_temperature(flux):
    return (flux/4/sigma_SB)**(1/4)
# <<< astro functions <<<