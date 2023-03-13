# class for creating compositional gradients
# packages
import numpy as np
from mesa_inlist_manager.astrophys import scaled_solar_ratio_mass_fractions, X_Sol, Y_Sol, Z_Sol, M_Earth_in_Jup

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

# compositional profiles
def lin(m:np.ndarray, m_1:float, m_2:float, f_1:float, f_2:float, **kwargs):

    # tests
    if m_2 < m_1:
        raise Exception("m_2 must be larger than m_1")
    elif m_1 < 0:
        raise Exception("m_1 needs to be >= 0")
    elif any(n < 0 for n in m):
        raise Exception("m should contain positive numbers only")
    
    # linear function f = a m + b
    a = - (f_2-f_1)/(m_1-m_2)
    b = - (m_2*f_1 - m_1*f_2)/(m_1 - m_2)

    return np.piecewise(m, [m < m_1, ((m_1 <= m) & (m <= m_2)), m > m_2], [lambda m: f_1, lambda m: a*m+b, f_2])

def Z_lin_M_z(m:np.ndarray, m_1:float, m_2:float, M_z:float, Z_atm:float, **kwargs):

    M_z = M_z*M_Earth_in_Jup
    # tests
    if m_2 < m_1:
        raise Exception("m_2 must be larger than m_1")
    elif m_1 < 0:
        raise Exception("m_1 needs to be >= 0")
    elif any(n < 0 for n in m):
        raise Exception("m should contain positive numbers only")
    elif M_z < 0:
        raise Exception("M_z needs to be >= 0")
    elif not 0<=Z_atm<= 1:
        raise Exception("Z_at needs to be between 0 and 1")

    # for some reason, I need to pass the lambda functions directly without predefining them ...
    return np.piecewise(m, [m < m_1, ((m_1 <= m) & (m <= m_2)), m > m_2], [lambda m:(-2*M_z+(-m_1+m_2)*Z_atm)/(m_1 - m_2), lambda m: (2*(-m+m_2)*M_z + (m_1-m_2)*(-2*m+m_1+m_2)*Z_atm)/(m_1-m_2)**2, Z_atm])

def stepwise(m, m_transition, f_1, f_2, **kwargs):

    # tests
    if any(n < 0 for n in m):
        raise Exception("m should contain positive numbers only")
    
    return np.piecewise(m, [m <= m_transition, m > m_transition], [f_1, f_2])

class CompositionalGradient:
    def __init__(self, method : str, M_p : float, iso_net = 'planets') -> None:
        
        # planet mass as input parameter
        self.M_p = M_p

        # set MESA reaction network
        # test
        if not iso_net in ['planets','basic']:
            raise Exception(f"iso_net={iso_net} not supported.")
        
        self.iso_net = iso_net

        # set gradient method
        if (method == 'Y_lin') or (method == 'Z_lin'):
            self.abu_profile = lin

        elif (method == 'Y_log') or (method == 'Z_log'):
            self.abu_profile = lambda m, m_1, m_2, log_f_1, log_f_2: 10**lin(m, m_1, m_2, log_f_1, log_f_2)

        elif method == 'Z_lin_M_z':
            self.abu_profile = Z_lin_M_z

        elif (method == 'Y_stepwise') or (method == 'Z_Stepwise'):
            self.abu_profile = stepwise

        else:
            raise Exception(f"method={method} not supported.")
        
        # set abundance scaling method
        if 'Z' in method:
            self.scaled_abundances = self._scaled_abundances_Z

        elif 'Y' in method:
            self.scaled_abundances = self._scaled_abundances_H_He

    # scaled abundances
    def _scaled_abundances_Z(self, Z:float):
        [X, Y, Z] = scaled_solar_ratio_mass_fractions(Z)
        if self.iso_net == 'basic':
            f = lambda el: X/X_Sol if el=="H" else (Y/Y_Sol if el in ["He3","He4"] else Z/Z_Sol) 
            abu = {}
            abu.update((el,X_el*f(el)) for el, X_el in X_el_basic.items())
        elif self.iso_net == 'planets':
            abu = {"H":X, "He4":Y, "O16":Z}

        return abu
    
    # scaled abundances for pure H-He
    def _scaled_abundances_H_He(self, Y:float):
        X = 1-Y
        if self.iso_net == 'basic':
            f = lambda el: X/X_Sol if el=="H" else (Y/Y_Sol if el in ["He3","He4"] else 0.) 
            abu = {}
            abu.update((el,X_el*f(el)) for el, X_el in X_el_basic.items())
        elif self.iso_net == 'planets':
            abu = {"H":X, "He4":Y, "O16":0.}
        return abu


    # create file for relax_initial_composition

    def m(self, n_bins = 10000, **kwargs):
        return np.linspace(0, self.M_p, n_bins)

    def _create_composition_list(self, *args, **kwargs):
        
        # list of Z(m) (or Y(m))
        mass_bins = self.m(**kwargs)
        abu_list = self.abu_profile(mass_bins, *args, **kwargs)

        l = []
        for i, m_bin in enumerate(mass_bins):
            # creates list [mass_bin, X_H(mass_bin), ..., X_Mg24(mass_bin)]
            l.append([(self.M_p-m_bin)/self.M_p, *self.scaled_abundances(abu_list[i]).values()])

        # reverse order for MESA's relax_inital_composition format
        return np.flip(l, 0)

    def create_relax_inital_composition_file(self, *args, **kwargs):

        """

        Creates a file for `MESA`'s `relax_inital_composition functionality`. The `**kwargs` depend upon the method used.
        For Z_lin_M_z, we have

        Parameters
        ----------
        
        m_1 : float
            lower bound for linear profile
        m_2 : float
            upper bound for linear profile
        M_z : float
            core mass
        Z_atm : float
            envelope metallicity before mixing
        """
        
        relax_composition_filename='relax_composition_file.dat'

        # comp_list = [[mass_bin, spec_1, spec_2, ..., spec_N], ...]
        comp_list = self._create_composition_list(*args, **kwargs)
        num_points = len(comp_list)
        num_species = len(comp_list[0])-1

        with open(relax_composition_filename, 'w') as file:
            file.write(f"{num_points}  {num_species}\n")
            for l in comp_list:
                str_version = [f'{el:.16e}' for el in l]
                file.write('  '.join(str_version)+'\n')

        print(f'{relax_composition_filename} was created successfully.')

    

