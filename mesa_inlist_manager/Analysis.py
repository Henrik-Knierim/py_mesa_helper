# class for analyzing the output of a MESA run
import numpy as np
import mesa_reader as mr
import os
import glob
from tabulate import tabulate
from mesa_inlist_manager.astrophys import M_Earth_in_g, M_Earth_in_Sol
from mesa_inlist_manager.CompositionalGradient import CompositionalGradient


class Analysis:

    def __init__(self, src = 'LOGS') -> None:
        # location of the LOGS directory
        self.src = src
    
    def heterogeneity(self, model_number = -1, **kwargs):
        """Gives the deviation of the metallicity from a homogeneous profile."""

        # init log object
        log = mr.MesaLogDir(self.src)
        # gets the last profile
        profile = log.profile_data(model_number = model_number)

        # get average metallicty
        if model_number == -1:
            Z_avg = log.history.data('average_o16')[-1]
        else:
            Z_avg = log.history.data_at_model_number('average_o16', m_num = model_number) # check definition

        # get the metallicity profile
        dm = profile.dm
        Z = profile.z

        # calculate the deviation
        variance = np.sum(np.power(Z - Z_avg, 2) * dm) / np.sum(dm)

        return np.sqrt(variance)
    
    def relative_atmospheric_metallicity(self):
        """Gives the ratio between the atmopsheric metallicity (Z_atm) and the average metallicity (Z_avg)."""

        # init log object
        log = mr.MesaLogDir(self.src)
        
        # get the last profile
        profile = log.profile_data()

        # get the average metallicity
        Z_avg = log.history.data('average_o16')[-1]
        
        # select the metallicty where convection starts
        Z_atm = profile.z[profile.sch_stable == 0][0]
        
        # calculate the relative atmospheric metallicity
        return Z_atm/Z_avg
    
    def dlogZ_dlogP(self, **kwargs):
        """Gives the gradient of the metallicity profile."""

        # init log object
        log = mr.MesaLogDir(self.src)
        
        # get profile
        profile = log.profile_data(**kwargs)

        # get the pressure and metallicity profiles
        logP = profile.logP
        Z = profile.z
        logZ = np.log10(Z)

        # calculate the gradient
        return np.gradient(logZ, logP)
    
    def dlogmu_dlogP(self, **kwargs):
        """Gives the gradient of the mean molecular weight profile."""
        
        # init log object
        log = mr.MesaLogDir(self.src)
        
        # get profile
        profile = log.profile_data(**kwargs)

        # get the pressure and mean molecular weight profiles
        logP = profile.logP
        mu = profile.mu
        logmu = np.log10(mu)

        # calculate the gradient
        return np.gradient(logmu, logP)
    
    def heavy_mass_error(self, M_p, s0, **kwargs)->float:
        """Returns the error in the heavy mass of the planet"""
        
        # init log object
        from mesa_inlist_manager.MesaRun import MesaRun
        path = MesaRun(self.src).create_logs_path_string(M_p, s0)
        log = mr.MesaLogDir(path)

        # get the heavy mass of the planet theoretically
        comp_grad = CompositionalGradient(M_p = M_p, **kwargs)
        M_z_in = comp_grad.M_z_tot(**kwargs) 

        # get the heavy mass of the planet at the end of the simulation
        M_z_out = log.history.data('total_mass_o16')[-1]/M_Earth_in_Sol

        return abs(1-M_z_out/M_z_in)
    
    def get_history_data(self, data_name, model_number = -1):
        """Returns the data from the history.data file."""

        if model_number == -1:
            get_data = lambda history, data_name: history.data(data_name)[-1]
        else:
            get_data = lambda history, data_name: history.data_at_model_number(data_name, m_num = model_number)

        history_file = mr.MesaData(os.path.join(self.src,'history.data'))
            
        # get the data for the i-th simulation
        data = get_data(history_file, data_name)

        return data
    
    def get_evolution_parameters(self, keys : list) -> dict:
        """Gets the specified paramters from the evolution_parameters.txt file."""
        from mesa_inlist_manager.Inlist import Inlist
        evolution_params = {}
        with open(os.path.join(self.src,'evolution_parameters.txt'), 'r') as file:
            for line in file:
                key, value = line.strip().split('\t')
                if key in keys:
                    evolution_params[key] = Inlist.python_format(value)

        return evolution_params

class MultipleSimulationAnalysis:
    """Class for analyzing multiple simulations."""

    def __init__(self, src = 'LOGS',**kwargs) -> None:
        
        # parent directory of the LOGS directories
        self.src = src

        # get the names of the LOGS directories
        self._select_simulations(**kwargs)

        # get the planetary masses and initial entropies
        self._get_planetary_mass_and_intial_entropy()

    def _select_simulations(self, M_p = None, s0 = None, **kwargs):
        """Select subset of simulations from the parent directory."""

        # get the names of the LOGS directories
        if M_p == None and s0 == None:
            self.simulations = glob.glob(self.src +'/'+ "M_p_*_s0_*")
        elif M_p == None:
            self.simulations = glob.glob(self.src +'/'+ "M_p_*_s0_{:.1f}".format(s0))
        elif s0 == None:
            print(self.src +'/'+ "M_p_{}_s0_*".format(M_p))
            self.simulations = glob.glob(self.src +'/'+ "M_p_{:.2f}_s0_*".format(M_p))
        else:
            self.simulations = glob.glob(self.src +'/'+ "M_p_{:.2f}_s0_{:.1f}".format(M_p, s0))

        # sort the simulations by mass
        self.simulations.sort(key = lambda x: float(x.split('_')[-3]))

    def _get_planetary_mass_and_intial_entropy(self):
        """Gives the masses of the planets."""

        self.planetary_mass = np.zeros(len(self.simulations))
        self.initial_entropy = np.zeros(len(self.simulations))

        for i, simulation in enumerate(self.simulations):
            parts = simulation.split('_')
            self.planetary_mass[i] = float(parts[-3])
            self.initial_entropy[i] = float(parts[-1])

    def get_history_data(self, data_name, model_number = -1):
        """Returns the n-th entry of the history data for each simulation."""

        analysis = Analysis()

        # init data that will be returned
        data = np.zeros(len(self.simulations))
        for i, simulation in enumerate(self.simulations):
            analysis.src = simulation
            data[i] = analysis.get_history_data(data_name, model_number = model_number)
        
        return data


    def heterogeneity(self, **kwargs):
        """Gives the deviation of the metallicity from a homogeneous profile."""

        # init analysis object
        analysis = Analysis()

        # get the homogeneity for each simulation
        heterogeneity = np.zeros(len(self.simulations))
        for i, simulation in enumerate(self.simulations):
            analysis.src = simulation
            heterogeneity[i] = analysis.heterogeneity(**kwargs)
        
        return heterogeneity

    def relative_atmospheric_metallicity(self):
        """Gives the ratio between the atmopsheric metallicity (Z_atm) and the average metallicity (Z_avg)."""

        # init analysis object
        analysis = Analysis()

        # get the relative atmospheric metallicity for each simulation
        relative_atmospheric_metallicity = np.zeros(len(self.simulations))
        for i, simulation in enumerate(self.simulations):
            analysis.src = simulation
            relative_atmospheric_metallicity[i] = analysis.relative_atmospheric_metallicity()
        
        return relative_atmospheric_metallicity
    
    def get_evolution_parameters(self, keys : list):
        """Gives the specified parameters in evolution_parameters.txt for each simulation."""
            
        # init analysis object
        analysis = Analysis()

        # get the evolution parameters for each simulation
        evolution_parameters = {
            'M_p' : self.planetary_mass,
            's0' : self.initial_entropy
        }

        # init the evolution parameters
        for key in keys:
            evolution_parameters[key] = np.zeros(len(self.simulations))

        # get the evolution parameters for each simulation
        for i, simulation in enumerate(self.simulations):
            analysis.src = simulation

            d = analysis.get_evolution_parameters(keys)
            for key in keys:
                evolution_parameters[key][i] = d[key]

        return evolution_parameters
    
    def print_simulation_results(self, **kwargs) -> None:
        """Prints the results of a number of simulations."""
        
        rel_M_z_err = np.zeros(len(self.simulations))
        Z_atm = np.zeros(len(self.simulations))
        age = np.zeros(len(self.simulations))
        Z_atm_Z_avg_ratio = self.relative_atmospheric_metallicity()
        heterogeneity = self.heterogeneity()

        for i, l in enumerate(self.simulations):

            # init LOGS directory
            log = mr.MesaLogDir(self.simulations[i])
            
            # get last profile file date
            profile = log.profile_data()
            Z = profile.z

            # select the metallicty where convection starts
            Z_atm[i] = Z[profile.sch_stable == 0][0]

            # get the age
            age[i] = log.history.star_age[-1]

            # mass error
            rel_M_z_err[i] = Analysis(self.src).heavy_mass_error(M_p=self.planetary_mass[i], s0=self.initial_entropy[i], **kwargs)
        
        print(tabulate(np.array([self.planetary_mass, self.initial_entropy, age, Z_atm, Z_atm_Z_avg_ratio, heterogeneity, rel_M_z_err]).T, headers = ['M_p', 's0', 'age', 'Z_atm', 'Z_atm/Z_avg','heterogeneity','rel_M_z_err'], floatfmt=(".2f", ".1f", ".2e", ".4f", ".4f",".4f", ".4f")))
