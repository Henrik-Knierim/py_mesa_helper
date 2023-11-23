# class for anything related to after the simulation has been run
# for example, analyzing, plotting, saving, etc.
import enum
import os
from unittest import result
import numpy as np
import mesa_reader as mr
from py import log
from mesa_inlist_manager.astrophys import M_Jup_in_g
class Simulation:
    """Class for anything related to after the simulation has been run. For example, analyzing, plotting, saving, etc."""

    def __init__(self, parent_dir = './LOGS', suite = '', sim = '', final_age = 5e9) -> None:
        
        # parent directory of the simulation
        self.parent_dir = parent_dir

        # simulation suite (if any)
        self.suite = suite

        # simulation (if any)
        self.sim = sim

        # full relative path to the suite and simulation
        self.suite_dir = os.path.join(self.parent_dir, self.suite)
        
        if suite != '':
            self.log_dirs = os.listdir(self.suite_dir)

        if self.sim != '':
            self.sim_dir = os.path.join(self.suite_dir, self.sim)


        # import sim results
        self._init_mesa_logs()
        self._init_mesa_histories()
        # delete simulations that do not converge
        self.remove_non_converged_simulations(final_age)

        self.n_simulations = len(self.histories)

    def _init_mesa_logs(self) -> None:
        """Initialzes `MesaLogDir` objects for each log directory in the simulation."""
        self.logs = {}
        # if we only consider one simulation, then we only have one log directory
        if hasattr(self, 'sim_dir'):
            self.logs[self.sim] = mr.MesaLogDir(self.sim_dir)
            # for convenience, we also create a log attribute for the log directory
            self.log = self.logs[self.sim]

        else:
            for log_dir in self.log_dirs:
                self.logs[log_dir] = mr.MesaLogDir(os.path.join(self.suite_dir, log_dir))

    def _init_mesa_histories(self) -> None:
        """Initializes `MesaData` objects for each history file in the simulation."""
        
        # make sure self.logs exists
        if not hasattr(self, 'logs'):
            self._init_mesa_logs()

        self.histories = {}
        for key, log in self.logs.items():
            self.histories[key] = log.history
        
        # if we only consider one simulation, then we only have one history file
        # for convenience, we then also create a history attribute for the history file
        if hasattr(self, 'sim_dir'):
            self.history = self.histories[self.sim]

    def get_suite_params(self, free_params):
        """Returns a dictonary of the free parameters of the simulation suite."""
        out = {log : {} for log in self.logs}
        for log in self.logs:
            sim_dir = log.split('/')[-1]
            for param in free_params:
                if param in sim_dir:
                    splitted_sim_dir = sim_dir.split(param)
                    value = float(splitted_sim_dir[1].split('_')[1])
                    out[log][param] =  value

                else:
                    raise ValueError(f'Parameter {param} not found in simulation directory {sim_dir}')

        # sort the data by the first free parameter
        out = dict(sorted(out.items(), key=lambda item: item[1][free_params[0]]))

        self.results = out

    # ------------------------------ #
    # --- Simulation Convergence --- #
    # ------------------------------ #

    def has_conserved_mass_fractions(self, tol = 1e-3, starting_model = 0, final_model = -1) -> bool:
        """Checks if the simulation has conserved mass fractions to a certain tolerance"""
        
        mass_keys = ['total_mass_h1', 'total_mass_he4', 'total_mass_o16']

        for key, history in self.histories.items():
            
            dm_h1 = history.data('total_mass_h1')[final_model] - history.get('total_mass_h1')[starting_model]
            dm_he4 = history.data('total_mass_he4')[final_model] - history.get('total_mass_he4')[starting_model]
            dm_o16 = history.data('total_mass_o16')[final_model] - history.get('total_mass_o16')[starting_model]
            
            if abs(dm_h1) > tol or abs(dm_he4) > tol or abs(dm_o16) > tol:
                return False
            
        return True
    
    def has_final_age(self, final_age, tol = 1e-3) -> dict:
        """Checks if the simulation has the final age to a certain tolerance."""
        
        out = {key : True for key in self.histories}
        for key, history in self.histories.items():
            
            if abs(history.data('star_age')[-1] - final_age) > tol:
                out[key] = False
            
        return out
    
    def remove_non_converged_simulations(self, final_age) -> None:
        """Removes all simulations that do not converge according to the criterion."""
        
        bools = self.has_final_age(final_age)

        for key, bool in bools.items():
            if not bool:
                del self.histories[key]
                del self.logs[key]
    
    
    # ------------------------------ #
    # ----- Simulation Results ----- #
    # ------------------------------ #
    def add_history_data(self, history_key, model = -1):
        """Adds `history_key` to `self.results`."""
        if hasattr(self, 'results'):
            for log_key in self.results:
                self.results[log_key][history_key] = self.histories[log_key].data(history_key)[model]

        else:
            raise ValueError('No results found. Run get_suite_params() first.')
    
    @staticmethod
    def integrate_profile(profile, quantity, m0 = 0, m1 = 1):
        """Integrates the profile quantity from m0 to m1 in Jupiter masses."""
        m_Jup = profile.data('mass_Jup')
        dm = profile.data('dm')[(m0 < m_Jup)&(m_Jup<m1)]/M_Jup_in_g
        quantity = profile.data(quantity)[(m0 < m_Jup)&(m_Jup<m1)]
        return np.dot(dm, quantity)
    
    @staticmethod
    def mean_profile_value(profile, quantity, m0 = 0, m1 = 1):
        """Integrates the profile quantity from m0 to m1 in Jupiter masses."""
        m_Jup = profile.data('mass_Jup')
        dm = profile.data('dm')[(m0 < m_Jup)&(m_Jup<m1)]/M_Jup_in_g
        quantity = profile.data(quantity)[(m0 < m_Jup)&(m_Jup<m1)]
        return np.dot(dm, quantity)/np.sum(dm)


    def add_integrated_profile_data(self, quantity, m0 = 0, m1 = 1, profile_number = -1, name = None):
        """Integrates the profile quantity from m0 to m1 and adds it to `self.results`."""
        if name is None:
            integrated_quantity_key = 'integrated_'+quantity
        else:
            integrated_quantity_key = name
        
        if hasattr(self, 'results'):
            for log_key in self.results:
                profile = self.logs[log_key].profile_data(profile_number = profile_number)
                self.results[log_key][integrated_quantity_key] = Simulation.integrate_profile(profile, quantity, m0, m1)

        else:
            raise ValueError('No results found. Run get_suite_params() first.')
        
    def add_mean_profile_data(self, quantity, m0 = 0, m1 = 1, profile_number = -1, name = None):
        """Computes the mean of the profile quantity from m0 to m1 and adds it to `self.results`."""
        
        if name is None:
            integrated_quantity_key = 'integrated_'+quantity
        else:
            integrated_quantity_key = name
        
        if hasattr(self, 'results'):
            for log_key in self.results:
                profile = self.logs[log_key].profile_data(profile_number = profile_number)
                self.results[log_key][integrated_quantity_key] = Simulation.mean_profile_value(profile, quantity, m0, m1)

        else:
            raise ValueError('No results found. Run get_suite_params() first.')
    
    def extract_results(self, result_keys):
        """Returns the keys specified in result_keys."""
        out = {}
        for key in result_keys:
            out[key] = np.zeros(self.n_simulations)
            for i, log_key in enumerate(self.results):
                out[key][i] = self.results[log_key][key]
        
        return out
    