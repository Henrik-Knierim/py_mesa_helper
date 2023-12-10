# class for anything related to after the simulation has been run
# for example, analyzing, plotting, saving, etc.
import matplotlib.pyplot as plt
import os
import numpy as np
import mesa_reader as mr
import pandas as pd
from mesa_inlist_manager.astrophys import M_Jup_in_g, specific_entropy

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
            # sort the log directories
            # maybe you need to modify this if you have multiple suite parameters
            self.log_dirs.sort()

        if self.sim != '':
            self.sim_dir = os.path.join(self.suite_dir, self.sim)


        # import sim results
        self._init_mesa_logs()
        self._init_mesa_histories()
        # delete simulations that do not converge
        self.remove_non_converged_simulations(final_age)

        self.n_simulations = len(self.histories)

        # add the simulation parameters to self.results
        if suite != '':
            self.results = pd.DataFrame({'log_dir': self.log_dirs})
        
        if self.sim != '':
            self.results = pd.DataFrame({'log_dir': [self.sim]})

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

    @staticmethod
    def extract_value(string, free_param : str):
        """Extracts the numerical value of `free_param` from `string`."""
        if type(free_param) != str:
            raise TypeError('free_params must be a string')
        elif free_param not in string:
            raise ValueError(f'Parameter {free_param} not found in string {string}')

        splitted_string = string.split(free_param)
        value = float(splitted_string[1].split('_')[1])
        return value
  
        
    def get_suite_params(self, free_params):
        """Add numerical value of `free_params` to the results DataFrame."""
        # make free_params a list if it is not already
        if type(free_params) != list:
            free_params = [free_params]

        out = {}
        for free_param in free_params:
            out[free_param] = [self.extract_value(log_dir, free_param) for log_dir in self.results['log_dir']]

        self.results = pd.concat([self.results, pd.DataFrame(out)], axis = 1)
        

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
        self.failed_simulations = [key for key, bool in bools.items() if not bool]
        
        for key, bool in bools.items():
            if not bool:
                del self.histories[key]
                del self.logs[key]
                del self.log_dirs[self.log_dirs.index(key)]
        
    def delete_suite(self):
        """Cleans the logs directory."""
        os.system(f'rm -r {self.suite_dir}')

    def delete_failed_simulations(self, final_age):
        """Deletes the failed simulations."""
        pass
    
    # ------------------------------ #
    # ----- Simulation Results ----- #
    # ------------------------------ #
    def add_history_data(self, history_keys, model = -1):
        """Adds `history_key` to `self.results`."""
        if type(history_keys) != list:
            history_keys = [history_keys]
        
        for history_key in history_keys:
            out = [self.histories[log_key].data(history_key)[model] for log_key in self.results['log_dir']]
            self.results[history_key] = out
    
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
        
        out = [Simulation.integrate_profile(self.logs[log_key].profile_data(profile_number = profile_number), quantity, m0, m1) for log_key in self.results['log_dir']]
        self.results[integrated_quantity_key] = out

        
    def add_mean_profile_data(self, quantity, m0 = 0, m1 = 1, profile_number = -1, name = None):
        """Computes the mean of the profile quantity from m0 to m1 and adds it to `self.results`."""
        
        if name is None:
            integrated_quantity_key = 'integrated_'+quantity
        else:
            integrated_quantity_key = name
        
        out = [Simulation.mean_profile_value(self.logs[log_key].profile_data(profile_number = profile_number), quantity, m0, m1) for log_key in self.results['log_dir']]
        self.results[integrated_quantity_key] = out

    # ------------------------------ #
    # -------- Plot Results -------- #
    # ------------------------------ #

    def profile_plot(self, x, y, profile_number = -1, ax = None, **kwargs):
        """Plots the profile data with (x, y) as the axes at `profile_number`."""
        if ax is None:
            fig, ax = plt.subplots()
        
        for log_key in self.logs:
            profile = self.logs[log_key].profile_data(profile_number = profile_number)
            ax.plot(profile.data(x), profile.data(y), **kwargs)
        
        ax.set(xlabel = x, ylabel = y)
        return ax
    
    def history_plot(self, x, y, ax = None, **kwargs):
        """Plots the history data with (x, y) as the axes."""
        if ax is None:
            fig, ax = plt.subplots()
        
        for log_key in self.logs:
            history = self.histories[log_key]
            ax.plot(history.data(x), history.data(y), **kwargs)
        
        ax.set(xlabel = x, ylabel = y)
        return ax
    
    # ------------------------------ #
    # ------- Static Methods ------- #
    # ------------------------------ #
    
    @staticmethod
    def mod_file_exists(mod_file = 'planet_relax_composition.mod'):
        """Returns True if the mod_file exits."""
        # test if the file ending is .mod
        if mod_file[-3:] != 'mod':
            raise ValueError('mod_file must end with .mod')
        return os.path.isfile(mod_file)