# class for anything related to after the simulation has been run
# for example, analyzing, plotting, saving, etc.
from ast import Raise
import matplotlib.pyplot as plt
import os
import numpy as np
import mesa_reader as mr
import pandas as pd
from mesa_inlist_manager.astrophys import M_Jup_in_g
from scipy.interpolate import interp1d

class Simulation:
    """Class for anything related to after the simulation has been run. For example, analyzing, plotting, saving, etc."""

    def __init__(self, parent_dir : str = './LOGS', suite : str = '', sim : str = '', check_age_convergence = False, delete_horribly_failed_simulations = False, **kwargs) -> None:
        """Initializes the Simulation object.
        Parameters
        ----------
        parent_dir : str, optional
            The parent directory of the simulation. The default is './LOGS'.
        suite : str, optional
            The simulation suite. The default is ''.
        sim : str, optional
            The simulation. The default is ''.
        check_age_convergence : bool, optional
            If True, then the simulations that do not converge to the final age are removed. The default is True.
        **kwargs : dict
            Keyword arguments for `self.remove_non_converged_simulations`. For example, `final_age` can be specified.
        """
        
        # parent directory of the simulation
        self.parent_dir = parent_dir

        # simulation suite (if any)
        if suite != '':
            self.suite = suite
            # full relative path to the suite and simulation
            self.suite_dir = os.path.join(self.parent_dir, self.suite)

        # simulation (if any)
        if sim != '':
            self.sim = sim


        
        if hasattr(self, 'suite'):
            self.log_dirs = os.listdir(self.suite_dir)
            # sort the log directories
            # maybe you need to modify this if you have multiple suite parameters
            self.log_dirs.sort()

        if hasattr(self, 'sim'):
            self.sim_dir = os.path.join(self.parent_dir, self.sim)


        # import sim results
        # first, delete horribly failed simulations
        self.delete_horribly_failed_simulations() if delete_horribly_failed_simulations else None
        
        # then, initialize the mesa logs and histories
        self._init_mesa_logs()
        self._init_mesa_histories()
        
        # delete simulations that do not converge to the final age
        # these did not fail horribly, but they did not converge to the final age
        if check_age_convergence:
            final_age = kwargs.get('final_age', 5e9)
            self.remove_non_converged_simulations(final_age)

        self.n_simulations = len(self.histories)

        # add the simulation parameters to self.results
        if hasattr(self, 'suite'):
            self.results = pd.DataFrame({'log_dir': self.log_dirs})
        
        if hasattr(self, 'sim'):
            self.results = pd.DataFrame({'log_dir': [self.sim]})

    # create a __str__ method that returns the name of the suite, or the name of the simulation if there is no suite
    def __str__(self):
        if hasattr(self, 'suite'):
            return self.suite
    
        elif hasattr(self, 'sim'):
            return self.sim
        else:
            Raise(NotImplementedError('The simulation does not have a suite or a simulation.'))
    
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

    @staticmethod
    def _is_profile_index_valid(path_to_profile_index):
        """Returns True if profiles.index contains more than 2 lines."""
        with open(path_to_profile_index, 'r') as f:
            lines = f.readlines()
        return len(lines) > 2

    def delete_horribly_failed_simulations(self):
        """Deletes all simulations that have a profiles.index file with less than 2 lines."""
        
        if hasattr(self, 'log_dirs'):
            for log_dir in self.log_dirs:
                path_to_profile_index = os.path.join(self.suite_dir, log_dir, 'profiles.index')
                if not self._is_profile_index_valid(path_to_profile_index):
                    os.system(f'rm -r {os.path.join(self.suite_dir, log_dir)}')
                    self.log_dirs.remove(log_dir)
                    
        else:
            path_to_profile_index = os.path.join(self.sim_dir, 'profiles.index')
            if not self._is_profile_index_valid(path_to_profile_index):
                os.system(f'rm -r {self.sim_dir}')
                self.sim = ''
                self.log_dirs.remove(self.sim)

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
    def add_history_data(self, history_keys, model = -1, age = None):
        """Adds `history_key` to `self.results`."""
        if type(history_keys) != list:
            history_keys = [history_keys]
        
        if age is None:
            if model == -1:
                # add the final model
                for history_key in history_keys:
                    out = [self.histories[log_key].data(history_key)[model] for log_key in self.results['log_dir']]
                    self.results[history_key] = out
            elif isinstance(model, int):
                for history_key in history_keys:
                    out = [self.histories[log_key].data_at_model_number(history_key, model) for log_key in self.results['log_dir']]
                    self.results[history_key] = out
            else:
                raise TypeError('model must be an integer')
            
        elif isinstance(age, (int, float)):
            # find the model number closest to the age
            for history_key in history_keys:
                out = np.zeros(len(self.results['log_dir']))
                for i, log_key in enumerate(self.results['log_dir']):
                    ages = self.histories[log_key].data('star_age')
                    parameter = self.histories[log_key].data(history_key)
                    model = np.argmin(np.abs(ages - age))
                    out[i] = parameter[model]
                self.results[history_key] = out


        else:
            raise TypeError('age must be None or a float')
    
    @staticmethod
    def integrate_profile(profile, quantity : str, m0 : float = 0.0, m1 : float = 1.0):
        """Integrates the profile quantity from m0 to m1 in Jupiter masses."""
        m_Jup = profile.data('mass_Jup')
        dm = profile.data('dm')[(m0 < m_Jup)&(m_Jup<m1)]/M_Jup_in_g
        quantity = profile.data(quantity)[(m0 < m_Jup)&(m_Jup<m1)]
        return np.dot(dm, quantity)
    
    @staticmethod
    def mean_profile_value(profile, quantity, m0 : float = 0., m1 : float = 1.):
        """Integrates the profile quantity from m0 to m1 in Jupiter masses."""
        m_Jup = profile.data('mass_Jup')
        dm = profile.data('dm')[(m0 < m_Jup)&(m_Jup<m1)]/M_Jup_in_g
        quantity = profile.data(quantity)[(m0 < m_Jup)&(m_Jup<m1)]
        return np.dot(dm, quantity)/np.sum(dm)

    
    def add_integrated_profile_data(self, quantity : str, m0 : float = 0.0, m1 : float = 1.0, profile_number : int = -1, name = None):
        """Integrates the profile quantity from m0 to m1 and adds it to `self.results`."""
        if name is None:
            integrated_quantity_key = 'integrated_'+quantity
        else:
            integrated_quantity_key = name
        
        out = [Simulation.integrate_profile(self.logs[log_key].profile_data(profile_number = profile_number), quantity, m0, m1) for log_key in self.results['log_dir']]
        self.results[integrated_quantity_key] = out

        
    def add_mean_profile_data(self, quantity : str, m0 : float = 0.0, m1 : float = 1.0, profile_number : int = -1, model_number : int = -1, name = None):
        """Computes the mean of the profile quantity from m0 to m1 and adds it to `self.results`."""
        
        if name is None:
            integrated_quantity_key = 'mean_'+quantity
        else:
            integrated_quantity_key = name
        
        out = [Simulation.mean_profile_value(self.logs[log_key].profile_data(profile_number = profile_number, model_number=model_number), quantity, m0, m1) for log_key in self.results['log_dir']]
        self.results[integrated_quantity_key] = out

    def get_mean_profile_value(self, quantity : str, m0 : float = 0.0, m1 : float = 1.0, profile_number : int = -1, model_number : int = -1, log_dir = None):
        """Returns the mean of the profile quantity from m0 to m1."""
        return Simulation.mean_profile_value(self.logs[log_dir].profile_data(profile_number = profile_number, model_number=model_number), quantity, m0, m1)

    def get_profile_data(self, quantity : str, profile_number : int = -1, log_dir = None, **kwargs) -> np.ndarray:
        """Returns the profile data for `quantity`."""
        
        if hasattr(self, 'sim'):
            return self.logs[self.sim].profile_data(profile_number = profile_number, **kwargs).data(quantity)
        
        elif hasattr(self, 'suite'):
            if log_dir is None:
                raise ValueError('log_dir must be specified.')
            else:
                return self.logs[log_dir].profile_data(profile_number = profile_number, **kwargs).data(quantity)
            
        else:
            raise NotImplementedError("The method 'get_profile_data' is currently only supported for one simulation or a suite of simulations where the log_dir is specified.")
        
    def get_profile_data_at_condition(self, quantity : str, condition : str, value : float, profile_number : int = -1, log_dir = None, **kwargs) -> np.float_ | int:
        """Returns the profile data for `quantity` where `condition` is `value`.
        
        Description
        -----------
        The routine finds the index where the condition is satisfied and returns the profile data at that index. It starts from the first index (surface) and moves to the last index (core).

        Parameters
        ----------
        quantity : str
            The quantity of the profile.
        condition : str
            The condition of the profile.
        value : float
            The value of the condition.
        profile_number : int, optional
            The profile number. The default is -1.
        log_dir : str, optional
            The log directory. The default is None.
        **kwargs : dict
            Keyword arguments for `MesaProfileData`.
        """

        # throw an error if value is a bool or a string
        if isinstance(value, (bool, str)):
            raise ValueError('value must be a float or an integer.')
        
        condition_data = self.get_profile_data(condition, profile_number, log_dir, **kwargs)
        index = np.argmin(np.abs(condition_data - value))
        return self.get_profile_data(quantity, profile_number, log_dir, **kwargs)[index]

    def get_profile_number_at_profile_header_value(self, quantity : str, value : float, log_dir = None, **kwargs) -> int:
        """Returns the profile number where the profile header value is `value`."""
        
        # create the profile_header_values dictionary if it does not exist
        if not hasattr(self, 'profile_header_values'):
            self.profile_header_values = {}
        
        # create the profile_header_values dictionary for the log directory if it does not exist
        if self.profile_header_values.get(log_dir) is None:
            self.profile_header_values[log_dir] = {}
        
        # create the profile_header_values dictionary for the quantity if it does not exist
        if self.profile_header_values[log_dir].get(quantity) is None:
            self.profile_header_values[log_dir][quantity] = np.array([self.logs[log_dir].profile_data(profile_number = i, **kwargs).header_data[quantity] for i in self.logs[log_dir].profile_numbers])

        return np.argmin(np.abs(self.profile_header_values[log_dir][quantity] - value))
    
    def add_profile_data_at_condition(self, quantity : str, condition : str, value : float, profile_number : int = -1, name = None) -> None:
        """Adds the profile data for `quantity` where `condition` is `value` to `self.results`."""
        
        if name is None:
            name = f'first_{quantity}_at_{condition}_{value}'
        
        out = [self.get_profile_data_at_condition(quantity, condition, value, profile_number, log_dir) for log_dir in self.results['log_dir']]
        self.results[name] = out

    def get_profile_header(self, data : str = "star_age", profile_number = -1, log_dir = None, **kwargs):
        """Returns the profile header."""
        
        if hasattr(self, 'sim'):
            return self.logs[self.sim].profile_data(profile_number = profile_number, **kwargs).header_data[data]
        elif hasattr(self, 'suite'):
            if log_dir is None:
                raise ValueError('log_dir must be specified.')
            else:
                return self.logs[log_dir].profile_data(profile_number = profile_number, **kwargs).header_data[data]
        else:
            raise NotImplementedError("The method 'get_profile_header' is currently only supported for one simulation or a suite of simulations where the log_dir is specified.")
     
    def interpolate_profile_data(self, x : str, y : str, profile_number : int = -1, **kwargs) -> None:
        """Creates an interpolation function for (x,y) at `self.interpolation[log, profile_number, x, y]`."""
        
        # create the interpolation dictionary if it does not exist
        if not hasattr(self, 'interpolation'):
            self.interpolation = {}

        for log_key in self.logs:
            # if the interpolation does not exist, create it
            if self.interpolation.get((log_key, profile_number, x, y)) is None:
                profile = self.logs[log_key].profile_data(profile_number = profile_number)
                x_data = profile.data(x)
                y_data = profile.data(y)
                self.interpolation[log_key,profile_number,x,y] = interp1d(x_data, y_data, **kwargs)
    
    def interpolate_history_data(self, x: str, y: str, scaling = ('lin', 'lin'), **kwargs) -> None:
        """Creates an interpolation function for (x,y) at `self.interpolation[log, x, y]`."""
        
        # throw an error if scaling is not a tuple that contains either 'lin' or 'log'
        if not isinstance(scaling, tuple):
            raise ValueError('scaling must be a tuple.')
        elif not all([scale in ['lin', 'log'] for scale in scaling]):
            raise ValueError('scaling must be either "lin" or "log"')

        if not hasattr(self, 'interpolation'):
            self.interpolation = {}
        
        x_lbl = 'log_' + x if scaling[0] == 'log' else x
        y_lbl = 'log_' + y if scaling[1] == 'log' else y

        for log_key in self.logs:
            if self.interpolation.get((log_key, x_lbl, y_lbl)) is None:
                history = self.histories[log_key]
                x_data = history.data(x) if scaling[0] == 'lin' else np.log10(history.data(x))
                y_data = history.data(y) if scaling[1] == 'lin' else np.log10(history.data(y))
                self.interpolation[log_key, x_lbl, y_lbl] = interp1d(x_data, y_data, **kwargs)

    def get_relative_difference_of_two_profiles_from_two_logs(self, x : str, y : str, profile_number : int = -1, log_reference = None, log_compare = None, **kwargs):
        """Returns the relative difference of two profiles.
        
        Parameters
        ----------
        x : str
            The x-axis of the profile.
        y : str
            The y-axis of the profile.
        profile_number : int, optional
            The profile number. The default is -1.
        log_reference : str, optional
            The reference log. The default is None.
        log_compare : str, optional
            The log that is compared to the reference log. The default is None.
        **kwargs : dict
            Keyword arguments for `scipy.interpolate.interp1d`.
        """
        
        # call the interpolation routine
        # if the interpolation object for [log, profile_number, x, y] already exists, then it is not created by the interpolation routine
        self.interpolate_profile_data(x, y, profile_number = profile_number, **kwargs)

        # get the x range
        # the minimum is the smallest common x value of the reference and compare log
        x_min = max(self.get_profile_data(quantity = x, profile_number=profile_number, log_dir = log_reference).min(), self.get_profile_data(quantity = x, profile_number=profile_number, log_dir = log_compare).min())

        # the maximum is the largest common x value of the reference and compare log
        x_max = min(self.get_profile_data(quantity = x, profile_number=profile_number, log_dir = log_reference).max(), self.get_profile_data(quantity = x, profile_number=profile_number, log_dir = log_compare).max())

        x_values = np.linspace(x_min, x_max, 1000)
        
        if hasattr(self, 'suite'):
            if log_reference is None or log_compare is None:
                raise ValueError('log_reference and log_compare must be specified.')
            else:
                y_reference = self.interpolation[log_reference, profile_number, x,y](x_values)
                y_compare = self.interpolation[log_compare, profile_number, x,y](x_values)
                return x_values, (y_compare - y_reference)/y_reference
        else:
            raise NotImplementedError("The method 'get_relative_difference_of_two_profiles' is currently only supported for a suite of simulations where the log_reference and log_compare are specified.")
            
    def get_relative_difference_of_two_profiles_from_one_log(self, x : str, y : str, profile_number_reference : int = -1, profile_number_compare : int = -1, log = None, **kwargs):
        """Returns the relative difference of two profiles.

        Parameters
        ----------
        x : str
            The x-axis of the profile.
        y : str
            The y-axis of the profile.
        profile_number_reference : int, optional
            The reference profile number. The default is -1.
        profile_number_compare : int, optional
            The profile number that is compared to the reference profile. The default is -1.
        log : str, optional
            The log. The default is None.
        **kwargs : dict
            Keyword arguments for `scipy.interpolate.interp1d`.
        """
            
        # make sure the interpolation object for [log, profile_number, x, y] exists
        self.interpolate_profile_data(x, y, profile_number = profile_number_reference, **kwargs)
        self.interpolate_profile_data(x, y, profile_number = profile_number_compare, **kwargs)

        # if sim exists, then there is only one log directory
        # hence, we can define log
        if hasattr(self, 'sim') and log is None:            
            log = self.sim
        # check whether log is a valid log directory
        elif log not in self.log_dirs:
            raise ValueError('log must be a valid log directory.')

            

        # get the x range
        # the minimum is the smallest common x value of the reference and compare log
        x_min = max(self.get_profile_data(quantity = x, profile_number=profile_number_reference, log_dir = log).min(), self.get_profile_data(quantity = x, profile_number=profile_number_compare, log_dir = log).min())

        # the maximum is the largest common x value of the reference and compare log
        x_max = min(self.get_profile_data(quantity = x, profile_number=profile_number_reference, log_dir = log).max(), self.get_profile_data(quantity = x, profile_number=profile_number_compare, log_dir = log).max())

        x_values = np.linspace(x_min, x_max, 1000)

        if log is not None:
            y_reference = self.interpolation[log,profile_number_reference,x,y](x_values)
            y_compare = self.interpolation[log,profile_number_compare,x,y](x_values)
            return x_values, (y_compare - y_reference)/y_reference
        else:
            raise ValueError('log must be specified.')
        
    # ------------------------------ #
    # -------- Plot Results -------- #
    # ------------------------------ #

    def profile_plot(self, x, y, profile_number = -1, ax = None, set_labels = False, **kwargs):
        """Plots the profile data with (x, y) as the axes at `profile_number`."""
        if ax is None:
            fig, ax = plt.subplots()
        
        for log_key in self.logs:
            profile = self.logs[log_key].profile_data(profile_number = profile_number)
            
            # set the label (optional)
            if set_labels:
                kwargs['label'] = log_key

            ax.plot(profile.data(x), profile.data(y), **kwargs)
            
        
        ax.set(xlabel = x, ylabel = y)
        return ax

    def profile_series_plot(self, x : str, y : str, profile_numbers : list = [-1], ax = None, set_labels : bool = False, log_dir : str | None = None, **kwargs):
        """Plots the profile data with (x, y) as the axes at multiple profile numbers."""
        if ax is None:
            fig, ax = plt.subplots()

        for profile_number in profile_numbers:
            profile = self.logs[log_dir].profile_data(profile_number = profile_number)
            if set_labels:
                kwargs['label'] = f"{profile.header_data['star_age']:.2e}"
            ax.plot(profile.data(x), profile.data(y), **kwargs)
            
        ax.set(xlabel = x, ylabel = y)
        return ax
    
    def profile_series_plot_at_condition(self, x : str, y : str, condition : str, values : list[float], ax = None, set_labels : bool = False, log_dir : str | None = None, **kwargs):
        """Plots the profile data with (x, y) as the axes at multiple profile numbers where `condition` is `values`."""
        if ax is None:
            fig, ax = plt.subplots()

        for value in values:
            profile_number = self.get_profile_number_at_profile_header_value(condition, value, log_dir = log_dir)
            profile = self.logs[log_dir].profile_data(profile_number = profile_number)
            if set_labels:
                kwargs['label'] = f"{self.profile_header_values[log_dir][condition][profile_number]:.2e}"
            ax.plot(profile.data(x), profile.data(y), **kwargs)
            
        ax.set(xlabel = x, ylabel = y)
        return ax
    
    def history_plot(self, x, y, ax = None, set_labels = False, **kwargs):
        """Plots the history data with (x, y) as the axes."""
        if ax is None:
            fig, ax = plt.subplots()
        
        for log_key in self.logs:
            
            # set the label (optional)
            if set_labels:
                kwargs['label'] = log_key

            history = self.histories[log_key]
            ax.plot(history.data(x), history.data(y), **kwargs) 
        
        ax.set(xlabel = x, ylabel = y)
        return ax
    
    def relative_difference_of_two_profiles_plot(self, x : str, y : str, profile_number_reference : int = -1, profile_number_compare : int = None, log_reference = None, log_compare = None, ax = None, **kwargs):
        """Plots the relative difference of two profiles with (x, y) as the axes.
        
        The rountine either compares two profiles from the same log, or two profiles from two different logs.

        Parameters
        ----------
        x : str
            The x-axis of the profile.
        y : str
            The y-axis of the profile.
        profile_number_reference : int, optional
            The reference profile number. The default is -1.
        profile_number_compare : int, optional
            The profile number that is compared to the reference profile. The default is None.
        log_reference : str, optional
            The reference log. The default is None.
        log_compare : str, optional
            The log that is compared to the reference log. The default is None.
        ax : matplotlib.axes, optional
            The axes. The default is None.
        **kwargs : dict
            Keyword arguments for `matplotlib.pyplot.plot`.
        """

        if ax is None:
            fig, ax = plt.subplots()
        
        # in the routine, you either specify log_reference and log_compare, or profile_number_reference and profile_number_compare
        # if both, profile_number_compare and log_compare are specified, then the routine raises an error
        if profile_number_compare is not None and log_compare is not None:
            raise ValueError('profile_number_compare and log_compare cannot be specified at the same time.')
        elif profile_number_compare is not None:
            x_values, y_values = self.get_relative_difference_of_two_profiles_from_one_log(x, y, profile_number_reference = profile_number_reference, profile_number_compare = profile_number_compare, log = log_reference, **kwargs)
        elif log_compare is not None:
            x_values, y_values = self.get_relative_difference_of_two_profiles_from_two_logs(x, y, profile_number = profile_number_reference, log_reference = log_reference, log_compare = log_compare, **kwargs)
        else:
            raise ValueError('profile_number_compare and log_compare cannot be both None.')
        
        ax.plot(x_values, y_values, label = y)
        ax.set(xlabel = x, ylabel = 'Relative difference')
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