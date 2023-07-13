# class for analyzing the output of a MESA run
import numpy as np
import mesa_reader as mr
import os
import glob
from tabulate import tabulate
from mesa_inlist_manager.astrophys import M_Earth_in_g, M_Earth_in_Sol, M_Jup_in_Sol
from mesa_inlist_manager.CompositionalGradient import CompositionalGradient


class Analysis:
    def __init__(self, src="LOGS") -> None:
        # location of the LOGS directory
        self.src = src

    def heterogeneity(self, **kwargs):
        """Gives the deviation of the metallicity from a homogeneous profile. If the gradient is in Y, the deviation is calculated in Y."""

        # init log object
        log = mr.MesaLogDir(self.src)
        # gets the last profile
        profile_data_kwargs = {key: value for key, value in kwargs.items() if key in log.profile_data.__code__.co_varnames}
        profile = log.profile_data(**profile_data_kwargs)

        # get the metallicity profile
        dm = profile.dm
        Z = profile.data(CompositionalGradient.data_string_by_method(kwargs.get("method")))
        Z_avg = np.average(Z, weights=dm)

        # calculate the deviation
        variance = np.sum(np.power(Z - Z_avg, 2) * dm) / np.sum(dm)

        return np.sqrt(variance)

    def heterogeneity_evolution(self, **kwargs):
        """Gives the evolution of the heterogeneity."""

        # init log object
        log = mr.MesaLogDir(self.src)
        # get the model numbers
        model_numbers = log.model_numbers
        model_numbers = np.trim_zeros(model_numbers, "f")

        t = np.array(
            [
                log.history.data_at_model_number("star_age", m_num=model_number)
                for model_number in model_numbers
            ]
        )

        # get the heterogeneity
        heterogeneity = np.array(
            [
                self.heterogeneity(model_number=model_number, **kwargs)
                for model_number in model_numbers
            ]
        )

        return t, heterogeneity

    def relative_atmospheric_metallicity(self,**kwargs):
        """Gives the ratio between the atmopsheric metallicity (Z_atm) and the average metallicity (Z_avg)."""

        # init log object
        log = mr.MesaLogDir(self.src)

        # get the last profile
        profile = log.profile_data()

        # get the average metallicity
        history_string = CompositionalGradient.data_string_by_method(kwargs.get("method", "Z"), 'average')
        Z_avg = log.history.data("history_string")[-1]

        # select the metallicty where convection starts
        Z_atm = profile.z[profile.sch_stable == 0][0]

        # calculate the relative atmospheric metallicity
        return Z_atm / Z_avg

    def dlogZ_dlogP(self, **kwargs):
        """Gives the gradient of the metallicity profile. If the gradient is in Y, the gradient is calculated in logY."""

        # init log object
        log = mr.MesaLogDir(self.src)

        # get profile
        profile_data_kwargs = {key: value for key, value in kwargs.items() if key in log.profile_data.__code__.co_varnames}

        profile = log.profile_data(**profile_data_kwargs)

        # get the pressure and metallicity profiles
        logP = profile.logP
        y_coord = profile.data(CompositionalGradient.data_string_by_method(kwargs.get("method", "Z")))

        # careful: might be logY if gradient is in Y
        logZ = np.log10(y_coord)

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

    def heavy_mass_error(self, M_p, s0, **kwargs) -> float:
        """Returns the error in the heavy mass of the planet"""

        # init log object
        from mesa_inlist_manager.Inlist import Inlist

        path = Inlist.create_logs_path_string(logs_src= self.src, M_p = M_p, s0 = s0, logs_style =['M_p', 's0'])
        log = mr.MesaLogDir(path)

        # get the heavy mass of the planet theoretically
        comp_grad = CompositionalGradient(M_p=M_p, **kwargs)
        M_z_in = comp_grad.M_z_tot(**kwargs)

        history_string = CompositionalGradient.data_string_by_method(kwargs.get("method"), 'total_mass')
        # get the heavy mass of the planet at the end of the simulation
        
        M_z_out = log.history.data(history_string)[-1] / M_Earth_in_Sol

        return abs(1 - M_z_out / M_z_in)

    def get_history_data(self, data_name, model_number=-1):
        """Returns the data from the history.data file."""

        if model_number == -1:
            get_data = lambda history, data_name: history.data(data_name)[-1]
        else:
            get_data = lambda history, data_name: history.data_at_model_number(
                data_name, m_num=model_number
            )

        history_file = mr.MesaData(os.path.join(self.src, "history.data"))

        # get the data for the i-th simulation
        data = get_data(history_file, data_name)

        return data
    
    def get_profile_data(self, data_name : str, **kwargs) -> np.ndarray:
        """Returns the data from the profile corresponding the profile number."""

        # init log object
        log = mr.MesaLogDir(log_path=self.src)

        # get profile
        profile = log.profile_data(**kwargs)
        
        data = profile.data(data_name)

        return data

    def get_evolution_parameters(self, keys: list) -> dict:
        """Gets the specified paramters from the evolution_parameters.txt file."""
        from mesa_inlist_manager.Inlist import Inlist

        evolution_params = {}
        with open(os.path.join(self.src, "evolution_parameters.txt"), "r") as file:
            for line in file:
                key, value = line.strip().split("\t")
                if key in keys:
                    evolution_params[key] = Inlist.python_format(value)

        return evolution_params


class MultipleSimulationAnalysis:
    """Class for analyzing multiple simulations."""

    def __init__(self, src="LOGS", free_params = ['M_p','s_0'], **kwargs) -> None:
        # parent directory of the LOGS directories
        self.src = src

        # get the names of the LOGS directories
        if free_params == ['M_p','s_0']:
            self._select_simulations(**kwargs)

        elif free_params == ['m_core']:
            self._select_simulations_custom_pattern(
                pattern = "m_core_*",
                sort_key = lambda x: float(x.split("_")[-1])
                )
            
            self.core_mass = np.array([float(s.split("_")[-1]) for s in self.simulations])
            

        # get the planetary masses and initial entropies
        self._get_planetary_mass_and_entropy()

    def _select_simulations_custom_pattern(self, pattern, sort_key=None):
        """Select subset of simulations from the parent directory."""

        # get the names of the LOGS directories
        self.simulations = glob.glob(self.src + "/" + pattern)
        
        if sort_key != None:
            self.simulations.sort(key=sort_key)

    def _select_simulations(self, M_p=None, s0=None, **kwargs):
        """Select subset of simulations from the parent directory."""

        # get the names of the LOGS directories
        if M_p == None and s0 == None:
            self.simulations = glob.glob(self.src + "/" + "M_p_*_s0_*")
        elif M_p == None:
            self.simulations = glob.glob(self.src + "/" + "M_p_*_s0_{:.1f}".format(s0))
        elif s0 == None:
            self.simulations = glob.glob(self.src + "/" + "M_p_{:.2f}_s0_*".format(M_p))
        else:
            self.simulations = glob.glob(
                self.src + "/" + "M_p_{:.2f}_s0_{:.1f}".format(M_p, s0)
            )

        # sort the simulations by mass
        self.simulations.sort(key=lambda x: float(x.split("_")[-3]))

    def _get_planetary_mass_and_entropy(self):
        """Gives the masses and entropies of the planets."""

        # masses
        self.planetary_mass = self.get_history_data("star_mass")/M_Jup_in_Sol

        # entropies
        self.total_entropy = self.get_profile_data("entropy", metric=np.sum)
        self.center_entropy = self.get_profile_data("entropy", metric=lambda l: l[-1])

        # check if the initial entropy is in the simulation name
        if 's0' in self.simulations[0].split('/')[-1]:
            self.initil_entropy = np.zeros(len(self.simulations))

            for i, simulation in enumerate(self.simulations):
                parts = simulation.split("_")
                self.initial_entropy[i] = float(parts[-1])

                    
    def get_history_data(self, data_name, model_number=-1):
        """Returns the n-th entry of the history data for each simulation."""

        analysis = Analysis()

        # init data that will be returned
        data = np.zeros(len(self.simulations))
        for i, simulation in enumerate(self.simulations):
            analysis.src = simulation
            data[i] = analysis.get_history_data(data_name, model_number=model_number)

        return data
    
    def get_profile_data(self, data_name : str, metric, **kwargs) -> np.ndarray:
        """Gives the profile entry of data_name that fulfills the metric for each simulation."""
            
        analysis = Analysis()
        
        # init data that will be returned
        data = []
        for simulation in self.simulations:
            analysis.src = simulation

            # get the data and apply the metric
            data.append(metric(analysis.get_profile_data(data_name, **kwargs)))
    
        return np.stack(data)

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
            relative_atmospheric_metallicity[
                i
            ] = analysis.relative_atmospheric_metallicity()

        return relative_atmospheric_metallicity

    def get_evolution_parameters(self, keys: list):
        """Gives the specified parameters in evolution_parameters.txt for each simulation."""

        # init analysis object
        analysis = Analysis()

        # get the evolution parameters for each simulation
        evolution_parameters = {"M_p": self.planetary_mass, "s0": self.initial_entropy}

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
            rel_M_z_err[i] = Analysis(self.src).heavy_mass_error(
                M_p=self.planetary_mass[i], s0=self.initial_entropy[i], **kwargs
            )

        print(
            tabulate(
                np.array(
                    [
                        self.planetary_mass,
                        self.initial_entropy,
                        age,
                        Z_atm,
                        Z_atm_Z_avg_ratio,
                        heterogeneity,
                        rel_M_z_err,
                    ]
                ).T,
                headers=[
                    "M_p",
                    "s0",
                    "age",
                    "Z_atm",
                    "Z_atm/Z_avg",
                    "heterogeneity",
                    "rel_M_z_err",
                ],
                floatfmt=(".2f", ".1f", ".2e", ".4f", ".4f", ".4f", ".4f"),
            )
        )
