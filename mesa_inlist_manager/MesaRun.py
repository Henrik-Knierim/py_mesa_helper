import os
import mesa_reader as mr
from mesa_inlist_manager.Inlist import Inlist
from mesa_inlist_manager.Analysis import Analysis, MultipleSimulationAnalysis
from mesa_inlist_manager.astrophys import specific_entropy
from typing import Callable
import numpy as np

class MesaRun:
    """Utilities for MESA simulations."""

    def __init__(self, src = 'LOGS') -> None:
        
        self.src = src

    # This function should be reworked. It must be more general.
    # The relevant inlist options could be others than mass and entropy.
    def export_setup(self, M_p : float, s0 : float, output_file : str = 'failed_simulations.txt'):
        """Writes setup to file in logs parent directory"""

        with open(self.src +'/'+ output_file, 'a+') as file:
            
            # check if file is empty
            lengt_of_file = os.stat(file.name).st_size 
            
            # if file is empty, write header
            if lengt_of_file == 0:
                file.write(f"# M_p\ts0\n")
                lengt_of_file = 1

            file.write(f"{M_p:.2f}\t{s0:.1f}\n")
    
    def clean_logs(self, M_p, s0)->None:
        """Cleans the logs directory"""
        
        os.system(f'rm {Inlist.create_logs_path_string(logs_src= self.src, M_p = M_p, s0 = s0, logs_style =["M_p", "s0"])}/*')

    def delete_failed_model_dirs(self, failed_simulations_file = 'failed_simulations.txt')->None:
        """Deletes the model directories of failed simulations according to failed_simulations.txt"""
            
        # read failed_simulations.txt
        try:
            with open(self.src+ '/' + failed_simulations_file, 'r') as file:
                failed_sims = file.readlines()
                # delete model directories
                # [1:] because first line is header
                for line in failed_sims[1:]:
                    M_p = float(line.split('\t')[0])
                    s0 = float(line.split('\t')[1])
                    logs_path = f'{self.src}/M_p_{M_p:.2f}_s0_{s0:.1f}'
                    print("deleting: ", logs_path)
                    os.system(f'rm -r {logs_path}')

        except FileNotFoundError:
            print(f"No file {failed_simulations_file} found.")

    def delete_failed_simulations_file(self)->None:
        """Deletes the file containing the failed simulations, typically failed_simulations.txt"""
        try:
            os.system(f'rm {self.src}/failed_simulations.txt')
        except FileNotFoundError:
            print("No failed_simulations.txt found.")

    # convergence tests
    def ageQ(self, age_tolerance = 1e-3, **kwargs)->bool:
        """Returns True if the simulation has reached the desired age."""
        
        # get the age of the model(s)
        suiteQ = kwargs.get('suiteQ', False)
        if suiteQ:
            age = MultipleSimulationAnalysis(src=self.src, **kwargs).get_history_data('star_age')
        else:
            age  = Analysis(src=self.src).get_history_data('star_age')
        
        # test if age is within tolerance
        # if final_age_method is not specified, use inlist
        method = kwargs.get('final_age_method', 'inlist')

        if method == 'inlist':
            inlist_name = kwargs.get('inlist_name', 'inlist_evolve')
            with Inlist(inlist_name) as inlist:
                t_final = inlist.read_option("max_age")

        # if final_age is specified, use that
        elif method == 'input':
            # if final_age is not specified, return False
            t_final = kwargs.get('final_age', 0)
        
        test = (t_final*(1-age_tolerance) < age) &  (age < t_final*(1+age_tolerance))
        return test
    
    def heavy_mass_convergedQ(self, M_p, s0, heavy_mass_tol = 1e-2, **kwargs)->bool:
        """Checks if the heavy mass of the planet has converged"""
        # lazy import to avoid circular import
        from mesa_inlist_manager.Analysis import Analysis
        return Analysis(self.src).heavy_mass_error(M_p, s0, **kwargs) < heavy_mass_tol
    
    @staticmethod
    def relaxedQ(mod_file = 'planet_relax_composition.mod'):
        """Returns True if the mod_file exits."""
        return os.path.isfile(mod_file)
    
    @staticmethod
    def create_relax_entropy_file_homogeneous(s_kerg, relax_entropy_filename : str ='relax_entropy_file.dat'):
        """Creates a relax entropy file for s(m) = s_kerg."""
        s = specific_entropy(s_kerg)
        with open(relax_entropy_filename, 'w') as file:
            file.write('1\n')
            file.write(f'1  {Inlist.fortran_format(s)}')
        print(f"Created entropy profile with s_kerg = {s_kerg}")
    
    @staticmethod
    def _create_relax_entropy_list(s_of_m_kerg : Callable, n_points : int = 1000) -> np.ndarray:
        """Creates a list of mass and entropy values for the relax entropy file."""

        out = np.zeros((n_points, 2))
        mass_bins = np.linspace(0, 1, n_points)
        for i,m in enumerate(mass_bins):
            out[i,0] = 1-m # mass fraction q starting from M_p
            out[i,1] = specific_entropy(s_of_m_kerg(m))

        # test wether any entropy values are negative
        if np.any(out[:,1] < 0):
            raise ValueError("Entropy values must be positive.")
        
        # flip array to have increasing mass
        return np.flip(out, axis=0)
    
    @staticmethod
    def create_relax_entropy_file(s_of_m_kerg : Callable, relax_entropy_filename : str='relax_entropy_file.dat', n_points : int = 1000) -> None:
        """Creates a relax entropy file for s(m) = s_of_m(m).
        
        Parameters
        ----------
        s_of_m_kerg : Callable
            A function that returns the entropy as a function of mass in kergs. The function is expected take m/M_p as an argument, i.e., relative in mass.
        relax_entropy_filename : str, optional
            The name of the relax entropy file. The default is 'relax_entropy_file.dat'.
        n_points : int, optional
            The number of points to use in the entropy profile. The default is 1000.
        """

        # tests
        if not callable(s_of_m_kerg):
            raise TypeError("s_of_m_kerg must be a function.")
        if n_points < 1:
            raise ValueError("n_points must be at least 1.")
        
        # create list of mass and entropy values
        s_list = MesaRun._create_relax_entropy_list(s_of_m_kerg, n_points)

        with open(relax_entropy_filename, 'w') as file:
            # write header: number of points
            file.write(f"{n_points}\n")
            for l in s_list:
                str_version = [f'{el:.16e}' for el in l]
                file.write('  '.join(str_version)+'\n')

        print(f'{relax_entropy_filename} was created successfully.')

    @staticmethod
    def create_inital_model(M_p:float, s0 : float = None, R_ini : float = None,
                            initial_model_method : str = 'entropy', inlist_name : str = 'inlist_create',
                            run_exe: str = 'rn_create')->None:
        """Creates the initial model for the planet.
        
        This function creates an initial model for the planet based on the specified parameters.

        Parameters:
            M_p (float): Mass of the planet in Jupiter masses.
            s0 (float, optional): Initial entropy in k_b/bary. Required if initial_model_method is 'entropy'.
            R_ini (float, optional): Initial radius of the planet in Jupiter radii. Required if initial_model_method is 'radius'.
            initial_model_method (str, optional): Method to determine the initial model.
                - 'entropy': The initial entropy value (s0) is used to determine the model.
                - 'radius': The initial radius (R_ini) is used to determine the model.
                (default: 'entropy')
            inlist_name (str, optional): Name of the inlist file to be modified. (default: 'inlist_create')
            run_exe (str, optional): Name of the executable to run the model. (default: 'rn_create')

        Raises:
            ValueError: If s0 is not specified and initial_model_method is 'entropy',
                        or if R_ini is not specified and initial_model_method is 'radius'.

        Returns:
            None

        Notes:
            - This function provides a convenient way to create an initial model for the planet.
            - The inlist file specified by inlist_name is used for storing the initial model configuration.
            - The created model is executed using the run_exe executable.

        Example usage:
            create_initial_model(1.0, s0=10)
        """

        if initial_model_method == 'entropy' and s0 is None:
            raise ValueError("s0 must be specified if initial_parameter == 'entropy'")
        elif initial_model_method == 'radius' and R_ini is None:
            raise ValueError("R_ini must be specified if initial_parameter == 'radius'")


        with Inlist(inlist_name) as inlist:

            if initial_model_method == 'entropy':
                inlist.set_initial_mass_in_M_Jup(M_p)
                inlist.set_initial_entropy_in_kergs(M_p, s0)

                inlist.set_option('center_entropy_lower_limit', s0)
                inlist.set_option('entropy_1st_try_for_create_initial_model', s0)

            elif initial_model_method == 'radius':
                inlist.set_initial_mass_in_M_Jup(M_p)
                inlist.set_initial_radius_in_R_Jup(R_ini)
                
            os.system('./'+run_exe)

    @staticmethod
    def run_inlist_single_value(inlist_name: str, option: str, value, run_exe: str = "rn") -> None:
        """
        Run a MESA inlist with a single parameter changed.

        Parameters:
            inlist_name (str): The name of the MESA inlist file.
            option (str): The name of the parameter to be changed in the inlist.
            value: The new value to be assigned to the specified parameter.
            run_exe (str, optional): The name of the executable file to run the MESA inlist.
                Default is "rn".

        Returns:
            None

        Raises:
            None

        Usage:
            Run a MESA inlist file with a single parameter changed. The function opens the inlist file specified by
            'inlist_name', changes the value of the parameter identified by 'option' to the provided 'value',
            runs the MESA inlist using the executable specified by 'run_exe', and prints a confirmation message
            indicating the inlist name, the changed option, and the new value. After the inlist is run, 
            it is restored to its original state.
        """
        with Inlist(inlist_name) as inlist:
            inlist.set_option(option, value)
            os.system('./' + run_exe)
            print(f"Ran {inlist.name} with {option} set to {Inlist.fortran_format(value)}.")


        # @staticmethod
        # # same as run_inlist_single_paramter but for a list of values
        # def run_inlist_multiple_value(option: str, values: list, run_command='./rn', logs_parent_directory="../LOGS", inlist_logs=None):

        #     for v in values:
        #         log_value = f'{logs_parent_directory}/{option}/{v}'

        #         # check where to save the file
        #         # e.g., you change inlist_core but the LOGS are saved in inlist_evolve

        #         if inlist_logs != None:
        #             Inlist(inlist_logs).set_option('log_directory', log_value)
        #         else:
        #             self.set_option('log_directory', log_value)

        #         self.run_inlist_single_value(option, v, run_command)
    
            
        
            

        