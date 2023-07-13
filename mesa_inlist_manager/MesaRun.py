import os
import mesa_reader as mr
from mesa_inlist_manager.Inlist import Inlist
from mesa_inlist_manager.astrophys import specific_entropy

class MesaRun:
    """Utilities for MESA simulations."""

    MIXING_PARAMTERS = [
            'mix_factor',
            'alpha_semiconvection',
            'thermohaline_coeff',
            'num_cells_for_smooth_gradL_composition_term',
            'conv_dP_term_factor',
            'overshoot_alpha', # only used when > 0.  if <= 0, then use `mixing_length_alpha` instead.
            'smooth_convective_bdy',
            # exp overshooting
            'overshoot_f_above_nonburn_core',
            'overshoot_f0_above_nonburn_core',
            'overshoot_f_above_nonburn_shell',    # 1d-2
            'overshoot_f0_above_nonburn_shell',   # 2d-3
            'overshoot_f_below_nonburn_shell',
            'overshoot_f0_below_nonburn_shell',
            'column_depth_for_irradiation',
            'irradiation_flux',
            'max_years_for_timestep',
            'use_Ledoux_criterion',
            'mesh_delta_coeff',
            'initial_age',
            'kappa_file_prefix',
            'kappa_lowT_prefix',
            'use_lnS_for_eps_grav',
            'include_dmu_dt_in_eps_grav',
            'use_dEdRho_form_for_eps_grav',
            'eosDT_use_linear_interp_for_X'
        ]
    
    EVOLUTION_OPTIONS = [
        'tol_correction_norm',
        'tol_max_correction',
        'max_years_for_timestep',
        'mesh_delta_coeff',
        'mesh_logX_species(1)',
        'mesh_logX_min_for_extra(1)',
        'mesh_dlogX_dlogP_extra(1)',
        'mesh_dlogX_dlogP_full_on(1)',
        'mesh_dlogX_dlogP_full_off(1)',
        'column_depth_for_irradiation',
        'irradiation_flux'
        ]

    def __init__(self, src = 'LOGS') -> None:
        
        self.src = src

    def read_mixing_params(self, inlist_name : str = 'inlist_evolve')->None:
        """Sets mixing parameters in inlists"""

        # read in mixing parameters from inlist and store in dict
        self.mixing_params = {}
        for option in self.MIXING_PARAMTERS:
            value = Inlist(inlist_name).read_option(option)
            self.mixing_params[option] = value
            # release Inlist object
            Inlist.delete_latest_instance()
    
    def export_mixing_parameters(self):
        """Writes mixing parameters to file in logs parent directory"""
        
        # read mixing parameters if not already done
        if 'self.mixing_params' not in locals(): self.read_mixing_params()

        with open(self.src + '/setup.txt', 'w') as file:
            for key, value in self.mixing_params.items():
                file.write(f"{key}\t{value}\n")
        print("Mixing parameters exported to setup.txt")

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

    def export_evolution_options(self, M_p : float, s0 : float, inlist_name : str = 'inlist_evolve')->None:
        """Writes evolution options to evolution_parameters.txt inside the LOGS directory."""
        
        path = Inlist.create_logs_path_string(logs_src= self.src, M_p = M_p, s0 = s0, logs_style =['M_p', 's0'])
        file = os.path.join(path, 'evolution_parameters.txt')
        with open(file, 'w') as file:
            for option in self.EVOLUTION_OPTIONS:
                value = Inlist(inlist_name).read_option(option)
                file.write(f"{option}\t{value}\n")
                # release Inlist object
                Inlist.delete_latest_instance()
                
        print(f"Evolution options of {path} exported to evolution_parameters.txt")
    
    def clean_logs(self, M_p, s0)->None:
        """Cleans the logs directory"""
        
        os.system(f'rm {Inlist.create_logs_path_string(logs_src= self.src, M_p = M_p, s0 = s0, logs_style =["M_p", "s0"])}/*')

    def delete_failed_model_dirs(self)->None:
        """Deletes the model directories of failed simulations according to failed_simulations.txt"""
            
        # read failed_simulations.txt
        try:
            with open(self.src+ '/' + 'failed_simulations.txt', 'r') as file:
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
            print("No failed_simulations.txt found.")

    def delete_failed_simulations_file(self)->None:
        """Deletes the file containing the failed simulations, typically failed_simulations.txt"""
        try:
            os.system(f'rm {self.src}/failed_simulations.txt')
        except FileNotFoundError:
            print("No failed_simulations.txt found.")

    # convergence tests
    def ageQ(self, M_p, s0, inlist_name : str = 'inlist_evolve', age_tolerance = 1e-3)->bool:
        """Returns True if the simulation has reached the desired age."""
        
        # get the age of the model
        logs_path = Inlist.create_logs_path_string(logs_src= self.src, M_p = M_p, s0 = s0, logs_style =['M_p', 's0'])
        history = mr.MesaData(logs_path+'/history.data')
        age = history.data('star_age')[-1]

        t_final = Inlist(inlist_name).read_option("max_age")

        # release Inlist object
        Inlist.delete_latest_instance()

        return t_final*(1-age_tolerance) < age < t_final*(1+age_tolerance)
    
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
    def create_relax_initial_entropy_file(s_kerg, relax_entropy_filename='relax_entropy_file.dat'):
        s = specific_entropy(s_kerg)
        with open(relax_entropy_filename, 'w') as file:
            file.write('1\n')
            file.write(f'1  {Inlist.fortran_format(s)}')
        print(f"Created entropy profile with s_kerg = {s_kerg}")

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
    
            
        
            

        