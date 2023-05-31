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
            'overshoot_f0_below_nonburn_shell'
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
    
    def export_mixing_parameters(self):
        """Writes mixing parameters to file in logs parent directory"""
        
        # read mixing parameters if not already done
        if 'self.mixing_params' not in locals(): self.read_mixing_params()

        with open(self.src + '/setup.txt', 'w') as file:
            for key, value in self.mixing_params.items():
                file.write(f"{key}\t{value}\n")

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

    def create_logs_path_string(self, M_p: float, s0: float) -> str:
        """Creates the LOGS path string."""
        return f'{self.src}/M_p_{M_p:.2f}_s0_{s0:.1f}'
    
    def clean_logs(self, M_p, s0)->None:
        """Cleans the logs directory"""
        
        os.system(f'rm {self.create_logs_path_string(M_p, s0)}/*')

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

    # convergence tests
    def ageQ(self, M_p, s0, inlist_name : str = 'inlist_evolve', age_tolerance = 1e-3)->bool:
        """Returns True if the simulation has reached the desired age."""
        
        # get the age of the model
        logs_path = self.create_logs_path_string(M_p, s0)
        history = mr.MesaData(logs_path+'/history.data')
        age = history.data('star_age')[-1]

        t_final = Inlist(inlist_name).read_option("max_age")

        return t_final*(1-age_tolerance) < age < t_final*(1+age_tolerance)
    
    def heavy_mass_convergedQ(self, M_p, s0, heavy_mass_tol = 1e-2, **kwargs)->bool:
        """Checks if the heavy mass of the planet has converged"""
        # lazy import to avoid circular import
        from mesa_inlist_manager.Analysis import Analysis
        return Analysis().heavy_mass_error(M_p, s0, **kwargs) < heavy_mass_tol
    
    @staticmethod
    def relaxedQ(mod_file = 'planet_relax_composition.mod'):
        """Returns True if the mod_file exits."""
        return os.path.isfile(mod_file)
    
    @staticmethod
    def create_relax_inital_entropy_file(s_kerg, relax_entropy_filename='relax_entropy_file.dat'):
        s = specific_entropy(s_kerg)
        with open(relax_entropy_filename, 'w') as file:
            file.write('1\n')
            file.write(f'1  {Inlist.fortran_format(s)}')
        print(f"Created entropy profile with s_kerg = {s_kerg}")
            

        