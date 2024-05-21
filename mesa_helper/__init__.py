from mesa_helper.astrophys import *
from mesa_helper.Inlist import *
from mesa_helper.CompositionalGradient import *
from mesa_helper.rn import rn
from mesa_helper.Simulation import Simulation


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