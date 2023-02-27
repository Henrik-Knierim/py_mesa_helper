import os
import numpy as np
from mesa_inlist_manager.astrophys import *


class Inlist:
    def __init__(self, name):
        self.name = name

        with open(self.name, 'r') as file:
            self.original_inlist = file.read()

    def __str__(self):
        return self.name

    # finds existing option and changes it to the new value

    def change_lines(self, option: str, value):

        separator = "="

        with open(self.name, 'r') as file:

            lines = file.readlines()

            for i, l in enumerate(lines):
                if option in l:

                    # test if this is in fact the right option

                    # for ignoring fortran comments after the value
                    line_splitted = l.replace('!', separator)

                    # after split: 0:option, 1: value, 2: comment (if present)
                    line_splitted = line_splitted.split(separator)

                    # true if the occurence exactly matches with option
                    is_option = line_splitted[0].strip() == option
                    if is_option:
                        index_option = i

                        # fortran formatting
                        out = fortran_format(value)

                        new_line = l.replace(
                            line_splitted[1].strip(),
                            out,
                            1  # to change the value only
                        )
                        break

            lines[index_option] = new_line

        return lines

    # create lines with new option

    def create_lines(self, section: str, option: str, value):

        with open(self.name, 'r') as file:

            lines = file.readlines()

            # fortran formatting
            out = fortran_format(value)

            for i, l in enumerate(lines):
                if section in l:

                    index_section = i

                    break

            lines.insert(index_section + 2, f"\t{option} = {out}\n")

        return lines

    # sets options in inlist files

    def set_option(self, section: str, option: str, value):

        # check if the option is already present. If not, create it
        try:
            lines = self.change_lines(option, value)
        except:
            lines = self.create_lines(section, option, value)

        # write new lines into the inlist
        with open(self.name, 'w') as file:
            file.writelines(lines)

        print(f"Set {option} to {fortran_format(value)}")

    def restore_inlist(self):
        with open(self.name, 'w') as file:
            file.write(self.original_inlist)
        print("restored inlist to original version")

    # sets the option and runs the inlist
    def run_inlist_single_value(self, section: str, option: str, value, run_command='./rn'):

        # set the option
        self.set_option(section, option, value)

        # run the inlist
        os.system(run_command)

        print(f"Ran {self.name} with {option} set to {value}.")

        # restore the inlist to its original state
        self.restore_inlist()

    # same as run_inlist_single_paramter but for a list of values
    def run_inlist_multiple_value(self, section: str, option: str, values: list, run_command='./rn', logs_parent_directory="../LOGS", inlist_logs=None):

        for v in values:
            log_value = f"'{logs_parent_directory}/{option}/{v}'"

            # check where to save the file
            # e.g., you change inlist_core but the LOGS are saved in inlist_evolve

            if inlist_logs != None:
                Inlist(inlist_logs).set_option('&controls', 'log_directory', log_value)
            else:
                self.set_option('&controls', 'log_directory', log_value)

            self.run_inlist_single_value(section, option, v, run_command)

    def set_logs_path(self, logs_name, logs_parent_directory="LOGS", inlist_logs=None):
        
        logs_name = f'{logs_parent_directory}/{logs_name}'

        if inlist_logs==None:
            self.set_option('&control','log_directory', logs_name)

        else:
            Inlist(inlist_logs).set_option('&control','log_directory', logs_name)

    def set_logs_path_multiple_values(self, value, value_header_name='value', logs_parent_directory="LOGS"):
        
        # write (append) to 'folder_index'
        path = f'{logs_parent_directory}/folder.index'

        with open(path, 'a') as file:
            lengt_of_file = os.stat(file.name).st_size 
            
            # if file is empty, write header
            if lengt_of_file == 0:
                file.write(f"# id\t{value_header_name}\n")
                lengt_of_file = 1

            file.write(f"{lengt_of_file}\t{value}\n")

    # common inlist tasks

    def set_initial_mass_in_M_Jup(self, M_p_in_M_J:float):
        M_p_in_g = M_Jup_in_g*M_p_in_M_J
        self.set_option('&star_job', 'mass_in_gm_for_create_initial_model', M_p_in_g)

    def set_initial_radius_in_R_Jup(self, R_p_in_R_J:float):
        R_p_in_cm = R_Jup_in_cm*R_p_in_R_J
        self.set_option('&star_job', 'radius_in_cm_for_create_initial_model', R_p_in_cm)

class MultipleInlists:

    def __init__(self) -> None:
        pass

    def run_multiple_inlists(self, inlist_dict: dict, run_command='./rn'):

        # set value to inlist option
        for inlist in inlist_dict.keys():
            Inlist(inlist).set_option(
                inlist_dict[inlist]['section'],
                inlist_dict[inlist]['option'],
                inlist_dict[inlist]['value']
            )

        # run option
        os.system(run_command)

        self.restore_inlist()

    # def __init__(self, inlist_dict):

    #     self.inlist_dict = inlist_dict
    #     # create a dictionary using the Inlist class
    #     for key, value in self.inlist_dict.items():
    #         inlist_dict[key] = Inlist(value)


# inlist methods

# set core mass, planet mass, and metallicity (for opacity)
# opacity option only works if implemented for 'use_other_kap'
def set_mass_and_metallicity(
        inlist_create, inlist_core, inlist_kap,
        Z,      # mass fraction
        M_p,    # in M_J
        use_other_kap=False
):

    # core mass
    inlist_core_instance = Inlist(inlist_core)
    option_core = 'new_core_mass'
    m_core = Z*M_p*M_Jup_in_Sol  # convert to M_Sol
    inlist_core_instance.set_option('&star_job', option_core, m_core)

    # envelope mass
    inlist_create_instance = Inlist(inlist_create)
    option_create = 'mass_in_gm_for_create_initial_model'
    m_env = M_Jup_in_g - m_core * M_Sol_in_g
    inlist_create_instance.set_option('&star_job', option_create, m_env)

    # relax mass option for low mass planets

    # opacity
    if use_other_kap:
        inlist_kap_instance = Inlist(inlist_kap)
        option_kap = 'x_ctrl(1)'
        inlist_kap_instance.set_option('&controls', option_kap, Z)


def run_inlists(set_function, set_function_input, values, run_command='./rn'):

    for v in values:
        set_function(set_function_input, v)
        os.system(run_command)


def fortran_format(x):
    if (type(x) == float) or (type(x) == np.float32) or (type(x) == np.float64):
        log = np.log10(x)
        exponent = int(np.floor(log))
        prefactor = 10**(log-exponent)
        out = f'{prefactor:.6f}d{exponent}'

    else:
        out = str(x)
    return out

# create file for relax_initial_composition
def _create_composition_list(m, M_p, method_input, method, iso_net):
    # get Z or Y values for mass bins in m
    if method == 'grad_Z_lin_M_z':
        [m_1, m_2, M_z, Z_atm] = method_input
        # Z values:
        abu_list = grad_Z_lin_M_z(m, m_1, m_2, M_z, Z_atm)
        def abu_func(Z): return scaled_heavy_mass_abundaces(Z, iso_net)
    elif method == 'grad_Z_lin_Z_1_Z_2':
        [m_1, m_2, Z_1, Z_2] = method_input
        # Z values
        abu_list = grad_Z_lin_Z_1_Z_2(m, m_1, m_2, Z_1, Z_2)
        def abu_func(Z): return scaled_heavy_mass_abundaces(Z, iso_net)
    elif method == 'grad_Z_stepwise':
        [m_transition, Z_inner, Z_outer] = method_input
        # Z values:
        abu_list = grad_Z_stepwise(m, m_transition, Z_inner, Z_outer)
        def abu_func(Z): return scaled_heavy_mass_abundaces(Z, iso_net)
    elif method == 'grad_Z_log':
        [m_1, m_2, log_Z_1, log_Z_2] = method_input
        # Z values
        abu_list = grad_Z_log(m, m_1, m_2, log_Z_1, log_Z_2)
        def abu_func(Z): return scaled_heavy_mass_abundaces(Z, iso_net)
    elif method == 'grad_Y_lin':
        [m_1, m_2, Y_1, Y_2] = method_input
        # Y values:
        abu_list = grad_Y_lin(m, m_1, m_2, Y_1, Y_2)
        def abu_func(Y): return scaled_H_He_mass_abundances(Y, iso_net)
    elif method == 'grad_Y_stepwise':
        [m_transition, Y_inner, Y_outer] = method_input
        # Y values:
        abu_list = grad_Y_stepwise(m, m_transition, Y_inner, Y_outer)
        def abu_func(Y): return scaled_H_He_mass_abundances(Y, iso_net)
    l = []
    for i, mass_bin in enumerate(m):
        # creates list [mass_bin, X_H(mass_bin), ..., X_Mg24(mass_bin)]
        l.append([(M_p-mass_bin)/M_p, *abu_func(abu_list[i]).values()])
    # reverse order for MESA's relax_inital_composition format
    return np.flip(l, 0)

def create_relax_inital_composition_file(m, M_p, method_input, method, iso_net, relax_composition_filename='relax_composition_file.dat'):
    comp_list = _create_composition_list(
        m, M_p, method_input, method, iso_net)
    num_points = len(m)
    # comp_list = [[mass_bin, spec_1, spec_2, ..., spec_N], ...]
    num_species = len(comp_list[1])-1
    with open(relax_composition_filename, 'w') as file:
        file.write(f"{num_points}  {num_species}\n")
        for l in comp_list:
            #str_version = [str(el) for el in l]
            str_version = [f'{el:.16e}' for el in l]
            file.write('  '.join(str_version)+'\n')