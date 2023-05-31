import os
import numpy as np
from mesa_inlist_manager.astrophys import *

class Inlist:
    """Class for reading and writing inlist files."""
    
    # keeps track of all instances of Inlist
    instances = []

    def __init__(self, name):
        """Initializes an Inlist object."""
        # add instance to list of instances
        self.instances.append(self)

        # name of the inlist file
        self.name = name

        with open(self.name, 'r') as file:
            self.original_inlist = file.read()

    def __str__(self):
        return f'Inlist({self.name})'

    def read_option(self, option: str):
        """Reads the value of an option in an inlist file."""
        with open(self.name, 'r') as file:
            lines = file.readlines()

            for l in lines:
                if option in l:
                    line_splitted = l.replace('!', '=')  # for ignoring fortran comments after the value
                    line_splitted = line_splitted.split('=')
                    # python formatting
                    out = line_splitted[1].strip()
                    out = Inlist.python_format(out)
            
            # if the option is not found, return None
            try:
                out
            except:
                out = None

        return out
    
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
                        out = Inlist.fortran_format(value)

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
            out = Inlist.fortran_format(value)

            for i, l in enumerate(lines):
                if section in l:

                    index_section = i

                    break

            lines.insert(index_section + 2, f"\t{option} = {out}\n")

        return lines

    # sets options in inlist files

    def set_option(self, section: str, option: str, value):
        
        # conversion such that output is in ''
        if type(value)==str:
            value = f"'{value}'"

        # check if the option is already present. If not, create it
        try:
            lines = self.change_lines(option, value)
        except:
            lines = self.create_lines(section, option, value)

        # write new lines into the inlist
        with open(self.name, 'w') as file:
            file.writelines(lines)

        print(f"Set {option} to {Inlist.fortran_format(value)}")

    def restore_inlist(self):
        with open(self.name, 'w') as file:
            file.write(self.original_inlist)
        print(f"restored {self} to original version")

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
            log_value = f'{logs_parent_directory}/{option}/{v}'

            # check where to save the file
            # e.g., you change inlist_core but the LOGS are saved in inlist_evolve

            if inlist_logs != None:
                Inlist(inlist_logs).set_option('&controls', 'log_directory', log_value)
            else:
                self.set_option('&controls', 'log_directory', log_value)

            self.run_inlist_single_value(section, option, v, run_command)

    def set_logs_path(self, logs_name, logs_parent_directory="LOGS", inlist_logs=None):
        
        logs_path = f'{logs_parent_directory}/{logs_name}'

        if inlist_logs==None:
            self.set_option('&control','log_directory', logs_path)

        else:
            Inlist(inlist_logs).set_option('&control','log_directory', logs_path)

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

    def set_initial_mass_in_M_Jup(self, M_p_in_M_J:float) -> None:
        M_p_in_g = M_Jup_in_g*M_p_in_M_J
        self.set_option('&star_job', 'mass_in_gm_for_create_initial_model', M_p_in_g)

    def set_initial_radius_in_R_Jup(self, R_p_in_R_J:float) -> None:
        R_p_in_cm = R_Jup_in_cm*R_p_in_R_J
        self.set_option('&star_job', 'radius_in_cm_for_create_initial_model', R_p_in_cm)

    def set_initial_entropy_in_kergs(self, M_p : float, s0 : float, **kwargs) -> None:
        """Sets entropy for the inital model
        
        This function computes the inital radius that approximately corresponds to `s0`.
        The resulting value is then set for 'radius_in_cm_for_create_initial_model'.
        """
        R_ini = initial_radius(M_p, s0, **kwargs)
        R_p_in_cm = R_ini * R_Jup_in_cm
        self.set_option('&star_job', 'radius_in_cm_for_create_initial_model', R_p_in_cm)

    def set_convergence_tolerances(self, convergence_tolerances = 'tight', **kwargs)->None:
        """Sets the convergence tolerances of the inlists."""

        tol_correction_norm, tol_max_correction = Inlist.convergence_tolerance_options(convergence_tolerances)
        self.set_option('&controls','tol_correction_norm', tol_correction_norm)
        self.set_option('&controls','tol_max_correction', tol_max_correction)

    @classmethod
    def restore_all_instances(cls):
        for instance in cls.instances:
            instance.restore_inlist()

    @staticmethod
    def fortran_format(x):
        if (type(x) == float) or (type(x) == np.float32) or (type(x) == np.float64):
            log = np.log10(x)
            exponent = int(np.floor(log))
            prefactor = 10**(log-exponent)
            out = f'{prefactor:.6f}d{exponent}'

        else:
            out = str(x)
        return out
    
    @staticmethod
    def python_format(x):
        """Converts a fortran number to a python number"""
        try:
            return int(x)
        except:
            try:
                return float(x.replace('d', 'e'))
            except:
                # check if bool
                if x == ".true.":
                    return True
                elif x == ".false.":
                    return False
                else:
                    return x
                
    @staticmethod
    def convergence_tolerance_options(convergence_tolerances):
        """Returns common convergence tolerances."""
        if convergence_tolerances == "tight":
            tol_correction_norm = 1e-4
            tol_max_correction = 3e-2 

        elif convergence_tolerances == "medium":
            tol_correction_norm = 1e-3
            tol_max_correction = 8e-2

        elif convergence_tolerances == "loose":
            tol_correction_norm = 1e-2
            tol_max_correction = 3e-1
            
        elif convergence_tolerances == "very_loose":
            tol_correction_norm = 5e-2
            tol_max_correction = 5e-1

        return [tol_correction_norm, tol_max_correction]

# class MultipleInlists:

#     def __init__(self) -> None:
#         pass

#     def run_multiple_inlists(self, inlist_dict: dict, run_command='./rn'):

#         # set value to inlist option
#         for inlist in inlist_dict.keys():
#             Inlist(inlist).set_option(
#                 inlist_dict[inlist]['section'],
#                 inlist_dict[inlist]['option'],
#                 inlist_dict[inlist]['value']
#             )

#         # run option
#         os.system(run_command)

#         self.restore_inlist()

#     # def __init__(self, inlist_dict):

#     #     self.inlist_dict = inlist_dict
#     #     # create a dictionary using the Inlist class
#     #     for key, value in self.inlist_dict.items():
#     #         inlist_dict[key] = Inlist(value)

# def run_inlists(set_function, set_function_input, values, run_command='./rn'):

#     for v in values:
#         set_function(set_function_input, v)
#         os.system(run_command)

def create_relax_inital_entropy_file(s_kerg, relax_entropy_filename='relax_entropy_file.dat'):
    s = specific_entropy(s_kerg)
    with open(relax_entropy_filename, 'w') as file:
        file.write('1\n')
        file.write(f'1  {Inlist.fortran_format(s)}')
    print(f"Created entropy profile with s_kerg = {s_kerg}")


