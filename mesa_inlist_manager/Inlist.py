import os
import re
import numpy as np
import shutil
from mesa_inlist_manager.astrophys import *
from typing import Callable


class Inlist:
    """Class for reading and writing inlist files."""

    # keeps track of all instances of Inlist
    instances: list = []

    def __init__(self, name: str, version: str = "23.05.1") -> None:
        """Initializes an Inlist object."""
        # add instance to list of instances
        self.instances.append(self)

        # name of the inlist file
        self.name: str = name

        # version of MESA
        self.version: str = version

        # backup original inlist
        with open(self.name, "r") as file:
            self.original_inlist: str = file.read()

    def __str__(self):
        return f"Inlist({self.name})"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):

        # restore original inlist
        self.restore_inlist()

        # delete instance from list of instances
        self.instances.remove(self)

    ### changin options ###

    # checks section of the option

    def _optionQ(self, section: str, option: str) -> bool:
        """Checks if the option is in the given section."""
        src: str = path.join(resources_dir, self.version, f"{section}.defaults")
        with open(src, "r") as file:
            for line in file:
                # remove whitespace
                line = line.strip()
                # check if line starts with option
                if line.startswith(option):
                    return True
        return False

    @staticmethod
    def _is_x_ctrl(string: str) -> bool:
        """Check if a string is in the form 'x_ctrl(i)', where i is an integer between 1 and 99."""
        pattern: str = r"^x_ctrl\((?:[1-9]|[1-9][0-9])\)$"
        return bool(re.match(pattern, string))

    @staticmethod
    def _is_x_logical_ctrl(string: str) -> bool:
        """Check if a string is in the form 'x_logical_ctrl(i)', where i is an integer between 1 and 99."""
        pattern: str = r"^x_logical_ctrl\((?:[1-9]|[1-9][0-9])\)$"
        return bool(re.match(pattern, string))

    def _get_section_of_option(self, option: str) -> str:
        """Returns the section of an option."""
        if self._optionQ("controls", option):
            return "&controls"
        elif self._optionQ("star_job", option):
            return "&star_job"
        elif self._optionQ("pgstar", option):
            return "&pgstar"
        elif Inlist._is_x_ctrl(
            option
        ):  # x_ctrl(i) is a special case not covered by the defaults files
            return "&controls"
        elif Inlist._is_x_logical_ctrl(
            option
        ):  # x_logical_ctrl(i) is a special case not covered by the defaults files
            return "&controls"
        else:
            raise ValueError(f"Option {option} not found in MESA {self.version}.")

    def read_option(self, option: str) -> float | int | str | bool | None:
        """Reads the value of an option in an inlist file."""
        with open(self.name, "r") as file:
            lines: list[str] = file.readlines()

            for l in lines:
                # pick out the line with the option
                if l.strip().startswith(option):

                    line_replaced: str = l.replace(
                        "!", "="
                    )  # for ignoring fortran comments after the value

                    line_splitted: list[str] = line_replaced.split("=")

                    # python formatting
                    out: float | int | str | bool = line_splitted[1].strip()

                    out: float | int | str | bool = Inlist.python_format(out)
                    return out

        return None

    # finds existing option and changes it to the new value

    def change_lines(self, option: str, value) -> list[str]:

        separator: str = "="

        with open(self.name, "r") as file:

            lines: list[str] = file.readlines()

            for i, l in enumerate(lines):
                if option in l:

                    # test if this is in fact the right option

                    # for ignoring fortran comments after the value
                    line_replace: str = l.replace("!", separator)

                    # after split: 0:option, 1: value, 2: comment (if present)
                    line_splitted: list[str] = line_replace.split(separator)

                    # true if the occurence exactly matches with option
                    is_option: bool = line_splitted[0].strip() == option
                    if is_option:
                        index_option: int = i

                        # fortran formatting
                        out: str = Inlist.fortran_format(value)

                        new_line: str = line_splitted[0] + separator + out + "\n"

                        break

            lines[index_option] = new_line

        return lines

    # create lines with new option
    def _create_lines_explicit_section(
        self, section: str, option: str, value
    ) -> list[str]:
        """Creates lines for a new option in an inlist file, where the section is explicitly given."""
        with open(self.name, "r") as file:

            lines: list[str] = file.readlines()

            # fortran formatting
            out: str = Inlist.fortran_format(value)

            for i, l in enumerate(lines):
                if section in l:

                    index_section: int = i

                    break

            lines.insert(index_section + 2, f"\t{option} = {out}\n")

        return lines

    def create_lines(self, option: str, value) -> list[str]:
        """Creates lines for a new option in an inlist file."""

        section: str = self._get_section_of_option(option)
        return self._create_lines_explicit_section(section, option, value)

    # sets options in inlist files

    def set_option(self, option: str, value) -> None:

        # conversion such that output is in ''
        if type(value) == str:
            value = f"'{value}'"

        # check if the option is already present. If not, create it
        try:
            lines = self.change_lines(option, value)
        except:
            lines = self.create_lines(option, value)

        # write new lines into the inlist
        with open(self.name, "w") as file:
            file.writelines(lines)

        print(f"Set {option} to {Inlist.fortran_format(value)}")

    def restore_inlist(self) -> None:
        """Restores the inlist to its original version."""
        with open(self.name, "w") as file:
            file.write(self.original_inlist)
        print(f"restored {self} to original version")

    def save_inlist(self) -> None:
        """Copies the inlist into the LOGS directory of the run."""
        try:
            log_directory: str = os.path.normpath(self.read_option("log_directory"))
            shutil.copy(self.name, log_directory)
            print(f"saved {self.name} to {self.read_option('log_directory')} directory")
        except FileNotFoundError:
            raise ValueError(f"The file {self.name} does not exist.")
        except PermissionError:
            raise ValueError(
                f"Permission denied to copy {self.name} to {log_directory}."
            )
        except Exception as e:
            raise ValueError(f"An unexpected error occurred: {str(e)}")

    # common inlist tasks

    def set_logs_path(self, **kwargs) -> None:
        """Sets the path for the logs directory.

        Parameters:
            All parameters are inherited from Inlist.create_logs_path_string()

        Returns:
            None

        Example usage:
            set_logs_path(logs_style = 'm_core', m_core = 25.0)
        """
        logs_path = Inlist.create_logs_path_string(**kwargs)
        self.set_option("log_directory", logs_path)

    def set_initial_mass_in_M_Jup(self, M_p_in_M_J: float) -> None:
        M_p_in_g: float = M_Jup_in_g * M_p_in_M_J
        self.set_option("mass_in_gm_for_create_initial_model", M_p_in_g)

    def set_initial_radius_in_R_Jup(self, R_p_in_R_J: float) -> None:
        R_p_in_cm: float = R_Jup_in_cm * R_p_in_R_J
        self.set_option("radius_in_cm_for_create_initial_model", R_p_in_cm)

    def set_initial_entropy_in_kergs(self, M_p: float, s0: float, **kwargs) -> None:
        """Sets entropy for the inital model

        This function computes the inital radius that approximately corresponds to `s0`.
        The resulting value is then set for 'radius_in_cm_for_create_initial_model'.
        """
        R_ini: float = initial_radius(M_p, s0, **kwargs)
        R_p_in_cm: float = R_ini * R_Jup_in_cm
        self.set_option("radius_in_cm_for_create_initial_model", R_p_in_cm)

    def set_convergence_tolerances(
        self, convergence_tolerances="very_tight", **kwargs
    ) -> None:
        """Sets the convergence tolerances of the inlists."""

        tol_correction_norm, tol_max_correction = Inlist.convergence_tolerance_options(
            convergence_tolerances
        )
        self.set_option("tol_correction_norm", tol_correction_norm)
        self.set_option("tol_max_correction", tol_max_correction)

    @classmethod
    def restore_all_instances(cls) -> None:
        for instance in cls.instances:
            instance.restore_inlist()

        Inlist.delete_all_instances()

    # delete all instances of Inlist
    @classmethod
    def delete_all_instances(cls) -> None:
        """Deletes all instances of Inlist."""
        cls.instances = []

    @classmethod
    def delete_latest_instance(cls) -> None:
        """Deletes the last instance of Inlist."""
        if cls.instances:
            cls.instances.pop()

    @staticmethod
    def fortran_format(x: float | bool | str | np.floating) -> str:
        """Converts a python type to a fortran type"""
        if isinstance(x, (float, np.floating)):

            # Check if the number is zero float
            if x == 0.0:
                return "0d0"

            log: np.floating = np.log10(x)
            exponent: int = int(np.floor(log))
            prefactor = 10 ** (log - exponent)

            # Check if the prefactor is an integer or has trailing zeros
            if prefactor.is_integer():
                prefactor = int(prefactor)

            out = f"{prefactor:.6g}d{exponent}"  # Use the 'g' format specifier

        elif isinstance(x, bool):
            if x:
                out = ".true."
            else:
                out = ".false."

        else:
            out = str(x)

        return out

    @staticmethod
    def python_format(x) -> float | int | str | bool:
        """Converts a fortran number to a python number"""
        try:
            return int(x)
        except:
            try:
                return float(x.replace("d", "e"))
            except:
                # check if bool
                if x == ".true.":
                    return True
                elif x == ".false.":
                    return False
                else:
                    # check if string and remove quotes
                    if x[0] == "'" and x[-1] == "'":
                        return x[1:-1]
                    else:
                        return x

    # define subroutines for creating paths to logs directories
    @staticmethod
    def _from_style_to_string(style, value) -> str:
        """Converts a style and a value to a string."""
        if isinstance(value, str):
            return f"{style}_{value}"
        else:
            return f"{style}_{value:.2f}"

    @staticmethod
    def _from_styles_to_string(styles, **kwargs) -> str:
        """Converts a list of styles and a list of values to a string."""
        out = ""
        for style in styles:
            style_value = kwargs.get(style, None)
            if style_value is None:
                raise ValueError(f"{style} must be given if logs_style is {styles}")

            out += (
                Inlist._from_style_to_string(style, style_value) + "_"
            )  # add underscore to separate styles

        return out[:-1]

    @staticmethod
    def create_logs_path_string(
        logs_src: str = "LOGS",
        suite_dir="",
        suite_style=None,
        logs_style=None,
        **kwargs,
    ) -> str:
        """
        Returns the path to the logs directory.

        Parameters:
            logs_src (str): The parent directory for logs. Default is "LOGS".
            suite_dir (str): The parent directory for the suite. Default is "".
            logs_style: Style of logs. Can be None, str, or list.
                - If None, the 'logs_name' keyword argument must be provided.
                - If 'logs_style' is 'id', the 'inlist_name' and 'option' keyword arguments must be provided.
                - If 'logs_style' is str, the 'logs_style' keyword argument must be provided,
                and its value will be appended to the 'logs_name' in the format 'logs_style_logs_value'.
                - If 'logs_style' is list, the 'logs_style' keyword argument must be provided as a list of strings,
                and the values corresponding to each style will be appended to the 'logs_name' in the format
                'style1_value1_style2_value2_...'.

            **kwargs: Additional keyword arguments that may be required based on the 'logs_style' value.

        Returns:
            logs_path (str): The path to the logs directory.

        Raises:
            ValueError: If the required keyword arguments are not provided or if 'logs_style' is an invalid value.

        Usage:
            This method generates a path to the logs directory based on the provided 'logs_style' and additional
            keyword arguments. It supports different scenarios:

            - If 'logs_style' is None, the 'logs_name' keyword argument must be provided. The 'logs_name' will be used
            as the name of the logs directory.

            - If 'logs_style' is 'id', the 'inlist_name' and 'option' keyword arguments must be provided. The method
            will read the value of the specified 'option' from the given MESA inlist file and append it as a new
            entry to the 'folder.index' file in the logs directory. The 'logs_name' will be set to the index of
            the appended entry.

            - If 'logs_style' is str, the 'logs_style' keyword argument must be provided. The method will look for
            another keyword argument with the same name as the 'logs_style' value. The 'logs_value' will be appended
            to the 'logs_name' in the format 'logs_style_logs_value'.

            - If 'logs_style' is list, the 'logs_style' keyword argument must be provided as a list of strings. The
            method will look for keyword arguments corresponding to each style in the list. The values will be appended
            to the 'logs_name' in the format 'style1_value1_style2_value2_...'.

            If any of the required keyword arguments are missing or if 'logs_style' has an invalid value, a ValueError
            will be raised.

        Examples:
            >>> # Example 1: logs_style is None
            >>> logs_path = Inlist.create_logs_path_string(logs_style=None, logs_name='test')
            >>> print(logs_path)
            LOGS/test

            >>> # Example 2: logs_style is 'id'
            >>> logs_path = Inlist.create_logs_path_string(logs_style='id', inlist_name='inlist', option='initial_mass')
            >>> print(logs_path)
            LOGS/1

            >>> # Example 3: logs_style is str
            >>> logs_path = Inlist.create_logs_path_string(logs_style='initial_mass', initial_mass=1.0)
            >>> print(logs_path)
            LOGS/initial_mass_1.0

            >>> # Example 4: logs_style is list
            >>> logs_path = Inlist.create_logs_path_string(logs_style=['initial_mass', 'metallicity'], initial_mass=1.0, metallicity=0.02)
            >>> print(logs_path)
            LOGS/initial_mass_1.0_metallicity_0.02
        """

        # define custom suite_dir if suite_style is given
        if isinstance(suite_style, list):
            suite_dir = Inlist._from_styles_to_string(suite_style, **kwargs)
        elif isinstance(suite_style, str):
            suite_dir = Inlist._from_styles_to_string([suite_style], **kwargs)
        else:
            suite_dir = suite_dir

        if logs_style is None:
            # get logs_name
            logs_name = kwargs.get("logs_name", None)
            if logs_name is None:
                raise ValueError("logs_name must be given if logs_style is None.")

        elif logs_style == "id":

            # get inlist and option
            inlist_name = kwargs.get("inlist_name", None)
            option = kwargs.get("option", None)

            # tests
            if option is None:
                raise ValueError("option must be given if logs_style is 'id'.")
            if inlist_name is None:
                raise ValueError("inlist_name must be given if logs_style is 'id'.")

            with Inlist(inlist_name) as inlist:
                value = inlist.read_option(option)

            # write (append) to 'folder_index'
            path = os.path.join(logs_src, "folder.index")

            with open(path, "a") as file:
                lengt_of_file = os.stat(file.name).st_size

                # if file is empty, write header
                if lengt_of_file == 0:
                    file.write(f"# id\t{option}\n")
                    lengt_of_file = 1

                file.write(f"{lengt_of_file}\t{value}\n")

            logs_name = f"{lengt_of_file}"

        elif isinstance(logs_style, str):
            # get logs_name
            logs_value = kwargs.get(logs_style, None)
            if logs_value is None:
                raise ValueError(
                    f"{logs_style} must be given if logs_style is {logs_style}"
                )
            if isinstance(logs_value, (str,int)):
                logs_name = f"{logs_style}_{logs_value}"
            elif isinstance(logs_value, float):
                logs_name = f"{logs_style}_{kwargs.get(logs_style, None):.2f}"
            else:
                raise ValueError(f"{logs_value} must be a string, integer or a float.")

        elif isinstance(logs_style, list):
            logs_name = Inlist._from_styles_to_string(logs_style, **kwargs)

        else:
            raise ValueError(
                f"logs_style must be None, str, or list. Got {logs_style}."
            )

        logs_path = os.path.join(logs_src, suite_dir, logs_name)

        return logs_path

    @staticmethod
    def convergence_tolerance_options(convergence_tolerances: str) -> list:
        """Returns common convergence tolerances."""
        if convergence_tolerances == "very_tight":
            tol_correction_norm = 1e-4
            tol_max_correction = 3e-2

        elif convergence_tolerances == "tight":
            tol_correction_norm = 1e-4
            tol_max_correction = 8e-2

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

    @staticmethod
    def set_multiple_options(options_dict: dict) -> None:
        """Sets options in multiple inlist files.

        Initializes inlist objects and sets the options accordingly.

        Parameters
        ----------
        options : dict
            dictionary with options of the format {'inlist_name': {'option_name': 'option_value', ...}, ...}

        Returns
        -------
        None

        Notes
        -----
        This function does not restore the inlist files to their original versions. Use Inlist.restore_all_instances() for that.

        Examples
        --------
        >>> Inlist.set_multiple_options({'inlist_pgstar': {'pgstar_flag': True}, 'inlist_evolve': {'log_directory': 'LOGS/test'}})
        """

        for inlist_name, options in options_dict.items():

            # initialize inlist
            inlist = Inlist(inlist_name)

            # go through all options and set them
            for option_name, option_value in options.items():
                inlist.set_option(option_name, option_value)

    @staticmethod
    def create_relax_entropy_file_homogeneous(
        s_kerg, relax_entropy_filename: str = "relax_entropy_file.dat"
    ):
        """Creates a relax entropy file for s(m) = s_kerg."""
        s = specific_entropy(s_kerg)
        with open(relax_entropy_filename, "w") as file:
            file.write("1\n")
            file.write(f"1  {Inlist.fortran_format(s)}")
        print(f"Created entropy profile with s_kerg = {s_kerg}")

    @staticmethod
    def _create_relax_entropy_list(
        s_of_m_kerg: Callable, n_points: int = 1000
    ) -> np.ndarray:
        """Creates a list of mass and entropy values for the relax entropy file."""

        out = np.zeros((n_points, 2))
        mass_bins = np.linspace(0, 1, n_points)
        for i, m in enumerate(mass_bins):
            out[i, 0] = 1 - m  # mass fraction q starting from M_p
            out[i, 1] = specific_entropy(s_of_m_kerg(m))

        # test wether any entropy values are negative
        if np.any(out[:, 1] < 0):
            raise ValueError("Entropy values must be positive.")

        # flip array to have increasing mass
        return np.flip(out, axis=0)

    @staticmethod
    def create_relax_entropy_file(
        s_of_m_kerg: Callable,
        relax_entropy_filename: str = "relax_entropy_file.dat",
        n_points: int = 1000,
    ) -> None:
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
        s_list = Inlist._create_relax_entropy_list(s_of_m_kerg, n_points)

        with open(relax_entropy_filename, "w") as file:
            # write header: number of points
            file.write(f"{n_points}\n")
            for l in s_list:
                str_version = [f"{el:.16e}" for el in l]
                file.write("  ".join(str_version) + "\n")

        print(f"{relax_entropy_filename} was created successfully.")
