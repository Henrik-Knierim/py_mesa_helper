import os
import numpy as np
import shutil
from mesa_helper.astrophys import *
from mesa_helper.utils import *
from typing import Callable

OptionType = float | bool | str | int | np.floating


# TODO: Fix bug where a ! inside a string in an inlist file is not read correctly
class Inlist:
    """Class for reading and writing inlist files."""

    # TODO: verbose option and debug option
    def __init__(self, name: str, **kwargs) -> None:
        """Initializes an Inlist object."""

        # name of the inlist file
        self.name: str = name

        # backup original inlist
        with open(self.name, "r") as file:
            self.original_inlist: str = file.read()

        # define path to where the program looks for the MESA options files
        self._set_mesa_options_path(kwargs.get("mesa_options_path", None))

        # control how much information is printed
        self.verbose: bool = kwargs.get("verbose", False)
        self.debug: bool = kwargs.get("debug", False)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:

        # restore original inlist
        self.restore_inlist()

    ### *changin options ###

    def _set_mesa_options_path(self, path: str | None = None) -> None:
        """Sets path to the MESA options files."""
        if path is None:
            try:
                self.mesa_options_path = os.path.join(
                    os.environ["MESA_DIR"], "star", "defaults"
                )
            except KeyError:
                raise ValueError("The environment variable $MESA_DIR is not defined.")
        else:
            self.mesa_options_path = path

    def _is_option(self, section: str, option: str) -> bool:
        """Checks if `option` is in `section` of `MESA`.

        Parameters
        ----------
        section : str
            Section of the option. For example, 'controls', 'star_job', 'pgstar'.
        option : str
            Option to check.

        """

        src: str = os.path.join(self.mesa_options_path, f"{section}.defaults")
        with open(src, "r") as file:
            for line in file:

                # remove whitespace
                line = line.strip()

                # if the option contains a paranthesis, only check the part before the paranthesis
                if "(" in option:
                    option = option.split("(")[0]

                # check if line starts with option
                if line.startswith(option):
                    return True
        return False

    def _get_section_of_option(self, option: str) -> str:
        """Returns the section of an option."""
        if self._is_option("controls", option):
            return "&controls"
        elif self._is_option("star_job", option):
            return "&star_job"
        elif self._is_option("pgstar", option):
            return "&pgstar"
        else:
            raise ValueError(f"Option {option} not found.")

    def read_option(self, option: str) -> OptionType | None:
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

                    out: float | int | str | bool = Inlist._python_format(out)
                    return out

        return None

    def _change_lines(self, option: str, value) -> list[str]:
        """Tries to find `option` in the inlist file and changes it to `value`.

        Parameters
        ----------
        option : str
            Option to change.
        value : _type_
            Value to set.

        Returns
        -------
        list[str]
            List of lines of the inlist file with the new option value.
        """
        separator: str = "="

        with open(self.name, "r") as file:

            lines: list[str] = file.readlines()

            for i, l in enumerate(lines):
                if option in l:

                    # test if this is in fact the right option

                    # for ignoring fortran comments after the value
                    line_replace: str = l.replace("!", separator)

                    # after split: 0: option, 1: value, 2: comment (if present)
                    line_splitted: list[str] = line_replace.split(separator)

                    # true if the occurence exactly matches with option
                    is_option: bool = line_splitted[0].strip() == option
                    if is_option:
                        index_option: int = i

                        # fortran formatting
                        out: str = Inlist._fortran_format(value)

                        new_line: str = (
                            line_splitted[0] + " " + separator + " " + out + "\n"
                        )

                        break

            lines[index_option] = new_line

        return lines

    def _create_lines_explicit_section(
        self, section: str, option: str, value: OptionType
    ) -> list[str]:
        """Creates lines for a new option in an inlist file, where the section is explicitly given."""
        with open(self.name, "r") as file:

            lines: list[str] = file.readlines()

            # fortran formatting
            out: str = Inlist._fortran_format(value)

            for i, l in enumerate(lines):
                if section in l:

                    index_section: int = i

                    break

            lines.insert(index_section + 2, f"\t{option} = {out}\n")

        return lines

    def _create_lines(self, option: str, value: OptionType) -> list[str]:
        """Creates lines for an `option` assuming it is not present in the inlist file."""

        section: str = self._get_section_of_option(option)
        return self._create_lines_explicit_section(section, option, value)

    def set_option(self, option: str, value: OptionType) -> None:
        """Sets an option in an inlist file.

        Parameters
        ----------
        option : str
            The option to set in the inlist file.
        value : float | int | str | bool
            The value to set for the option.
        """
        # conversion such that output is in ''
        if type(value) == str:
            value = f"'{value}'"

        # check if the option is already present. If not, create it
        try:
            lines = self._change_lines(option, value)
        except:
            lines = self._create_lines(option, value)

        # write new lines into the inlist
        with open(self.name, "w") as file:
            file.writelines(lines)

        print(f"Set {option} to {Inlist._fortran_format(value)}") if self.verbose else None

    def set_multiple_options(self, **options: OptionType) -> None:
        """Sets multiple options in an inlist file.

        Parameters
        ----------
        options_dict : dict
            Dictionary with options of the format {'option_name': option_value, ...}

        Examples
        --------
        >>> set_multiple_options({'pgstar_flag': True, 'log_directory': 'LOGS/test'})
        """

        # go through all options and set them
        for option_name, option_value in options.items():
            self.set_option(option_name, option_value)

    def restore_inlist(self) -> None:
        """Restores the inlist to its original version."""
        with open(self.name, "w") as file:
            file.write(self.original_inlist)
        print(f"restored {self} to original version") if self.verbose else None

    def save_inlist(self, target_directory: str | None = None) -> None:
        """Copies the inlist into `target_directory`. The default is the `log_directory`."""

        if target_directory is None:
            log_directory: str = os.path.normpath(self.read_option("log_directory"))
        else:
            log_directory = target_directory

        try:
            shutil.copy(self.name, log_directory)
            (
                print(
                    f"saved {self.name} to {self.read_option('log_directory')} directory"
                )
                if self.verbose
                else None
            )
        except FileNotFoundError:
            raise ValueError(f"The file {self.name} does not exist.")
        except PermissionError:
            raise ValueError(
                f"Permission denied to copy {self.name} to {log_directory}."
            )
        except Exception as e:
            raise ValueError(f"An unexpected error occurred: {str(e)}")

    ### *common inlist tasks ###

    def set_initial_mass_in_M_Jup(self, M_p_in_M_J: float) -> None:
        """Given the mass of the planet in Jupiter masses, sets the mass of the planet in grams."""
        M_p_in_g: float = M_Jup_in_g * M_p_in_M_J
        self.set_option("mass_in_gm_for_create_initial_model", M_p_in_g)

    def set_initial_radius_in_R_Jup(self, R_p_in_R_J: float) -> None:
        """Given the radius of the planet in Jupiter radii, sets the radius of the planet in centimeters."""
        R_p_in_cm: float = R_Jup_in_cm * R_p_in_R_J
        self.set_option("radius_in_cm_for_create_initial_model", R_p_in_cm)

    # Todo: test this function
    def set_initial_abundances(self, gradient: str, value: float, scaling: Callable | None = None) -> None:

        # check if gradient is 'Y' or 'Z'
        validate_option(gradient, ["Y", "Z"])

        if scaling is None and gradient == "Z":
            scaling = lambda Z: scaled_solar_ratio_mass_fractions(Z=value)

        elif scaling is None and gradient == "Y":
            scaling = lambda Y: (1-Y, Y, 0.0)

        elif scaling is None:
            # something went wrong
            raise ValueError("scaling must be given if gradient is not 'Y' or 'Z'.")
            
        X, Y, Z = scaling(value)

        self.set_option("initial_z", Y)
        self.set_option("initial_y", Z)


    @staticmethod
    def _fortran_float(value: float|np.floating)-> str:
        # Check if the number is zero float
        if value == 0.0:
            prefactor: float | np.floating = 0.0
            exponent: int = 0
            sign: int = 1
        else:
            log: np.floating = np.log10(abs(value))
            exponent = int(np.floor(log))
            prefactor = 10 ** (log - exponent)
            sign = np.sign(value)

        # Check if the prefactor is an integer or has trailing zeros
        if prefactor.is_integer():
            prefactor = int(prefactor)

        return f"{sign*prefactor:.6g}d{exponent}"
    
    @staticmethod
    def _fortran_format(x: float | bool | str | int | np.floating) -> str:
        """Converts a python input to a fortran output (as str)."""
        if isinstance(x, (float, np.floating)):
            out = Inlist._fortran_float(x)

        elif isinstance(x, bool):
            if x:
                out = ".true."
            else:
                out = ".false."

        else:
            out = str(x)

        return out

    @staticmethod
    def _python_format(x: str) -> float | int | str | bool:
        """Converts a fortran input string to a python output."""
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
        if isinstance(value, (str, int)):
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
    def _directory_name_from_no_style(
        directory: str | None = None, empty_string_is_valid: bool = False
    ) -> str:
        """Creates a directory name when no style is given."""

        if directory is None and empty_string_is_valid:
            return ""
        elif directory is None:
            raise ValueError("`directory` must be given if `directory_style` is None.")
        elif isinstance(directory, str):
            return directory
        else:
            raise ValueError(f"{directory} must be a string.")

    @staticmethod
    def _directory_name_from_id(
        inlist_name: str, option: str, index_file_path: str
    ) -> str:
        """Creates a directory name based on the id option."""
        with Inlist(inlist_name) as inlist:
            value = inlist.read_option(option)

        # write (append) to 'folder_index'
        path = os.path.join(index_file_path, "folder.index")

        with open(path, "a") as file:
            lengt_of_file = os.stat(file.name).st_size

            # if file is empty, write header
            if lengt_of_file == 0:
                file.write(f"# id\t{option}\n")
                lengt_of_file = 1

            file.write(f"{lengt_of_file}\t{value}\n")

        return f"{lengt_of_file}"

    @staticmethod
    def _directory_name_from_str(directory_style: str, **kwargs) -> str:
        """Creates a directory name based on a style."""
        directory_value = kwargs.get(directory_style, None)
        if directory_value is None:
            raise ValueError(
                f"{directory_style} must be given if directory_style is {directory_style}"
            )
        if isinstance(directory_value, (str, int)):
            return f"{directory_style}_{directory_value}"
        elif isinstance(directory_value, float):
            return f"{directory_style}_{kwargs.get(directory_style, None):.2f}"
        else:
            raise ValueError(f"{directory_value} must be a string, integer or a float.")

    @staticmethod
    def _create_directory_name_from_style(
        directory_style: str | list[str] | None = None,
        **kwargs,
    ) -> str:
        """
        Creates a directory name according to the style specified in `directory_style`.

        Parameters
        ----------
        directory_style : str, list of str, or None, optional
            Variables to define the logs directory structure. Can be None, str, or list.

        **kwargs
            Additional keyword arguments that may be required based on the 'directory_style' value.

        Returns
        -------
        str
            The path to the logs directory.

        Raises
        ------
        ValueError
            If the required keyword arguments are not provided or if 'directory_style' is an invalid value.

        Examples
        --------
        This method generates a path to the logs directory based on the provided 'directory_style' and additional
        keyword arguments. It supports different scenarios:

        - If 'directory_style' is None, the 'directory' keyword argument must be provided. The 'directory' will be used
        as the name of the directory.

        - If 'directory_style' is 'id', the 'inlist_name' and 'option' keyword arguments must be provided. The method
        will read the value of the specified 'option' from the given MESA inlist file and append it as a new
        entry to the 'folder.index' file in the logs directory. The 'directory' will be set to the index of
        the appended entry.

        - If 'directory_style' is str, the 'directory_style' keyword argument must be provided. The method will look for
        another keyword argument with the same name as the 'directory_style' value. The 'directory_style' will be appended
        to the 'directory_style' in the format 'directory_style_logs_value'.

        - If 'directory_style' is list, the 'directory_style' keyword argument must be provided as a list of strings. The
        method will look for keyword arguments corresponding to each variable in the list. The values will be appended
        to the 'directory' in the format 'var1_value1_var2_value2_...'.

        If any of the required keyword arguments are missing or if 'directory_style' has an invalid value, a ValueError
        will be raised.

        Examples
        --------
        >>> # Example 1: directory_style is None
        >>> directory = Inlist._create_directory_name_from_style(directory_style=None, directory='test')
        >>> print(directory)
        test

        >>> # Example 2: directory_style is 'id'
        >>> directory = Inlist._create_directory_name_from_style(folder_style='id', inlist_name='inlist', option='initial_mass')
        >>> print(directory)
        1

        >>> # Example 3: directory_style is str
        >>> directory = Inlist._create_directory_name_from_style(folder_style='initial_mass', initial_mass=1.0)
        >>> print(directory)
        initial_mass_1.0

        >>> # Example 4: directory_style is list
        >>> directory = Inlist._create_directory_name_from_style(folder_style=['initial_mass', 'metallicity'], initial_mass=1.0, metallicity=0.02)
        >>> print(directory)
        initial_mass_1.0_metallicity_0.02
        """

        # optional arguments
        verbose = kwargs.get("verbose", False)
        if verbose:
            print(Inlist._create_directory_name_from_style.__name__)
            print(f"\tkwargs", *kwargs)

        # directory name directly given as an argument
        if directory_style is None:
            print("\tdirectory_style is None") if verbose else None
            directory = Inlist._directory_name_from_no_style(
                kwargs.get("directory", None),
                kwargs.get("empty_string_is_valid", False),
            )

        # name by id
        elif directory_style == "id":

            # get inlist and option
            inlist_name: str | None = kwargs.get("inlist_name", None)
            option: str | None = kwargs.get("option", None)
            index_file_path: str | None = kwargs.get("index_file_path", None)

            # tests
            for arg in [inlist_name, option, index_file_path]:
                if not isinstance(arg, str):
                    raise ValueError(f"{arg} must be a string.")

            directory = Inlist._directory_name_from_id(
                inlist_name, option, index_file_path
            )

        elif isinstance(directory_style, str):
            directory = Inlist._directory_name_from_str(directory_style, **kwargs)

        elif isinstance(directory_style, list):
            directory = Inlist._from_styles_to_string(directory_style, **kwargs)

        else:
            raise ValueError(
                f"logs_style must be None, str, or list. Got {directory_style}."
            )

        print(f"\tdirectory = {directory}") if verbose else None
        return directory

    @staticmethod
    def create_logs_path(
        logs_parent_dir: str = "LOGS",
        logs_style: str | list[str] | None = None,
        series_style: str | list[str] | None = None,
        **kwargs,
    ) -> str:
        """
        Creates a path for the output of the MESA simulations.

        This method generates a path to the logs directory based on the provided 'logs_style' and additional
        keyword arguments. It supports different scenarios:

        - If 'logs_style' is None, the 'logs_dir' keyword argument must be provided. The 'logs_dir' will be used
        as the name of the logs directory.

        - If 'logs_style' is 'id', the 'inlist_name' and 'option' keyword arguments must be provided. The method
        will read the value of the specified 'option' from the given MESA inlist file and append it as a new
        entry to the 'folder.index' file in the logs directory. The 'logs_dir' will be set to the index of
        the appended entry.

        - If 'logs_style' is str, the 'logs_style' keyword argument must be provided. The method will look for
        another keyword argument with the same name as the 'logs_style' value. The 'logs_value' will be appended
        to the 'logs_dir' in the format 'logs_style_logs_value'.

        - If 'logs_style' is list, the 'logs_style' keyword argument must be provided as a list of strings. The
        method will look for keyword arguments corresponding to each variable in the list. The values will be appended
        to the 'logs_dir' in the format 'var1_value1_var2_value2_...'.

        If any of the required keyword arguments are missing or if 'logs_style' has an invalid value, a ValueError
        will be raised.

        The same logic applies to the 'series_style' keyword argument, which is used to define the directory for a series

        Parameters
        ----------
        logs_parent_dir : str, optional
            The parent directory for logs. Default is "LOGS".
        logs_style : str, list of str, or None, optional
            Variables to define the logs directory structure. Can be None, str, or list.
            - If None, the 'logs_dir' keyword argument must be provided.
            - If 'logs_style' is 'id', the 'inlist_name' and 'option' keyword arguments must be provided.
            - If 'logs_style' is str, the 'logs_style' keyword argument must be provided,
            and its value will be appended to the 'logs_dir' in the format 'logs_style_logs_value'.
            - If 'logs_style' is list, the 'logs_style' keyword argument must be provided as a list of strings,
            and the values corresponding to each variable will be appended to the 'logs_dir' in the format
            'var1_value1_var2_value2_...'.
        series_dir : str or None, optional
            The directory for the series of simulations. Default is None.
        series_style : str, list of str, or None, optional
            Variables to define the directory for a series of simulations. Can be None, str, or list. Follows the same logic as 'logs_style'.
        **kwargs
            Additional keyword arguments that may be required based on the 'logs_style' value.

        Returns
        -------
        str
            The path to the logs directory.

        Raises
        ------
        ValueError
            If the required keyword arguments are not provided or if 'logs_style' is an invalid value.

        Examples
        --------

        >>> # Example 1: logs_style is None
        >>> logs_path = Inlist.create_logs_path(logs_style=None, logs_dir='test')
        >>> print(logs_path)
        LOGS/test

        >>> # Example 2: logs_style is 'id'
        >>> logs_path = Inlist.create_logs_path(logs_style='id', inlist_name='inlist', option='initial_mass')
        >>> print(logs_path)
        LOGS/1

        >>> # Example 3: logs_style is str
        >>> logs_path = Inlist.create_logs_path(logs_style='initial_mass', initial_mass=1.0)
        >>> print(logs_path)
        LOGS/initial_mass_1.0

        >>> # Example 4: logs_style is list
        >>> logs_path = Inlist.create_logs_path(logs_style=['initial_mass', 'metallicity'], initial_mass=1.0, metallicity=0.02)
        >>> print(logs_path)
        LOGS/initial_mass_1.0_metallicity_0.02
        """

        # get the series directory
        directory = kwargs.get("series_dir", None)
        series_dir = Inlist._create_directory_name_from_style(
            series_style,
            index_file_path=logs_parent_dir,
            empty_string_is_valid=True,
            directory=directory,
            **kwargs,
        )

        # get the logs directory
        parent_dir = series_dir if series_dir != "" else logs_parent_dir
        directory = kwargs.get("logs_dir", None)
        logs_dir = Inlist._create_directory_name_from_style(
            logs_style, index_file_path=parent_dir, directory=directory, **kwargs
        )

        logs_path = os.path.join(logs_parent_dir, series_dir, logs_dir)

        return logs_path

    def set_logs_path(self, **kwargs) -> None:
        """
        Sets the path for the logs directory.

        Parameters
        ----------
        **kwargs : dict
            All parameters are inherited from `Inlist.create_logs_path_string()`.

        Returns
        -------
        None

        Examples
        --------
        >>> set_logs_path(logs_style='m_core', m_core=25.0)
        # sets the log directory to 'LOGS/m_core_25.0'
        """
        logs_path = Inlist.create_logs_path(**kwargs)
        self.set_option("log_directory", logs_path)

    @staticmethod
    def create_model_filename(**kwargs) -> str:
        """Creates a model filename using the functionality from `create_logs_path`."""
        
        mod_parent_dir = kwargs.get("parent_dir", "")
        mod_file  = Inlist.create_logs_path(logs_parent_dir = mod_parent_dir, **kwargs)
        mod_file += ".mod"

        return mod_file

    # ? Do we need this function?
    @staticmethod
    def set_multiple_options_for_multiple_inlists(options_dict: dict) -> None:
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
            file.write(f"1  {Inlist._fortran_format(s)}")
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
