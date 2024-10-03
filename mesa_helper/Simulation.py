# class for anything related to after the simulation has been run
# for example, analyzing, plotting, saving, etc.
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import os
import numpy as np
import mesa_reader as mr
import pandas as pd
from param import Callable
from mesa_helper.astrophys import M_Jup_in_g, M_Sol_in_g, M_Earth_in_g
from scipy.interpolate import interp1d
from functools import lru_cache
from typing import Tuple


class Simulation:
    """Class for anything related to a single simulation has been run. For example, analyzing, plotting, saving, etc."""

    def __init__(
        self,
        simulation_dir: str,
        parent_dir: str = "./LOGS",
        verbose: bool = False,
    ) -> None:
        """Initializes the Simulation object.
        Parameters
        ----------
        parent_dir : str, optional
            The parent directory of the simulation. The default is './LOGS'.
        simulation_dir : str, optional
            The simulation. The default is ''.
        check_age_convergence : bool, optional
            If True, then the simulations that do not converge to the final age are removed. The default is True.
        **kwargs : dict
            Keyword arguments for `self.remove_non_converged_simulations`. For example, `final_age` can be specified.
        """
        self.verbose = verbose

        # parent directory of the simulation
        self.parent_dir = parent_dir

        self.sim = simulation_dir

        self.sim_dir = os.path.join(self.parent_dir, self.sim)

        # then, initialize the mesa logs and histories
        self.log: mr.MesaLogDir = mr.MesaLogDir(self.sim_dir)
        self.history: mr.MesaData = self.log.history

        # if the history is None, then the simulation did not run successfully
        if self.history is None:
            raise ValueError(f"The simulation {self.sim} did not run successfully.")

        # data frame for extracting results
        self.results = pd.DataFrame({"log_dir": [self.sim]})

    # create a __str__ method that returns the name of the suite, or the name of the simulation if there is no suite
    def __str__(self):
        return self.sim

    @staticmethod
    def _extract_value(string, free_param: str):
        """Extracts the value of `free_param` from `string` where the values are separated by an underscore."""
        if type(free_param) != str:
            raise TypeError("free_params must be a string")
        elif free_param not in string:
            raise ValueError(f"Parameter {free_param} not found in string {string}")

        splitted_string = string.split(free_param)

        # convert value either to a float or keep it as a string
        try:
            value = float(splitted_string[1].split("_")[1])
        except:
            value = splitted_string[1].split("_")[1:]
            # join the list of strings to a single string
            value = "_".join(value)

        return value

    # *------------------------------ #
    # *--- Simulation Convergence --- #
    # *------------------------------ #

    @staticmethod
    def _is_profile_index_valid(path_to_profile_index):
        """Returns True if profiles.index contains more than 2 lines."""
        with open(path_to_profile_index, "r") as f:
            lines = f.readlines()
        return len(lines) > 2

    def check_if_conserved(
        self,
        quantity: str,
        model_number_initial: int = 1,
        model_number_final=-1,
        relative_tolerance: float = 1e-3,
    ) -> bool:
        # """Checks if the quantity is conserved to a certain tolerance."""
        """Checks if the quantity is conserved to a certain tolerance.

        Parameters
        ----------
        quantity : str
            quantity to check
        model_number_initial : int, optional
            initial model number at which the quantity is read, by default 1
        model_number_final : int, optional
            final model number at which the quantity is read, by default -1
        relative_tolerance : float, optional
            tolerance below which a quantitiy is considered to be conserved, by default 1e-3

        Returns
        -------
        bool
            True if the quantity is conserved to the given tolerance
        """

        if model_number_initial < 0:
            model_number_initial = self.log.model_numbers[model_number_initial]
        if model_number_final < 0:
            model_number_final = self.log.model_numbers[model_number_final]

        quantity_start: np.floating | np.integer = self.history.data_at_model_number(
            quantity, model_number_initial
        )
        quantity_end: np.floating | np.integer = self.history.data_at_model_number(
            quantity, model_number_final
        )

        return (
            np.abs(quantity_end - quantity_start) / quantity_start < relative_tolerance
        )

    def check_value(
        self,
        quantity: str,
        value: float,
        model_number: int = -1,
        relative_tolerance: float = 1e-3,
    ) -> bool:
        """Checks if the quantity is equal to a certain value to a certain tolerance.

        Parameters
        ----------
        quantity : str
            quantity to check
        value : float
            value to compare to
        model_number : int, optional
            model number at which to evaluate the quantity, by default -1
        relative_tolerance : float, optional
            tolerance below which a quantitiy is considered to be conserved, by default 1e-3

        Returns
        -------
        bool
            True if the quantity is conserved to the given tolerance
        """

        if model_number < 0:
            model_number = self.history.model_number[model_number]

        quantity_value: np.floating | np.integer = self.history.data_at_model_number(
            quantity, model_number
        )

        return np.abs(quantity_value - value) / value < relative_tolerance

    # ------------------------------ #
    # ----- Simulation Results ----- #
    # ------------------------------ #

    @lru_cache
    def get_history_data_at_condition(
        self,
        quantity: str,
        condition: str,
        value: int | float,
        check_tolerance: bool = False,
        relative_tolerance: float = 1e-3,
    ) -> np.float_ | int:
        """Returns the history data for `quantity` where the quantity is closest to `value`.

        Parameters
        ----------
        quantity : str
            The key to the history data that we want to retrieve.
        condition : str
            The quantity that should be closest to `value`.
        value : int | float
            The value that the quantity should be closest to.
        check_tolerance : bool, optional
            If True, then the quantity is checked to be within the given tolerance. The default is False.
        relative_tolerance : float, optional
            The relative tolerance for the quantity. The default is 1e-3.
        """

        if condition == "model_number":
            if value < 0:
                value = self.history.model_number[value]
            return self.history.data_at_model_number(quantity, value)

        # find the index of the closest value to the age
        quantities = self.history.data(condition)

        # find the index of the closest value to the age
        index = np.argmin(np.abs(quantities - value))
        model_number = self.history.model_number[index]

        if check_tolerance:
            if not self.check_value(
                condition,
                value,
                model_number=model_number,
                relative_tolerance=relative_tolerance,
            ):
                raise ValueError(
                    f"The quantity {condition} is not within the given tolerance of {relative_tolerance:.2e}."
                )

        return self.history.data_at_model_number(quantity, model_number)

    def add_history_data(
        self,
        history_keys: str | list[str],
        condition: str = "model_number",
        value: int | float = -1,
    ) -> None:
        """Adds `history_keys` to `self.results`.

        Parameters
        ----------
        history_keys : str | list[str]
            The history keys to add to `self.results`. Can either be a string or a list of strings.
        condition : str, optional
            The quantity that should be closest to `value`. The default is 'model_number'.
        value : int | float, optional
            The value that the quantity should be closest to. The default is -1.
        """

        history_keys = [history_keys] if isinstance(history_keys, str) else history_keys

        for history_key in history_keys:
            out = [
                self.get_history_data_at_condition(
                    quantity=history_key, condition=condition, value=value
                )
            ]
            self.results[history_key] = out

    def get_profile_data_sequence(
        self,
        quantity: str,
        model_numbers: list[int] | np.ndarray | None = None,
        profile_numbers: list[int] | np.ndarray | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        """Returns the profile data for `quantity` for all profile numbers in `profile_numbers`."""

        if model_numbers is None and profile_numbers is None:
            indices: np.ndarray = self.log.profile_numbers
            f = lambda i: self.log.profile_data(profile_number=i, **kwargs).data(
                quantity
            )

        elif model_numbers is not None:
            indices = model_numbers
            f = lambda i: self.log.profile_data(model_number=i, **kwargs).data(quantity)

        elif profile_numbers is not None:
            indices = profile_numbers
            f = lambda i: self.log.profile_data(profile_number=i, **kwargs).data(
                quantity
            )

        return [f(i) for i in indices]

    @staticmethod
    def integrate_profile(
        profile, quantity: str, q0: float = 0.0, q1: float = 1.0, mass_unit: str = "g"
    ) -> np.float_:
        """Returns the profile quantity integrated in dm from the relative mass coordinate q0 to q1.

        Parameters
        ----------
        profile : mesa_reader.MesaProfileData
            The profile data object.
        quantity : str
            The quantity to integrate.
        q0 : float, optional
            The lower limit of the integration. The default is 0.0.
        q1 : float, optional
            The upper limit of the integration. The default is 1.0.
        mass_unit : str, optional
            The mass unit of the quantity. The default is 'g'. Other options are 'M_Jup', 'M_Sol', and 'M_Earth'.
        """

        normalizations = {
            "M_Jup": M_Jup_in_g,
            "M_Sol": M_Sol_in_g,
            "M_Earth": M_Earth_in_g,
            "g": 1.0,
        }
        normalization = normalizations[mass_unit]

        q = profile.data("mass") / profile.header_data["star_mass"]
        criterion = (q0 < q) & (q < q1)

        dm = profile.data("dm")[criterion] / normalization
        quantity = profile.data(quantity)[criterion]

        return np.dot(dm, quantity)

    @staticmethod
    def mean_profile_value(
        profile, quantity: str, q0: float = 0.0, q1: float = 1.0
    ) -> np.float_:
        """Returns the mean of the profile quantity from the relative mass coordinate q0 to q1.

        Parameters
        ----------
        profile : mesa_reader.MesaProfileData
            The profile data object.
        quantity : str
            quantity to integrate.
        q0 : float, optional
            The lower limit of the integration, by default 0.0
        q1 : float, optional
            The upper limit of the integration, by default 1.0

        Returns
        -------
        np.float_
            The mean of the profile quantity from q0 to q1.
        """

        q = profile.data("mass") / profile.header_data["star_mass"]
        criterion = (q0 < q) & (q < q1)

        dm = profile.data("dm")[criterion]
        quantity = profile.data(quantity)[criterion]

        return np.dot(dm, quantity) / np.sum(dm)

    def integrate(
        self,
        quantity: str,
        model_number: int = -1,
        profile_number: int = -1,
        q0: float = 0.0,
        q1: float = 1.0,
        mass_unit: str = "g",
        **kwargs,
    ) -> np.float_:
        """Returns the integrated quantity from the relative mass coordinate q0 to q1 for the profile number."""
        p: mr.MesaData = self.log.profile_data(
            model_number=model_number, profile_number=profile_number, **kwargs
        )
        return Simulation.integrate_profile(p, quantity, q0, q1, mass_unit)

    def mean(
        self,
        quantity: str,
        model_number: int = -1,
        profile_number: int = -1,
        q0: float = 0.0,
        q1: float = 1.0,
        **kwargs,
    ) -> np.float_:
        """Returns the mean of the quantity from the relative mass coordinate q0 to q1 for the profile number."""
        p: mr.MesaData = self.log.profile_data(
            model_number=model_number, profile_number=profile_number, **kwargs
        )
        return Simulation.mean_profile_value(p, quantity, q0, q1)

    def add_profile_data(
        self,
        quantity: str,
        q0: float = 0.0,
        q1: float = 1.0,
        profile_number: int = -1,
        kind: str = "integrated",
        name: str | None = None,
        mass_unit: str = "g",
        **kwargs,
    ) -> None:
        """Adds proccessed profile quantity to `self.results`.

        The profile quantity is either integrated or the mean is taken from q0 to q1.

        Parameters
        ----------
        quantity : str
            The quantity to integrate.
        q0 : float, optional
            lower relative mass coordinate, by default 0.0
        q1 : float, optional
            upper relative mass coordinate, by default 1.0
        profile_number : int, optional
            which profile to integrate, by default -1
        kind : str, optional
            whether to integrate or take the mean, by default "integrated"
        name : str | None, optional
            name of the results coulmn, by default None
        mass_unit : str, optional
            mass unit of the quantity, by default 'g'. Only relevant for integrated quantities.
        **kwargs : dict
            keyword arguments for `mesa_reader.MesaProfileData`.

        Examples
        --------
        >>> sim.add_profile_data('entropy', kind='integrated', mass_unit='M_Jup')
        >>> sim.add_profile_data('entropy', kind='mean', name='mean_temperature')
        """
        if name is None:
            integrated_quantity_key = kind + "_" + quantity
        else:
            integrated_quantity_key = name

        if kind == "integrated":
            out = [
                self.integrate(
                    quantity,
                    profile_number=profile_number,
                    q0=q0,
                    q1=q1,
                    mass_unit=mass_unit,
                    **kwargs,
                )
            ]
        elif kind == "mean":
            out = [
                self.mean(
                    quantity, profile_number=profile_number, q0=q0, q1=q1, **kwargs
                )
            ]
        else:
            raise ValueError("kind must be either 'integrated' or 'mean'.")

        self.results[integrated_quantity_key] = out

    def get_profile_data_at_condition(
        self,
        quantity: str,
        condition: str,
        value: float | int,
        profile_number: int = -1,
        **kwargs,
    ) -> np.float_ | int:
        """Returns the profile data for `quantity` where `condition` is `value`.

        Description
        -----------
        The routine finds the index where the condition is satisfied and returns the profile data at that index. It starts from the first index (surface) and moves to the last index (core).

        Parameters
        ----------
        quantity: str
            The key to the profile data that we want to retrieve.
        condition : str
            The quantity that should be closest to `value`.
        value : float
            The value that the quantity should be closest to.
        profile_number : int, optional
            The profile number. The default is -1.
        log_dir : str, optional
            The log directory. The default is None.
        **kwargs : dict
            Keyword arguments for `MesaProfileData`.
        """

        # throw an error if value is a bool or a string
        if isinstance(value, (bool, str)):
            raise ValueError("value must be a float or an integer.")

        profile = self.log.profile_data(profile_number=profile_number, **kwargs)
        data = profile.data(condition)
        index = np.argmin(np.abs(data - value))

        return profile.data(quantity)[index]

    def add_profile_data_at_condition(
        self,
        quantity: str,
        condition: str,
        value: float,
        profile_number: int = -1,
        name=None,
        **kwargs,
    ) -> None:
        """Adds `quantity` to `self.results` where `condition` is closest to `value` of the specified `profile_number`."""

        if name is None:
            name = f"{quantity}_at_{condition}_{value}"

        out = [
            self.get_profile_data_at_condition(
                quantity, condition, value, profile_number=profile_number, **kwargs
            )
        ]
        self.results[name] = out

    @lru_cache
    def _create_profile_header_df(self, quantity: str, **kwargs) -> None:
        """Creates a dictionary for the profile header values.

        Parameters
        ----------
        quantity : str
            The quantity to extract from the profile header.
        """

        if not hasattr(self, "profile_header_df"):
            self.profile_header_df = pd.DataFrame()
            # always initialize the header with the model numbers
            model_numbers = [
                self.log.profile_data(profile_number=i, **kwargs).header_data[
                    "model_number"
                ]
                for i in self.log.profile_numbers
            ]
            self.profile_header_df["model_number"] = model_numbers

        data = [
            self.log.profile_data(profile_number=i, **kwargs).header_data[quantity]
            for i in self.log.profile_numbers
        ]

        self.profile_header_df[quantity] = data

    @staticmethod
    def _clostest_quantity(
        df: pd.DataFrame, quantity: str, condition: str, value: int | float
    ) -> np.float_ | int:
        """Returns the quantity in a DataFrame where condition is closest to value."""
        return df.iloc[(df[condition] - value).abs().argsort()[:1]][quantity].values[0]

    @lru_cache
    def get_profile_at_header_condition(
        self, condition: str, value: float | int, **kwargs
    ) -> mr.MesaData:
        """Returns the profile data for `quantity` where the profile header `condition` is closest to `value`."""

        # check if the profile header values exist in `self.profile_header_df`
        # if not, then create it
        if not hasattr(self, "profile_header_df"):
            self._create_profile_header_df(condition, **kwargs)

        elif condition not in self.profile_header_df.columns:
            self._create_profile_header_df(condition, **kwargs)

        model_number = self._clostest_quantity(
            self.profile_header_df, "model_number", condition, value
        )

        return self.log.profile_data(model_number=model_number, **kwargs)

    def get_profile_data_at_header_condition(
        self, quantity: str, condition: str, value: float | int, **kwargs
    ):
        """Returns the profile data for `quantity` where the profile header `condition` is closest to `value`."""
        return self.get_profile_at_header_condition(condition, value, **kwargs).data(
            quantity
        )

    def get_profile_data_at_header_condition_sequence(
        self, quantity: str, condition: str, values: list[float] | list[int], **kwargs
    ):
        """Returns the profile data for `quantity` where the profile header `condition` is closest to each value in `values`."""
        return [
            self.get_profile_data_at_header_condition(
                quantity, condition, value, **kwargs
            )
            for value in values
        ]

    def get_integrated_profile_data_sequence(
        self,
        quantity: str,
        q0: float = 0.0,
        q1: float = 1.0,
        model_numbers: list[int] | np.ndarray | None = None,
        profile_numbers: list[int] | np.ndarray | None = None,
        mass_unit: str = "g",
        **kwargs,
    ):
        """Returns the integrated profile quantity from the relative mass coordinates q0 to q1 for all profile numbers or model numbers specified."""

        if profile_numbers == None and model_numbers == None:
            return [
                self.integrate(
                    quantity,
                    profile_number=i_p,
                    q0=q0,
                    q1=q1,
                    mass_unit=mass_unit,
                    **kwargs,
                )
                for i_p in self.log.profile_numbers
            ]
        elif model_numbers != None:
            return [
                self.integrate(
                    quantity,
                    model_number=i_m,
                    q0=q0,
                    q1=q1,
                    mass_unit=mass_unit,
                    **kwargs,
                )
                for i_m in model_numbers
            ]
        elif profile_numbers != None:
            return [
                self.integrate(
                    quantity,
                    profile_number=i_p,
                    q0=q0,
                    q1=q1,
                    mass_unit=mass_unit,
                    **kwargs,
                )
                for i_p in profile_numbers
            ]

    def get_mean_profile_data_sequence(
        self,
        quantity: str,
        q0: float = 0.0,
        q1: float = 1.0,
        model_numbers: list[int] | np.ndarray | None = None,
        profile_numbers: list[int] | np.ndarray | None = None,
        **kwargs,
    ):
        """Returns the mean of the profile quantity from the relative mass coordinates q0 to q1 for all profile numbers or model numbers specified."""

        if profile_numbers == None and model_numbers == None:
            return [
                self.mean(quantity, profile_number=i_p, q0=q0, q1=q1, **kwargs)
                for i_p in self.log.profile_numbers
            ]
        elif model_numbers != None:
            return [
                self.mean(quantity, model_number=i_m, q0=q0, q1=q1, **kwargs)
                for i_m in model_numbers
            ]
        elif profile_numbers != None:
            return [
                self.mean(quantity, profile_number=i_p, q0=q0, q1=q1, **kwargs)
                for i_p in profile_numbers
            ]

    @lru_cache
    def interpolate_profile_data(
        self, x: str, y: str, model_number: int = -1, profile_number: int = -1, **kwargs
    ) -> interp1d:
        """Returns a interpolation function for the quantities (x,y).

        Note that this method retrieves the profile data via `mesa_reader.MesaProfileData.data` and then interpolates the data using `scipy.interpolate.interp1d`.
        Hence, you can use the same 'log'-syntax as in `mesa_reader.MesaProfileData.data` for logarithmic interpolation.
        """

        x_data = self.log.profile_data(
            model_number=model_number, profile_number=profile_number
        ).data(x)
        y_data = self.log.profile_data(
            model_number=model_number, profile_number=profile_number
        ).data(y)

        return interp1d(x_data, y_data, **kwargs)

    @lru_cache
    def interpolate_history_data(self, x: str, y: str, **kwargs) -> interp1d:
        """Returns an interpolation function for the quantities (x,y).

        Note that this method retrieves the profile data via `mesa_reader.MesaProfileData.data` and then interpolates the data using `scipy.interpolate.interp1d`.
        Hence, you can use the same 'log'-syntax as in `mesa_reader.MesaProfileData.data` for logarithmic interpolation.
        """

        x_data = self.history.data(x)
        y_data = self.history.data(y)

        return interp1d(x_data, y_data, **kwargs)

    def get_relative_difference(
        self,
        x: str,
        y: str,
        profile_number_reference: int = -1,
        profile_number_compare: int = -1,
        model_number_reference: int = -1,
        model_number_compare: int = -1,
        **kwargs,
    ):
        """Returns the relative difference of two profiles using interpolation.

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
        **kwargs : dict
            Keyword arguments for `scipy.interpolate.interp1d`.
        """

        # make sure the interpolation object for [log, profile_number, x, y] exists
        f_ref = self.interpolate_profile_data(
            x,
            y,
            model_number=model_number_reference,
            profile_number=profile_number_reference,
            **kwargs,
        )

        f_compare = self.interpolate_profile_data(
            x,
            y,
            model_number=model_number_compare,
            profile_number=profile_number_compare,
            **kwargs,
        )

        # get the x range
        # the minimum is the smallest common x value of the reference and compare log
        x_ref = self.log.profile_data(
            model_number=model_number_reference, profile_number=profile_number_reference
        ).data(x)
        x_compare = self.log.profile_data(
            model_number=model_number_compare, profile_number=profile_number_compare
        ).data(x)

        x_min = max(
            x_ref.min(),
            x_compare.min(),
        )

        # the maximum is the largest common x value of the reference and compare log
        x_max = min(
            x_ref.max(),
            x_compare.max(),
        )

        n_bins = kwargs.get("n_bins", 1000)
        x_values = np.linspace(x_min, x_max, n_bins)

        return x_values, (f_compare(x_values) - f_ref(x_values)) / f_ref(x_values)

    # ------------------------------ #
    # ------- Export Results ------- #
    # ------------------------------ #

    def export_history_data(self, filename: str, columns: list[str], **kwargs) -> None:
        """Exports the quantities in `columns` to a csv file."""

        # get the history data
        df = pd.DataFrame({column: self.history.data(column) for column in columns})

        # add index = False to kwargs if not specified
        if kwargs.get("index") is None:
            kwargs["index"] = False

        df.to_csv(filename, **kwargs)

    def export_profile_data(
        self,
        filename: str,
        columns: list[str],
        method="profile_number",
        model_number: int = -1,
        profile_number: int = -1,
        condition: str | None = None,
        value: int | float | None = None,
        **kwargs,
    ) -> None:
        """Exports the quantities in `columns` to a csv file.

        Parameters
        ----------
        filename : str
            The filename of the csv file.
        columns : list[str]
            The columns to be exported.
        method : str, optional
            The method to extract the profile data. The default is 'profile_number'. Available options are 'profile_number' and 'profile_header_condition'.
            For 'profile_number', the profile data is extracted at the profile number specified by `profile_number`.
            For 'profile_header_condition', the profile data is extracted at the profile number where the profile header `condition` is closest to `value`.
        **kwargs : dict
            Keyword arguments for `pd.DataFrame.to_csv`.

        Description
        -----------
        The routine loads the profile data of all logs and creates one joint DataFrame where arg_log_key is the column name.

        Examples
        --------
        >>> columns = ['mass', 'radius', 'temperature']
        >>> sim.export_profile_data('profile_data.csv', columns)

        """

        # get the profile data
        if method == "profile_number":
            df = pd.DataFrame(
                {
                    column: self.log.profile_data(profile_number=profile_number).data(
                        column
                    )
                    for column in columns
                }
            )
        elif method == "profile_header_condition":
            df = pd.DataFrame(
                {
                    column: self.get_profile_data_at_header_condition(
                        column, condition, value
                    )
                    for column in columns
                }
            )

        # add index = False to kwargs if not specified
        if kwargs.get("index") is None:
            kwargs["index"] = False

        df.to_csv(
            filename,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["profile_number", "condition", "value"]
            },
        )

    # * ------------------------------ #
    # * -------- Plot Results -------- #
    # * ------------------------------ #

    def profile_plot(
        self,
        x: str,
        y: str,
        model_number: int = -1,
        profile_number: int = -1,
        fig: plt.Figure | None = None,
        ax: Axes | None = None,
        set_label: bool = False,
        **kwargs
    ) -> Tuple[plt.Figure, Axes]:
        """Plots the profile data with (x, y) as the axes for a specified profile.

        You can specify the profile either by `model_number` or `profile_number`.

        Parameters
        ----------
        x : str
            The profile quantity for the x-axis.
        y : str
            The profile quantity for the y-axis.
        model_number : int, optional
            The model number of the profile, by default -1
        profile_number : int, optional
            The profile number, by default -1
        fig : plt.Figure, optional
            The figure. The default is None.
        ax : Axes, optional
            The axes. The default is None. If None, a new figure is created.
        set_label : bool, optional
            If true, add label to plot, by default False

        Returns
        -------
        Tuple[plt.Figure, Axes]
            The figure and axes of the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()

        data = self.log.profile_data(
            model_number=model_number, profile_number=profile_number
        )

        if set_label:
            kwargs["label"] = self.sim

        ax.plot(data.data(x), data.data(y), **kwargs)

        ax.set(xlabel=x, ylabel=y)
        
        return fig, ax

    def profile_series_plot(
        self,
        x: str,
        y: str,
        model_numbers: list[int] | np.ndarray  = [-1],
        profile_numbers: list[int] | np.ndarray = [-1],
        fig: plt.Figure | None = None,
        ax: Axes | None = None,
        set_labels: bool = False,
        **kwargs,
    ):
        """Plots the profile data with (x, y) as the axes at multiple profile numbers.

        You can specify the profile either by `model_number` or `profile_number`.

        Parameters
        ----------
        x : str
            The profile quantity for the x-axis.
        y : str
            The profile quantity for the y-axis.
        model_numbers : list[int] | np.ndarray, optional
            The model numbers to plot. The default is [-1].
        profile_numbers : list[int] | np.ndarray, optional
            The profile numbers to plot. The default is [-1].
        fig : plt.Figure, optional
            The figure. The default is None.
        ax : Axes | None, optional
            The axes. The default is None. If None, a new figure is created.
        set_labels : bool, optional
            If true, add labels to the plot, by default False. The labels are the age of the star.

        Returns
        -------
        Tuple[plt.Figure, Axes]
            The figure and axes of the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()


        if model_numbers != [-1]:
            for i in model_numbers:
                label = self.log.profile_data(model_number=i).header_data['star_age'] if set_labels else None
                label = f'{label:.2e}' if label is not None else None
                self.profile_plot(x, y, model_number=i, ax=ax, label = label , **kwargs)

        elif profile_numbers != [-1]:
            for i in profile_numbers:
                label = self.log.profile_data(profile_number=i).header_data['star_age'] if set_labels else None
                label = f'{label:.2e}' if label is not None else None
                self.profile_plot(x, y, profile_number=i, ax=ax, label = label, **kwargs)

        else:
            self.profile_plot(x, y, ax=ax, set_label=set_labels, **kwargs)

        ax.set(xlabel=x, ylabel=y)
        return fig, ax
    
    @staticmethod
    def data_mask(x: np.ndarray, y: np.ndarray, filter_x: Callable | None = None, filter_y: Callable | None = None) -> np.ndarray:
        """Returns the data mask for the quantities x and y."""

        mask_x = np.ones_like(x, dtype=bool) if filter_x is None else filter_x(x)
        mask_y = np.ones_like(y, dtype=bool) if filter_y is None else filter_y(y)

        return mask_x & mask_y
        

    def history_plot(self, x: str, y: str, fig: plt.Figure | None = None, ax: Axes | None = None, set_label: bool = False, filter_x: Callable | None = None, filter_y: Callable | None = None, **kwargs):
        """Plots the history data with (x, y) as the axes.
        
        Parameters
        ----------
        x : str
            The x-axis of the history data.
  
        y : str
            The y-axis of the history data.

        fig : plt.Figure, optional  
            The figure. The default is None.

        ax : Axes, optional
            The axes. The default is None. If None, a new figure is created.

        set_label : bool, optional
            If true, add label to plot, by default False

        filter_x : Callable | None, optional
            A function that filters the x-values. The default is None.
        
        filter_y : Callable | None, optional
            A function that filters the y-values. The default is None.

        **kwargs : dict
            Keyword arguments for `matplotlib.pyplot.plot`.

        Returns
        -------
        Tuple[plt.Figure, Axes]
            The figure and axes of the plot.

        Examples
        --------
        >>> sim.history_plot('star_age', 'Teff', filter_x=lambda x: x < 1e7)

        """

        if ax is None:
            fig, ax = plt.subplots()

        if set_label:
            kwargs["label"] = self.sim

        x_values = self.history.data(x)
        y_values = self.history.data(y)
        mask = Simulation.data_mask(x_values, y_values, filter_x, filter_y)

        ax.plot(x_values[mask], y_values[mask], **kwargs)

        ax.set(xlabel=x, ylabel=y)

        return fig, ax

    def relative_difference_of_two_profiles_plot(
        self,
        x: str,
        y: str,
        profile_number_reference: int = -1,
        profile_number_compare: int = -1,
        model_number_reference: int = -1,
        model_number_compare: int = -1,
        fig: plt.Figure | None = None,
        ax: Axes | None = None,
        set_label: bool = False,
        **kwargs
    ):
        """Plots the relative difference of two profiles using interpolation.

        Parameters
        ----------
        x : str
            The profile quantity for the x-axis.
        y : str
            The profile quantity for the y-axis.
        profile_number_reference : int, optional
            The reference profile number. The default is -1.
        profile_number_compare : int, optional
            The profile number that is compared to the reference profile. The default is -1.
        model_number_reference : int, optional
            The reference model number. The default is -1.
        model_number_compare : int, optional
            The model number that is compared to the reference model. The default is -1.
        fig : plt.Figure | None, optional
            The figure. The default is None.
        ax : Axes | None, optional
            The axes. The default is None. If None, a new figure is created.
        set_label : bool, optional
            If true, add label to plot, by default False

        Returns
        -------
        Tuple[plt.Figure, Axes]
            The figure and axes of the plot.
        """

        if ax is None:
            fig, ax = plt.subplots()

        if set_label:
            kwargs["label"] = self.sim

        # TODO: Add a label option to show the different ages of the profiles

        x_values, y_values = self.get_relative_difference(
            x,
            y,
            profile_number_reference=profile_number_reference,
            profile_number_compare=profile_number_compare,
            model_number_reference=model_number_reference,
            model_number_compare=model_number_compare,
            **kwargs,
        )

        ax.plot(x_values, y_values, **kwargs)

        ax.set(xlabel=x, ylabel=y)

        return fig, ax

    def mean_profile_sequence_plot(
        self,
        x: str,
        y: str,
        q0: float = 0.0,
        q1: float = 1.0,
        fig: plt.Figure | None = None,
        ax: Axes | None = None,
        set_labels: bool = False,
        model_numbers: list[int] | np.ndarray | None = None,
        profile_numbers: list[int] | np.ndarray | None = None,
        **kwargs
    ):
        """Plots a sequence of mean profile values with (x, y) as the axes."""

        if ax is None:
            fig, ax = plt.subplots()

        # x is often model_number, profile_number, or star_age
        # We need to deal with all of these cases
        # TODO: Add a more general method that looks for the header_data or a corresponding history file

        if "star_age" in x:
            self._create_profile_header_df(x)
            # in the Pandas.DataFrame, select the x values that are at the model numbers
            if model_numbers is not None:
                # if -1 is in model_numbers, replace it with the last model number
                model_numbers = np.array(model_numbers)
                model_numbers[model_numbers == -1] = self.log.model_numbers[-1]
                model_numbers = model_numbers.tolist()
                print(model_numbers)
                x_vals = self.profile_header_df.query('model_number == @model_numbers')['star_age']
                print(x_vals)
            else:
                # TODO: Add a method for profile_numbers
                raise ValueError("If x is 'star_age', then model_numbers must be specified.")

        elif x == "model_number":
            x_vals = model_numbers

        elif x == "profile_number":
            x_vals = profile_numbers

        else:
            x_vals = self.get_mean_profile_data_sequence(
                x,
                q0=q0,
                q1=q1,
                model_numbers=model_numbers,
                profile_numbers=profile_numbers,
                **kwargs,
            )        

        y_vals = self.get_mean_profile_data_sequence(
                y,
                q0=q0,
                q1=q1,
                model_numbers = model_numbers,
                profile_numbers = profile_numbers,
                **kwargs,
            )

        ax.plot(x_vals, y_vals, **kwargs)

        return fig, ax
