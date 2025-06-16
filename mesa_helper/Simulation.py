# class for anything related to after the simulation has been run
# for example, analyzing, plotting, saving, etc.
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import os
import numpy as np
import mesa_reader as mr
import pandas as pd
from typing import Callable, Tuple
from mesa_helper.astrophys import (
    M_Jup_in_g,
    M_Sol_in_g,
    M_Earth_in_g,
    _compute_mean,
    _integrate,
)
from mesa_helper.utils import single_data_mask, multiple_data_mask, extract_expression
from functools import lru_cache


# ? Should I have just one path as an input that leads directly to the simulation directory?
class Simulation:
    """Class for anything related to a single simulation that has been run. For example, analyzing, plotting, saving, etc."""

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
        # test that verbose is a boolean
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean.")
        self.verbose = verbose

        # parent directory of the simulation
        if not isinstance(parent_dir, str):
            raise TypeError("parent_dir must be a string.")
        self.parent_dir = parent_dir

        # test that simulation_dir is a string
        if not isinstance(simulation_dir, str):
            raise TypeError("simulation_dir must be a string.")
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
    
    def is_converged(self, keys: str | list[str], function: Callable | None = None, filter: Callable | list[Callable] | None = None) -> bool:
        data, mask = self._composite_data(
            keys = keys,
            function = function,
            filter = filter,
            kind = "history"
        )
        return any(data[mask])

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
    ) -> np.float64 | int:
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
            # if we use the pythonic way of counting model numbers from the end
            # then we need to convert the negative value to a positive one
            if value < 0:
                value = self.history.model_number[value]
            return self.history.data_at_model_number(quantity, value)

        if condition == "profile_number":
            # see model_number
            if value < 0:
                value = self.log.profile_numbers[value]
            i_m = self.log.model_with_profile_number(value)
            return self.history.data_at_model_number(quantity, i_m)

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
        key_names: str | list[str] | None = None,
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
        key_names : str | list[str] | None, optional
            The names of the results columns. If None, then the history keys are used. The default is None.
        """

        history_keys = [history_keys] if isinstance(history_keys, str) else history_keys

        if key_names is None:
            key_names = history_keys
        elif isinstance(key_names, str) and len(history_keys) == 1:
            key_names = [key_names]
        elif len(key_names) != len(history_keys):
            raise ValueError("key_names must have the same length as history_keys.")

        for key_name, history_key in zip(key_names, history_keys):
            out = [
                self.get_history_data_at_condition(
                    quantity=history_key, condition=condition, value=value
                )
            ]
            self.results[key_name] = out

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

    def integrate(
        self,
        keys: str | list,
        model_number: int = -1,
        profile_number: int = -1,
        unit: str | float | None = None,
        function_x: Callable | None = None,
        function_y: Callable | None = None,
        filter_x: Callable | list[Callable] | None = None,
        filter_y: Callable | list[Callable] | None = None,
        kind: str = "profile",
        **kwargs,
    ) -> np.float64:
        """Returns the integral of `keys` or a function thereof.

        Parameters
        ----------
        keys : str | list
            The keys to compute the mean of.
        model_number : int, optional
            The model number. The default is -1.
        profile_number : int, optional
            The profile number. The default is -1.
        unit : str | float | None, optional
            The unit of the integrated quantity. The default is None. If None, then the unit is not changed. If a string, then the unit is converted to the specified unit. If a float, then the result is devided by the float.
        function_x : Callable | None, optional
            The function to apply to the variable that is integrated. The default is None.
        function_y : Callable | None, optional
            The function to apply to `keys`. The default is None.
        filter_x : Callable | list[Callable] | None, optional
            The filter for the variable that is integrated. The default is None.
        filter_y : Callable | list[Callable] | None, optional
            The filter for `keys`. The default is None.
        kind : str, optional
            The kind of data to use. The default is 'profile'. The other option is 'history'.

        Returns
        -------
        np.float64
            The integral of the quantity.

        Raises
        ------
        ValueError
            If `kind` is not 'profile' or 'history'.
        """

        dx_dict = {"profile": "dm", "history": "star_age"}

        # raise an error if kind is not 'profile' or 'history'
        if kind not in dx_dict.keys():
            raise ValueError("kind must be either 'profile' or 'history'.")

        dx, mask_dx = self._composite_data(
            keys=dx_dict[kind],
            function=function_x,
            filter=filter_x,
            kind=kind,
            model_number=model_number,
            profile_number=profile_number,
        )

        f, mask_f = self._composite_data(
            keys=keys,
            function=function_y,
            filter=filter_y,
            kind=kind,
            model_number=model_number,
            profile_number=profile_number,
        )

        mask = mask_f & mask_dx

        return _integrate(f[mask], dx[mask], unit=unit)

    def mean(
        self,
        keys: str | list[str],
        model_number: int = -1,
        profile_number: int = -1,
        function_x: Callable | None = None,
        function_y: Callable | None = None,
        filter_x: Callable | list[Callable] | None = None,
        filter_y: Callable | list[Callable] | None = None,
        kind: str = "profile",
        **kwargs,
    ) -> np.float64:
        """Returns the mean of `keys` or a function thereof.

        Parameters
        ----------
        keys : str | list
            The keys to compute the mean of.
        model_number : int, optional
            The model number. The default is -1.
        profile_number : int, optional
            The profile number. The default is -1.
        function_x : Callable | None, optional
            The function to apply to the variable that is integrated. The default is None.
        function_y : Callable | None, optional
            The function to apply to `keys`. The default is None.
        filter_x : Callable | list[Callable] | None, optional
            The filter for the variable that is integrated. The default is None.
        filter_y : Callable | list[Callable] | None, optional
            The filter for `keys`. The default is None.
        kind : str, optional
            The kind of data to use. The default is 'profile'. The other option is 'history'.

        Returns
        -------
        np.float64
            The mean of the quantity.

        Raises
        ------
        ValueError
            If `kind` is not 'profile' or 'history'.

        Examples
        --------
        >>> sim.mean('entropy')
        # returns the mean of the entropy for the last profile
        >>> sim.mean(['x', 'y'], function_y = lambda x, y: 1-x-y, profile_number=1)
        # returns the mean of Z = 1-X-Y for the first  profile
        """

        dx_dict = {"profile": "dm", "history": "star_age"}

        # raise an error if kind is not 'profile' or 'history'
        if kind not in dx_dict.keys():
            raise ValueError("kind must be either 'profile' or 'history'.")

        dx, mask_dx = self._composite_data(
            keys=dx_dict[kind],
            function=function_x,
            filter=filter_x,
            kind=kind,
            model_number=model_number,
            profile_number=profile_number,
        )
        print(f"mean: dx: {dx[mask_dx]}") if self.verbose else None

        f, mask_f = self._composite_data(
            keys=keys,
            function=function_y,
            filter=filter_y,
            kind=kind,
            model_number=model_number,
            profile_number=profile_number,
        )

        print(f"mean: f: {f[mask_f]}") if self.verbose else None

        mask = mask_f & mask_dx

        return _compute_mean(f[mask], dx[mask])

    def add_profile_data(
        self,
        keys: str | list,
        model_number: int = -1,
        profile_number: int = -1,
        kind: str | None = "integrate",
        function_x: Callable | None = None,
        function_y: Callable | None = None,
        filter_x: Callable | list[Callable] | None = None,
        filter_y: Callable | list[Callable] | None = None,
        name: str | None = None,
        unit: str | float | None = None,
        **kwargs,
    ) -> None:
        """Adds the processed profile keys to `self.results`.

        Parameters
        ----------
        keys : str | list
            The keys that are input to the function.
        model_number : int, optional
            The model number. The default is -1.
        profile_number : int, optional
            The profile number. The default is -1.
        kind : str, optional
            The kind of function to apply to the keys. If 'integrate', the keys are integrated. If 'mean', the keys are averaged. If None, then function_y is applied directly to the keys. The default is 'integrate'.
        function_x : Callable | None, optional
            The function to apply to the variable that is either integrated or used to compute the mean. If kind is neither 'integrate' nor 'mean', then function_x is not used. The default is None.
        function_y : Callable | str | None, optional
            The function to apply to the keys. The default is None.
        filter_x : Callable | list[Callable] | None, optional
            The filter for the variable that is is either integrated or used to compute the mean. If kind is neither 'integrate' nor 'mean', then filter_x is not used. The default is None.
        filter_y : Callable | list[Callable] | None, optional
            The filter for the keys. The default is None.
        name : str | None, optional
            The name of the results column. The default is None.
        unit : str | float | None, optional
            The unit of the integrated quantity. The default is None. If None, then the unit is not changed. If a string, then the unit is converted to the specified unit. If a float, then the result is devided by the float. If function is not 'integrate', then the unit is not used.
        **kwargs : dict
            Keyword arguments for the function.
        """

        # * tests

        # kind must be either 'integrate', 'mean', or None
        if kind not in ["integrate", "mean", None]:
            raise ValueError("kind must be either 'integrate', 'mean', or None.")

        # if kind is not callable, then keys must be a string
        if isinstance(kind, str) and not isinstance(keys, str):
            raise ValueError("If kind is not a function, then keys must be a string.")

        # * name of the results column
        if name is None and isinstance(kind, str):
            result_key = kind + "_" + keys

        # if function is Callable, i.e., a function, then name must be specified
        elif name is None and isinstance(function_y, Callable):
            result_key = extract_expression(function_y)

        else:
            result_key = name

        if kind == "integrate":
            print(f"add_profile_data: integrate {keys}") if self.verbose else None
            value = self.integrate(
                keys=keys,
                model_number=model_number,
                profile_number=profile_number,
                unit=unit,
                function_x=function_x,
                function_y=function_y,
                filter_x=filter_x,
                filter_y=filter_y,
                kind="profile",
                **kwargs,
            )
        elif kind == "mean":
            print(f"add_profile_data: mean {keys}") if self.verbose else None
            value = self.mean(
                keys=keys,
                model_number=model_number,
                profile_number=profile_number,
                function_x=function_x,
                function_y=function_y,
                filter_x=filter_x,
                filter_y=filter_y,
                kind="profile",
                **kwargs,
            )

        elif isinstance(kind, Callable):
            print(f"add_profile_data: function {keys}") if self.verbose else None
            value = self._composite_data(
                keys=keys,
                function=function_y,
                filter=filter_y,
                kind="profile",
                model_number=model_number,
                profile_number=profile_number,
            )

        else:
            raise ValueError("kind must 'integrated', 'mean', or a function.")

        out = [value]

        print(f"add_profile_data: {result_key} = {out}") if self.verbose else None
        self.results[result_key] = out
        print("add_profile_data finished") if self.verbose else None

    def get_profile_data_at_condition(
        self,
        quantity: str,
        condition: str,
        value: float | int,
        profile_number: int = -1,
        **kwargs,
    ) -> np.float64 | int:
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
    ) -> np.float64 | int:
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
        keys: str | list,
        model_numbers: list[int] | np.ndarray | None = None,
        profile_numbers: list[int] | np.ndarray | None = None,
        unit: str | float | None = None,
        function_x: Callable | None = None,
        function_y: Callable | None = None,
        filter_x: Callable | list[Callable] | None = None,
        filter_y: Callable | list[Callable] | None = None,
        **kwargs,
    ):
        """Returns the integration of the profile data for `keys` for all profile numbers or model numbers specified.

        Parameters
        ----------
        keys : str | list
            The keys to compute the mean of.
        model_numbers : list[int] | np.ndarray | None, optional
            The model numbers. The default is None.
        profile_numbers : list[int] | np.ndarray | None, optional
            The profile numbers. The default is None.
        unit : str | float | None, optional
            The unit of the integrated quantity. The default is None. If None, then the unit is not changed. If a string, then the unit is converted to the specified unit. If a float, then the result is devided by the float.
        function_x : Callable | None, optional
            The function to apply to the variable that is integrated. The default is None.
        function_y : Callable | None, optional
            The function to apply to `keys`. The default is None.
        filter_x : Callable | list[Callable] | None, optional
            The filter for the variable that is integrated. The default is None.
        filter_y : Callable | list[Callable] | None, optional
            The filter for `keys`. The default is None.
        """

        def local_integrate(model_number: int = -1, profile_number: int = -1):
            return self.integrate(
                keys,
                model_number=model_number,
                profile_number=profile_number,
                unit=unit,
                function_x=function_x,
                function_y=function_y,
                filter_x=filter_x,
                filter_y=filter_y,
                **kwargs,
            )

        if profile_numbers == None and model_numbers == None:
            return [
                local_integrate(profile_number=i_p) for i_p in self.log.profile_numbers
            ]

        elif model_numbers != None:
            return [local_integrate(model_number=i_m) for i_m in model_numbers]

        elif profile_numbers != None:
            return [local_integrate(profile_number=i_p) for i_p in profile_numbers]

    def get_mean_profile_data_sequence(
        self,
        keys: str | list,
        model_numbers: list[int] | np.ndarray | None = None,
        profile_numbers: list[int] | np.ndarray | None = None,
        function_x: Callable | None = None,
        function_y: Callable | None = None,
        filter_x: Callable | list[Callable] | None = None,
        filter_y: Callable | list[Callable] | None = None,
        **kwargs,
    ):
        """Returns the mean of the profile data for `keys` for all profile numbers or model numbers specified.

        Parameters
        ----------
        keys : str | list
            The keys to compute the mean of.
        model_numbers : list[int] | np.ndarray | None, optional
            The model numbers. The default is None.
        profile_numbers : list[int] | np.ndarray | None, optional
            The profile numbers. The default is None.
        function_x : Callable | None, optional
            The function to apply to the variable that is integrated. The default is None.
        function_y : Callable | None, optional
            The function to apply to `keys`. The default is None.
        filter_x : Callable | list[Callable] | None, optional
            The filter for the variable that is integrated. The default is None.
        filter_y : Callable | list[Callable] | None, optional
            The filter for `keys`. The default is None.
        """

        def local_mean(model_number: int = -1, profile_number: int = -1):
            return self.mean(
                keys,
                model_number=model_number,
                profile_number=profile_number,
                function_x=function_x,
                function_y=function_y,
                filter_x=filter_x,
                filter_y=filter_y,
                **kwargs,
            )

        if profile_numbers == None and model_numbers == None:
            return [local_mean(profile_number=i_p) for i_p in self.log.profile_numbers]

        elif model_numbers != None:
            return [local_mean(model_number=i_m) for i_m in model_numbers]

        elif profile_numbers != None:
            return [local_mean(profile_number=i_p) for i_p in profile_numbers]

    def _interpolate_mesa_data(
        self,
        x: str,
        y: str,
        kind: str,
        model_number: list[int] | np.ndarray | None = None,
        profile_number: list[int] | np.ndarray | None = None,
        function_x: Callable | None = None,
        function_y: Callable | None = None,
        filter_x: Callable | None = None,
        filter_y: Callable | None = None,
        **kwargs,
    ) -> Callable:
        """Returns an interpolation function for the quantities (x,y).

        Note that this method retrieves the profile data via `mesa_reader.MesaProfileData.data` and then interpolates the data using `numpy.interp`.
        Hence, you can use the same 'log'-syntax as in `mesa_reader.MesaProfileData.data` for logarithmic interpolation.

        Parameters
        ----------
        x : str
            The x-axis of the data.
        y : str
            The y-axis of the data.
        kind : str
            The kind of data to use. Either 'profile' or 'history'.
        model_number : list[int] | np.ndarray | None, optional
            The model numbers. The default is None.
        profile_number : list[int] | np.ndarray | None, optional
            The profile numbers. The default is None.
        function_x : Callable | None, optional
            The function to apply to the x-axis data. The default is None.
        function_y : Callable | None, optional
            The function to apply to the y-axis data. The default is None.
        filter_x : Callable | None, optional
            The filter for the x-axis data. The default is None.
        filter_y : Callable | None, optional
            The filter for the y-axis data. The default is None.
        **kwargs : dict
            Keyword arguments for `numpy.interp`.

        """

        data_x, mask_x = self._composite_data(
            x,
            function_x,
            filter_x,
            kind=kind,
            model_number=model_number,
            profile_number=profile_number,
        )
        data_y, mask_y = self._composite_data(
            y,
            function_y,
            filter_y,
            kind=kind,
            model_number=model_number,
            profile_number=profile_number,
        )

        mask = mask_x & mask_y

        return lambda x: np.interp(x, data_x[mask], data_y[mask], **kwargs)

    @lru_cache
    def interpolate_profile_data(
        self,
        x: str,
        y: str,
        model_number: list[int] | np.ndarray | None = None,
        profile_number: list[int] | np.ndarray | None = None,
        function_x: Callable | None = None,
        function_y: Callable | None = None,
        filter_x: Callable | None = None,
        filter_y: Callable | None = None,
        **kwargs,
    ) -> Callable:
        """Returns an interpolation function for the quantities (x,y).

        Note that this method retrieves the profile data via `mesa_reader.MesaProfileData.data` and then interpolates the data using `numpy.interp`.
        Hence, you can use the same 'log'-syntax as in `mesa_reader.MesaProfileData.data` for logarithmic interpolation.

        Parameters
        ----------
        x : str
            The x-axis of the data.
        y : str
            The y-axis of the data.
        model_number : list[int] | np.ndarray | None, optional
            The model numbers. The default is None.
        profile_number : list[int] | np.ndarray | None, optional
            The profile numbers. The default is None.
        function_x : Callable | None, optional
            The function to apply to the x-axis data. The default is None.
        function_y : Callable | None, optional
            The function to apply to the y-axis data. The default is None.
        filter_x : Callable | None, optional
            The filter for the x-axis data. The default is None.
        filter_y : Callable | None, optional
            The filter for the y-axis data. The default is None.
        **kwargs : dict
            Keyword arguments for `numpy.interp`.

        """

        f = self._interpolate_mesa_data(
            x=x,
            y=y,
            kind="profile",
            model_number=model_number,
            profile_number=profile_number,
            function_x=function_x,
            function_y=function_y,
            filter_x=filter_x,
            filter_y=filter_y,
            **kwargs,
        )

        return f

    @lru_cache
    def interpolate_history_data(
        self,
        x: str,
        y: str,
        function_x: Callable | None = None,
        function_y: Callable | None = None,
        filter_x: Callable | None = None,
        filter_y: Callable | None = None,
        **kwargs,
    ) -> Callable:
        """Returns an interpolation function for the quantities (x,y).

        Note that this method retrieves the profile data via `mesa_reader.MesaProfileData.data` and then interpolates the data using `numpy.interp`.
        Hence, you can use the same 'log'-syntax as in `mesa_reader.MesaProfileData.data` for logarithmic interpolation.

        Parameters
        ----------
        x : str
            The x-axis of the data.
        y : str
            The y-axis of the data.
        function_x : Callable | None, optional
            The function to apply to the x-axis data. The default is None.
        function_y : Callable | None, optional
            The function to apply to the y-axis data. The default is None.
        filter_x : Callable | None, optional
            The filter for the x-axis data. The default is None.
        filter_y : Callable | None, optional
            The filter for the y-axis data. The default is None.
        **kwargs : dict
            Keyword arguments for `numpy.interp`.

        """

        f = self._interpolate_mesa_data(
            x=x,
            y=y,
            kind="history",
            function_x=function_x,
            function_y=function_y,
            filter_x=filter_x,
            filter_y=filter_y,
            **kwargs,
        )

        return f

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
            Keyword arguments for `numpy.interp`.
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

    def export_history_data(
        self,
        columns: list[str],
        filename: str = "history.csv",
        functions: Callable | list[Callable] | None = None,
        filters: Callable | list[Callable] | None = None,
        **kwargs,
    ) -> None:
        """Exports the quantities in `columns` to a csv file."""

        data = {}
        masks = {}
        filters = [None] * len(columns) if filters is None else filters
        functions = [None] * len(columns) if functions is None else functions

        for column, function, filter in zip(columns, functions, filters):
            data[column], masks[column] = self._composite_data(keys = column, function=function, filter = filter, kind = 'history')

        mask = np.all(list(masks.values()), axis = 0)
        data = {column: data[column][mask] for column in columns}
        df = pd.DataFrame(data)
    
        # add index = False to kwargs if not specified
        if kwargs.get("index") is None:
            kwargs["index"] = False

        df.to_csv(filename, **kwargs)

    def export_profile_data(
        self,
        columns: list[str],
        filename: str = "profile_data.csv",
        method: str = "index",
        model_number: int = -1,
        profile_number: int = -1,
        condition: str | None = None,
        value: int | float | None = None,
        **kwargs,
    ) -> None:
        """Exports the quantities in `columns` to a csv file.

        Parameters
        ----------
        columns : list[str]
            The columns to be exported.
        filename : str
            The filename of the csv file.
        method : str, optional
            The method to extract the profile data. The default is 'profile_number'. Available options are 'profile_number' and 'profile_header_condition'.
            For 'index', the profile data is extracted at the profile number specified by `profile_number` or `model_number`.
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
        if method == "index":
            df = pd.DataFrame(
                {
                    column: self.log.profile_data(
                        model_number=model_number, profile_number=profile_number
                    ).data(column)
                    for column in columns
                }
            )

        elif method == "profile_header_condition":

            # raise an error if condition or value is not specified
            if condition is None or value is None:
                raise ValueError(
                    "condition and value must be specified for method='profile_header_condition'."
                )

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
    def _composite_data(
        self,
        keys: str | list,
        function: Callable | None,
        filter: Callable | list[Callable] | None = None,
        kind: str = "history",
        model_number: int = -1,
        profile_number: int = -1,
    ) -> Tuple[np.ndarray, np.ndarray]:

        if kind == "history":
            mesa_data = self.history
        elif kind == "profile":
            mesa_data = self.log.profile_data(
                model_number=model_number, profile_number=profile_number
            )
        else:
            raise ValueError("kind must be either 'history' or 'profile'.")

        if isinstance(keys, str):

            print("_composite_data: key is a string") if self.verbose else None

            values = mesa_data.data(keys)
            mask = single_data_mask(values, filter)

            if function is not None:
                values = function(values)

        elif isinstance(keys, list):
            print("_composite_data: key is a list") if self.verbose else None

            values = [mesa_data.data(key) for key in keys]
            mask = multiple_data_mask(values, filter)

            if function is None:
                raise ValueError("function must be specified if keys is a list.")
            else:
                (
                    print(f"_composite_data: function is not None")
                    if self.verbose
                    else None
                )
                values = function(*values)

        return values, mask

    def composition_plot(
        self,
        x: str | list,
        y: str | list,
        kind: str,
        model_number: int = -1,
        profile_number: int = -1,
        function_x: Callable | None = None,
        function_y: Callable | None = None,
        fig: plt.Figure | None = None,
        ax: Axes | None = None,
        set_label: bool = False,
        set_axes_labels: bool = False,
        filter_x: Callable | list[Callable] | None = None,
        filter_y: Callable | list[Callable] | None = None,
        **kwargs,
    ) -> Tuple[plt.Figure, Axes]:
        """Plots function_y(*y) as a function of function_x(*x) for the  MesaData object defined by `kind`.

        Parameters
        ----------
        x : str | list
            The x-axis of the plot. If a list, then the list should contain the quantities for the x-axis, which are then combined using `function_x`.
        y : str | list
            The y-axis of the plot. If a list, then the list should contain the quantities for the y-axis, which are then combined using `function_y`.
        kind : str
            The kind of data. Either 'history' or 'profile'.
        model_number : int, optional
            The model number if using `kind = 'profile'`. The default is -1.
        profile_number : int, optional
            The profile number if using `kind = 'profile'`. The default is -1.
        function_x : Callable | None, optional
            A function that combines the x-values. It must take as many inputs as there are x values. The default is None.
        function_y : Callable | None, optional
            A function that combines the y-values. It must take as many inputs as there are y values. The default is None.
        fig : plt.Figure | None, optional
            The figure. The default is None.
        ax : Axes | None, optional
            The axes. The default is None. If None, a new figure is created.
        set_label : bool, optional
            If true, add label to plot, by default False
        set_axes_labels : bool, optional
            If true, tries to set the axis labels automatically, by default False
        filter_x : Callable | list[Callable] | None, optional
            A function that filters the x-values. The default is None.
        filter_y : Callable | list[Callable] | None, optional
            A function that filters the y-values. The default is None.
        **kwargs : dict
            Keyword arguments for `matplotlib.pyplot.plot`.

        """

        if ax is None:
            fig, ax = plt.subplots()

        if set_label:
            kwargs["label"] = self.sim

        print("composition_plot: get x data") if self.verbose else None
        data_x, mask_x = self._composite_data(
            x,
            function_x,
            filter_x,
            kind=kind,
            model_number=model_number,
            profile_number=profile_number,
        )
        (
            print(f"composition_plot: first five data_x entries: {data_x[:5]}")
            if self.verbose
            else None
        )
        (
            print(f"composition_plot: first five mask_x entries: {mask_x[:5]}")
            if self.verbose
            else None
        )

        print("composition_plot: get y data") if self.verbose else None
        data_y, mask_y = self._composite_data(
            y,
            function_y,
            filter_y,
            kind=kind,
            model_number=model_number,
            profile_number=profile_number,
        )
        (
            print(f"composition_plot: first five data_y entries: {data_y[:5]}")
            if self.verbose
            else None
        )
        (
            print(f"composition_plot: first five mask_y entries: {mask_y[:5]}")
            if self.verbose
            else None
        )

        mask = mask_x & mask_y
        (
            print(f"composition_plot: first five mask entries: {mask[:5]}")
            if self.verbose
            else None
        )

        ax.plot(data_x[mask], data_y[mask], **kwargs)

        if not set_axes_labels:
            return fig, ax

        if function_x is None:
            x_label = x
        else:
            x_label = extract_expression(function_x)

        if function_y is None:
            y_label = y
        else:
            y_label = extract_expression(function_y)

        ax.set(xlabel=x_label, ylabel=y_label)

        return fig, ax

    def profile_plot(
        self,
        x: str,
        y: str,
        model_number: int = -1,
        profile_number: int = -1,
        fig: plt.Figure | None = None,
        ax: Axes | None = None,
        filter_x: Callable | None = None,
        filter_y: Callable | None = None,
        set_label: bool = False,
        set_axes_labels: bool = False,
        **kwargs,
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
        set_axes_labels : bool, optional
            If true, tries to set the axis labels automatically, by default False
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
        """
        fig, ax = self.composition_plot(
            x,
            y,
            kind="profile",
            model_number=model_number,
            profile_number=profile_number,
            fig=fig,
            ax=ax,
            set_label=set_label,
            set_axes_labels=set_axes_labels,
            filter_x=filter_x,
            filter_y=filter_y,
            **kwargs,
        )

        return fig, ax

    def profile_composition_plot(
        self,
        x: str | list,
        y: str | list,
        model_number: int = -1,
        profile_number: int = -1,
        function_x: Callable | None = None,
        function_y: Callable | None = None,
        fig: plt.Figure | None = None,
        ax: Axes | None = None,
        set_label: bool = False,
        set_axes_labels: bool = False,
        filter_x: Callable | list[Callable] | None = None,
        filter_y: Callable | list[Callable] | None = None,
        **kwargs,
    ):
        """Plots function_y(*y) as a function of function_x(*x) for the profile data specified by the model number or profile number.

        Parameters
        ----------
        x : str | list
            The x-axis of the plot. If a list, then the list should contain the quantities for the x-axis, which are then combined using `function_x`.
        y : str | list
            The y-axis of the plot. If a list, then the list should contain the quantities for the y-axis, which are then combined using `function_y`.
        model_number : int, optional
            The model number of the profile. The default is -1.
        profile_number : int, optional
            The profile number. The default is -1.
        function_x : Callable | None, optional
            A function that combines the x-values. It must take as many inputs as there are x values. The default is None.
        function_y : Callable | None, optional
            A function that combines the y-values. It must take as many inputs as there are y values. The default is None.
        fig : plt.Figure | None, optional
            The figure. The default is None.
        ax : Axes | None, optional
            The axes. The default is None. If None, a new figure is created.
        set_label : bool, optional
            If true, add label to plot, by default False
        set_axes_labels : bool, optional
            If true, tries to set the axis labels automatically, by default False
        filter_x : Callable | list[Callable] | None, optional
            A function that filters the x-values. The default is None.
        filter_y : Callable | list[Callable] | None, optional
            A function that filters the y-values. The default is None.

        """

        fig, ax = self.composition_plot(
            x,
            y,
            kind="profile",
            model_number=model_number,
            profile_number=profile_number,
            function_x=function_x,
            function_y=function_y,
            fig=fig,
            ax=ax,
            set_label=set_label,
            set_axes_labels=set_axes_labels,
            filter_x=filter_x,
            filter_y=filter_y,
            **kwargs,
        )

        return fig, ax

    def profile_series_plot(
        self,
        x: str,
        y: str,
        model_numbers: list[int] | np.ndarray = [-1],
        profile_numbers: list[int] | np.ndarray = [-1],
        function_x: Callable | None = None,
        function_y: Callable | None = None,
        fig: plt.Figure | None = None,
        ax: Axes | None = None,
        set_labels: bool = False,
        set_axes_labels: bool = False,
        filter_x: Callable | list[Callable] | None = None,
        filter_y: Callable | list[Callable] | None = None,
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
        set_axes_labels : bool, optional
            If true, tries to set the axis labels automatically, by default False

        Returns
        -------
        Tuple[plt.Figure, Axes]
            The figure and axes of the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()

        if model_numbers != [-1]:
            for i in model_numbers:
                label = (
                    self.log.profile_data(model_number=i).header_data["star_age"]
                    if set_labels
                    else None
                )
                label = f"{label:.2e}" if label is not None else None
                self.profile_composition_plot(
                    x,
                    y,
                    model_number=i,
                    function_x=function_x,
                    function_y=function_y,
                    ax=ax,
                    label=label,
                    set_axes_labels=set_axes_labels,
                    filter_x=filter_x,
                    filter_y=filter_y,
                    **kwargs,
                )

        elif profile_numbers != [-1]:
            for i in profile_numbers:
                label = (
                    self.log.profile_data(profile_number=i).header_data["star_age"]
                    if set_labels
                    else None
                )
                label = f"{label:.2e}" if label is not None else None
                self.profile_composition_plot(
                    x,
                    y,
                    profile_number=i,
                    function_x=function_x,
                    function_y=function_y,
                    ax=ax,
                    label=label,
                    set_axes_labels=set_axes_labels,
                    filter_x=filter_x,
                    filter_y=filter_y,
                    **kwargs,
                )

        else:
            self.profile_composition_plot(
                x,
                y,
                function_x=function_x,
                function_y=function_y,
                ax=ax,
                set_label=set_labels,
                set_axes_labels=set_axes_labels,
                filter_x=filter_x,
                filter_y=filter_y,
                **kwargs,
            )

        return fig, ax

    def history_plot(
        self,
        x: str,
        y: str,
        fig: plt.Figure | None = None,
        ax: Axes | None = None,
        set_label: bool = False,
        set_axes_labels: bool = False,
        filter_x: Callable | None = None,
        filter_y: Callable | None = None,
        **kwargs,
    ):
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

        set_axes_labels : bool, optional
            If true, tries to set the axis labels automatically, by default False

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

        fig, ax = self.composition_plot(
            x,
            y,
            kind="history",
            fig=fig,
            ax=ax,
            set_label=set_label,
            set_axes_labels=set_axes_labels,
            filter_x=filter_x,
            filter_y=filter_y,
            **kwargs,
        )

        return fig, ax

    def history_composition_plot(
        self,
        x: str | list,
        y: str | list,
        function_x: Callable | None = None,
        function_y: Callable | None = None,
        fig: plt.Figure | None = None,
        ax: Axes | None = None,
        set_label: bool = False,
        set_axes_labels: bool = False,
        filter_x: Callable | list[Callable] | None = None,
        filter_y: Callable | list[Callable] | None = None,
        **kwargs,
    ):
        """Plots function_y(*y) as a function of function_x(*x) for the history data.

        Parameters
        ----------
        x : str | list
            The x-axis of the plot. If a list, then the list should contain the quantities for the x-axis, which are then combined using `function_x`.
        y : str | list
            The y-axis of the plot. If a list, then the list should contain the quantities for the y-axis, which are then combined using `function_y`.
        function_x : Callable | None, optional
            A function that combines the x-values. It must take as many inputs as there are x values. The default is None.
        function_y : Callable | None, optional
            A function that combines the y-values. It must take as many inputs as there are y values. The default is None.
        fig : plt.Figure | None, optional
            The figure. The default is None.
        ax : Axes | None, optional
            The axes. The default is None. If None, a new figure is created.
        set_label : bool, optional
            If true, add label to plot, by default False
        set_axes_labels : bool, optional
            If true, tries to set the axis labels automatically, by default False
        filter_x : Callable | list[Callable] | None, optional
            A function that filters the x-values. The default is None.
        filter_y : Callable | list[Callable] | None, optional
            A function that filters the y-values. The default is None.

        """

        fig, ax = self.composition_plot(
            x,
            y,
            kind="history",
            function_x=function_x,
            function_y=function_y,
            fig=fig,
            ax=ax,
            set_label=set_label,
            set_axes_labels=set_axes_labels,
            filter_x=filter_x,
            filter_y=filter_y,
            **kwargs,
        )

        return fig, ax

    def history_ratio_plot(
        self,
        x: str,
        y_numerator: str,
        y_denominator: str,
        fig: plt.Figure | None = None,
        ax: Axes | None = None,
        set_label: bool = False,
        set_axes_labels: bool = False,
        filter_x: Callable | None = None,
        filter_y_numerator: Callable | None = None,
        filter_y_denominator: Callable | None = None,
        **kwargs,
    ):
        """Plots y_numerator / y_denominator as a function of x for the history data.

        Parameters
        ----------
        x : str
            The x-axis of the history data.
        y_numerator : str
            The history quantatiy that's the numerator of the ratio.
        y_denominator : str
            The history quantatiy that's the denominator of the ratio.
        fig : plt.Figure | None, optional
            The figure. The default is None.
        ax : Axes | None, optional
            The axes. The default is None. If None, a new figure is created.
        set_label : bool, optional
            If true, add label to plot, by default False
        set_axes_labels : bool, optional
            If true, tries to set the axis labels automatically, by default False
        filter_x : Callable | None, optional
            A function that filters the x-values. The default is None.
        filter_y_numerator : Callable | None, optional
            A function that filters the y_numerator-values. The default is None.
        filter_y_denominator : Callable | None, optional
            A function that filters the y_denominator-values. The default is None.
        """

        if ax is None:
            fig, ax = plt.subplots()

        fig, ax = self.history_composition_plot(
            x,
            [y_numerator, y_denominator],
            function_y=lambda y_numerator, y_denominator: y_numerator / y_denominator,
            fig=fig,
            ax=ax,
            set_label=set_label,
            set_axes_labels=set_axes_labels,
            filter_x=filter_x,
            filter_y=[filter_y_numerator, filter_y_denominator],
            **kwargs,
        )

        # ax.set(xlabel=x, ylabel=f"{y_numerator} / {y_denominator}")

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
        **kwargs,
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
        x: str | list,
        y: str | list,
        function_dm: Callable | None = None,
        function_x: Callable | None = None,
        function_y: Callable | None = None,
        filter_dm: Callable | list[Callable] | None = None,
        filter_x: Callable | list[Callable] | None = None,
        filter_y: Callable | list[Callable] | None = None,
        fig: plt.Figure | None = None,
        ax: Axes | None = None,
        model_numbers: list[int] | np.ndarray | None = None,
        profile_numbers: list[int] | np.ndarray | None = None,
        **kwargs,
    ) -> Tuple[plt.Figure, Axes]:
        """Plots the mean profile data with (x, y) as the axes for a sequence of model numbers or profile numbers.

        Parameters
        ----------
        x : str | list
            The history quantity for the x-axis. If a list, then the list should contain the quantities for the x-axis, which are then combined using `function_x`.
        y : str | list
            The profile quantites for the y-axis. If a list, then the list should contain the quantities for the y-axis, which are then combined using `function_y`.
        function_dm : Callable | None, optional
            A function that is applied to the mass bin size. The default is None.
        function_x : Callable | None, optional
            A function that combines the x-values. It must take as many inputs as there are x values. The default is None.
        function_y : Callable | None, optional
            A function that combines the y-values. It must take as many inputs as there are y values. The default is None.
        filter_dm : Callable | list[Callable] | None, optional
            A function that filters the mass bin size. The default is None.
        filter_x : Callable | list[Callable] | None, optional
            A function that filters the x-values. The default is None.
        filter_y : Callable | list[Callable] | None, optional
            A function that filters the y-values. The default is None.
        fig : plt.Figure | None, optional
            The figure. The default is None.
        ax : Axes | None, optional
            The axes. The default is None. If None, a new figure is created.
        model_numbers : list[int] | np.ndarray | None, optional
            The model numbers to plot. The default is None.
        profile_numbers : list[int] | np.ndarray | None, optional
            The profile numbers to plot. The default is None.

        Returns
        -------
        Tuple[plt.Figure, Axes]
            The figure and axes of the plot.
        """

        if ax is None:
            fig, ax = plt.subplots()

        # x is often model_number, profile_number, or a history quantity

        if x == "model_number":
            x_vals = model_numbers

        elif x == "profile_number":
            x_vals = profile_numbers

        elif isinstance(x, str):
            self._create_profile_header_df(x)
            # in the Pandas.DataFrame, select the x values that are at the model numbers
            if model_numbers is not None:
                # if -1 is in model_numbers, replace it with the last model number
                model_numbers = np.array(model_numbers)
                model_numbers[model_numbers == -1] = self.log.model_numbers[-1]
                model_numbers = model_numbers.tolist()
                x_vals = self.profile_header_df.query("model_number == @model_numbers")[
                    x
                ]

            # TODO: Add a method for profile_numbers
            else:
                # if model_numbers is None, then we plot all model numbers
                model_numbers = self.log.model_numbers.tolist()
                x_vals = self.profile_header_df[x]

        else:
            x_vals = self.get_mean_profile_data_sequence(
                x,
                model_numbers=model_numbers,
                profile_numbers=profile_numbers,
                function_x=function_dm,
                function_y=function_x,
                filter_x=filter_dm,
                filter_y=filter_x,
                **kwargs,
            )

        y_vals = self.get_mean_profile_data_sequence(
            y,
            model_numbers=model_numbers,
            profile_numbers=profile_numbers,
            function_x=function_dm,
            function_y=function_y,
            filter_x=filter_dm,
            filter_y=filter_y,
            **kwargs,
        )

        ax.plot(x_vals, y_vals, **kwargs)
        ax.set(xlabel=x, ylabel=f"Mean {y}")

        return fig, ax

    # TODO: Add a profile sequence plot for different header conditions
