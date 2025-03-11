# class for anything related to after the simulation has been run
# for example, analyzing, plotting, saving, etc.
from functools import lru_cache
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import os
import numpy as np
import pandas as pd
from typing import Callable, Tuple
from mesa_helper.Simulation import Simulation
from mesa_helper.utils import sort_list_by_variable


# Todo: add a method to remove simulations from the class
class SimulationSeries:
    """Class for anything related to a series of simulations after they finished. For example, analyzing, plotting, saving, etc."""

    def __init__(
        self,
        series_dir: str,
        delete_horribly_failed_simulations = False,
        key_to_sort: str | None = None,
        **kwargs,
    ) -> None:
        """Initializes the Simulation object.
        Parameters
        ----------
        suite : str, optional
            The simulation suite. The default is ''.
        check_age_convergence : bool, optional
            If True, then the simulations that do not converge to the final age are removed. The default is True.
        key_to_sort : str, optional
            The key to sort the simulations by. If none, then sorts for the first free parameter. The default is None.
        **kwargs : dict
            Keyword arguments for `self.remove_non_converged_simulations`. For example, `final_age` can be specified.
        """
        self.verbose = kwargs.get("verbose", False)

        self.series_dir = series_dir

        # get the log directories while ignoring hidden directories
        # TODO: Make this more robust
        self.log_dirs = [
            log_dir
            for log_dir in os.listdir(self.series_dir)
            if not log_dir.startswith(".")
        ]

        # sort the log directories
        self.log_dirs = sort_list_by_variable(self.log_dirs, key_to_sort)

        # import sim results
        # first, delete horribly failed simulations
        (
            self.delete_horribly_failed_simulations()
            if delete_horribly_failed_simulations
            else None
        )

        # then, initialize the mesa logs and histories
        self._init_Simulation()

        self.n_simulations = len(self.simulations)

        # add the simulation parameters to self.results
        self.results = pd.DataFrame({"log_dir": self.log_dirs})

    # create a __str__ method that returns the name of the suite, or the name of the simulation if there is no suite
    def __str__(self):
        return self.series_dir

    def _init_Simulation(self) -> None:
        """Initialzes `Simulation` objects for each log directory in the simulation."""
        self.simulations = {}

        for log_dir in self.log_dirs:
            
            if self.verbose:
                print("_init_Simulation: parent_dir = ", self.series_dir)
                print("_init_Simulation: simulation_dir = ", log_dir)

            self.simulations[log_dir] = Simulation(simulation_dir = log_dir, parent_dir=self.series_dir)

    @staticmethod
    def extract_value(string, free_param: str):
        """Extracts the value of `free_param` from `string`."""
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

    def add_params(self, free_params):
        """Add value of `free_params` to the results DataFrame."""
        # make free_params a list if it is not already
        if type(free_params) != list:
            free_params = [free_params]

        out = {}
        for free_param in free_params:
            out[free_param] = [
                self.extract_value(log_dir, free_param)
                for log_dir in self.results["log_dir"]
            ]

        self.results = pd.concat([self.results, pd.DataFrame(out)], axis=1)

    # ------------------------------ #
    # --- Simulation Convergence --- #
    # ------------------------------ #

    @staticmethod
    def _is_profile_index_valid(path_to_profile_index):
        """Returns True if profiles.index contains more than 2 lines."""
        with open(path_to_profile_index, "r") as f:
            lines = f.readlines()
        return len(lines) > 2

    def delete_horribly_failed_simulations(self):
        """Deletes all simulations that have a profiles.index file with less than 2 lines."""

        for log_dir in self.log_dirs:
            path_to_profile_index = os.path.join(
                self.series_dir, log_dir, "profiles.index"
            )
            if not self._is_profile_index_valid(path_to_profile_index):
                os.system(f"rm -r {os.path.join(self.series_dir, log_dir)}")
                self.log_dirs.remove(log_dir)

    def remove(self, log_dir):
        """Removes the simulation with `log_dir` from the SimulationSeries."""

        self.results = self.results[self.results["log_dir"] != log_dir]
        del self.simulations[log_dir]
        del self.log_dirs[self.log_dirs.index(log_dir)]

    def apply_filter(
        self,
        quantity: str,
        value: float | int,
        model_number: int = -1,
        relative_tolerance: float = 1e-3,
    ) -> None:
        """Filters the SimulationSeries based on a condition."""

        for log_dir in self.log_dirs:
            sim = self.simulations[log_dir]
            fulfils_criterion: bool = sim.check_value(
                quantity, value, model_number, relative_tolerance
            )

            if not fulfils_criterion:
                (
                    print(f"Removing {log_dir} from the SimulationSeries.")
                    if self.verbose
                    else None
                )
                self.remove(log_dir)

    # ------------------------------ #
    # ----- Simulation Results ----- #
    # ------------------------------ #

    def merge_results(self, dfs: list[pd.DataFrame]) -> None:
        """Merges the results of the simulations in `dfs` with the results of the SimulationSeries."""

        combined_df = pd.concat(dfs).groupby("log_dir", as_index=False).first()
        self.results = pd.merge(self.results, combined_df, on="log_dir", how="left")

    def add_history_data(
        self,
        history_keys: str | list[str],
        condition: str = "model_number",
        value: int | float = -1,
        key_names: str | list[str] | None = None
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

        if isinstance(history_keys, str):
            history_keys = [history_keys]

        if key_names is None:
            key_names = history_keys
        elif isinstance(key_names, str) and len(history_keys) == 1:
            key_names = [key_names]
        elif len(key_names) != len(history_keys):
            raise ValueError("key_names must have the same length as history_keys.")
        
        # remove the history key if it is already in the results
        # also remove the key name if it is already in the results
        for key_name, history_key in zip(key_names, history_keys):
            if key_name in self.results.columns:
                history_keys.remove(history_key)
                key_names.remove(key_name)

        print("history_keys = ", history_keys) if self.verbose else None

        for history_key in history_keys:
            [
                self.simulations[log_dir].add_history_data(history_key, condition, value, key_name)
                for log_dir in self.log_dirs
            ]
            dfs = [self.simulations[log_dir].results for log_dir in self.log_dirs]
            filtered_dfs = [
                df[
                    [
                        col
                        for col in df.columns
                        if col not in self.results.columns or col == "log_dir"
                    ]
                ]
                for df in dfs
            ]
            print("dfs = ", filtered_dfs) if self.verbose else None
            self.merge_results(filtered_dfs)

    # TODO: Make the function reevaluate if inputs like the mass unit are changed
    @lru_cache
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
    ):
        """Computes the integrated quantity from q0 to q1 and adds it to `self.results`."""
        if name is None:
            name = kind + "_" + quantity

        [
            self.simulations[log_dir].add_profile_data(
                quantity=quantity,
                q0=q0,
                q1=q1,
                profile_number=profile_number,
                kind=kind,
                name=name,
                mass_unit=mass_unit,
            )
            for log_dir in self.log_dirs
        ]
        dfs = [self.simulations[log_dir].results for log_dir in self.log_dirs]
        filtered_dfs = [
            df[
                [
                    col
                    for col in df.columns
                    if col not in self.results.columns or col == "log_dir"
                ]
            ]
            for df in dfs
        ]
        self.merge_results(filtered_dfs)

    @lru_cache
    def add_profile_data_at_condition(
        self,
        quantity: str,
        condition: str,
        value: float,
        profile_number: int = -1,
        name: str | None = None,
    ) -> None:
        """Adds `quantity` to `self.results` where `condition` is closest to `value` of the specified `profile_number`.

        Parameters
        ----------
        quantity : str
            The quantity to add to `self.results`.
        condition : str
            The condition that should be closest to `value`.
        value : float
            The value that the condition should be closest to.
        profile_number : int, optional
            The profile number. The default is -1.
        name : str, optional
            The name of the quantity in `self.results`. The default is None.
        """

        if name is None:
            name = f"{quantity}_at_{condition}_{value}"

        [
            self.simulations[log_dir].add_profile_data_at_condition(
                quantity, condition, value, profile_number, name
            )
            for log_dir in self.log_dirs
        ]
        dfs = [self.simulations[log_dir].results for log_dir in self.log_dirs]
        filtered_dfs = [
            df[
                [
                    col
                    for col in df.columns
                    if col not in self.results.columns or col == "log_dir"
                ]
            ]
            for df in dfs
        ]
        self.merge_results(filtered_dfs)

    def get_relative_difference_of_two_simulations(
        self,
        x: str,
        y: str,
        log_key_reference: str,
        log_key_compare: str,
        profile_number_reference: int = -1,
        profile_number_compare: int = -1,
        model_number_reference: int = -1,
        model_number_compare: int = -1,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns the relative difference of two simulations at the same profile number.

        Parameters
        ----------
        x : str
            The profile quantity on the x-axis.
        y : str
            The profile quantity on the y-axis.
        log_key_reference : str
            The reference log key.
        log_key_compare : str
            The log key that is compared to the reference log.
        profile_number_reference : int, optional
            The reference profile number. The default is -1.
        profile_number_compare : int, optional
            The profile number that is compared to the reference profile. The default is -1.
        model_number_reference : int, optional
            The reference model number. The default is -1.
        model_number_compare : int, optional
            The model number that is compared to the reference model. The default is -1.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            First element is a numpy array of x values, second element is the relative difference of the two simulations at these x values.
        """

        # reference simulation
        sim_reference = self.simulations[log_key_reference]
        f_reference = sim_reference.interpolate_profile_data(
            x,
            y,
            model_number=model_number_reference,
            profile_number=profile_number_reference,
            **kwargs,
        )
        x_min_reference, x_max_reference = f_reference.x.min(), f_reference.x.max()

        # compare simulation
        sim_compare = self.simulations[log_key_compare]
        f_compare = sim_compare.interpolate_profile_data(
            x,
            y,
            model_number=model_number_compare,
            profile_number=profile_number_compare,
            **kwargs,
        )
        x_min_compare, x_max_compare = f_compare.x.min(), f_compare.x.max()

        # find the common x values
        x_min = max(x_min_reference, x_min_compare)
        x_max = min(x_max_reference, x_max_compare)

        n_bins = kwargs.get("n_bins", 1000)
        x_scale = kwargs.get("x_scale", "linear")
        if x_scale == "linear":
            x_values = np.linspace(x_min, x_max, n_bins)
        elif x_scale == "log":
            x_values = np.logspace(np.log10(x_min), np.log10(x_max), n_bins)

        y_reference = f_reference(x_values)
        y_compare = f_compare(x_values)

        relative_difference = (y_compare - y_reference) / y_reference

        return x_values, relative_difference

    # ------------------------------ #
    # ------- Export Results ------- #
    # ------------------------------ #

    def export_history_data(
        self,
        columns: list[str],
        file_labeling: Callable[[str], str] | None = None,
        filters: Callable | list[Callable] | None = None,
        **kwargs,
    ) -> None:
        """Exports the history quantities in `columns` to a csv file for every simulation in the SimulationSeries instance.

        Parameters
        ----------
        columns : list[str]
            The columns to be exported.
        file_labeling : Callable[[str], str] | None, optional
            A function that labels the output files. The input of the function is the simulation key. If none, writes the files into the current directory with `<simulation name>.csv`. The default is None.
        **kwargs : dict
            Keyword arguments for `pd.DataFrame.to_csv`.
        """

        if file_labeling is None:
            file_labeling = lambda x: x

        for sim in self.simulations.values():
            filename = file_labeling(sim.sim) + ".csv"
            sim.export_history_data(columns=columns, filename=filename, filters = filters, **kwargs)

    def export_profile_data(
        self,
        columns: list[str],
        file_labeling: Callable[[str], str] | None = None,
        method="index",
        model_number: int = -1,
        profile_number: int = -1,
        condition: str | None = None,
        value: int | float | None = None,
        **kwargs,
    ) -> None:
        """Exports the profile quantities in `columns` to a csv file for every simulation in the SimulationSeries instance.

        Parameters
        ----------
        columns : list[str]
            The columns to be exported.
        file_labeling : Callable[[str], str] | None, optional
            A function that labels the file. The default is None. The input of the function is the simulation key.
        method : str, optional
            The method to export the profile data. The default is 'index'.
        model_number : int, optional
            The model number. The default is -1.
        profile_number : int, optional
            The profile number. The default is -1.
        condition : str | None, optional
            The condition that should be closest to `value`. The default is None.
        value : int | float | None, optional
            The value that the condition should be closest to. The default is None.
        """
        if file_labeling is None:
            file_labeling = lambda x: x

        for sim in self.simulations.values():
            filename = file_labeling(sim.sim) + ".csv"
            sim.export_profile_data(
                columns=columns,
                filename=filename,
                method=method,
                model_number=model_number,
                profile_number=profile_number,
                condition=condition,
                value = value,
                **kwargs
            )

    # ------------------------------ #
    # -------- Plot Results -------- #
    # ------------------------------ #

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
        """Plots the profile data with (x, y) as the axes.

        Parameters
        ----------
        x : str
            The profile quantity for the x-axis.
        y : str
            The profile quantity for the y-axis.
        model_number : int, optional
            The model number of the series profiles. The default is -1.
        profile_number : int, optional
            The profile number of the series profiles. The default is -1.
        fig : plt.Figure | None, optional
            The figure. The default is None.
        ax : Axes | None, optional
            The axes. The default is None.
        set_label : bool, optional
            If True, then the label is set. The default is False.

        Returns
        -------
        Tuple[plt.Figure, Axes]
            The figure and the axes.
        """
        if ax is None:
            fig, ax = plt.subplots()
    
        for log_key, sim in self.simulations.items():
            sim.profile_plot(
                x,
                y,
                model_number=model_number,
                profile_number=profile_number,
                fig=fig,
                ax=ax,
                set_label=set_label,
                **kwargs,
            )

        return fig, ax

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

        """
        if ax is None:
            fig, ax = plt.subplots()

        for log_key, sim in self.simulations.items():
            sim.history_plot(x, y, fig=fig, ax=ax, set_label=set_label, filter_x=filter_x, filter_y=filter_y, **kwargs)

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
        filter_x: Callable | list[Callable] | None = None,
        filter_y: Callable | list[Callable] | None = None,
        **kwargs,
    ):
        """Plots y_numerator / y_denominator as a function of x for the history data.

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
        filter_x : Callable | list[Callable] | None, optional
            A function that filters the x-values. The default is None.
        filter_y : Callable | list[Callable] | None, optional
            A function that filters the y-values. The default is None.

        """

        if ax is None:
            fig, ax = plt.subplots()

        for sim in self.simulations.values():
            sim.history_composition_plot(
                x,
                y,
                function_x=function_x,
                function_y=function_y,
                fig=fig,
                ax=ax,
                set_label=set_label,
                filter_x=filter_x,
                filter_y=filter_y,
                **kwargs,
            )

        return fig, ax
    
    def history_ratio_plot(self, x: str, y_numerator: str, y_denominator: str, fig: plt.Figure | None = None, ax: Axes | None = None, set_label: bool = False, filter_x: Callable | None = None, filter_y_numerator: Callable | None = None, filter_y_denominator: Callable | None = None, **kwargs):
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
        filter_x : Callable | None, optional
            A function that filters the x-values. The default is None.
        filter_y_numerator : Callable | None, optional
            A function that filters the y_numerator-values. The default is None.
        filter_y_denominator : Callable | None, optional
            A function that filters the y_denominator-values. The default is None.
        """

        if ax is None:
            fig, ax = plt.subplots()

        for sim in self.simulations.values():
            sim.history_ratio_plot(x, y_numerator, y_denominator, fig=fig, ax=ax, set_label=set_label, filter_x=filter_x, filter_y_numerator=filter_y_numerator, filter_y_denominator=filter_y_denominator, **kwargs)

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

        if ax is None:
            fig, ax = plt.subplots()

        for sim in self.simulations.values():
            sim.profile_composition_plot(
                x = x,
                y = y,
                model_number = model_number,
                profile_number = profile_number,
                function_x = function_x,
                function_y = function_y,
                fig = fig,
                ax = ax,
                set_label = set_label,
                set_axes_labels = set_axes_labels,
                filter_x = filter_x,
                filter_y = filter_y,
                **kwargs,
            )

        return fig, ax

    def relative_difference_of_two_simulations_plot(
        self,
        x: str,
        y: str,
        log_key_reference: str,
        log_key_compare: str,
        profile_number_reference: int = -1,
        profile_number_compare: int = -1,
        model_number_reference: int = -1,
        model_number_compare: int = -1,
        fig: plt.Figure | None = None,
        ax: Axes | None = None,
        **kwargs
    ) -> Tuple[plt.Figure, Axes]:
        """Plots the relative difference of two simulations at the same profile number.

        Parameters
        ----------
        x : str
            The profile quantity on the x-axis.
        y : str
            The profile quantity for which the relative difference is calculated.
        log_key_reference : str
            The reference log key.
        log_key_compare : str
            The log key that is compared to the reference log.
        profile_number_reference : int, optional
            The reference profile number. The default is -1.
        profile_number_compare : int, optional
            The profile number that is compared to the reference profile. The default
        model_number_reference : int, optional
            The reference model number. The default is -1.
        model_number_compare : int, optional
            The model number that is compared to the reference model. The default is -1.
        fig : plt.Figure | None, optional
            The figure. The default is None.
        ax : Axes | None, optional
            The axes. The default is None. If None, a new figure is created.

        Returns
        -------
        Tuple[plt.Figure, Axes]
            The figure and axes of the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()

        x_data, y_data = self.get_relative_difference_of_two_simulations(
            x,
            y,
            log_key_reference,
            log_key_compare,
            profile_number_reference,
            profile_number_compare,
            model_number_reference,
            model_number_compare,
            **kwargs
            )

        ax.plot(x_data, y_data, **kwargs)
        ax.set_xlabel(x)
        ax.set_ylabel(f'Relative Difference of {y}')

        return fig, ax


    # Todo: update to new method in Simulation
    def mean_profile_sequence_plot(
        self,
        x: str,
        y: str,
        q0: float = 0.0,
        q1: float = 1.0,
        fig: plt.Figure | None = None,
        ax: Axes | None = None,
        model_numbers: list[int] | np.ndarray | None = None,
        profile_numbers: list[int] | np.ndarray | None = None,
        **kwargs
    ):
        """Plots a sequence of mean profile values with (x, y) as the axes."""

        if ax is None:
            fig, ax = plt.subplots()

        for log_key, sim in self.simulations.items():
            sim.mean_profile_sequence_plot(x, y, q0, q1, fig=fig, ax=ax, model_numbers=model_numbers, profile_numbers=profile_numbers, **kwargs)
        
        return fig, ax