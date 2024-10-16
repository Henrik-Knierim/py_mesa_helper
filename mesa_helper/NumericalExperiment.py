from mesa_helper.Inlist import Inlist
from mesa_helper.Rn import Rn
import os

# TODO: Test this class
class NumericalExperiment(Inlist, Rn):
    """Modifies rn-files and inlists"""

    def __init__(self, inlist_name: str, rn_name: str, verbose: bool = False) -> None:
        """Initializes inlist and rn-file modifier.

        Parameters
        ----------
        inlist_name : str
            file name of the inlist
        rn_name : str
            file name of the rn-script
        """

        Inlist.__init__(self, inlist_name, verbose)
        Rn.__init__(self, rn_name, verbose)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore_inlist()
        self.restore_rn()

    def rename_save_mod_file(self, save_model_filename: str) -> None:
        """Renames the `save_model_filename` and adjusts the rn-file accordingly."""

        self.set_mod_name(save_model_filename)
        self.set_option("save_model_filename", save_model_filename)

    def evolve(
        self,
        do_restart: bool = False,
        photo: str | None = None,
        save_inlist: bool = False,
        **options
    ) -> None:
        """Runs the evolution of the model.

        Parameters
        ----------
        options : dict
            Options to be set in the inlist
        do_restart : bool, optional
            If True, restart the evolution, else start from scratch, by default False
        photo : str | None, optional
            The photo that should be used for the restart. If None, uses the latest photo, by default None
        save_inlist : bool, optional
            If True, saves the inlist, by default False
        """

        self.set_multiple_options(**options)

        # if `save_model_filename` is in the options, call `rename_save_mod_file`
        if "save_model_filename" in options:
            self.rename_save_mod_file(options["save_model_filename"])

        self.run(do_restart=do_restart, photo=photo)

        self.save_inlist() if save_inlist else None

    def evolve_with_logs_style(
        self,
        do_restart: bool = False,
        photo: str | None = None,
        save_inlist: bool = False,
        save_final_model: bool = False,
        save_final_photo: bool = False,
        logs_parent_dir: str = "LOGS",
        logs_style: str | list[str] | None = None,
        series_style: str | list[str] | None = None,
        logs_kwargs: dict = {},
        **options
    ) -> None:
        """Runs the evolution of the model with logs_style = 2.

        Parameters
        ----------
        options : dict
            Options to be set in the inlist
        do_restart : bool, optional
            If True, restart the evolution, else start from scratch, by default False
        photo : str | None, optional
            The photo that should be used for the restart. If None, uses the latest photo, by default None
        save_inlist : bool, optional
            If True, saves the inlist, by default False
        """
        logs_dir = Inlist.create_logs_path(logs_parent_dir = logs_parent_dir, logs_style = logs_style, series_style = series_style, **logs_kwargs)
        options["log_directory"] = logs_dir

        # try to get the save_model_filename and change the output path
        # ! Currently, this will produce an error if the save_model_filename neither in the options nor in the inlist
        # ! I need to modify the read_option method to return the default value if the key is not found
        save_model_filename = options.get("save_model_filename", self.read_option("save_model_filename"))
        options["save_model_filename"] = os.path.join(logs_dir, save_model_filename) if save_model_filename else None

        # save the final photo if requested
        if save_final_photo:
            options['save_photo_when_terminate'] = True
            options['filename_for_photo_when_terminate'] = logs_dir + '/final_photo'

        self.evolve(do_restart = do_restart, photo = photo, save_inlist = save_inlist, **options)
