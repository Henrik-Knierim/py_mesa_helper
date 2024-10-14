from mesa_helper.Inlist import Inlist
from mesa_helper.Rn import Rn

# TODO: Test this class
class NumericalExperiment(Inlist, Rn):
    """Modifies rn-files and inlists"""
    
    def __init__(self, inlist_name : str, rn_name : str, verbose: bool = False) -> None:
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