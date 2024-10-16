import os
class Rn:
    """Modifies rn-files"""

    def __init__(self, name : str, verbose: bool = False) -> None:
        """Initializes rn-file modifier.

        Parameters
        ----------
        name : str
            file name of the rn-script
        """
        self.verbose = verbose

        # name/path to rn script that should be modified
        self.name = name
        
        # save rn file to restore later
        with open(self.name, 'r') as file:
            self.original_rn_script = file.read()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        
        # restore original inlist
        self.restore_rn()

    def restore_rn(self) -> None:
        """Restores run script to original version."""
        with open(self.name, 'w') as file:
            file.write(self.original_rn_script)
        print("restored inlist to original version") if self.verbose else None

    def _change_mod_name(self, mod_file_name : str):
        with open(self.name, 'r') as file:
            
            # store content in list
            lines = file.readlines()
            
            key = 'do_one'

            for i, l in enumerate(lines):
                if key in l:
                    # split line at whitespaces
                    line_splitted = l.split()

                    is_key = line_splitted[0] == key
                    if is_key:
                        index_key = i
                        new_line = l.replace(
                            line_splitted[2],
                            mod_file_name
                        )
                        break
                    
            lines[index_key] = new_line

        return lines
    
    def set_mod_name(self, mod_file_name : str):
        """Changes mod-file name"""
        # get modified lines
        lines = self._change_mod_name(mod_file_name)

        # write new lines into the inlist
        with open(self.name, 'w') as file:
            file.writelines(lines)

        print(f"Set mod-file name to {mod_file_name}") if self.verbose else None

    def run(self, do_restart : bool = False, photo : str | None = None) -> None:
        """Runs the rn-script"""
        
        if do_restart and photo is not None:
            os.system(f"./re {photo}")
        elif do_restart:
            os.system("./re")
        else:
            os.system(f'./{self.name}')
