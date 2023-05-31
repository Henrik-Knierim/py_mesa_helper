class rn:
    """Modifies rn-files"""

    def __init__(self, name : str) -> None:

        # name/path to rn script that should be modified
        self.name = name
        
        # save rn file to restore later
        with open(self.name, 'r') as file:
            self.original_rn_script = file.read()

    def restore_rn(self):
        """Restores run script to original version."""
        with open(self.name, 'w') as file:
            file.write(self.original_rn_script)
        print("restored inlist to original version")

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

        print(f"Set mod-file name to {mod_file_name}")
