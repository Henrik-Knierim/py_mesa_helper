import os
import matplotlib.pyplot as plt


class inlist:
    def __init__(self, name):
        self.name = name

        with open(self.name, 'r') as file:
            self.original_inlist = file.read()

    def __str__(self):
        return self.name

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
                        new_line = l.replace(
                            line_splitted[1].strip(), 
                            str(value),
                            1 # to change the value only
                            )
                        break

            lines[index_option] = new_line

        return lines

    # create lines with new option

    def create_lines(self, section: str, option: str, value):

        with open(self.name, 'r') as file:

            lines = file.readlines()

            for i, l in enumerate(lines):
                if section in l:

                    index_section = i

                    break

            lines.insert(index_section + 2, f"\t{option} = {value}\n")

        return lines

    # sets options in inlist files

    def set_option(self, section: str, option: str, value):

        # check if the option is already present. If not, create it
        try:
            lines = self.change_lines(option, value)
        except:
            lines = self.create_lines(section, option, value)

        # write new lines into the inlist
        with open(self.name, 'w') as file:
            file.writelines(lines)

        print(f"Set {option} to {value}")

    def restore_inlist(self):
        with open(self.name, 'w') as file:
            file.write(self.original_inlist)
        print("restored inlist to original version")

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
    def run_inlist_multiple_value(self, section: str, option: str, values: list, run_command='./rn', logs_parent_directory="../LOGS"):

        for v in values:
            log_value = f"'{logs_parent_directory}/{option}/{v}'"
            self.set_option('&controls', 'log_directory', log_value)
            self.run_inlist_single_value(section, option, v, run_command)
