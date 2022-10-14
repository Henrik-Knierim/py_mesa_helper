class inlist:
    def __init__(self, name):
        self.name = name

    
    def __str__(self):
        return self.name

    
    # finds existing option and changes it to the new value

    def change_lines(self, option : str, value):
        
        delimiter = "="

        with open(self.name, 'r') as file:

            lines = file.readlines()

            for i,l in enumerate(lines):
                if option in l:

                    # test if this is in fact the right option

                    line_splitted = l.split(delimiter)
                    # true if the occurence exactly matches with option
                    is_option = line_splitted[0].strip() == option                    
                    if is_option:
                        index_option = i
                        new_line = l.replace(line_splitted[1].strip(), str(value))
                        break
        
            lines[index_option] = new_line

        return lines

    # create lines with new option

    def create_lines(self, section : str, option : str, value):
        
        with open(self.name, 'r') as file:

            lines = file.readlines()

            for i,l in enumerate(lines):
                if section in l:

                        index_section = i

                        break
        
            lines.insert(index_section + 2, f"\t{option} = {value}\n")

        return lines

    # sets options in inlist files

    def set_option(self, section : str , option : str, value):

        # check if the option is already present. If not, create it
        try:
            lines = self.change_lines(option, value)
        except:
            lines = self.create_lines(section, option, value)


        # write new lines into the inlist
        with open(self.name, 'w') as file:
            file.writelines(lines)
        
        print(f"Changed {option} to {value}")

        def change_lines(self, option : str, value):
        
            delimiter = "="

            with open(self.name, 'r') as file:

                lines = file.readlines()

                for i,l in enumerate(lines):
                    if option in l:

                        # test if this is in fact the right option

                        line_splitted = l.split(delimiter)
                        # true if the occurence exactly matches with option
                        is_option = line_splitted[0].strip() == option                    
                        if is_option:
                            index_option = i
                            new_line = l.replace(line_splitted[1].strip(), str(value))
                            break
        
            lines[index_option] = new_line

        return lines

    # I could have some kind of clean function such that the inlist doesn't contain too many options.
    # idea: one attribute saves the initial inlist and another function restores the initial inlist after the code ran