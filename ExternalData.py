# class for handeling external data
import numpy as np

from os import path
resources_dir = path.join(path.dirname(__file__), 'resources')


class HotJupiterData:

    data_path = path.join(resources_dir, 'HJ_data.csv')

    def __init__(self):

        self.name, self.radius, self.mass,\
            self.Teq, self.stellar_FeH, self.stellar_age = np.loadtxt(
                self.data_path,
                dtype={
                    'names': (
                        'Planet_Name',
                        'Rp',
                        'Mp',
                        'Teq',
                        '[Fe/H]_star',
                        'Age_star'
                    ),
                    'formats': (
                        'U34',
                        np.single,
                        np.single,
                        np.single,
                        np.single,
                        np.single
                    )
                },
                skiprows=1,
                usecols=(0, 10, 14, 26, 42, 47),
                delimiter=',',
                unpack=True,

                # converting missing values to -1 so I can delete them later
                converters={
                    10: lambda s: float(s.strip() or -1),
                    14: lambda s: float(s.strip() or -1),
                    26: lambda s: float(s.strip() or -1),
                    42: lambda s: float(s.strip() or -1),
                    47: lambda s: float(s.strip() or -1)
                }
            )

        # pop missing values from data
     
        ## find the missing indices
        missing_value_indices = np.array([],dtype=int)
        for key in self.__dict__.keys():
            if key != 'name':
                missing_value_indices = np.append(
                    missing_value_indices,
                    np.where(self.__dict__[key] == -1)[0]
                    )

        ## deleting the respective elements
        for key, value in self.__dict__.items():
            self.__dict__[key] = np.delete(value, missing_value_indices)

