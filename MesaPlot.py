# module to plot MESA logs
# making everything class bases - work in progess

import matplotlib.pyplot as plt
import mesa_reader as mr
import numpy as np

class MesaPlot:

    def create_LOGS_list(self):
        logs_list = [f'{self.logs_origin}/{self.option}/{v}' for v in self.values]
        return logs_list

    def read_final_quantities(self):
        
        qunatity_list = []

        for l in self.logs_list:
            history = mr.MesaData(l + '/history.data')
            final_quantity = history.data(self.quantity)[-1]
            qunatity_list.append(final_quantity)
            
        return np.array(qunatity_list)

    def __init__(
        self,
        quantity,
        values,
        option,
        logs_origin = '../LOGS'
        ):
        
        # quantity we want to plot
        self.quantity = quantity

        # option in MESA we investigated
        self.option = option

        # values we varied
        self.values = values

        # path to the LOGS parent directory
        self.logs_origin = logs_origin

        # list of paths to the individual LOGS directories
        self.logs_list = self.create_LOGS_list()

        # reads list of the final value of self.quantity

        self.final_quantities = self.read_final_quantities()

    
    def valid_kwargs(self, kwargs, opts):
        return {key: kwargs[key] for key in kwargs.keys() & opts}

    def pop_unvalid_kwargs(self, kwargs, *invalid_options):
        opts = set()
        opts = opts.union(*invalid_options)
    
        kwargs_key_set = set(kwargs.keys())
        return {key: kwargs[key] for key in kwargs_key_set.difference(opts)}

    def convert_list(self, attribute, conversion):
        
        old_values = getattr(self, attribute)

        if type(conversion) == float or type(conversion) == int:
            values = conversion * old_values
        
        elif callable(conversion):
            values = conversion(old_values)

        setattr(self, attribute, values)

    def final_quantity_plot(
        self,
        *args,
        **kwargs
        ):

        plt.xlabel(self.option)
        plt.ylabel(self.quantity)

        xData = self.values

        yData = self.final_quantities
        
        plt.plot(xData, yData, *args, **kwargs)