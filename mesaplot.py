# module to plot MESA logs

import matplotlib.pyplot as plt
import mesa_reader as mr
import numpy as np

def valid_kwargs(kwargs, opts):
    return {key: kwargs[key] for key in kwargs.keys() & opts}

def pop_unvalid_kwargs(kwargs, *invalid_options):
    opts = set()
    opts = opts.union(*invalid_options)
    
    kwargs_key_set = set(kwargs.keys())
    return {key: kwargs[key] for key in kwargs_key_set.difference(opts)}

opts_get_final_quant = {'conversion_factor'}

def get_final_quant(path, quant, conversion_factor = 1):

    if type(path) == str:
        history = mr.MesaData(path + '/history.data')
        final_quantity = history.data(quant)[-1]

        if conversion_factor != 1:
            final_quantity = conversion_factor * final_quantity

    if type(path) == list:
        
        final_quantity = []

        for p in path:
            final_quantity.append(get_final_quant(p, quant, conversion_factor))
        
    return final_quantity

opts_create_LOGS_list = {'path'}

def create_LOGS_list(values, option, path = '../LOGS'):

    return [f'{path}/{option}/{v}' for v in values]

def final_quantity_plot(
    quant, # quantity you want to plot
    values, # values used for the MESA run
    option, # option for which the values were used
    *args,
    **kwargs
    ):

    plt.xlabel(option)
    plt.ylabel(quant)

    xData = values
    

    logs = create_LOGS_list(
        values,
        option,
        **valid_kwargs(kwargs, opts_create_LOGS_list)
        )

    yData = get_final_quant(
        logs,
        quant,
        **valid_kwargs(kwargs, opts_get_final_quant)
        )
    
    
    # because I don't have a set of all possible plt.plot kwargs,
    # I simply pop all the others
    plt.plot(
        xData,
        yData,
        *args,
        **pop_unvalid_kwargs(
            kwargs,
            opts_create_LOGS_list,
            opts_get_final_quant
            )
        )