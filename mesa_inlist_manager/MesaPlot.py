# module to plot MESA logs
# making everything class bases - work in progess

import matplotlib.pyplot as plt
import mesa_reader as mr
import numpy as np
from mesa_inlist_manager.Analysis import Analysis, MultipleSimulationAnalysis
from mesa_inlist_manager.astrophys import Z_Sol, M_Jup_in_Sol, R_Jup_in_Sol
from mesa_inlist_manager.MesaRun import MesaRun

class MesaPlot:

    def __init__(self, src = 'LOGS', **kwargs) -> None:
        self.src = src

        # get the atmopsheric metallicity
        if 'Z_atm' in kwargs:
            self.Z_atm = kwargs['Z_atm']
        else:
            self.Z_atm = Z_Sol

    def _dlogZ_dlogP_two_axes_plot(self, ax, ax2, **kwargs)-> None:
        """Plots the gradient of the metallicity profile as a function of logP."""

        # init log object
        log = mr.MesaLogDir(self.src)
        
        # get the profile
        profile = log.profile_data(**kwargs)
        
        # get the pressure and metallicity profile
        logP = profile.logP
        z = profile.z

        # get the derrivative of the metallicity profile
        dlogZ_dlogP = Analysis(self.src).dlogZ_dlogP(**kwargs)
        
        # create the first plot
        ax.plot(logP, dlogZ_dlogP, 'b-')
        ax.set_xlabel('logP [dyn/cm^2]')
        ax.set_ylabel('dlogZ/dlogP', color = 'b')
        ax.set_yscale('log')
        ax.tick_params('y', colors='b')
    
        # create the second plot
        ax2.plot(logP, z, 'r-')
        ax2.set_ylabel('Z', color = 'r')
        ax2.tick_params('y', colors='r')

    def dlogmu_dlogP_two_axes_plot(self, M_p, s0, **kwargs)-> None:
        """Plots the gradient of the metallicity profile as a function of logP."""

        # init log object
        logs_path = MesaRun(self.src).create_logs_path_string(M_p, s0)
        logs = mr.MesaLogDir(logs_path)
        profile = logs.profile_data(**kwargs)
        
        # get the pressure and mean molecular weight profile
        logP = profile.logP
        mu = profile.mu

        # get the derrivative of the metallicity profile
        dlogmu_dlogP = Analysis(logs_path).dlogmu_dlogP(**kwargs)
        
        # create the first plot
        fig, ax1 = plt.subplots()
        ax1.plot(logP, dlogmu_dlogP, 'b-')
        ax1.set_xlabel('logP [dyn/cm^2]')
        ax1.set_ylabel('dlogµ/dlogP', color = 'b')
        ax1.set_yscale('log')
        ax1.tick_params('y', colors='b')

        # create the second plot
        ax2 = ax1.twinx()
        ax2.plot(logP, mu, 'r-')
        ax2.set_ylabel('µ', color = 'r')
        ax2.tick_params('y', colors='r')
    
    def _mesh_distribution_plot(self, ax, ax2, **kwargs)-> None:
        """Plots the mesh distribution as a function of logP."""

        # init log object
        log = mr.MesaLogDir(self.src)
        
        # get the profile
        profile = log.profile_data(**kwargs)
        
        # get the pressure and metallicity profile
        logP = profile.logP
        z = profile.z

        # create the first plot
        ax.hist(logP, bins = 100, color = 'b')
        ax.set_xlabel('logP [dyn/cm^2]')
        ax.set_ylabel('Number of mesh points', color = 'b')
        ax.tick_params('y', colors='b')

        # create the second plot
        ax2.plot(logP, z, 'r-')
        ax2.set_ylabel('Z', color = 'r')
        ax2.tick_params('y', colors = 'r')
    
    def mesh_overview(self)-> None:
        """Plots a grid of the relevant mesh metrics for the start and the end of the simulation."""

        # Create a 2x2 grid of subplots
        fig, axs = plt.subplots(2, 2, figsize = (9, 6))

        # Plot data in each subplot
        self._dlogZ_dlogP_two_axes_plot(axs[0, 0], axs[0, 0].twinx(), profile_number = 1)
        self._mesh_distribution_plot(axs[0, 1], axs[0, 1].twinx(), profile_number = 1)
        self._dlogZ_dlogP_two_axes_plot(axs[1, 0], axs[1, 0].twinx())
        self._mesh_distribution_plot(axs[1, 1], axs[1, 1].twinx())

        # Increase the distance between columns
        #plt.subplots_adjust(hspace=0.3, wspace=0.3)
        line = plt.Line2D([0.05,.95],[0.5,0.5], transform=fig.transFigure, color="black", linewidth = .5)
        fig.add_artist(line)

        # Add column headings
        axs[0, 0].set_title('Start of evolution')
        axs[0, 1].set_title('Start of evolution')
        axs[1, 0].set_title('End of evolution')
        axs[1, 1].set_title('End of evolution')

        # Adjust spacing between subplots
        plt.tight_layout()
        
    def heterogeneity_plot(self, initial_entropies : list, **kwargs)-> None:
        """Plots the heterogeneity of multiple simulations as a function of planetary mass."""
        
        model_number = kwargs.get("model_number", None)

        for s0 in initial_entropies:
            # initalize MultiSimulationAnalysis object
            sim = MultipleSimulationAnalysis(self.src, s0 = s0)

            # style the plot
            label = kwargs.get("label", f'{s0:.1f} kb/mu')

            # plot the heterogeneity
            if model_number == 1:
                plt.plot(sim.planetary_mass, sim.heterogeneity(**kwargs), label = f'initial')
            else:
                plt.plot(sim.planetary_mass, sim.heterogeneity(**kwargs), label = label)
        
        plt.xlabel('planetary mass [M_J]')
        plt.ylabel('heterogeneity')
        plt.legend()

    def relative_atmospheric_metallicity_plot(self, inital_entropies : list, **kwargs)-> None:
        """Plots the realtive atmospheric metallicity of multiple simulations as a function of planetary mass."""

        for s0 in inital_entropies:
            # initalize MultiSimulationAnalysis object
            sim = MultipleSimulationAnalysis(self.src, s0=s0)

            # style the plot
            label = kwargs.get("label", f'{s0:.1f} kb/mu')
            # plot the relative atmospheric metallicity
            plt.plot(sim.planetary_mass, sim.relative_atmospheric_metallicity(), label = label)
        
        plt.xlabel('planetary mass [M_J]')
        plt.ylabel('relative atmospheric metallicity')
        plt.legend()

    def metallicity_profile_plot(self, M_p, s0, showFirstProfileQ = True)->None:
        """Plots the metallicity profile at the beginning and the end of the simulation."""

        # create the plot
        fig, ax = plt.subplots()
        self._metallicity_profile_plot_axes_allocation(ax, M_p, s0, showFirstProfileQ)


    def _metallicity_profile_plot_axes_allocation(self, ax, M_p, s0, showFirstProfileQ = True)-> None:
        """Populates the input ax with a metallicity profile plot."""

        logs_path = MesaRun(self.src).create_logs_path_string(M_p, s0)
        logs = mr.MesaLogDir(logs_path)

        if showFirstProfileQ:
            profile_first = logs.profile_data(profile_number=1)
            m_first = profile_first.mass
            Z_first = profile_first.z
            ax.plot(m_first/M_Jup_in_Sol, Z_first, label = '0')
        
        final_profile = logs.profile_data()
        m_final = final_profile.mass
        
        # get planetary data
        Z_final = final_profile.z
        age = logs.history.data('star_age')[-1]/1e9
        R_final = logs.history.radius[-1]/R_Jup_in_Sol
        Z_avg = logs.history.data('average_o16')[-1]
    
        ax.plot(m_final/M_Jup_in_Sol, Z_final, label = f'{age:.2f}')
        
        ax.set_xlim(0, M_p*1.02)
        ax.set_title(f'Mixing results for {M_p:.2f} M_J, R_final = {R_final:.2f} R_J') 
        ax.set_ylabel("$Z$")
        ax.set_xlabel("mass [$M_J$]")
        
        # plot the average metallicity
        ax.axhline(y=Z_avg, color = 'k', linestyle = 'dotted')
        ax.text(M_p*0.02, Z_avg+0.012, 'Z_avg')
        
        # plot the atmospheric metallicity
        ax.axhline(y=self.Z_atm, color = 'k', linestyle = 'dashed')
        ax.text(M_p*0.02, self.Z_atm+0.012, 'Z_atm')
        ax.legend(title='age [Gyr]')

    def metallicity_profile_mass_evolution_grid_plot(self, M_ps : list, s0s : list)-> None:
        """Plots a grid of metallicity profiles for different inital conditions."""

        # Create a 2x2 grid of subplots
        fig, axs = plt.subplots(2, 2, figsize = (9, 6))

        # Plot data in each subplot
        for M_p, s0, ax in zip(M_ps, s0s, axs.flat):
            self._metallicity_profile_plot_axes_allocation(ax, M_p, s0)

        # Adjust spacing between subplots
        plt.tight_layout()


    def convection_stability_gradient_plot(self, M_p, s0, plot_grad_radQ = True, **kwargs)->None:
        """Plots the adiabatic temperature, Ledoux, and radiative gradient."""

        logs_path = MesaRun(self.src).create_logs_path_string(M_p, s0)
        logs = mr.MesaLogDir(logs_path)

        # get the data
        profile = logs.profile_data(**kwargs)
        m = profile.mass
        grad_ad = profile.grada
        grad_Ledoux = profile.gradL
        grad_rad = profile.gradr

        # plot the gradients
        #plt.plot(m/M_Jup_in_Sol, grad_ad, label = 'grad_ad')
        #plt.plot(m/M_Jup_in_Sol, grad_Ledoux, label = 'grad_Ledoux')
        plt.plot(m/M_Jup_in_Sol, grad_Ledoux, label = f'{M_p:.2f} M_J')
        if plot_grad_radQ: plt.plot(m/M_Jup_in_Sol, grad_rad, label = 'grad_rad')

        plt.xlabel('mass [$M_J$]')
        plt.ylabel('Ledoux gradient')
        #plt.title(f'Convection stability for {M_p:.2f} M_J, {s0:.1f} kb/mu')

    def mean_molecular_weight_plot(self, M_p, s0, **kwargs)->None:
        """Plots the mean molecular weight as a function of mass coordinate."""

        logs_path = MesaRun(self.src).create_logs_path_string(M_p, s0)
        logs = mr.MesaLogDir(logs_path)

        # get the data
        profile = logs.profile_data(**kwargs)
        m = profile.mass
        mu = profile.mu

        # plot the gradients
        plt.plot(m/M_Jup_in_Sol, mu)

        plt.xlabel('mass [$M_J$]')
        plt.ylabel('mean molecular weight')
        plt.title(f'Mean molecular weight for {M_p:.2f} M_J, {s0:.1f} kb/mu')

    def relative_cumulative_energy_error_plot(self, initial_entropies : list, **kwargs)->None:
        """Plots the relative cumulative energy error as a function of planetary mass."""
        
        for s0 in initial_entropies:
            # initalize MultiSimulationAnalysis object
            sim = MultipleSimulationAnalysis(self.src, s0 = s0)
            x = sim.planetary_mass
            y = sim.get_history_data('rel_cumulative_energy_error')
            label = kwargs.get("label", f'{s0:.1f} kb/mu')
            plt.plot(x, y, label = label)
            
        plt.xlabel('planetary mass [M_J]')
        plt.ylabel('relative cumulative energy error')
        plt.legend()
        
""" The following code needs some additional work. 
class MesaPlot:

    def __init__(
        self,
        logs_list,
        logs_origin = 'LOGS/'
        ):

        # list of paths to the individual LOGS directories
        self.logs_list = logs_list

        # path to the LOGS parent directory
        self.logs_origin = logs_origin

    def final_quantities(self, quantity):
        
        quantity_list = []

        for l in self.logs_list:
            history = mr.MesaData(self.logs_origin + l + '/history.data')
            final_quantity = history.data(quantity)[-1]
            quantity_list.append(final_quantity)
            
        return np.array(quantity_list)
    
    def convert_list(self, attribute, conversion):
        
        old_values = getattr(self, attribute)

        if type(conversion) == float or type(conversion) == int:
            values = conversion * old_values
        
        elif callable(conversion):
            values = conversion(old_values)

        setattr(self, attribute, values)

    def final_quantity_plot(
        self,
        x,
        y,
        *args,
        **kwargs
        ):

        xData = self.final_quantities(x)
        yData = self.final_quantities(y)

        plt.xlabel(x)
        plt.ylabel(y)
        
        plt.plot(xData, yData, *args, **kwargs)

class MesaPlotOption():

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
        logs_origin = 'LOGS'
        ):
        
        super().__init__()
        
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
        
        print(xData)
        yData = self.final_quantities
        print(yData)

        plt.plot(xData, yData, *args, **kwargs)
"""