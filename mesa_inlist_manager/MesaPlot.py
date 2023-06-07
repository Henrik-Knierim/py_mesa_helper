# module to plot MESA logs
# making everything class bases - work in progess

import matplotlib.pyplot as plt
import mesa_reader as mr
import numpy as np
import os
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
                plt.plot(sim.planetary_mass, sim.heterogeneity(**kwargs), 'o-', label = f'initial')
            else:
                plt.plot(sim.planetary_mass, sim.heterogeneity(**kwargs), 'o-', label = label)
        
        plt.xlabel('planetary mass [M_J]')
        plt.ylabel('heterogeneity')
        plt.legend()

    def heterogeneity_evolution_plot(self)-> None:
        """Plots the heterogeneity evolution of a simulation."""

        # init history file
        t, h = Analysis(self.src).heterogeneity_evolution()

        # get the energy error
        history_file = mr.MesaData(os.path.join(self.src,'history.data'))

        age = history_file.data('star_age')
        # rel_E_err = history_file.data('rel_error_in_energy_conservation')
        abs_rel_E_err = history_file.data('rel_cumulative_energy_error')

        # plot both quantities on the same axis
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('age [yr]')
        ax1.set_ylabel('heterogeneity')
        ax1.plot(t, h, 'b-')
        ax1.tick_params('y', colors='b')
        ax1.set_xscale('log')

        # plot the relative error
        ax2 = ax1.twinx()
        ax2.plot(age, abs_rel_E_err, 'r-')
        ax2.set_ylabel('relative cumulative energy error', color='r')
        ax2.tick_params('y', colors='r')
        ax2.set_xscale('log')

        # Add gridlines at specific positions
        grid_positions = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        for pos in grid_positions:
            ax2.axhline(y=pos, color='r', linestyle='--', linewidth=0.5, alpha=0.5)

        ax2.set_ylim(0, 0.35)


    def relative_atmospheric_metallicity_plot(self, inital_entropies : list, **kwargs)-> None:
        """Plots the realtive atmospheric metallicity of multiple simulations as a function of planetary mass."""

        for s0 in inital_entropies:
            # initalize MultiSimulationAnalysis object
            sim = MultipleSimulationAnalysis(self.src, s0=s0)

            # style the plot
            label = kwargs.get("label", f'{s0:.1f} kb/mu')
            # plot the relative atmospheric metallicity
            plt.plot(sim.planetary_mass, sim.relative_atmospheric_metallicity(), 'o-', label = label)
        
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


    def _convection_gradient_plot(self, ax, M_p, s0, grad_radQ = False, x_axis='m', **kwargs)->None:
        """Plots the adiabatic temperature, Ledoux, and radiative gradient."""

        # set the logs path
        # if not defined, use the default logs path
        kwargs.get('logs_path', MesaRun(self.src).create_logs_path_string(M_p, s0))
        logs = mr.MesaLogDir(kwargs['logs_path'])
        

        # get the data
        profile_data_kwargs = {key: value for key, value in kwargs.items() if key in logs.profile_data.__code__.co_varnames}
        profile = logs.profile_data(**profile_data_kwargs)

        m = profile.mass
        logP = profile.logP
        grad_ad = profile.grada
        grad_Ledoux = profile.gradL
        grad_rad = profile.gradr
        grad_comp = profile.gradL_composition_term

        # plot the gradients
        if x_axis == 'm':
            ax.plot(m/M_Jup_in_Sol, grad_ad, label = 'grad_ad')
            ax.plot(m/M_Jup_in_Sol, grad_Ledoux, label = 'grad_Ledoux')
            ax.plot(m/M_Jup_in_Sol, grad_comp, label = 'grad_comp')
            if grad_radQ: ax.plot(m/M_Jup_in_Sol, grad_rad, label = 'grad_rad')
            ax.set_xlabel('mass [$M_J$]')

        elif x_axis == 'logP':
            ax.plot(logP, grad_ad, label = 'grad_ad')
            ax.plot(logP, grad_Ledoux, label = 'grad_Ledoux')
            ax.plot(logP, grad_comp, label = 'grad_comp')
            if grad_radQ: ax.plot(logP, grad_rad, label = 'grad_rad')
            ax.set_xlabel('logP [cgs]')
            ax.invert_xaxis()

        ax.set_yscale('log')
        ax.set_ylabel('gradients')
        ax.set_title(f'Convection gradients for {M_p:.2f} M_J, {s0:.1f} kb/mu')
        ax.legend()

    def _convection_stability_plot(self, ax, M_p, s0, x_axis='m', **kwargs)->None:
        """Plots the Schwarzschild and Ledoux criteria for convection stability."""
        
        # set the logs path
        # if not defined, use the default logs path
        kwargs.get('logs_path', MesaRun(self.src).create_logs_path_string(M_p, s0))
        logs = mr.MesaLogDir(kwargs['logs_path'])

        # get the data
        profile_data_kwargs = {key: value for key, value in kwargs.items() if key in logs.profile_data.__code__.co_varnames}
        profile = logs.profile_data(**profile_data_kwargs)

        m = profile.mass
        logP = profile.logP

        schwarzschild=profile.sch_stable
        ledoux=profile.ledoux_stable

        # plot the gradients
        if x_axis == 'm':
            ax.plot(m/M_Jup_in_Sol, schwarzschild, label = 'Schwarzschild criterion')
            ax.plot(m/M_Jup_in_Sol, ledoux, label = 'Ledoux criterion')
            ax.set_xlabel('mass [$M_J$]')

        elif x_axis == 'logP':
            ax.plot(logP, schwarzschild, label = 'Schwarzschild criterion')
            ax.plot(logP, ledoux, label = 'Ledoux criterion')
            ax.set_xlabel('logP [cgs]')
            ax.invert_xaxis()
     
        ax.set_ylabel('stability criteria')
        ax.set_title(f'Convection stability for {M_p:.2f} M_J, {s0:.1f} kb/mu')
        ax.legend()

    def convective_stability_overview_plot(self, M_p, s0, **kwargs)->None:
        """Plots a the convective gradients and the stability criteria in a grid of subplots for both, mass and pressure."""
        
        # Create a 2x2 grid of subplots
        fig, axs = plt.subplots(2, 2, figsize = (9, 6))

        # Plot data in each subplot
        self._convection_gradient_plot(axs[0,0], M_p, s0, **kwargs)
        self._convection_gradient_plot(axs[0,1], M_p, s0, x_axis='logP', **kwargs)

        self._convection_stability_plot(axs[1,0], M_p, s0, **kwargs)
        self._convection_stability_plot(axs[1,1], M_p, s0, x_axis='logP', **kwargs)

        # Adjust spacing between subplots
        plt.tight_layout()

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
            plt.plot(x, y, 'o-', label = label)
            
        plt.xlabel('planetary mass [M_J]')
        plt.ylabel('relative cumulative energy error')
        plt.legend()

    def energy_error_evolution_plot(self, **kwargs)->None:
        """Plots the relative cumulative energy error and relative error for a single model."""
        
        # init history file
        history_file = mr.MesaData(os.path.join(self.src,'history.data'))

        # get the data
        age = history_file.data('star_age')
        rel_E_err = history_file.data('rel_error_in_energy_conservation')
        abs_rel_E_err = history_file.data('rel_cumulative_energy_error')

        # plot both quantities on the same axis
        fig, ax1 = plt.subplots()

        # plot the relative error
        ax1.plot(age, rel_E_err, 'b-', **kwargs)
        ax1.set_xlabel('age [yr]')
        ax1.set_ylabel('relative error in energy conservation', color='b')
        ax1.tick_params('y', colors='b')
        ax1.set_xscale('log')

        # plot the relative cumulative error
        ax2 = ax1.twinx()
        ax2.plot(age, abs_rel_E_err, 'r-', **kwargs)
        ax2.set_ylabel('relative cumulative energy error', color='r')
        ax2.tick_params('y', colors='r')
        # Add gridlines at specific positions
        grid_positions = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        for pos in grid_positions:
            ax2.axhline(y=pos, color='r', linestyle='--', linewidth=0.5, alpha=0.5)

        ax2.set_ylim(0, 0.35)
