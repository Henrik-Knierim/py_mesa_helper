# class for analyzing the output of a MESA run
import numpy as np
import mesa_reader as mr

class Analysis:

    def __init__(self, src = './LOGS/') -> None:
        # location of the LOGS directory
        self.src = src
    
    def homogeinity(self, model_number = -1):
        """Gives the deviation of the metallicity from a homogeneous profile."""

        # init log object
        log = mr.MesaLogDir(self.src)
        # gets the last profile
        profile = log.profile_data(model_number = model_number)

        # get average metallicty
        Z_avg = log.history.data('average o16')[model_number] # check definition

        # get the metallicity profile
        dm = profile.dm
        Z = profile.z

        # calculate the deviation
        variance = np.sum(np.power(Z - Z_avg, 2) * dm) / np.sum(dm)

        return np.sqrt(variance)
    
    def relative_atmospheric_metallicity(self):
        """Gives the ratio between the atmopsheric metallicity (Z_atm) and the average metallicity (Z_avg)."""

        # init log object
        log = mr.MesaLogDir(self.src)
        
        # gets the last profile
        profile = log.profile_data()

        # get the metallicity profile
        dm = profile.dm
        Z = profile.z

        # calculate the relative atmospheric metallicity
        return np.sum(Z * dm) / np.sum(dm)
        
