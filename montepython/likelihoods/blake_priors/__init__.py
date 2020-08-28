import os
import numpy as np
from montepython.likelihood_class import Likelihood_prior
from numpy.fft import fft, ifft , rfft, irfft , fftfreq
from numpy import exp, log, log10, cos, sin, pi, cosh, sinh , sqrt
from scipy.special import gamma,erf
from scipy import interpolate
from scipy.integrate import quad
import scipy.integrate as integrate
from scipy import special

class blake_priors(Likelihood_prior):

    # Add priors to Omega_m, norm and omega_b for no-BAO analysis.
    def __init__(self,path,data,command_line):

        Likelihood_prior.__init__(self,path,data,command_line)

    def loglkl(self, cosmo, data):

        omb = (data.mcmc_parameters['omega_b']['current'] *
                    data.mcmc_parameters['omega_b']['scale'])
        if 'A_s' in list(data.mcmc_parameters.keys()):
            norm = np.sqrt((data.mcmc_parameters['A_s']['current'] *
                        data.mcmc_parameters['A_s']['scale'])/2.0989e-9)
        else:
            norm = (data.mcmc_parameters['norm']['current'] *
                        data.mcmc_parameters['norm']['scale'])
        chi2 = 0.
        if self.norm_std>0:
            chi2 += (norm-self.norm_mean)**2./self.norm_std**2.
        if self.Omega_m_std>0:
            chi2 += (cosmo.Omega_m()-self.Omega_m_mean)**2./self.Omega_m_std**2.
        if self.omb_std>0:
            chi2 += (omb-self.omb_mean)**2./self.omb_std**2.

        loglkl = -0.5 * chi2

        return loglkl
