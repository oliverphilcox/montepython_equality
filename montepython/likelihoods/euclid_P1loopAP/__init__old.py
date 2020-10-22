import os
import numpy as np
import math
from montepython.likelihood_class import Likelihood_prior
from time import time


class euclid_P1loopAP(Likelihood_prior):

    # initialisation of the class is done within the parent Likelihood_prior. For
    # this case, it does not differ, actually, from the __init__ method in
    # Likelihood class.
    def __init__(self, path, data, command_line):
        """Load in datasets"""

        Likelihood_prior.__init__(self,path,data,command_line)

        self.zsize = len(self.z)

        print("Using %d redshifts"%self.zsize)

        # Read in k, P(k) + inverse covariance
        self.k = np.zeros(self.ksize,'float64')
        self.Pk0,self.Pk2,self.Pk4,self.invcov = [],[],[],[]

        for z_i in range(self.zsize):

            Pk0 = np.zeros(self.ksize,'float64')
            Pk2 = np.zeros(self.ksize,'float64')
            Pk4 = np.zeros(self.ksize,'float64')

            datafile = open(os.path.join(self.data_directory, self.measurements_file[z_i]), 'r')
            for i in range(self.ksize):
                line = datafile.readline()
                if z_i==0: self.k[i] = float(line.split()[0])
                Pk0[i] = float(line.split()[1])
                Pk2[i] = float(line.split()[2])
                Pk4[i] = float(line.split()[3])
            datafile.close()

            self.Pk0.append(Pk0)
            self.Pk2.append(Pk2)
            self.Pk4.append(Pk4)
            self.invcov.append(np.loadtxt(os.path.join(self.data_directory, self.covmat_file[z_i])))

        print("Using k in range [%.2f, %.2f]\n"%(self.k[0],self.k[-1]))

    def loglkl(self, cosmo, data):

        delta = np.zeros(3*self.ksize,'float64')

        # Load in parameters
        norm = (data.mcmc_parameters['norm']['current'] * data.mcmc_parameters['norm']['scale'])

        # read in (many) parameters
        b1 = [data.mcmc_parameters['b1_%d'%(i+1)]['current']*data.mcmc_parameters['b1_%d'%(i+1)]['scale'] for i in range(self.zsize)]
        b2 = [data.mcmc_parameters['b2_%d'%(i+1)]['current']*data.mcmc_parameters['b2_%d'%(i+1)]['scale'] for i in range(self.zsize)]
        bG2 = [data.mcmc_parameters['bG2_%d'%(i+1)]['current']*data.mcmc_parameters['bG2_%d'%(i+1)]['scale'] for i in range(self.zsize)]

        # fixed bGamma3 input - see Chudaykin+Ivanov '19 (or '20?) to see how these are chosen
        bGamma3 = [0.076666666666666744, 0.12047619047619047, 0.16428571428571431, 0.20809523809523806, 0.25190476190476191, 0.29571428571428576, 0.33952380952380962, 0.38333333333333347]

        css0 = [data.mcmc_parameters['css0_%d'%(i+1)]['current']*data.mcmc_parameters['css0_%d'%(i+1)]['scale'] for i in range(self.zsize)]
        css2 = [data.mcmc_parameters['css2_%d'%(i+1)]['current']*data.mcmc_parameters['css2_%d'%(i+1)]['scale'] for i in range(self.zsize)]
        css4 = [data.mcmc_parameters['css4_%d'%(i+1)]['current']*data.mcmc_parameters['css4_%d'%(i+1)]['scale'] for i in range(self.zsize)]

        Pshot = [data.mcmc_parameters['Pshot_%d'%(i+1)]['current']*data.mcmc_parameters['Pshot_%d'%(i+1)]['scale'] for i in range(self.zsize)]

        h = cosmo.h()

        # Load all power spectrum components at all k and z (this just does a for loop in cython for speed)
        k_input = np.vstack([self.k*h for _ in range(self.zsize)]).T
        pk_comp = cosmo.get_pk_comp(k_input[:,:,np.newaxis], np.asarray(self.z), self.ksize, self.zsize, 1)

        # Compute (theory - data) vector
        chi2 = 0.
        for z_i in range(self.zsize):

            fz = cosmo.scale_independent_growth_factor_f(self.z[z_i])

            # hexadecapole, ell = 4
            delta[2*self.ksize: 3*self.ksize] = (norm*pk_comp[:,:,z_i,0][20] +norm**2.*(pk_comp[:,:,z_i,0][27])+ b1[z_i]*norm**2.*pk_comp[:,:,z_i,0][28] + b1[z_i]**2.*norm**2.*pk_comp[:,:,z_i,0][29] + b2[z_i]*norm**2.*pk_comp[:,:,z_i,0][38] + bG2[z_i]*norm**2.*pk_comp[:,:,z_i,0][39] + 2.*css4[z_i]*norm*pk_comp[:,:,z_i,0][13]/h**2.)*h**3. - self.Pk4[z_i]

            # quadrupole, ell = 2
            delta[self.ksize: 2*self.ksize] = (norm*pk_comp[:,:,z_i,0][18] +norm**2.*(pk_comp[:,:,z_i,0][24])+ norm*b1[z_i]*pk_comp[:,:,z_i,0][19] +norm**2.*b1[z_i]*(pk_comp[:,:,z_i,0][25]) + b1[z_i]**2.*norm**2.*pk_comp[:,:,z_i,0][26] +b1[z_i]*b2[z_i]*norm**2.*pk_comp[:,:,z_i,0][34]+ b2[z_i]*norm**2.*pk_comp[:,:,z_i,0][35] + b1[z_i]*bG2[z_i]*norm**2.*pk_comp[:,:,z_i,0][36]+ bG2[z_i]*norm**2.*pk_comp[:,:,z_i,0][37] + (2.*b1[z_i]**2./3.+fz*b1[z_i]*8./7.+fz**2.*10./21.)*fz**1.*(3./2.)*css2[z_i]*norm*pk_comp[:,:,z_i,0][12]/h**2. + (2.*bG2[z_i]+0.8*bGamma3[z_i])*norm**2.*pk_comp[:,:,z_i,0][9])*h**3. - self.Pk2[z_i]

            # monopole, ell = 0
            delta[: self.ksize] = (norm*pk_comp[:,:,z_i,0][15] +norm**2.*(pk_comp[:,:,z_i,0][21])+ norm*b1[z_i]*pk_comp[:,:,z_i,0][16] +norm**2.*b1[z_i]*(pk_comp[:,:,z_i,0][22]) + norm*b1[z_i]**2.*pk_comp[:,:,z_i,0][17] +norm**2.*b1[z_i]**2.*(pk_comp[:,:,z_i,0][23]) + 0.25*norm**2.*b2[z_i]**2.*pk_comp[:,:,z_i,0][1] +b1[z_i]*b2[z_i]*norm**2.*pk_comp[:,:,z_i,0][30]+ b2[z_i]*norm**2.*pk_comp[:,:,z_i,0][31] + b1[z_i]*bG2[z_i]*norm**2.*pk_comp[:,:,z_i,0][32]+ bG2[z_i]*norm**2.*pk_comp[:,:,z_i,0][33]+b2[z_i]*bG2[z_i]*norm**2.*pk_comp[:,:,z_i,0][4]+ bG2[z_i]**2.*norm**2.*pk_comp[:,:,z_i,0][5] + css0[z_i]*(b1[z_i]**2./3.+fz*b1[z_i]*2./5.+fz**2./7.)*fz**2.*norm*pk_comp[:,:,z_i,0][11]/h**2. + (2.*bG2[z_i]+0.8*bGamma3[z_i])*norm**2.*(b1[z_i]*pk_comp[:,:,z_i,0][7]+pk_comp[:,:,z_i,0][8]))*h**3. + Pshot[z_i] - self.Pk0[z_i]

            chi2 += np.dot(delta,np.dot(self.invcov[z_i],delta))

        #print("chi2_euclidP=", chi2)
        loglkl = -0.5 * chi2

        return loglkl
