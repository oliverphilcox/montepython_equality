import os
import numpy as np
import math
from montepython.likelihood_class import Likelihood_prior
from time import time


class euclid_P1loopAP(Likelihood_prior):

    def __init__(self,path,data,command_line):
        """ Load in datasets """

        Likelihood_prior.__init__(self,path,data,command_line)

        # Read spectra
        self.k = np.zeros(self.ksize, 'float64')
        self.Pk0 = np.zeros(
            (self.ksize, self.zsize), 'float64')
        self.Pk2 = np.zeros(
            (self.ksize, self.zsize), 'float64')
        self.Pk4 = np.zeros(
            (self.ksize, self.zsize), 'float64')
        for z_i in range(self.zsize):
            datafile = open(os.path.join(self.data_directory, self.measurements_file[z_i]), 'r')
            for i in range(self.ksize):
                line = datafile.readline()
                self.k[i] = float(line.split()[0])
                self.Pk0[i,z_i] = float(line.split()[1])
                self.Pk2[i,z_i] = float(line.split()[2])
                self.Pk4[i,z_i] = float(line.split()[3])
            datafile.close()

        #Read covmats
        self.cov = np.zeros(
            (3*self.ksize, 3*self.ksize, self.zsize), 'float64')
        self.logdetcov = np.zeros(self.zsize, 'float64')
        for z_i in range(self.zsize):
            self.cov[:,:,z_i] = np.loadtxt(os.path.join(self.data_directory, self.covmat_file[z_i]))
            self.logdetcov[z_i] = np.linalg.slogdet(self.cov[:,:,z_i])[1]

        # Read fiducial parameters
        # These are similar to the Chudaykin+Ivanov'19 paper (but not identical)
        # NB: we have weak Gaussian priors on nuisance parameters for marg. reasons
        self.z = np.asarray(self.z)
        self.b1fid = 0.9+0.4*self.z
        self.bG2fid = -2./7.*(self.b1fid-1.)
        self.b2fid = -0.704172-0.207993*self.z+0.183023*self.z**2.-0.00771288*self.z**3. + 4./3.*self.bG2fid
        self.bGamma3fid = 23./42.*(self.b1fid-1.)

        self.css0fid = np.array([1.0178509219947594,0.8401126369322178,0.7013255454416957,0.5921183685386798,0.5053079557189303,0.4355295623781029,0.37881018626306023,0.33220469789341484])
        self.css2fid = np.array([27.856972601961836,22.99255637919754,19.19417282261483,16.20534482316387,13.829480893360198,11.919756444032291,10.367436676673229,9.091918047609248])
        self.css4fid = np.array([-1.285706427782854,-1.0611949098091173,-0.8858848995052998,-0.7479389918383325,-0.6382837335397015,-0.5501426051091827,-0.4784970773849182,-0.4196269868127345])
        # NB: we don't include b4 term here
        self.b4fid = np.zeros(self.zsize,'float64')
        self.Pshotfid = np.array([261.4844988639507,481.44987939873755,847.6127559057418,1438.3881888390754,2556.0824045202717,4770.217643650718,8541.758519347533,14744.767249760793])

    def loglkl(self, cosmo, data):

        # Read in MCMC parameters
        # NB: norm = (A_s / A_s,fid)^{1/2} here, as in BOSS
        norm = data.mcmc_parameters['norm']['current']*data.mcmc_parameters['norm']['scale']

        b1 = [data.mcmc_parameters['b1_%d'%i]['current'] * data.mcmc_parameters['b1_%d'%i]['scale'] for i in range(1,self.zsize+1)]
        b2 = [data.mcmc_parameters['b2_%d'%i]['current'] * data.mcmc_parameters['b2_%d'%i]['scale'] for i in range(1,self.zsize+1)]
        bG2 = [data.mcmc_parameters['bG2_%d'%i]['current'] * data.mcmc_parameters['bG2_%d'%i]['scale'] for i in range(1,self.zsize+1)]
        bGamma3 = self.bGamma3fid

        # Set marginalized parameters equal to fiducial values
        css0 = self.css0fid
        css2 = self.css2fid
        css4 = self.css4fid
        Pshot = np.zeros(self.zsize,'float64')

        # Shot-noise parameters. a0 isn't used here (identical to Pshot).
        # This is for shot-noise P_stoch(k) = (Pshot + a0 + a2 k^2)
        a0 = 0.
        a2 = 0. # fiducial value

        #### standard deviations for marginalized parameters ######
        ## NB: we fix a0 = 0 here but vary a2 in [mu, sigma] = [0, Pshot_fid].
        if self.test:
            css0sig = 0.
            css2sig = 0.
            css4sig = 0.
            b4sig = 0.
            Pshotsig = np.zeros(self.zsize,'float64')
            bGamma3sig = 0.
            a0sig = 0.
            a2sig = np.zeros(self.zsize,'float64')
        else:
            css0sig = 10.
            css2sig = 10.
            css4sig = 10.
            b4sig = 0. # not marginalized
            Pshotsig = 0.3*self.Pshotfid
            bGamma3sig = 1.
            a0sig = 0. # not marginalized
            if self.include_a2:
                a2sig = self.Pshotfid
            else:
                a2sig = np.zeros(self.zsize,'float64') # not marginalized

        h = cosmo.h()
        chi2 = 0.

        for z_i in range(self.zsize):

            ### COMPUTE THEORY MODEL
            fz = cosmo.scale_independent_growth_factor_f(self.z[z_i])
            all_theory = cosmo.get_pk_mult(self.k*h,self.z[z_i],self.ksize)

            # Monopole
            theory0 = ((norm**2*all_theory[15]
                + norm**4*(all_theory[21])
                + norm**1*b1[z_i]*all_theory[16]
                + norm**3*b1[z_i]*(all_theory[22])
                + norm**0*b1[z_i]**2*all_theory[17]
                + norm**2*b1[z_i]**2*all_theory[23]
                + 0.25*norm**2*b2[z_i]**2*all_theory[1]
                + b1[z_i]*b2[z_i]*norm**2*all_theory[30]
                + b2[z_i]*norm**3*all_theory[31]
                + b1[z_i]*bG2[z_i]*norm**2*all_theory[32]
                + bG2[z_i]*norm**3*all_theory[33]
                + b2[z_i]*bG2[z_i]*norm**2*all_theory[4]
                + bG2[z_i]**2*norm**2*all_theory[5]
                + 2.*css0[z_i]*norm**2*all_theory[11]/h**2
                + (2.*bG2[z_i]+0.8*bGamma3[z_i]*norm)*norm**2*(b1[z_i]*all_theory[7]+norm*all_theory[8]))*h**3
                + Pshot[z_i]
                #+ a0*(self.k/0.45)**2.
                + a2*(1./3.)*(self.k/0.45)**2.
                #+ fz**2*b4[z_i]*self.k**2*(norm**2*fz**2/9. + 2.*fz*b1[z_i]*norm/7. + b1[z_i]**2/5)*(35./8.)*all_theory[13]*h
                )

            # Quadrupole
            theory2 = ((norm**2*all_theory[18]
                + norm**4*(all_theory[24])
                + norm**1*b1[z_i]*all_theory[19]
                + norm**3*b1[z_i]*(all_theory[25])
                + b1[z_i]**2*norm**2*all_theory[26]
                + b1[z_i]*b2[z_i]*norm**2*all_theory[34]
                + b2[z_i]*norm**3*all_theory[35]
                + b1[z_i]*bG2[z_i]*norm**2*all_theory[36]
                + bG2[z_i]*norm**3*all_theory[37]
                + 2.*css2[z_i]*norm**2*all_theory[12]/h**2
                + (2.*bG2[z_i]+0.8*bGamma3[z_i]*norm)*norm**3*all_theory[9])*h**3
                + a2*(2./3.)*(self.k/0.45)**2.
                #+ fz**2*b4[z_i]*self.k**2*((norm**2*fz**2*70. + 165.*fz*b1[z_i]*norm+99.*b1[z_i]**2)*4./693.)*(35./8.)*all_theory[13]*h
                )

            # Hexadecapole
            theory4 = ((norm**2*all_theory[20]
	            + norm**4*all_theory[27]
                + b1[z_i]*norm**3*all_theory[28]
                + b1[z_i]**2*norm**2*all_theory[29]
                + b2[z_i]*norm**3*all_theory[38]
                + bG2[z_i]*norm**3*all_theory[39]
                + 2.*css4[z_i]*norm**2*all_theory[13]/h**2)*h**3
                #+ fz**2*b4[z_i]*self.k**2*(norm**2*fz**2*48./143. + 48.*fz*b1[z_i]*norm/77.+8.*b1[z_i]**2/35.)*(35./8.)*all_theory[13]*h
                )

            # Residual
            x = np.hstack([theory0-self.Pk0[:,z_i],theory2-self.Pk2[:,z_i],theory4-self.Pk4[:,z_i]])

            ## COMPUTE DERIVATIVES W.R.T. LINEAR PARAMETERS
            dtheory4_dcss0 = np.zeros_like(self.k)
            dtheory4_dcss2 = np.zeros_like(self.k)
            dtheory4_dcss4 = 2.*norm**2*all_theory[13]*h
            #dtheory4_db4 = fz**2*self.k**2*(norm**2*fz**2*48./143. + 48.*fz*b1[z_i]*norm/77.+8.*b1[z_i]**2/35.)*(35./8.)*all_theory[13]*h
            dtheory4_dPshot = np.zeros_like(self.k)
            dtheory4_dbGamma3 = np.zeros_like(self.k)
            #dtheory4_da0 = np.zeros_like(self.k)
            dtheory4_da2 = np.zeros_like(self.k)

            dtheory2_dcss0 = np.zeros_like(self.k)
            dtheory2_dcss2 = 2.*norm**2*all_theory[12]*h
            dtheory2_dcss4 = np.zeros_like(self.k)
            #dtheory2_db4 = fz**2*self.k**2*((norm**2*fz**2*70. + 165.*fz*b1[z_i]*norm+99.*b1[z_i]**2)*4./693.)*(35./8.)*all_theory[13]*h
            dtheory2_dPshot = np.zeros_like(self.k)
            dtheory2_dbGamma3 = 0.8*norm*norm**3*all_theory[9]*h**3
            #dtheory2_da0 = np.zeros_like(self.k)
            dtheory2_da2 = (2./3.)*(self.k/0.45)**2.

            dtheory0_dcss0 = (2.*norm**2.*all_theory[11]/h**2.)*h**3.
            dtheory0_dcss2 = np.zeros_like(self.k)
            dtheory0_dcss4 = np.zeros_like(self.k)
            #dtheory0_db4 = fz**2.*self.k**2.*(norm**2.*fz**2./9. + 2.*fz*b1[z_i]*norm/7. + b1[z_i]**2./5)*(35./8.)*all_theory[13]*h
            dtheory0_dPshot = np.ones_like(self.k)
            dtheory0_dbGamma3 = 0.8*norm*norm**2*(b1[z_i]*all_theory[7]+norm*all_theory[8])*h**3
            #dtheory0_da0 = (self.k/0.45)**2.
            dtheory0_da2 = (1./3.)*(self.k/0.45)**2.

            # Stack arrays
            dtheory_dcss0 = np.hstack([dtheory0_dcss0,dtheory2_dcss0,dtheory4_dcss0])
            dtheory_dcss2 = np.hstack([dtheory0_dcss2,dtheory2_dcss2,dtheory4_dcss2])
            dtheory_dcss4 = np.hstack([dtheory0_dcss4,dtheory2_dcss4,dtheory4_dcss4])
            #dtheory_db4 = np.hstack([dtheory0_db4,dtheory2_db4,dtheory4_db4])
            dtheory_dPshot = np.hstack([dtheory0_dPshot,dtheory2_dPshot,dtheory4_dPshot])
            dtheory_dbGamma3 = np.hstack([dtheory0_dbGamma3,dtheory2_dbGamma3,dtheory4_dbGamma3])
            #dtheory_da0 = np.hstack([dtheory0_da0,dtheory2_da0,dtheory4_da0])
            dtheory_da2 = np.hstack([dtheory0_da2,dtheory2_da2,dtheory4_da2])

            # Compute marginalized covariance matrix
            marg_cov = (self.cov[:,:,z_i]
                + css0sig**2*np.outer(dtheory_dcss0,dtheory_dcss0)
                + css2sig**2*np.outer(dtheory_dcss2,dtheory_dcss2)
                + css4sig**2*np.outer(dtheory_dcss4,dtheory_dcss4)
                #+ b4sig**2*np.outer(dtheory_db4, dtheory_db4)
                + Pshotsig[z_i]**2*np.outer(dtheory_dPshot,dtheory_dPshot)
                + bGamma3sig**2*np.outer(dtheory_dbGamma3,dtheory_dbGamma3)
                #+ a0sig**2*np.outer(dtheory_da0,dtheory_da0)
                + a2sig[z_i]**2*np.outer(dtheory_da2,dtheory_da2))

            ## ADD TO CHI^2
            chi2 += np.inner(x,np.inner(np.linalg.inv(marg_cov),x))
            chi2 += np.linalg.slogdet(marg_cov)[1] - self.logdetcov[z_i]
            # Add priors on nuisance parameters
            chi2 += (b2[z_i]/norm - self.b2fid[z_i])**2./1.**2. + (bG2[z_i]/norm - self.bG2fid[z_i])**2./1.**2.


        #print("chi2_euclidP=", chi2)
        loglkl = -0.5 * chi2

        return loglkl
