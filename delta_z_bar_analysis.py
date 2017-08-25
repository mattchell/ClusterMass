# This is to find a relation between cluster mass and the average redshift at a given angular radii
# (c) mattchelldavis

from astropy.io import fits
import numpy as np

import matplotlib as mpl
mpl.use('Agg') # When I don't want to display graphs

import matplotlib.pyplot as plt
import sys
import treecorr
import time

from colossus.cosmology import cosmology as csmg
#csmg.setCosmology('WMAP9', {'flat':True,'Om0':0.2}) # setting cosmology in colossus

# my other files
import error as err
import cosmology as cosmo
import fitting

class Delta_z_bar:

    def __init__(self):
        # <<--Important overarching variable Definitions-->>
        # coorilation function
        self.max_sep = 2
        self.nbins = 7
        # data limits
        self.clust_z_min = 0.1
        self.clust_z_max = 0.33
        self.gal_z_min = 0.4

        self.load_data()
        
        return None


    # Getting the specific covarience matrix
    def get_Covar_Mat(self, covar_type, read_from_file):
        if covar_type == 'JK':
            Cov_mat = err.JackKnife(self.nn, self.cat_g, self.b_g_rel[0:3].copy(), self.c_rel.copy(), self.mean_z, self.sep, self.nbins, self.min_sep, self.max_sep, n=self.n, ncen=70, read_from_file=read_from_file)

        elif covar_type == 'Randoms':
            n_clusters = self.c_rel[0].size
            Cov_mat = err.Randoms(n_clusters, self.clust_z_min, self.clust_z_max, self.nn, self.cat_g, self.mean_z, self.sep, self.nbins, self.min_sep, self.max_sep, n=self.n, read_from_file=read_from_file)
            
        elif covar_type == 'Analytic':
            Cov_mat = err.Analytic(self.b_g_rel[2].copy(), self.n, self.N, self.nbins)

        else:
            raise ValueError("Didn't recognize the type of Covariance matrix you provided")
        
        return Cov_mat


    def load_data(self):
        # getting cluster and bacground galaxy data
        clusters = fits.getdata('/calvin1/mattchell200/redmapper_catalogs/sdss/v5.10/dr8_run_redmapper_v5.10_lgt20_catalog.fit');

        back_gal = fits.getdata('/calvin1/mattchell200/fits/sdss_galaxies_dr12_cut.fit')
        # print clusters.dtype.names
        # imposing limits on cluster data
        print('cutting clusters...')
        c = np.array([clusters['RA'].copy(),clusters['DEC'].copy(),clusters['Z_LAMBDA'].copy(),clusters['LAMBDA_CHISQ'].copy()]).T
        self.c_rel = np.array([[clust[0],clust[1],clust[2],cosmo.easy_D_A(clust[2]),clust[3]] for clust in c if self.clust_z_min < clust[2] < self.clust_z_max and clust[3] >= 20.]).T
        n_clusters = self.c_rel[0].size
        print("{0} clusters.".format(n_clusters))

        # imposing limits on galaxy data
        print('cutting galaxies...')
        b_g = np.array([back_gal['RA'].copy(),back_gal['DEC'].copy(),back_gal['Z_NOQSO'].copy(),back_gal['i'].copy(),back_gal['g'].copy(),back_gal['r'].copy(),back_gal['i_color'].copy(),back_gal['g_color'].copy(),back_gal['r_color'].copy(),back_gal['i_psf'].copy(),back_gal['i_mod'].copy(),back_gal['i_fib2'].copy(),back_gal['z_mod'].copy(),back_gal['z_psf'].copy()]).T
        self.b_g_rel = np.array([gal for gal in b_g if self.gal_z_min < gal[2]]).T
        n_galaxies = self.b_g_rel[0].size
        print("{0} galaxies.".format(n_galaxies))
        
        # set probability distributions for calcualtion of expected A value
        self.FitSignal = fitting.FittingSignal(self.c_rel[2].copy(), self.b_g_rel[2].copy(), lamb=self.c_rel[4].copy(), clust_z_min=self.clust_z_min, clust_z_max=self.clust_z_max, lamb_min=20., lamb_max=210., gal_z_min=self.gal_z_min, gal_z_max=1.)

        return None


    # Creates measured signal using treecorr
    def correlate_data(self):
        cat_c = treecorr.Catalog(ra=self.c_rel[0].copy(), dec=self.c_rel[1].copy(), r=self.c_rel[3].copy(), ra_units='deg', dec_units='deg')
        self.nn = treecorr.NNCorrelation(min_sep=self.min_sep, max_sep=self.max_sep, nbins=self.nbins, metric='Rlens')

        self.cat_g = treecorr.Catalog(ra=self.b_g_rel[0].copy(), dec=self.b_g_rel[1].copy(), w=np.power(self.b_g_rel[2].copy(), self.n), ra_units='deg', dec_units='deg')
        self.nn.process(cat_c, self.cat_g)

        z_bar = self.nn.weight.copy()/self.nn.npairs.copy()
        self.mean_z = np.average(np.power(self.b_g_rel[2], self.n))
        self.del_z_bar = z_bar-self.mean_z
        self.N = self.nn.npairs.copy()
        self.sep = self.nn.meanr.copy()

        return None


    def display_cuts():
        plt.plot(b_g_rel[3], b_g_rel[11], ',b')
        plt.xlabel(r'$ i $-band Magnitudes')
        plt.ylabel(r'$ i_{psf} $ Magnitudes')
        plt.subplots_adjust(bottom=0.15, left=0.15)
        plt.show()

        return None


    def compare_diagonals(self, min_sep, n):
        self.n = n
        self.min_sep = min_sep
        self.correlate_data()
        
        JK_Cov_Mat = self.get_Covar_Mat('JK', True)
        R_Cov_Mat = self.get_Covar_Mat('Randoms', True)
        A_Cov_Mat = self.get_Covar_Mat('Analytic', True)

        JK_sig = np.sqrt(np.asarray(JK_Cov_Mat.diagonal())[0, :])
        R_sig = np.sqrt(np.asarray(R_Cov_Mat.diagonal())[0, :])
        A_sig = np.sqrt(np.asarray(A_Cov_Mat.diagonal())[0, :])
        
        width = 2
        fig, ax = plt.subplots(1)
        JK, = plt.plot(self.sep, JK_sig, linewidth=width, color=(0.4, 0.4, 0.95))
        R, = plt.plot(self.sep, R_sig, linewidth=width, color=(0.95, 0.35, 0.35))
        A, = plt.plot(self.sep, A_sig, linewidth=width, color=(0.55, 0.95, 0.))
        plt.legend([JK, R, A], [r'Jack Knife', 'Randoms', 'Analytic'])
        plt.xlabel(r'$R \ (Mpc \ h^{-1})$')
        plt.ylabel('Standard Deviation of Diagonals')
        plt.xscale('log')
        plt.xlim([self.min_sep, self.max_sep])
        plt.subplots_adjust(left=0.15, right=0.95, bottom=0.18, top=0.9)
        labels = [str(label) for label in ax.get_xticks().tolist()]
        ax.set_xticklabels(labels)
        plt.ticklabel_format(style='sci', scilimits=(-2, 2), axis='y')
        plt.savefig('auto/diagonals/diags_{0}.png'.format(self.n))
        plt.close()

        return None


    # Runs the damn thing
    def run(self, min_sep, n, covar_type, read_from_file):
        
        self.min_sep = min_sep
        self.n = n
        
        # print specs so I know what I'm looking at in the output file
        print('n = {0}\nR_min = {1}\nR_max = {2}'.format(self.n, self.min_sep, self.max_sep))
        
        # start a timer
        program_start = time.time()
        
        # get signal from the data
        print('find correlation function...')
        self.correlate_data()

        # Calcualtion of error on the signal
        Cov_mat = self.get_Covar_Mat(covar_type, read_from_file)
        
        # fit the data
        print("fitting signal...")
        A, var_A = self.FitSignal.fit_for_A(self.sep.copy(), self.del_z_bar.copy(), self.N.copy(), self.b_g_rel.copy(), self.c_rel.copy(), self.n, Cov_mat.copy(), covar_type, self.nbins)

        r = np.logspace(np.log10(self.min_sep), np.log10(self.max_sep), 50, endpoint=True)
        A_p = A+np.power(var_A,0.5)
        A_m = A-np.power(var_A,0.5)
        model = self.FitSignal.model_delta_z_bar(A, r)
        model_p = self.FitSignal.model_delta_z_bar(A_p, r)
        model_m = self.FitSignal.model_delta_z_bar(A_m, r)

        # ploting avg redshift as a function of r
        sig = np.power(np.array([Cov_mat[i,i] for i in range(self.nbins)]),0.5)
        
        fig = plt.figure(1,figsize=(7.5,7),dpi=80)
        ax = fig.add_subplot(111)
        plt.axhline(y=0.0, xmin=0.0, xmax=1.0, linewidth=1, color='k')
        data_graph = plt.errorbar(self.sep, self.del_z_bar, yerr=sig, fmt='o', color='b', capsize=3)
        ylims = ax.get_ylim()
        plt.fill_between(r, model_m, model_p, color='r', alpha=0.25)
        model_graph, = plt.plot(r, model, 'r')
        sig_label, = plt.fill(np.NaN, np.NaN, 'r', alpha=0.25)

        if n == 1:
            plt.ylabel(r'$\Delta \bar{z}(R)$', fontsize=14)
        else:
            plt.ylabel(r'$\Delta \bar{z^{'+str(n)+'}}(R)$', fontsize=14)

        plt.xscale('log')
        plt.xlim([self.min_sep, self.max_sep])
        plt.ylim(ylims)
        plt.xlabel(r'$R \ (Mpc \ h^{-1})$', fontsize=14) 
        
        plt.subplots_adjust(hspace=0.26,wspace=0.56,right=0.95,left=0.16,top=0.93,bottom=0.15)
        labels = [str(label) for label in ax.get_xticks()]
        ax.set_xticklabels(labels)
        minor_labels = [str(label) if str(label*np.power(0.1, np.floor(np.log10(label)))) in ['2.0', '3.0', '4.0', '6.0'] else '' for label in ax.get_xticks(minor=True)]
        ax.set_xticklabels(minor_labels, minor=True)
        plt.legend([data_graph, model_graph, sig_label], [r'$\Delta \bar{z}$ Signal', 'NSF model', r'1 $\sigma$'])
        plt.savefig("auto/signal_fits/delta_z_bar_fit_NFW_{0}_{1}.png".format(self.n, covar_type))
        plt.close()

        # ploting r*avg redshift as a function of r
        fig = plt.figure(1,figsize=(7.5,7),dpi=80)
        ax = fig.add_subplot(111)
        plt.axhline(y=0.0, xmin=0.0, xmax=1.0, linewidth=1, color='k')
        data_graph = plt.errorbar(self.sep, self.sep*self.del_z_bar, yerr=self.sep*sig, fmt='o', color='b', capsize=3)
        plt.fill_between(r, r*model_m, r*model_p, color='r', alpha=0.25)
        model_graph, = plt.plot(r, r*model, 'r')
        sig_label, = plt.fill(np.NaN, np.NaN, 'r', alpha=0.25)

        if n == 1:
            plt.ylabel(r'$R \Delta \bar{z}(R) \ (Mpc \ h^{-1})$', fontsize=14)
        else:
            plt.ylabel(r'$R \Delta \bar{z^{'+str(n)+'}}(R) \ (Mpc \ h^{-1})$', fontsize=14)
                
        plt.xscale('log')
        plt.xlim([self.min_sep, self.max_sep])
        plt.xlabel(r'$R \ (Mpc \ h^{-1})$', fontsize=14) 

        plt.subplots_adjust(hspace=0.26,wspace=0.56,right=0.95,left=0.16,top=0.93,bottom=0.15)
        labels = [str(label) for label in ax.get_xticks().tolist()]
        ax.set_xticklabels(labels)
        plt.legend([data_graph, model_graph, sig_label], [r'$\Delta \bar{z}$ Signal', 'NSF model', r'1 $\sigma$'])
        plt.savefig("auto/signal_fits/delta_z_bar_r_fit_NFW_{0}_{1}.png".format(self.n, covar_type))
        plt.close()
        
        program_end = time.time()
        run_time = program_end-program_start
        
        print("done")
        print("total run time: "+str(run_time)+" sec")
        print("             or "+str(run_time/60.)+" min")
        print("             or "+str(run_time/3600.)+" hr\n")
            
        return None
