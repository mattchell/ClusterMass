# This is to find a relation between cluster mass and the average redshift at a given angular radii
# (c) mattchelldavis

from astropy.io import fits
import numpy as np

import matplotlib as mpl
mpl.use('Agg') # When I don't want to display graphs

import matplotlib.pyplot as plt
import kmeans_radec as kmrd
import sys
from scipy import integrate
import treecorr
import time
import scipy

from colossus.cosmology import cosmology as csmg
#csmg.setCosmology('WMAP9', {'flat':True,'Om0':0.2}) # setting cosmology in colossus

from colossus.halo import concentration as concen
# my other files
import error as err
import cosmology as cosmo
import ploting
import fitting_NFW

# <<--Important overarching variable Definitions-->>
# coorilation function
min_sep = 0.2 #0.428
max_sep = 7 #2.86
nbins = 10
# data limits
clust_z_min = 0.1
clust_z_max = 0.33
gal_z_min = 0.4
# <<--End of line-->>

program_start = time.time()

# getting cluster and bacground galaxy data
clusters = fits.getdata('/calvin1/mattchell200/redmapper_catalogs/sdss/v5.10/dr8_run_redmapper_v5.10_lgt20_catalog.fit');

back_gal = fits.getdata('/calvin1/mattchell200/fits/sdss_galaxies_dr12_cut.fit')
# print clusters.dtype.names
# imposing limits on cluster data
print('cutting clusters...')
c = np.array([clusters['RA'].copy(),clusters['DEC'].copy(),clusters['Z_LAMBDA'].copy(),clusters['LAMBDA_CHISQ'].copy()]).T
c_rel = np.array([[clust[0],clust[1],clust[2],cosmo.easy_D_A(clust[2]),clust[3]] for clust in c if clust_z_min < clust[2] < clust_z_max and clust[3] > 20.]).T
n_clusters = c_rel[0].size
print("{0} clusters.".format(n_clusters))

# imposing limits on galaxy data
print('cutting galaxies...')
b_g = np.array([back_gal['RA'].copy(),back_gal['DEC'].copy(),back_gal['Z_NOQSO'].copy(),back_gal['i'].copy(),back_gal['g'].copy(),back_gal['r'].copy(),back_gal['i_color'].copy(),back_gal['g_color'].copy(),back_gal['r_color'].copy(),back_gal['i_psf'].copy(),back_gal['i_mod'].copy(),back_gal['i_fib2'].copy(),back_gal['z_mod'].copy(),back_gal['z_psf'].copy()]).T
b_g_rel = np.array([gal for gal in b_g if gal_z_min < gal[2]]).T
n_galaxies = b_g_rel[0].size
print("{0} galaxies.".format(n_galaxies))

'''
b_g_rel_north = np.asarray([gal for gal in b_g_rel.T if 100 < gal[0] < 300]).T
b_g_rel_south = np.asarray([gal for gal in b_g_rel.T if 100 > gal[0] or gal[0] > 300]).T

plt.plot(b_g_rel_north[0],b_g_rel_north[1],"b,")
plt.plot(b_g_rel_south[0],b_g_rel_south[1],"r,")
plt.title("Galaxies in the North (blue) and South (red)")
plt.xlabel("RA")
plt.ylabel("Dec")
plt.xlim(0.,360.)
plt.ylim(-90.,90.)
plt.show()

plt.hist(c_rel[4],50,range=(15.,205.),normed=True,color="blue",alpha=0.3)
#plt.hist(b_g_rel_south[2],50,range=(0.2,1.2),normed=True,color="red",alpha=0.3)
plt.title("Galaxy Redshift Distribution in the North (blue) and South (red)")
plt.xlabel(r"$z$",fontsize=16)
plt.ylabel(r"Probability")
plt.yscale("log")
plt.show()
'''
# set probability distributions for calcualtion of expected mu value
fitting_NFW.set_p_of_c(c_rel[2].copy(),clust_z_min,clust_z_max)
fitting_NFW.set_p_of_g(b_g_rel[2].copy(),gal_z_min)

# using treecorr corrilation function for average z at pysical seperation
print('finding correlation function...')

cat_c = treecorr.Catalog(ra=c_rel[0].copy(), dec=c_rel[1].copy(), r=c_rel[3].copy(), ra_units='deg', dec_units='deg')
nn = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, metric='Rlens')

del_z_bar = np.zeros(nbins)
N = np.zeros(nbins)
cat_g = treecorr.Catalog(ra=b_g_rel[0].copy(), dec=b_g_rel[1].copy(), w=b_g_rel[2].copy(), ra_units='deg', dec_units='deg')
nn.process(cat_c,cat_g)

z_bar = nn.weight.copy()/nn.npairs.copy()
mean_z = np.average(b_g_rel[2])
del_z_bar = z_bar-mean_z
N = nn.npairs.copy()
sep = nn.meanr.copy()
print(N)

# Calcualtion of error on the signal
Cov_mat = err.jack_knife(nn,b_g_rel[0:3].copy(),c_rel.copy(),mean_z,nbins,ncen=100,method='calculate')
#Cov_mat = err.randoms_cluster_errors(n_clusters,clust_z_min,clust_z_max,nn,cat_g,mean_z,nbins,method='calculate')
print(Cov_mat.I/np.max(Cov_mat.I))

# quick significance checker
'''
mean_2_z = np.average(np.power(b_g_rel[2],2.))
chi_2 = np.sum(np.power(del_z_bar,2.)*N/(mean_2_z-np.power(mean_z,2.)))
print("chi_2: {0}".format(chi_2))
chi_2_dist = lambda x: np.power(x,nbins/2.-1)*np.exp(-x/2.)/(np.power(2.,nbins/2.)*scipy.special.gamma(nbins/2.))
tot_prob = scipy.integrate.quad(chi_2_dist,0,chi_2)
print("total probability: {0}".format(tot_prob))
print("Significance: {0}".format(scipy.special.erfinv(tot_prob[0])*np.power(2,0.5)))
'''
print("fitting signal...")
M_0, var_M_0 = fitting_NFW.multiple_profiles(sep,del_z_bar,N,b_g_rel.copy(),c_rel.copy(),Cov_mat)

r = np.logspace(np.log10(min_sep),np.log10(max_sep),30,endpoint=True)
model = []
model_p = []
model_m = []
M_p = M_0+np.power(var_M_0,0.5)
M_m = M_0-np.power(var_M_0,0.5)
for r_i in r:
    model.append(fitting_NFW.expect_delta_z_bar_of_Sig(M_0,r_i))
    model_p.append(fitting_NFW.expect_delta_z_bar_of_Sig(M_p,r_i))
    model_m.append(fitting_NFW.expect_delta_z_bar_of_Sig(M_m,r_i))
model = np.asarray(model)
model_p = np.asarray(model_p)
model_m = np.asarray(model_m)

'''
# ploting signal to noise as a function of mode number
n = np.linspace(.5,5.,16,endpoint=True)
print(n)
sig_noise = []
signal = []
noise = []
for k in n:
    del_z_n_bar, sig = fitting.other_modes(b_g_rel,cat_c,nn,k)
    A_n, sig_A = fitting.fit_powr_law(sep,del_z_n_bar,sig)
    sig_noise.append(A_n/sig_A)
    signal.append(A_n)
    noise.append(sig_A)
s_n = np.asarray(sig_noise)
signal = np.asarray(signal)
noise = np.asarray(noise)

plt.plot(n,s_n)
plt.xlabel('n')
plt.ylabel('Signal/Noise')
plt.title('Checking Signal to Noise') 
plt.show()

plt.plot(n,signal,'r')
plt.plot(n,noise,'b')
plt.xlabel('n')
plt.ylabel('Signal (red) and Noise (blue)')
plt.title('Signal And Noise')
plt.show()
'''

# ploting avg redshift as a function of r
old_sig = np.power((np.average(np.power(b_g_rel[2],2.))-np.power(np.average(b_g_rel[2]),2.))/N,0.5)
sig = np.power(np.array([Cov_mat[i,i] for i in range(sep.size)]),0.5)
print(sig/old_sig)
plt.figure(1,figsize=(7.5,7),dpi=80)
plt.errorbar(sep, del_z_bar, yerr=sig, fmt='o', color='b')
plt.plot(r,model,'r')
plt.plot(r,model_p,'r',alpha=0.5)
plt.plot(r,model_m,'r',alpha=0.5)
plt.ylabel(r'$\Delta \bar z$', fontsize=14)
plt.axhline(y=0.0, xmin=0.0, xmax=1.0, linewidth=1.5, color='k')
plt.xscale('log')
plt.xlim([min_sep, max_sep])
plt.xlabel(r'$R \ (Mpc)$') 

plt.subplots_adjust(hspace=0.26,wspace=0.56,right=0.95,left=0.16,top=0.93,bottom=0.10)
plt.suptitle('Delta Average Redshift vs. Physical Radius')
plt.savefig("auto/delta_z_bar_fit_NFW_JK.png")
plt.close()

# ploting r*avg redshift as a function of r
plt.figure(1,figsize=(7.5,7),dpi=80)
plt.errorbar(sep, sep*del_z_bar, yerr=sig*sep, fmt='o', color='b')
plt.plot(r,r*model,'r')
plt.plot(r,r*model_p,'r',alpha=0.5)
plt.plot(r,r*model_m,'r',alpha=0.5)
plt.ylabel(r'$R \Delta \bar z \ (Mpc)$', fontsize=14)
plt.axhline(y=0.0, xmin=0.0, xmax=1.0, linewidth=1.5, color='k')
plt.xscale('log')
plt.xlim([min_sep, max_sep])
plt.xlabel(r'$R \ (Mpc)$') 

plt.subplots_adjust(hspace=0.26,wspace=0.56,right=0.95,left=0.16,top=0.93,bottom=0.10)
plt.suptitle('Delta Average Redshift*Radius vs. Radius')
plt.savefig("auto/delta_z_bar_r_fit_NFW_JK.png")
plt.close()

'''
ploting.histogram(b_g_rel[2],100,range=(0.,3.),color='r',xlabel='Redshift',ylabel='counts',title='Log of Background Galaxies Histogram',yscale='log')

ploting.plot_posit(b_g_rel[0],b_g_rel[1],color='r')
'''
program_end = time.time()
run_time = program_end-program_start

print("done")
print("total run time: "+str(run_time)+" sec")
print("             or "+str(run_time/60.)+" min")
print("             or "+str(run_time/3600.)+" hr")

