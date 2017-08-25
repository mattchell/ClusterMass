# This is to find a relation between cluster mass and the average redshift at a given angular radii
# (c) mattchelldavis

from astropy.io import fits
import numpy as np

import matplotlib as mpl
# mpl.use('Agg') # When I don't want to display graphs

import matplotlib.pyplot as plt
import treecorr
import time

from colossus.cosmology import cosmology as csmg
#csmg.setCosmology('WMAP9', {'flat':True,'Om0':0.2}) # setting cosmology in colossus

from colossus.halo import concentration as concen
# my other files
import cosmology as cosmo

# <<--Important overarching variable Definitions-->>
# coorilation function
min_sep = 0.1
max_sep = 7.
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
c = np.asarray([clusters['RA'].copy(),clusters['DEC'].copy(),clusters['Z_LAMBDA'].copy(),clusters['LAMBDA_CHISQ'].copy()]).T
c_rel = np.asarray([[clust[0],clust[1],clust[2],cosmo.easy_D_A(clust[2]),clust[3]] for clust in c if clust_z_min < clust[2] < clust_z_max and clust[3] > 20.]).T

# imposing limits on galaxy data
print('cutting galaxies...')
b_g = np.asarray([back_gal['RA'].copy(),back_gal['DEC'].copy(),back_gal['Z_NOQSO'].copy(),back_gal['i'].copy(),back_gal['g'].copy(),back_gal['r'].copy(),back_gal['i_color'].copy(),back_gal['g_color'].copy(),back_gal['r_color'].copy(),back_gal['i_psf'].copy(),back_gal['i_mod'].copy(),back_gal['i_fib2'].copy(),back_gal['z_mod'].copy(),back_gal['z_psf'].copy()]).T
b_g_rel = np.asarray([gal for gal in b_g if gal_z_min < gal[2]]).T

# using treecorr corrilation function for average z at pysical seperation
print('finding correlation function...')

cat_c = treecorr.Catalog(ra=c_rel[0].copy(), dec=c_rel[1].copy(), r=c_rel[3].copy(), ra_units='deg', dec_units='deg')
nn = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, metric='Rlens')

del_z_bar = np.zeros(nbins)
N = np.zeros(nbins)
cat_g = treecorr.Catalog(ra=b_g_rel[0].copy(), dec=b_g_rel[1].copy(), w=b_g_rel[2].copy(), ra_units='deg', dec_units='deg')
nn.process(cat_c,cat_g)

z_bar = nn.weight/nn.npairs
mean_z = np.average(b_g_rel[2])
del_z_bar = z_bar-mean_z
N = nn.npairs
sep = nn.meanr
print(N)

# ploting avg redshift as a function of r
plt.figure(1,figsize=(7.5,7),dpi=80)
sig = np.power((np.average(np.power(b_g_rel[2],2.))-np.power(np.average(b_g_rel[2]),2.))/N,0.5)
plt.errorbar(sep, del_z_bar, yerr=sig, fmt='o', color='b')
plt.ylabel(r'$\Delta \bar z$', fontsize=14)
plt.axhline(y=0.0, xmin=0.0, xmax=1.0, linewidth=1.5, color='k')
plt.xscale('log')
plt.xlabel(r'$R (Mpc)$') 

plt.subplots_adjust(hspace=0.26,wspace=0.56,right=0.95,left=0.16,top=0.93,bottom=0.10)
plt.suptitle('Delta Average Redshift vs. Physical Radius')
plt.show()

# ploting r*avg redshift as a function of r
plt.figure(1,figsize=(7.5,7),dpi=80)
sig = np.power((np.average(np.power(b_g_rel[2],2.))-np.power(np.average(b_g_rel[2]),2.))/N,0.5)
plt.errorbar(sep, sep*del_z_bar, yerr=sig*sep, fmt='o', color='b')
plt.ylabel(r'$R \Delta \bar z (Mpc)$', fontsize=14)
plt.axhline(y=0.0, xmin=0.0, xmax=1.0, linewidth=1.5, color='k')
plt.xscale('log')
plt.xlabel(r'$R (Mpc)$') 

plt.subplots_adjust(hspace=0.26,wspace=0.56,right=0.95,left=0.16,top=0.93,bottom=0.10)
plt.suptitle('Delta Average Redshift*Radius vs. Radius')
plt.show()

program_end = time.time()
run_time = program_end-program_start

print("done")
print("total run time: "+str(run_time)+" sec")
print("             or "+str(run_time/60.)+" min")
print("             or "+str(run_time/3600.)+" hr")
