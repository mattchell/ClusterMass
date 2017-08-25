# This is to find a relation between cluster mass and the average redshift at a given angular radii
# (c) mattchelldavis
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import kmeans_radec as kmrd
import sys
from scipy import integrate
import treecorr
import time
# my other files
import error as err
import cosmology as cosmo
import ploting
import fitting

# <<--Important overarching variable Definitions-->>
# coorilation function
min_sep = 0.1
max_sep = 7.
nbins = 10
n = 1
# <<--End of line-->>

program_start = time.time()

# getting cluster and bacground galaxy data
clusters = fits.getdata('/calvin1/mattchell200/redmapper_catalogs/sdss/v5.10/dr8_run_redmapper_v5.10_lgt5_catalog.fit');

back_gal = fits.getdata('/calvin1/mattchell200/fits/sdss_galaxies_dr12_cut.fit')
print back_gal.dtype.names
# imposing limits on data
print('cutting clusters...')
c = np.asarray([clusters['RA'].copy(),clusters['DEC'].copy(),clusters['Z_LAMBDA'].copy()]).T
c_rel = np.asarray([[clust[0],clust[1],clust[2],cosmo.easy_D_A(clust[2])] for clust in c if clust[2] > 0.1 and clust[2] < 0.33 and clust[0] > 100.0 and clust[0] < 300.0]).T

print('cutting galaxies...')
b_g = np.asarray([back_gal['RA'].copy(),back_gal['DEC'].copy(),back_gal['Z_NOQSO'].copy(),back_gal['i'].copy(),back_gal['g'].copy(),back_gal['r'].copy(),back_gal['i_color'].copy(),back_gal['g_color'].copy(),back_gal['r_color'].copy(),back_gal['i_psf'].copy(),back_gal['i_mod'].copy(),back_gal['i_fib2'].copy(),back_gal['z_mod'].copy(),back_gal['z_psf'].copy()]).T
b_g_rel = np.asarray([gal for gal in b_g if gal[2] >= 0.4 and gal[0] > 100. and gal[0] < 300.]).T

print("finding equal partitions for the bins in i...")
total_i = b_g_rel[3].size
num_per_bin = int(total_i/n)
i_in_order = np.sort(b_g_rel[3])
i_bin_bounds = np.asarray(np.linspace(17.5,19.9,n+1,endpoint=True))
for j in range(1,n):
    i_bin_bounds[j] = i_in_order[num_per_bin*j]

print("finding galaxy indecies...")
b_g_rel = np.asarray([np.append(gal,fitting.i_index(gal[3],i_bin_bounds)) for gal in b_g_rel.T]).T

b_g_rel_binned = [0 for i in range(n)]
avg_i_in_bin = np.zeros(n)
for i in range(n):
    b_g_rel_binned[i] = np.asarray([gal for gal in b_g_rel.T if gal[14] == i]).T
    avg_i_in_bin[i] = np.average(b_g_rel_binned[i][3])


fitting.set_p_of_c(c_rel[2])
fitting.set_p_of_g_binned(b_g_rel_binned,n)

# fitting.find_d_delta_i_bar_d_mu(b_g_rel,avg_i_in_bin,n,i_bin_bounds)

# using treecorr corrilation function for average z at pysical seperation
print('finding correlation function...')

cat_c = treecorr.Catalog(ra=c_rel[0].copy(), dec=c_rel[1].copy(), r=c_rel[3].copy(), ra_units='deg', dec_units='deg')
nn = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, metric='Rlens')

del_z_bar = np.zeros((n,nbins))
N = np.zeros((n,nbins))
for i in range(n):
    cat_g = treecorr.Catalog(ra=b_g_rel_binned[i][0].copy(), dec=b_g_rel_binned[i][1].copy(), w=b_g_rel_binned[i][3].copy(), ra_units='deg', dec_units='deg')
    nn.process(cat_c,cat_g)
    
    i_bar = nn.weight/nn.npairs
    mean_i = np.average(b_g_rel_binned[i][3])
    del_z_bar[i] = i_bar-mean_i
    N[i] = nn.npairs
    print(mean_i,i_bar)
sep = nn.meanr
print(N)

print("fitting signal...")
A, var_A = fitting.multiple_profiles_i(sep,del_z_bar,N,b_g_rel_binned,c_rel[2],n)
output = open("auto/output.txt","r+")
data = ["n = "+str(n)+"\n","A = "+str(A)+"\n","var_A = "+str(var_A)+"\n","\n"]
output.seek(0,2)
output.writelines(data)
output.close()

r = np.logspace(-1,1,30,endpoint=True)
model_mu = []
model_mu_p = []
model_mu_m = []
for r_i in r:
    model_mu.append(fitting.calc_expect_mu(r_i,A,n))
    model_mu_p.append(fitting.calc_expect_mu(r_i,A+np.power(var_A,0.5),n))
    model_mu_m.append(fitting.calc_expect_mu(r_i,A-np.power(var_A,0.5),n))
model_mu = np.asarray(model_mu)
model_mu_p = np.asarray(model_mu_p)
model_mu_m = np.asarray(model_mu_m)

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
# ploting avg redshift as a function of theta
plt.figure(1,figsize=(8,8),dpi=80)
if int(np.power(n,0.5)) == np.power(n,0.5):
    row = int(np.power(n,0.5))
    col = row
else:
    small_n = int(np.power(n,0.5))
    big_n = int(np.power(n,0.5))+1
    if n <= small_n*big_n:
        row = big_n
        col = small_n
    else:
        row = big_n
        col = big_n
model = fitting.delta_i_bar_of_mu(model_mu,n)
model_p = fitting.delta_i_bar_of_mu(model_mu_p,n)
model_m = fitting.delta_i_bar_of_mu(model_mu_m,n)
for k in range(n):
    plot_n = int(float(k-(k%row))/float(row) + col*(k%row) + 1)
    sig = np.power((np.average(np.power(b_g_rel_binned[k][2],2.))-np.power(np.average(b_g_rel_binned[k][2]),2.))/N[k],0.5)
    plt.subplot(row,col,plot_n)
    plt.errorbar(sep, del_z_bar[k], yerr=sig, fmt='o', color='b')
    plt.plot(r,model.T[k],'r')
    plt.plot(r,model_p.T[k],'r',alpha=0.5)
    plt.plot(r,model_m.T[k],'r',alpha=0.5)
    plt.axhline(y=0.0, xmin=0.0, xmax=1.0, linewidth=1.5, color='k')
    plt.xscale('log')
    if k != 0:
        plt.ylabel(r'$\delta \bar i_{'+str(k+1)+'}$', fontsize=18)
    else:
        plt.ylabel(r'$\delta \bar i$', fontsize=18)
    if (k+1)%row == 0 or k+1 == n:
        plt.xlabel('r (Mpc)') 

plt.subplots_adjust(wspace=0.4,right=0.95,left=0.1,bottom=0.1,top=0.95)
plt.suptitle('Delta Average i vs. Physical Radius')
# plt.savefig("auto/delta_z_bar_pair_fit_"+str(n)+".png")
plt.show()
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

