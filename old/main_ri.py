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
import fitting_ri

# <<--Important overarching variable Definitions-->>
# coorilation function
min_sep = 0.1
max_sep = 7.
nbins = 10
# number of i-mag bins
n = 3
# data limits
clust_z_min = 0.1
clust_z_max = 0.33
gal_z_min = 0.4
# <<--End of line-->>

program_start = time.time()

# getting cluster and bacground galaxy data
clusters = fits.getdata('/calvin1/mattchell200/redmapper_catalogs/sdss/v5.10/dr8_run_redmapper_v5.10_lgt20_catalog.fit');

back_gal = fits.getdata('/calvin1/mattchell200/fits/sdss_galaxies_dr12_cut.fit')
print back_gal.dtype.names
# imposing limits on cluster data
print('cutting clusters...')
c = np.asarray([clusters['RA'].copy(),clusters['DEC'].copy(),clusters['Z_LAMBDA'].copy()]).T
c_rel = np.asarray([[clust[0],clust[1],clust[2],cosmo.easy_D_A(clust[2])] for clust in c if clust_z_min < clust[2] < clust_z_max]).T

# imposing limits on galaxy data
print('cutting galaxies...')
b_g = np.asarray([back_gal['RA'].copy(),back_gal['DEC'].copy(),back_gal['Z_NOQSO'].copy(),back_gal['i'].copy(),back_gal['g'].copy(),back_gal['r'].copy(),back_gal['i_color'].copy(),back_gal['g_color'].copy(),back_gal['r_color'].copy(),back_gal['i_psf'].copy(),back_gal['i_mod'].copy(),back_gal['i_fib2'].copy(),back_gal['z_mod'].copy(),back_gal['z_psf'].copy()]).T
b_g_rel = np.asarray([gal for gal in b_g if gal_z_min < gal[2]]).T

# binning data by (r-i) band magnitude
print("finding equal partitions for the bins in (r-i)...")
total_ri = b_g_rel[8].size
num_per_bin = int(total_ri/n)
ri_in_order = np.sort(b_g_rel[8]-b_g_rel[6])
ri_bin_bounds = np.asarray(np.linspace(0.5,2.,n+1,endpoint=True))
for j in range(1,n):
    ri_bin_bounds[j] = ri_in_order[num_per_bin*j]

print("finding galaxy indecies...")
b_g_rel = np.asarray([np.append(gal,fitting_ri.ri_index(gal[8]-gal[6],ri_bin_bounds)) for gal in b_g_rel.T]).T

b_g_rel_binned = [0 for i in range(n)]
avg_z_in_bin = np.zeros(n)
for i in range(n):
    b_g_rel_binned[i] = np.asarray([gal for gal in b_g_rel.T if gal[14] == i]).T
    avg_z_in_bin[i] = np.average(b_g_rel_binned[i][2])

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

plt.hist(b_g_rel_north[2],50,range=(0.2,1.2),normed=True,color="blue",alpha=0.3)
plt.hist(b_g_rel_south[2],50,range=(0.2,1.2),normed=True,color="red",alpha=0.3)
plt.title("Galaxy Redshift Distribution in the North (blue) and South (red)")
plt.xlabel(r"$z$",fontsize=16)
plt.ylabel(r"Probability")
plt.yscale("log")
plt.show()
'''
# set probability distrebutions for calcualtion of expected mu value
fitting_ri.set_p_of_c(c_rel[2],clust_z_min,clust_z_max)
fitting_ri.set_p_of_g_binned(b_g_rel_binned,n,gal_z_min)

# fitting_ri.find_d_delta_z_bar_d_mu(b_g_rel,avg_z_in_bin,n,ri_bin_bounds)

# using treecorr corrilation function for average z at pysical seperation
print('finding correlation function...')

cat_c = treecorr.Catalog(ra=c_rel[0].copy(), dec=c_rel[1].copy(), r=c_rel[3].copy(), ra_units='deg', dec_units='deg')
nn = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, metric='Rlens')

del_z_bar = np.zeros((n,nbins))
N = np.zeros((n,nbins))
for i in range(n):
    cat_g = treecorr.Catalog(ra=b_g_rel_binned[i][0].copy(), dec=b_g_rel_binned[i][1].copy(), w=b_g_rel_binned[i][2].copy(), ra_units='deg', dec_units='deg')
    nn.process(cat_c,cat_g)

    z_bar = nn.weight/nn.npairs
    mean_z = np.average(b_g_rel_binned[i][2])
    del_z_bar[i] = z_bar-mean_z
    N[i] = nn.npairs
sep = nn.meanr
print(N)

print("fitting signal...")
A, var_A = fitting_ri.multiple_profiles(sep,del_z_bar,N,b_g_rel_binned,c_rel[2],n)
#output = open("auto/output.txt","r+")
#data = ["n = "+str(n)+"\n","A = "+str(A)+"\n","var_A = "+str(var_A)+"\n","\n"]
#output.seek(0,2)
#output.writelines(data)
#output.close()

r = np.logspace(-1,1,30,endpoint=True)
model_mu = []
model_mu_p = []
model_mu_m = []
for r_i in r:
    model_mu.append(fitting_ri.calc_expect_mu(r_i,A,n))
    model_mu_p.append(fitting_ri.calc_expect_mu(r_i,A+np.power(var_A,0.5),n))
    model_mu_m.append(fitting_ri.calc_expect_mu(r_i,A-np.power(var_A,0.5),n))
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
# ploting avg redshift as a function of r
plt.figure(1,figsize=(7.5,7),dpi=80)
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
model = fitting_ri.delta_z_bar_of_mu(model_mu,n)
model_p = fitting_ri.delta_z_bar_of_mu(model_mu_p,n)
model_m = fitting_ri.delta_z_bar_of_mu(model_mu_m,n)
for k in range(n):
    plot_n = int(float(k-(k%row))/float(row) + col*(k%row) + 1)
    sig = np.power((np.average(np.power(b_g_rel_binned[k][2],2.))-np.power(np.average(b_g_rel_binned[k][2]),2.))/N[k],0.5)
    plt.subplot(row,col,plot_n)
    plt.errorbar(sep, del_z_bar[k], yerr=sig, fmt='o', color='b')
    plt.plot(r,model.T[k],'r')
    plt.plot(r,model_p.T[k],'r',alpha=0.5)
    plt.plot(r,model_m.T[k],'r',alpha=0.5)
    if n != 1:
        plt.ylabel(r'$\Delta \bar z_{'+str(k+1)+'}$', fontsize=14)
    else:
        plt.ylabel(r'$\Delta \bar z$', fontsize=14)
    plt.axhline(y=0.0, xmin=0.0, xmax=1.0, linewidth=1.5, color='k')
    plt.xscale('log')
    if (k+1)%row == 0 or k+1 == n:
        plt.xlabel(r'$r$ (Mpc)') 

plt.subplots_adjust(hspace=0.26,wspace=0.56,right=0.95,left=0.16,top=0.93,bottom=0.10)
plt.suptitle('Delta Average Redshift vs. Physical Radius')
#plt.savefig("auto/delta_z_bar_pair_fit_"+str(n)+".png")
#plt.close()
plt.show()

# ploting r*avg redshift as a function of r
plt.figure(1,figsize=(7.5,7),dpi=80)
for k in range(n):
    plot_n = int(float(k-(k%row))/float(row) + col*(k%row) + 1)
    sig = np.power((np.average(np.power(b_g_rel_binned[k][2],2.))-np.power(np.average(b_g_rel_binned[k][2]),2.))/N[k],0.5)
    plt.subplot(row,col,plot_n)
    plt.errorbar(sep, sep*del_z_bar[k], yerr=sig*sep, fmt='o', color='b')
    plt.plot(r,r*model.T[k],'r')
    plt.plot(r,r*model_p.T[k],'r',alpha=0.5)
    plt.plot(r,r*model_m.T[k],'r',alpha=0.5)
    if n != 1:
        plt.ylabel(r'$\Delta \bar z_{'+str(k+1)+'} r$', fontsize=14)
    else:
        plt.ylabel(r'$\Delta \bar z r$', fontsize=14)
    plt.axhline(y=0.0, xmin=0.0, xmax=1.0, linewidth=1.5, color='k')
    plt.xscale('log')
    if (k+1)%row == 0 or k+1 == n:
        plt.xlabel(r'$r$ (Mpc)') 

plt.subplots_adjust(hspace=0.26,wspace=0.56,right=0.95,left=0.16,top=0.93,bottom=0.10)
plt.suptitle('Delta Average Redshift*Radius vs. Radius')
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

