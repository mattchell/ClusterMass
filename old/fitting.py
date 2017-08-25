# Dealing with fitting and signal to noise information
# (c) mattchelldavis

import numpy as np
import treecorr
import error as err
import cosmology as cosmos
import ploting
import matplotlib.pyplot as plt
from math import gamma
import emcee
import corner
import scipy.optimize as opt
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def fit_powr_law(r,del_z,sig_y,alpha=-1.,R_0=1.):
    # fitting the Data to the function of del_z = A*(r/R_o)^alpha
    # fits vertically and finds best A value
    var_y = np.power(sig_y,2.)
    a = np.sum(np.power(r/R_0,alpha)*del_z/var_y)
    b = np.sum(np.power(r/R_0,2.*alpha)/var_y)
    A = a/b
    sig_A = np.sqrt(1/b)
    return (A, sig_A)
# setting filler data for galaxy redshift distrbution
n_bins_g = 31
z_step_g = np.linspace(0.4,1.0,n_bins_g,endpoint=True)
dz_g = z_step_g[1]-z_step_g[0]
z_bounds_g = np.linspace(0.4-dz_g/2.,1.0+dz_g/2.,n_bins_g+1,endpoint=True)

# setting filler data for cluster redshift distrebution
n_bins_c = 31
z_step_c = np.linspace(0.1,0.33,n_bins_c+1,endpoint=True)
dz_c = z_step_c[1]-z_step_c[0]
z_bounds_c = np.linspace(0.1-dz_c/2.,0.33+dz_c/2.,n_bins_c+1,endpoint=True)

def int_coeff(i,j,n,m):
    if i != 0 and i != n-1 and j != 0 and j != m-1:
        return 1.
    elif (i == 0 or i == n-1) and  (j == 0 or j == m-1):
        return 7./12.
    else:
        return 3./4.

p_grid_flat = np.zeros(n_bins_g*n_bins_c)
g_grid_flat = np.zeros(n_bins_g*n_bins_c)
c_grid_flat = np.zeros(n_bins_g*n_bins_c)
coeff_grid_flat = np.zeros(n_bins_g*n_bins_c)
def set_p_of_grid(z_g,z_c):
    global g_grid_flat, c_grid_flat, p_grid_flat, coeff_grid_flat
    p_g = np.zeros(n_bins_g)
    p_c = np.zeros(n_bins_c)
    p_grid = np.zeros((n_bins_g,n_bins_c))
    g_grid = np.zeros((n_bins_g,n_bins_c))
    c_grid = np.zeros((n_bins_g,n_bins_c))
    coeff_grid = np.zeros((n_bins_g,n_bins_c))
    for i in range(n_bins_g):
        check_g = [z_i for z_i in z_g if z_bounds_g[i] < z_i < z_bounds_g[i+1]]
        p_g[i] = len(check_g)/dz_g
    for j in range(n_bins_c):
        check_c = [z_i for z_i in z_c if z_bounds_c[j] < z_i < z_bounds_c[j+1]]
        p_c[j] = len(check_c)/dz_c
    for i in range(n_bins_g):
        for j in range(n_bins_c):
            p_grid[i][j] = p_g[i]*p_c[j]
            g_grid[i][j] = z_bounds_g[i]+0.5*dz_g
            c_grid[i][j] = z_bounds_c[j]+0.5*dz_c
            coeff_grid[i][j] = int_coeff(i,j,n_bins_g,n_bins_c)
    '''fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(g_grid,c_grid,p_grid,rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf)
    plt.show()'''
    integral = dz_g*dz_c*np.sum(coeff_grid*p_grid)
    p_grid = p_grid/integral
    p_grid_flat = p_grid.flatten()
    g_grid_flat = g_grid.flatten()
    c_grid_flat = c_grid.flatten()
    coeff_grid_flat = coeff_grid.flatten()

p_c = np.zeros(n_bins_c)
def set_p_of_c(z_c,clust_z_min,clust_z_max):
    global p_c,z_step,dz_c,z_bounds
    z_step_c = np.linspace(clust_z_min,clust_z_max,n_bins_c+1,endpoint=True)
    dz_c = z_step_c[1]-z_step_c[0]
    z_bounds_c = np.linspace(clust_z_min-dz_c/2.,clust_z_max+dz_c/2.,n_bins_c+1,endpoint=True)
    for i in range(n_bins_c):
        check_c = [z_i for z_i in z_c if z_bounds_c[i] < z_i < z_bounds_c[i+1]]
        p_c[i] = len(check_c)/dz_c
    p_c = p_c/simp_integral(dz_c,p_c)

p_g = []
def set_p_of_g_binned(b_g_rel_binned,n,gal_z_min):
    global p_g
    z_step_g = np.linspace(gal_z_min,1.0,n_bins_g,endpoint=True)
    dz_g = z_step_g[1]-z_step_g[0]
    z_bounds_g = np.linspace(gal_z_min-dz_g/2.,1.0+dz_g/2.,n_bins_g+1,endpoint=True)
    for i in range(n):
        p_g_bin = np.zeros(n_bins_g)
        for j in range(n_bins_g):
            check_g = [gal[2] for gal in b_g_rel_binned[i].T if z_bounds_g[j] < gal[2] < z_bounds_g[j+1]]
            p_g_bin[j] = len(check_g)/dz_g
        p_g.append(p_g_bin/simp_integral(dz_g,p_g_bin))

def find_d_delta_z_bar_d_mu(b_g_rel,avg_z_in_bin,n,i_bin_bounds):
    a_s = np.zeros(n)
    a_sigs = np.zeros(n)
    print("finding delta z bar dependence on mu...")
    mu = np.linspace(.95,1.,10,endpoint=False)
    z_bar_mu = np.zeros((10,n))
    var = np.zeros((10,n))
    for i in range(10):
        z_bar_i,var_i = calc_delta_z_bar_of_mu(b_g_rel,avg_z_in_bin,mu[i],n,i_bin_bounds)
        var[i] = var_i
        z_bar_mu[i] = z_bar_i
    # calcuating best fit line y = a(mu-1)
    print(z_bar_mu,var)
    for i in range(n):
        Y = np.matrix(z_bar_mu.T[i]).T
        A = np.matrix(mu-1.).T
        C = np.matrix(np.diag(var.T[i]))
        cov_B = (A.T*C.I*A).I
        B = cov_B*(A.T*C.I*Y)
        print(cov_B)
        a = B[0,0]
        print a
        a_sigs[i] = cov_B
        a_s[i] = a
        plt.plot(mu,z_bar_mu.T[i],"ro")
        plt.plot(mu,a*(mu-1),"b")
        plt.xlabel(r"$\mu$",size=20)
        plt.ylabel(r"$\Delta z_{"+str(i+1)+"} \bar$",size=20)
        plt.show()
    print(n,a_s,a_sigs)

def find_d_delta_i_bar_d_mu(b_g_rel,avg_z_in_bin,n,i_bin_bounds):
    a_s = np.zeros(n)
    a_sigs = np.zeros(n)
    print("finding delta i bar dependence on mu...")
    mu = np.linspace(.95,1.,10,endpoint=False)
    z_bar_mu = np.zeros((10,n))
    var = np.zeros((10,n))
    for i in range(10):
        z_bar_i,var_i = calc_delta_i_bar_of_mu(b_g_rel,avg_z_in_bin,mu[i],n,i_bin_bounds)
        var[i] = var_i
        z_bar_mu[i] = z_bar_i
    # calcuating best fit line y = a(mu-1)
    print(z_bar_mu,var)
    for i in range(n):
        Y = np.matrix(z_bar_mu.T[i]).T
        A = np.matrix(mu-1.).T
        C = np.matrix(np.diag(var.T[i]))
        cov_B = (A.T*C.I*A).I
        B = cov_B*(A.T*C.I*Y)
        print(cov_B)
        a = B[0,0]
        print a
        a_sigs[i] = cov_B
        a_s[i] = a
        plt.plot(mu,z_bar_mu.T[i],"ro")
        plt.plot(mu,a*(mu-1),"b")
        plt.xlabel(r"$\mu$",size=20)
        plt.ylabel(r"$\Delta z_{"+str(i+1)+"} \bar$",size=20)
        plt.show()
    print(n,a_s,a_sigs)


d_del_d_mu = [0.0332980199393,np.asarray([0.0473042986231,0.0485413729311]),np.asarray([0.0445653213182,0.0595901442039,0.0486843616686])]
d_del_d_mu_error = [7.64279366e-07,[],[]]

def delta_z_bar_of_mu(mu,n):
    x = np.asarray([d_del_d_mu[n-1]*(mu[i]-1) for i in range(mu.T[0].size)])
    return x

def delta_i_bar_of_mu(mu,n):
    x = np.asarray([-0.240520391666*(mu[i]-1) for i in range(mu.T[0].size)])
    return x

def simp_int_coeff(i,n):
    if i == 0 or i == n-i:
        return 1.
    elif i%2 == 0:
        return 2.
    else:
        return 4.

def simp_integral(dx,y):
    tot_y = y.size
    if tot_y%2 == 1:
        coeffs = np.asarray([simp_int_coeff(i,tot_y) for i in range(tot_y)])
        return dx/3.*np.sum(y*coeffs)
    else:
        print("For simpson's rule you need an odd number of y values")

def mu_SIS(z_c,z_g,r,A): # density(r) = A*r^-2
    return 1./(1.-2.*np.pi*A/(cosmos.Sig_cr(z_c,z_g)*r))

def mu_NFW(z_c,z_g,R,A,r_s): # density(r) = A/(r/r_s)/(1+r/r_s)^2
    sqrt_term = np.power(np.power(R/r_s,2.),0.5)
    kappa = 2.*r_s*A(sqrt_term-np.arctan(sqrt_term))/np.power(sqrt_term,3.)/cosmos.Sig_cr(z_c,z_g)
    return 1.+2.*kappa

def calc_expect_mu(r,A,n):
    mu_s = np.zeros(n)
    for k in range(n):
        mu_s[k] = dz_c*dz_g*np.sum(np.asarray([simp_int_coeff(j,n_bins_g)*p_g[k][j]*np.asarray([simp_int_coeff(i,n_bins_c)*p_c[i]*mu_SIS(z_step_c[i],z_step_g[j],r,A) for i in range(n_bins_c)]) for j in range(n_bins_g)]))/9.
    return mu_s

def calc_expect_mu_NFW(R,A,r_s,n):
    mu_s = np.zeros(n)
    for k in range(n):
        mu_s[k] = dz_c*dz_g*np.sum(np.asarray([simp_int_coeff(j,n_bins_g)*p_g[k][j]*np.asarray([simp_int_coeff(i,n_bins_c)*p_c[i]*mu_NFW(z_step_c[i],z_step_g[j],R,A,r_s) for i in range(n_bins_c)]) for j in range(n_bins_g)]))/9.
    return mu_s

def Fisher_matrix(A,sep,N,C_var):
    dA = 100
    n_tot = sep.size
    F = sum([N.T[i]*np.power((calc_expect_mu(sep[i],A-0.5*dA,1)-calc_expect_mu(sep[i],A+0.5*dA,1))/dA,2.) for i in range(n_tot)])*np.power(d_del_d_mu[0],2.)/C_var[0,0]
    return F

def multiple_profiles(sep,del_z_bar,N,b_g_rel_binned,c_z,n):
    C_var = np.matrix(np.diag(np.asarray([np.average(np.power(b_g_rel_binned[i][2],2.))-np.power(np.average(b_g_rel_binned[i][2]),2.) for i in range(n)])))
    C_inv = C_var.I
    def lnprob_n_fits(A,n):
        mu_s = np.asarray([calc_expect_mu(sep[i],A,n) for i in range(sep.size)])
        model_d_z_b = delta_z_bar_of_mu(mu_s,n)
        a = [float(np.matrix(del_z_bar.T[i]-model_d_z_b[i])*C_inv*np.matrix(np.diag(N.T[i]))*np.matrix(del_z_bar.T[i]-model_d_z_b[i]).T) for i in range(sep.size)]
        a = -0.5*sum(a)
        return a
    n_sect = 60
    A_step = np.linspace(0.,100000.,n_sect+1,endpoint=True)
    dA_step = A_step[1]-A_step[0]
    ln_prob = np.zeros(n_sect+1)
    for j in range(n_sect+1):
        ln_prob[j] = lnprob_n_fits(A_step[j],n)
    prob = np.exp(ln_prob)/simp_integral(dA_step,np.exp(ln_prob))
    print(A_step,prob)
    plt.title("Probability of A values")
    plt.plot(A_step, prob)
    plt.ylabel("Prob")
    plt.xlabel(r"A $\left ( \frac{M_{\odot}}{\mathrm{\mathsf{Mpc}}} \right )$")
    plt.subplots_adjust(left=0.16,bottom=0.14)
    #plt.savefig("auto/prob_of_A_"+str(n)+".png")
    #plt.close()
    plt.show()
    exp_A = simp_integral(dA_step,prob*A_step)
    var_A = simp_integral(dA_step,prob*np.power(A_step-exp_A,2.))
    print("A = "+str(exp_A))
    print("var_A = "+str(var_A))
    print("significance = "+str(exp_A/np.power(var_A,0.5)))
    Chi_2 = -2.0*lnprob_n_fits(exp_A,n)
    print("Chi Squared = "+str(Chi_2))
    F_sig = np.power(Fisher_matrix(exp_A,sep,N,C_var),-0.5)
    print("Fisher significance = "+str(exp_A/F_sig))
    return exp_A, var_A

def multiple_profiles_NFW(sep,del_z_bar,N,b_g_rel_binned,c_z,n):
    C_var = np.matrix(np.diag(np.asarray([np.average(np.power(b_g_rel_binned[i][2],2.))-np.power(np.average(b_g_rel_binned[i][2]),2.) for i in range(n)])))
    C_inv = C_var.I
    def lnprob_n_fits(A,r_s,n):
        mu_s = np.asarray([calc_expect_mu_NFW(sep[i],A,r_s,n) for i in range(sep.size)])
        model_d_z_b = delta_z_bar_of_mu(mu_s,n)
        a = [float(np.matrix(del_z_bar.T[i]-model_d_z_b[i])*C_inv*np.matrix(np.diag(N.T[i]))*np.matrix(del_z_bar.T[i]-model_d_z_b[i]).T) for i in range(sep.size)]
        a = -0.5*sum(a)
        return a
    n_sect = 60
    A_step = np.linspace(0.,100000.,n_sect+1,endpoint=True)
    dA_step = A_step[1]-A_step[0]
    ln_prob = np.zeros(n_sect+1)
    for j in range(n_sect+1):
        ln_prob[j] = lnprob_n_fits(A_step[j],n)
    prob = np.exp(ln_prob)/simp_integral(dA_step,np.exp(ln_prob))
    print(A_step,prob)
    plt.title("Probability of A values")
    plt.plot(A_step, prob)
    plt.ylabel("Prob")
    plt.xlabel(r"A $\left ( \frac{M_{\odot}}{\mathrm{\mathsf{Mpc}}} \right )$")
    plt.subplots_adjust(left=0.16,bottom=0.14)
    #plt.savefig("auto/prob_of_A_"+str(n)+".png")
    #plt.close()
    plt.show()
    exp_A = simp_integral(dA_step,prob*A_step)
    var_A = simp_integral(dA_step,prob*np.power(A_step-exp_A,2.))
    print("A = "+str(exp_A))
    print("var_A = "+str(var_A))
    print("significance = "+str(exp_A/np.power(var_A,0.5)))
    Chi_2 = -2.0*lnprob_n_fits(exp_A,n)
    print("Chi Squared = "+str(Chi_2))
    F_sig = np.power(Fisher_matrix(exp_A,sep,N,C_var),-0.5)
    print("Fisher significance = "+str(exp_A/F_sig))
    return exp_A, var_A

def multiple_profiles_i(sep,del_i_bar,N,b_g_rel_binned,c_z,n):
    C_var = np.matrix(np.diag(np.asarray([np.average(np.power(b_g_rel_binned[i][2],2.))-np.power(np.average(b_g_rel_binned[i][2]),2.) for i in range(n)])))
    C_inv = C_var.I
    def lnprob_n_fits(A,n):
        mu_s = np.asarray([calc_expect_mu(sep[i],A,n) for i in range(sep.size)])
        model_d_i_b = delta_i_bar_of_mu(mu_s,n)
        a = [float(np.matrix(del_i_bar.T[i]-model_d_i_b[i])*C_inv*np.matrix(np.diag(N.T[i]))*np.matrix(del_i_bar.T[i]-model_d_i_b[i]).T) for i in range(sep.size)]
        a = -0.5*sum(a)
        return a
    n_sect = 60
    A_step = np.linspace(-30000.,0.,n_sect+1,endpoint=True)
    dA_step = A_step[1]-A_step[0]
    ln_prob = np.zeros(n_sect+1)
    for j in range(n_sect+1):
        ln_prob[j] = lnprob_n_fits(A_step[j],n)
    prob = np.exp(ln_prob)/simp_integral(dA_step,np.exp(ln_prob))
    print(A_step,prob)
    plt.plot(A_step, prob)
    # plt.savefig("auto/prob_of_A_"+str(n)+".png")
    plt.show()
    exp_A = simp_integral(dA_step,prob*A_step)
    var_A = simp_integral(dA_step,prob*np.power(A_step-exp_A,2.))
    print("A = "+str(exp_A))
    print("var_A = "+str(var_A))
    print("significance = "+str(exp_A/np.power(var_A,0.5)))
    print("Chi^2 = "+str(-2*lnprob_n_fits(exp_A,n)))
    return exp_A, var_A

# gal[3]+del_mag < 19.9 and
# gal[3]+del_mag < 19.86+1.6*((gal[8]-gal[6])-(gal[7]-gal[8])/8.-0.8) and
# i_fib2_lensing(gal[11],gal[2],del_mag) < 21.5 and
# i_psf_lensing(gal[9],gal[2],del_mag)-(gal[10]+del_mag) > 0.2+0.2*(20.-(gal[10]+del_mag)) and
# z_psf_lensing(gal[13],gal[2],del_mag)-(gal[12]+del_mag) > 9.125-0.46*(gal[12]+del_mag)

def i_index(i,i_bin_bounds):
    return int(i_bin_bounds[i > i_bin_bounds].size-1)

def calc_delta_z_bar_of_mu(b_g_rel,avg_z,mu,n,i_bin_bounds):
    n_s = np.asarray(range(n))
    del_mag = -2.5*np.log10(mu)
    b_g_lens = np.asarray([np.append(gal[0:14],i_index(gal[3]+del_mag,i_bin_bounds)) for gal in b_g_rel.T if gal[3]+del_mag < 19.9 and gal[3]+del_mag < 19.86+1.6*((gal[8]-gal[6])-(gal[7]-gal[8])/8.-0.8) and i_fib2_lensing(gal[11],gal[2],del_mag) < 21.5 and i_psf_lensing(gal[9],gal[2],del_mag)-(gal[10]+del_mag) > 0.2+0.2*(20.-(gal[10]+del_mag)) and z_psf_lensing(gal[13],gal[2],del_mag)-(gal[12]+del_mag) > 9.125-0.46*(gal[12]+del_mag)]).T
    z_values = [np.asarray([gal[2] for gal in b_g_lens.T if gal[14]==i]) for i in n_s]
    # plt.plot(b_g_lens[3],b_g_lens[9],"k,",alpha=0.2)
    # plt.show()
    avg_z_lens = np.asarray([np.average(z_values[i]) for i in n_s])
    N = np.asarray([z_values[i].size for i in n_s])
    z_lens_2 = np.asarray([np.average(np.power(z_values[i],2.)) for i in n_s])
    var = (z_lens_2-np.power(avg_z_lens,2.))/N
    return avg_z_lens-avg_z,var

def calc_delta_i_bar_of_mu(b_g_rel,avg_i,mu,n,i_bin_bounds):
    n_s = np.asarray(range(n))
    del_mag = -2.5*np.log10(mu)
    b_g_lens = np.asarray([np.append(gal[0:14],i_index(gal[3]+del_mag,i_bin_bounds)) for gal in b_g_rel.T if gal[3]+del_mag < 19.9 and gal[3]+del_mag < 19.86+1.6*((gal[8]-gal[6])-(gal[7]-gal[8])/8.-0.8) and i_fib2_lensing(gal[11],gal[2],del_mag) < 21.5 and i_psf_lensing(gal[9],gal[2],del_mag)-(gal[10]+del_mag) > 0.2+0.2*(20.-(gal[10]+del_mag)) and z_psf_lensing(gal[13],gal[2],del_mag)-(gal[12]+del_mag) > 9.125-0.46*(gal[12]+del_mag)]).T
    i_values = [np.asarray([gal[3]+del_mag for gal in b_g_lens.T if gal[14]==i]) for i in n_s]
    avg_i_lens = np.asarray([np.average(i_values[i]) for i in n_s])
    N = np.asarray([i_values[i].size for i in n_s])
    i_lens_2 = np.asarray([np.average(np.power(i_values[i],2.)) for i in n_s])
    var = (i_lens_2-np.power(avg_i_lens,2.))/N
    return avg_i_lens-avg_i,var

def i_psf_lensing(i_psf,z,del_mag):
    # b_a = -4.4072352882 +/- 21.4745528772
    # b_b = 2.31175641404 +/- 1.38873092046
    # b_c = 20.3842561863 +/- 0.150368797443
    a_a = -3.57472592782 # +/- 21.4836316382
    a_b = -1.50680587953 # +/- 1.38657131458
    a_c = 0.41050188253  # +/- 0.150461969855
    return i_psf+del_mag*(a_a*np.power(z-0.575,2.)+a_b*(z-0.575)+a_c)

def i_psf_cmm_rel(b_g_rel,nbins):
    z_bounds = np.linspace(0.45,0.7,nbins+1,endpoint=True)
    sqr_a = np.zeros(nbins)
    sqr_b = np.zeros(nbins)
    dev_a = np.zeros(nbins)
    dev_b = np.zeros(nbins)
    z_step = np.zeros(nbins)
    def what_ind(z):
        check = [1 for bound in z_bounds if z > bound]
        return len(check)-1
    print('calculating values...')
    b_g_rel = np.asarray([np.append(gal,what_ind(gal[2])) for gal in b_g_rel.T]).T
    print('done.')
    def pivot(z):
        return -0.57465*np.exp(-6.82668*(z-0.45))+19.7321
    for j in range(nbins):
        b_g_rel_j = np.asarray([gal for gal in b_g_rel.T if gal[14] == j]).T
        # calcuating best fit line psf = a(cmm-19)+b
        if np.all(b_g_rel_j.shape > 1):
            i = b_g_rel_j[3]
            i_psf = b_g_rel_j[9]
            z_step[j] = (z_bounds[j+1]+z_bounds[j])/2.
            m = i.size
            Y = np.matrix(i_psf).T
            A = np.c_[np.matrix(np.ones(m)).T,np.matrix(i-pivot(z_step[j])).T]
            B = (A.T*A).I*(A.T*Y)
            sqr_b[j] = B[0,0]
            sqr_a[j] = B[1,0]
            def least_dev(x):
                return np.sum(np.abs(x[1]*(i-pivot(z_step[j]))+x[0]-i_psf))
            x_0 = np.asarray([sqr_b[j],sqr_a[j]])
            res = opt.minimize(least_dev, x_0, method='Nelder-Mead',options={'disp':True,'xatol':0.0001})
            dev_b[j] = res.x[0]
            dev_a[j] = res.x[1]
        i_step = np.linspace(17.5,20.,50,endpoint=True)
        if j == 0 or j == 17 or j == 99:
            '''plt.plot(i,i_psf,'.',color=(0.,0.,0.,0.3))
            plt.plot(i_step,sqr_a[j]*(i_step-pivot(z_step[j]))+sqr_b[j],'r')
            plt.plot(i_step,dev_a[j]*(i_step-pivot(z_step[j]))+dev_b[j],'b')
            plt.title(r"$i_{psf}$ vs. i for z = "+str(z_step[j]))
            plt.xlabel("i")
            plt.ylabel(r"$i_{psf}$")
            plt.show()'''
    def lnprob_for_i(x):
        if -100. < x[1] < 0.:
            a = -0.5*np.sum(np.power(x[0]*np.exp(x[1]*(z_step-0.45))+x[2]-means,2.))/0.0001
            return a
        else:
            return -np.inf
    # working on least square data for a
    m = sqr_a.size
    Y = np.matrix(sqr_a).T
    A = np.c_[np.matrix(np.ones(m)).T,np.matrix(z_step-0.575).T,np.matrix(np.power(z_step-0.575,2.)).T]
    cov_SQR = (A.T*A).I
    SQR = cov_SQR*(A.T*Y)
    print(cov_SQR)
    print(SQR)
    a_c_sqr = SQR[0,0]
    a_c_var = cov_SQR[0,0]
    a_b_sqr = SQR[1,0]
    a_b_var = cov_SQR[1,1]
    a_a_sqr = SQR[2,0]
    a_a_var = cov_SQR[2,2]
    # working on least deviation data a
    Y = np.matrix(dev_a).T
    DEV = (A.T*A).I*(A.T*Y)
    print(DEV)
    a_c_dev = DEV[0,0]
    a_b_dev = DEV[1,0]
    a_a_dev = DEV[2,0]
    # combining least square and least deviation for a
    a_a = (a_a_sqr+a_a_dev)/2.
    a_a_sig = np.power(a_a_var+np.power((a_a_sqr-a_a_dev)/2.,2.),0.5)
    a_b = (a_b_sqr+a_b_dev)/2.
    a_b_sig = np.power(a_b_var+np.power((a_b_sqr-a_b_dev)/2.,2.),0.5)
    a_c = (a_c_sqr+a_c_dev)/2.
    a_c_sig = np.power(a_c_var+np.power((a_c_sqr-a_c_dev)/2.,2.),0.5)
    print('a_a is '+str(a_a)+' +/- '+str(a_a_sig))
    print('a_b is '+str(a_b)+' +/- '+str(a_b_sig))
    print('a_c is '+str(a_c)+' +/- '+str(a_c_sig))
    plt.plot(z_step,sqr_a,'ro')
    plt.plot(z_step,a_a*np.power(z_step-0.575,2.)+a_b*(z_step-0.575)+a_c,'k')
    plt.plot(z_step,dev_a, 'bo')
    plt.title(r'z dependence of $a$ between $i$ and $i_{psf}$')
    plt.xlabel('z')
    plt.ylabel(r'$a$')
    plt.show()
    # working on least square for b
    Y = np.matrix(sqr_b).T
    cov_SQR = (A.T*A).I
    SQR = cov_SQR*(A.T*Y)
    print(SQR)
    b_c_sqr = SQR[0,0]
    b_c_var = cov_SQR[0,0]
    b_b_sqr = SQR[1,0]
    b_b_var = cov_SQR[1,1]
    b_a_sqr = SQR[2,0]
    b_a_var = cov_SQR[2,2]
    # working on least deviation for b
    Y = np.matrix(dev_b).T
    DEV = (A.T*A).I*(A.T*Y)
    print(DEV)
    b_c_dev = DEV[0,0]
    b_b_dev = DEV[1,0]
    b_a_dev = DEV[2,0]
    # combining least squares and least deviation for b
    b_a = (b_a_sqr+b_a_dev)/2.
    b_b = (b_b_sqr+b_b_dev)/2.
    b_c = (b_c_sqr+b_c_dev)/2.
    b_a_sig = np.power(b_a_var+np.power((b_a_sqr-b_a_dev)/2.,2.),0.5)
    b_b_sig = np.power(b_b_var+np.power((b_b_sqr-b_b_dev)/2.,2.),0.5)
    b_c_sig = np.power(b_c_var+np.power((b_c_sqr-b_c_dev)/2.,2.),0.5)
    print('b_a is '+str(b_a)+' +/- '+str(b_a_sig))
    print('b_b is '+str(b_b)+' +/- '+str(b_b_sig))
    print('b_c is '+str(b_c)+' +/- '+str(b_c_sig))
    plt.plot(z_step,sqr_b,'ro')
    plt.plot(z_step,b_a*np.power(z_step-0.575,2.)+b_b*(z_step-0.575)+b_c,'k')
    plt.plot(z_step,dev_b,'bo')
    plt.title(r'z dependence of $b$ between $i$ and $i_{psf}$')
    plt.xlabel('z')
    plt.ylabel(r'$b$')
    plt.show()

def i_fib2_lensing(i_fib2,z,del_mag):
    # b_a is -4.38195964126 +/- 21.4764771664
    # b_b is 2.25821363505 +/- 1.38951789428
    # b_c is 20.9840750622 +/- 0.150163657267
    a_a = -3.5487537242  # +/- 21.4765329904
    a_b = -1.38710326979 # +/- 1.38589704853
    a_c = 0.466890911038 # +/- 0.150525523687
    return i_fib2+del_mag*(a_a*np.power(z-0.575,2.)+a_b*(z-0.575)+a_c)

def i_fib2_cmm_rel(b_g_rel,nbins):
    z_bounds = np.linspace(0.45,0.7,nbins+1,endpoint=True)
    sqr_a = np.zeros(nbins)
    sqr_b = np.zeros(nbins)
    dev_a = np.zeros(nbins)
    dev_b = np.zeros(nbins)
    z_step = np.zeros(nbins)
    def what_ind(z):
        check = [1 for bound in z_bounds if z > bound]
        return len(check)-1
    print('calculating values...')
    b_g_rel = np.asarray([np.append(gal,what_ind(gal[2])) for gal in b_g_rel.T]).T
    print('done.')
    def pivot(z):
        return -0.57465*np.exp(-6.82668*(z-0.45))+19.7321
    for j in range(nbins):
        b_g_rel_j = np.asarray([gal for gal in b_g_rel.T if gal[14] == j]).T
        # calcuating best fit line psf = a(cmm-19)+b
        if np.all(b_g_rel_j.shape > 1):
            i = b_g_rel_j[3]
            i_fib2 = b_g_rel_j[11]
            z_step[j] = (z_bounds[j+1]+z_bounds[j])/2.
            m = i.size
            Y = np.matrix(i_fib2).T
            A = np.c_[np.matrix(np.ones(m)).T,np.matrix(i-pivot(z_step[j])).T]
            B = (A.T*A).I*(A.T*Y)
            sqr_b[j] = B[0,0]
            sqr_a[j] = B[1,0]
            def least_dev(x):
                return np.sum(np.abs(x[1]*(i-pivot(z_step[j]))+x[0]-i_fib2))
            x_0 = np.asarray([sqr_b[j],sqr_a[j]])
            res = opt.minimize(least_dev, x_0, method='Nelder-Mead',options={'disp':True,'xatol':0.0001})
            dev_b[j] = res.x[0]
            dev_a[j] = res.x[1]
        i_step = np.linspace(17.5,20.,50,endpoint=True)
        if j == 0 or j == 17 or j == 99:
            plt.plot(i,i_fib2,'.',color=(0.,0.,0.,0.3))
            plt.plot(i_step,sqr_a[j]*(i_step-pivot(z_step[j]))+sqr_b[j],'r')
            plt.plot(i_step,dev_a[j]*(i_step-pivot(z_step[j]))+dev_b[j],'b')
            plt.title(r"$i_{fib2}$ vs. i for z = "+str(z_step[j]))
            plt.xlabel("i")
            plt.ylabel(r"$i_{fib2}$")
            plt.show()
    # working on least square data for a
    m = sqr_a.size
    Y = np.matrix(sqr_a).T
    A = np.c_[np.matrix(np.ones(m)).T,np.matrix(z_step-0.575).T,np.matrix(np.power(z_step-0.575,2.)).T]
    cov_SQR = (A.T*A).I
    SQR = cov_SQR*(A.T*Y)
    print(cov_SQR)
    print(SQR)
    a_c_sqr = SQR[0,0]
    a_c_var = cov_SQR[0,0]
    a_b_sqr = SQR[1,0]
    a_b_var = cov_SQR[1,1]
    a_a_sqr = SQR[2,0]
    a_a_var = cov_SQR[2,2]
    # working on least deviation data a
    Y = np.matrix(dev_a).T
    DEV = (A.T*A).I*(A.T*Y)
    print(DEV)
    a_c_dev = DEV[0,0]
    a_b_dev = DEV[1,0]
    a_a_dev = DEV[2,0]
    # combining least square and least deviation for a
    a_a = (a_a_sqr+a_a_dev)/2.
    a_a_sig = np.power(a_a_var+np.power((a_a_sqr-a_a_dev)/2.,2.),0.5)
    a_b = (a_b_sqr+a_b_dev)/2.
    a_b_sig = np.power(a_b_var+np.power((a_b_sqr-a_b_dev)/2.,2.),0.5)
    a_c = (a_c_sqr+a_c_dev)/2.
    a_c_sig = np.power(a_c_var+np.power((a_c_sqr-a_c_dev)/2.,2.),0.5)
    print('a_a is '+str(a_a)+' +/- '+str(a_a_sig))
    print('a_b is '+str(a_b)+' +/- '+str(a_b_sig))
    print('a_c is '+str(a_c)+' +/- '+str(a_c_sig))
    plt.plot(z_step,sqr_a,'ro')
    plt.plot(z_step,a_a*np.power(z_step-0.575,2.)+a_b*(z_step-0.575)+a_c,'k')
    plt.plot(z_step,dev_a, 'bo')
    plt.title(r'z dependence of $a$ between $i$ and $i_{psf}$')
    plt.xlabel('z')
    plt.ylabel(r'$a$')
    plt.show()
    # working on least square for b
    Y = np.matrix(sqr_b).T
    cov_SQR = (A.T*A).I
    SQR = cov_SQR*(A.T*Y)
    print(SQR)
    b_c_sqr = SQR[0,0]
    b_c_var = cov_SQR[0,0]
    b_b_sqr = SQR[1,0]
    b_b_var = cov_SQR[1,1]
    b_a_sqr = SQR[2,0]
    b_a_var = cov_SQR[2,2]
    # working on least deviation for b
    Y = np.matrix(dev_b).T
    DEV = (A.T*A).I*(A.T*Y)
    print(DEV)
    b_c_dev = DEV[0,0]
    b_b_dev = DEV[1,0]
    b_a_dev = DEV[2,0]
    # combining least squares and least deviation for b
    b_a = (b_a_sqr+b_a_dev)/2.
    b_b = (b_b_sqr+b_b_dev)/2.
    b_c = (b_c_sqr+b_c_dev)/2.
    b_a_sig = np.power(b_a_var+np.power((b_a_sqr-b_a_dev)/2.,2.),0.5)
    b_b_sig = np.power(b_b_var+np.power((b_b_sqr-b_b_dev)/2.,2.),0.5)
    b_c_sig = np.power(b_c_var+np.power((b_c_sqr-b_c_dev)/2.,2.),0.5)
    print('b_a is '+str(b_a)+' +/- '+str(b_a_sig))
    print('b_b is '+str(b_b)+' +/- '+str(b_b_sig))
    print('b_c is '+str(b_c)+' +/- '+str(b_c_sig))
    plt.plot(z_step,sqr_b,'ro')
    plt.plot(z_step,b_a*np.power(z_step-0.575,2.)+b_b*(z_step-0.575)+b_c,'k')
    plt.plot(z_step,dev_b,'bo')
    plt.title(r'z dependence of $b$ between $i$ and $i_{psf}$')
    plt.xlabel('z')
    plt.ylabel(r'$b$')
    plt.show()

def z_psf_lensing(z_psf,z,del_mag):
    # b_a is -4.95649648383 +/- 21.4716326096
    # b_b is 1.8902895995 +/- 1.38617389848
    # b_c is 19.8969551834 +/- 0.150418087437
    a_a = -0.359690643443 # +/- 21.4738333702
    a_b = -1.08224600757  # +/- 1.38824216295
    a_c = 0.409516957636  # +/- 0.150052959636
    return z_psf+del_mag*(a_a*np.power(z-0.575,2.)+a_b*(z-0.575)+a_c)

def z_psf_zmm_rel(b_g_rel,nbins):
    z_bounds = np.linspace(0.45,0.7,nbins+1,endpoint=True)
    sqr_a = np.zeros(nbins)
    sqr_b = np.zeros(nbins)
    dev_a = np.zeros(nbins)
    dev_b = np.zeros(nbins)
    z_step = np.zeros(nbins)
    means = np.zeros(nbins)
    def what_ind(z):
        check = [1 for bound in z_bounds if z > bound]
        return len(check)-1
    print('calculating values...')
    b_g_rel = np.asarray([np.append(gal,what_ind(gal[2])) for gal in b_g_rel.T]).T
    print('done.')
    def pivot(z):
        return -0.55478*np.exp(-6.44553*(z-0.45))+19.30260
    for j in range(nbins):
        b_g_rel_j = np.asarray([gal for gal in b_g_rel.T if gal[14] == j]).T
        # calcuating best fit line psf = a(cmm-19)+b
        if np.all(b_g_rel_j.shape > 1):
            zm = b_g_rel_j[12]
            zm_psf = b_g_rel_j[13]
            z_step[j] = (z_bounds[j+1]+z_bounds[j])/2.
            means[j] = np.average(zm)
            m = zm.size
            Y = np.matrix(zm_psf).T
            A = np.c_[np.matrix(np.ones(m)).T,np.matrix(zm-pivot(z_step[j])).T]
            B = (A.T*A).I*(A.T*Y)
            sqr_b[j] = B[0,0]
            sqr_a[j] = B[1,0]
            def least_dev(x):
                return np.sum(np.abs(x[1]*(zm-pivot(z_step[j]))+x[0]-zm_psf))
            x_0 = np.asarray([sqr_b[j],sqr_a[j]])
            res = opt.minimize(least_dev, x_0, method='Nelder-Mead',options={'disp':True,'xatol':0.0001})
            dev_b[j] = res.x[0]
            dev_a[j] = res.x[1]
        zm_step = np.linspace(17.5,20.,50,endpoint=True)
        if j == 0 or j == 17 or j == 99:
            plt.plot(zm,zm_psf,'.',color=(0.,0.,0.,0.3))
            plt.plot(zm_step,sqr_a[j]*(zm_step-pivot(z_step[j]))+sqr_b[j],'r')
            plt.plot(zm_step,dev_a[j]*(zm_step-pivot(z_step[j]))+dev_b[j],'b')
            plt.title(r"$z_{psf}$ vs. $z_{mod}$ for z = "+str(z_step[j]))
            plt.xlabel(r"$z_{mod}$")
            plt.ylabel(r"$i_{psf}$")
            plt.show()
    # working on least square data for a
    m = sqr_a.size
    Y = np.matrix(sqr_a).T
    A = np.c_[np.matrix(np.ones(m)).T,np.matrix(z_step-0.575).T,np.matrix(np.power(z_step-0.575,2.)).T]
    cov_SQR = (A.T*A).I
    SQR = cov_SQR*(A.T*Y)
    print(cov_SQR)
    print(SQR)
    a_c_sqr = SQR[0,0]
    a_c_var = cov_SQR[0,0]
    a_b_sqr = SQR[1,0]
    a_b_var = cov_SQR[1,1]
    a_a_sqr = SQR[2,0]
    a_a_var = cov_SQR[2,2]
    # working on least deviation data a
    Y = np.matrix(dev_a).T
    DEV = (A.T*A).I*(A.T*Y)
    print(DEV)
    a_c_dev = DEV[0,0]
    a_b_dev = DEV[1,0]
    a_a_dev = DEV[2,0]
    # combining least square and least deviation for a
    a_a = (a_a_sqr+a_a_dev)/2.
    a_a_sig = np.power(a_a_var+np.power((a_a_sqr-a_a_dev)/2.,2.),0.5)
    a_b = (a_b_sqr+a_b_dev)/2.
    a_b_sig = np.power(a_b_var+np.power((a_b_sqr-a_b_dev)/2.,2.),0.5)
    a_c = (a_c_sqr+a_c_dev)/2.
    a_c_sig = np.power(a_c_var+np.power((a_c_sqr-a_c_dev)/2.,2.),0.5)
    print('a_a is '+str(a_a)+' +/- '+str(a_a_sig))
    print('a_b is '+str(a_b)+' +/- '+str(a_b_sig))
    print('a_c is '+str(a_c)+' +/- '+str(a_c_sig))
    plt.plot(z_step,sqr_a,'ro')
    plt.plot(z_step,a_a*np.power(z_step-0.575,2.)+a_b*(z_step-0.575)+a_c,'k')
    plt.plot(z_step,dev_a, 'bo')
    plt.title(r'z dependence of $a$ between $i$ and $i_{psf}$')
    plt.xlabel('z')
    plt.ylabel(r'$a$')
    plt.show()
    # working on least square for b
    Y = np.matrix(sqr_b).T
    cov_SQR = (A.T*A).I
    SQR = cov_SQR*(A.T*Y)
    print(SQR)
    b_c_sqr = SQR[0,0]
    b_c_var = cov_SQR[0,0]
    b_b_sqr = SQR[1,0]
    b_b_var = cov_SQR[1,1]
    b_a_sqr = SQR[2,0]
    b_a_var = cov_SQR[2,2]
    # working on least deviation for b
    Y = np.matrix(dev_b).T
    DEV = (A.T*A).I*(A.T*Y)
    print(DEV)
    b_c_dev = DEV[0,0]
    b_b_dev = DEV[1,0]
    b_a_dev = DEV[2,0]
    # combining least squares and least deviation for b
    b_a = (b_a_sqr+b_a_dev)/2.
    b_b = (b_b_sqr+b_b_dev)/2.
    b_c = (b_c_sqr+b_c_dev)/2.
    b_a_sig = np.power(b_a_var+np.power((b_a_sqr-b_a_dev)/2.,2.),0.5)
    b_b_sig = np.power(b_b_var+np.power((b_b_sqr-b_b_dev)/2.,2.),0.5)
    b_c_sig = np.power(b_c_var+np.power((b_c_sqr-b_c_dev)/2.,2.),0.5)
    print('b_a is '+str(b_a)+' +/- '+str(b_a_sig))
    print('b_b is '+str(b_b)+' +/- '+str(b_b_sig))
    print('b_c is '+str(b_c)+' +/- '+str(b_c_sig))
    plt.plot(z_step,sqr_b,'ro')
    plt.plot(z_step,b_a*np.power(z_step-0.575,2.)+b_b*(z_step-0.575)+b_c,'k')
    plt.plot(z_step,dev_b,'bo')
    plt.title(r'z dependence of $b$ between $i$ and $i_{psf}$')
    plt.xlabel('z')
    plt.ylabel(r'$b$')
    plt.show()

'''
phi_at_L_lim = []
L_lim = []
z = []
beta = []

def prepare_SIS_model_delta_z_bar(b_g_rel,z_g_min,z_g_max,n_bins):
    z_bounds = np.linspace(z_g_min,z_g_max,n_bins,endpoint=True)
    dz = (z_g_max-z_g_min)/n_bins
    for i in range(n_bins-1):
        z_in_bin = (z_bounds[i+1]+z_bounds[i])/2.
        L_mean_bin,phi_L,L_lim_i = phi(z_in_bin,dz,b_g_rel,100)
        if L_mean_bin.size > 2:
            near_i = np.amin(np.where(L_mean_bin>L_lim_i))
            global beta
            global phi_at_L_lim
            global L_lim
            global z
            beta.append(np.log(phi_L[near_i+1]/phi_L[near_i])/np.log(L_mean_bin[near_i+1]/L_mean_bin[near_i]))
            phi_at_L_lim.append(phi_L[near_i]/np.power(L_mean_bin[near_i]/L_lim_i,beta[near_i]))
            L_lim.append(L_lim_i)
            z.append(z_in_bin)
    phi_at_L_lim = np.asarray(phi_at_L_lim)
    L_lim = np.asarray(L_lim)
    z = np.asarray(z)
    beta = np.asarray(beta)
    print (z,phi_at_L_lim,L_lim,beta)

def SIS_model_delta_z_bar(r,A,c_rel,b_g_rel,z_steps,phi_at_L_lim,L_lim,beta):
    avr_z_c = np.average(c_rel[2])
    n_tot = float(b_g_rel[3].size)
    expect_z_n = np.average(b_g_rel[2])
    delta_n_tot = 0.
    expect_z_delta_n = 0.
    for i in range(z.size()-1):
        delta_n_tot = delta_n_tot + (z[i+1]-z[i])*(phi_at_L_lim[i]*L_lim[i]/(beta[i]+1)*(1-np.power(mu_SIS(avr_z_c,z[i],r,A),-beta[i]-1.))+phi_at_L_lim[i+1]*L_lim[i+1]/(beta[i+1]+1)*(1-np.power(mu_SIS(avr_z_c,z[i+1],r,A),-beta[i+1]-1.)))/2.
        expect_delta_n_tot = expect_delta_n_tot + (z[i+1]-z[i])*(z[i]*phi_at_L_lim[i]*L_lim[i]/(beta[i]+1)*(1-np.power(mu_SIS(avr_z_c,z[i],r,A),-beta[i]-1.))+z[i+1]*phi_at_L_lim[i+1]*L_lim[i+1]/(beta[i+1]+1)*(1-np.power(mu_SIS(avr_z_c,z[i+1],r,A),-beta[i+1]-1.)))/2.
    return (expect_z_delta_n-expect_z_n)/(1+n_tot/delta_n_tot)
'''
