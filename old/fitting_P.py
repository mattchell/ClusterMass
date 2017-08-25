# Dealing with fitting and signal to noise information
# (c) mattchelldavis

import numpy as np
import treecorr
from colossus.cosmology import cosmology as csmg
from colossus.halo import concentration as concen
csmg.setCosmology('WMAP9')
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

# setting filler data for galaxy redshift distrbution
n_bins_g = 31
z_step_g = np.linspace(0.4,1.0,n_bins_g,endpoint=True)
dz_g = z_step_g[1]-z_step_g[0]
z_bounds_g = np.linspace(0.4-dz_g/2.,1.0+dz_g/2.,n_bins_g+1,endpoint=True)

# setting filler data for cluster redshift distrebution
n_bins_c = 31
z_step_c = np.linspace(0.1,0.33,n_bins_c,endpoint=True)
dz_c = z_step_c[1]-z_step_c[0]
z_bounds_c = np.linspace(0.1-dz_c/2.,0.33+dz_c/2.,n_bins_c+1,endpoint=True)

# finding actual cluster probability distribution
p_c = np.zeros(n_bins_c)
def set_p_of_c(z_c,clust_z_min,clust_z_max):
    global p_c,z_step_c,dz_c,z_bounds_c
    z_step_c = np.linspace(clust_z_min,clust_z_max,n_bins_c+1,endpoint=True)
    dz_c = z_step_c[1]-z_step_c[0]
    z_bounds_c = np.linspace(clust_z_min-dz_c/2.,clust_z_max+dz_c/2.,n_bins_c+1,endpoint=True)
    for i in range(n_bins_c):
        num_c = np.sum([1. for z_i in z_c if z_bounds_c[i] <= z_i < z_bounds_c[i+1]])
        p_c[i] = num_c/dz_c
    p_c = p_c/simp_integral(dz_c,p_c)

# finding the cluster redshift and richness distribution
p_c_zr = np.zeros((n_bins_c,n_bins_c))
dlamb = 0.
lamb_step = 0.
def set_p_of_zr(c_rel,lamb_min,lamb_max):
    global p_c_zr,dlamb,lamb_step
    lamb_step = np.linspace(lamb_min,lamb_max,n_bins_c,endpoint=True)
    dlamb = lamb_step[1]-lamb_step[0]
    lamb_bounds = np.linspace(lamb_min-dlamb/2.,lamb_max+dlamb/2.,n_bins_c+1,endpoint=True)
    print(z_step_c,z_bounds_c)
    norm = np.zeros(n_bins_c)
    for i in range(n_bins_c):
        check_z_i = np.asarray([clust[4] for clust in c_rel.T if z_bounds_c[i] <= clust[2] < z_bounds_c[i+1]])
        for j in range(n_bins_c):
            check = [lamb_i for lamb_i in check_z_i if lamb_bounds[j] <= lamb_i < lamb_bounds[j+1]]
            p_c_zr[i][j] = len(check)/dlamb/dz_c
        norm[i] = simp_integral(dlamb,p_c_zr[i])
    p_c_zr = p_c_zr/simp_integral(dz_c,norm)

# finding actual galaxy probability distribution
p_g = np.zeros(n_bins_g)
def set_p_of_g(z_b_g,gal_z_min):
    global p_g,z_step_g,z_bounds_g
    z_step_g = np.linspace(gal_z_min,1.0,n_bins_g,endpoint=True)
    dz_g = z_step_g[1]-z_step_g[0]
    z_bounds_g = np.linspace(gal_z_min-dz_g/2.,1.0+dz_g/2.,n_bins_g+1,endpoint=True)
    for i in range(n_bins_g):
        num_g = sum([1. for z_i in z_b_g if z_bounds_g[i] <= z_i < z_bounds_g[i+1]])
        p_g[i] = num_g/dz_g
    p_g = p_g/simp_integral(dz_g,p_g)

M_scale = 0.7*10.**14.344
#calculate M_200 from lambda
def mass_richness(lamb,A):
    return A*M_scale*np.power(lamb/40.,1.33)

def L_lim(z):
    return np.power(10., -19.9/2.5+1+z)*np.power(cosmos.D_L(z)/0.01, 2.)

def beta(z):
    M_star, alpha = -22.27-1.23*(z-0.5), -1.35
    return L_lim(z)*np.power(10., 0.4*M_star)-alpha

def delta_z(z_c,M_200,R,mean_z):
    c_200 = concen.concentration(M_200,'200c',z_c,model='diemer15')
    Sigma = Sig_NFW(M_200,c_200,R)
    mu = np.power(1.-Sigma/cosmos.Sig_cr(z_c,z_step_g),2)
    top_integrand = np.power(mu,(beta(z_step_g)-1.))*p_g*z_step_g
    bottom_integrand = np.power(mu,(beta(z_step_g)-1.))*p_g
    #print(top_integrand, bottom_integrand)
    if np.any(np.isnan(top_integrand)):
        print("M_200 = {3} \nz_c = {4} \nSigma = {0} \nSigma_cr = {1} \nbeta = {2}".format(Sigma, cosmos.Sig_cr(z_c,z_step_g), beta(z_step_g), M_200, z_c))
        exit()
    return simp_integral(dz_g,top_integrand)/(mean_z*simp_integral(dz_g,bottom_integrand))-1.

def expect_delta_z(M_200,R,mean_z):
    return np.average(np.array([delta_z(z_i,M_200,R,mean_z) for z_i in z_step_c]))

# Simpson's method for approximating an integral
def simp_integral(dx,y):
    tot_y = y.size
    if tot_y%2 == 1:
        coeffs = np.asarray([simp_int_coeff(i,tot_y) for i in range(tot_y)])
        return dx/3.*np.sum(y*coeffs)
    else:
        print("For simpson's rule you need an odd number of y values")

# Simpson's coefficents
def simp_int_coeff(i,n):
    if i == 0 or i == n-i:
        return 1.
    elif i%2 == 0:
        return 2.
    else:
        return 4.

# The magnification of a cluster given its NFW parameters
def Sig_NFW(M_200,c_200,R): # density(r) = A/(r/r_s)/(1+r/r_s)^2
    rho_bar = 5.997e28 # critical density of universe [M_sol/Mpc^3]
    A = (200*rho_bar/3.)*c_200/(np.log(1+c_200)-c_200/(1+c_200))
    r_s = np.power(3*M_200/(4*np.pi*200*rho_bar),1./3.)/c_200
    if R > r_s:
        sqrt_term = np.power(np.power(R/r_s,2.)-1,0.5)
        Sigma = 2.*r_s*A*(sqrt_term-np.arctan(sqrt_term))/np.power(sqrt_term,3.)
    elif R == r_s:
        Sigma = 2*A*r_s/3
    else: 
        sqrt_term = np.power(1-np.power(R/r_s,2.),0.5)
        Sigma = 2.*r_s*A*(np.arctanh(sqrt_term)-sqrt_term)/np.power(sqrt_term,3.)
    return Sigma

# fitting the data to the model using MCMC
def multiple_profiles(sep,del_z_bar,N,mean_z,var_z):
    C_var = var_z/np.power(mean_z, 2.)
    C_inv = 1./C_var
    def lnprob(x):
        start = time.time()
        M_200 = x[0]
        if 0. < M_200 and M_200 < 1.e30:
            model_d_z_b = np.array([expect_delta_z(M_200,sep[i],mean_z) for i in range(sep.size)])
            a = float(np.matrix(model_d_z_b-del_z_bar)*C_inv*np.matrix(np.diag(N))*np.matrix(model_d_z_b-del_z_bar).T)
            print(time.time()-start,a)
            return -0.5*a
        else:
            return -np.inf
    ndim, nwalkers, nsteps = 1, 2, 1500
    p0 = np.asarray([[2.e13*np.random.randn()+1.e14] for i in range(nwalkers)])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    sampler.run_mcmc(p0, nsteps)
    samples = sampler.chain[:,50:,:].reshape((-1,ndim))
    print(samples,ndim)
    fig = corner.corner(samples, labels=[r'$M_{200}$'])
    plt.savefig("auto/corner_of_M_200.png")
    plt.close()
    plt.plot(sampler.chain[0].T[0])
    plt.savefig("auto/chain.png")
    plt.close()
    exp_M_200 = np.average(samples.T[0])
    var_M_200 = np.average(np.power(samples.T[0],2.))-np.power(exp_M_200,2.)
    print("M_200 = {0}".format(exp_M_200))
    print("var_M_200 = {0}".format(var_M_200))
    print("significance = {0}".format(exp_M_200/np.power(var_M_200,0.5)))
    Chi_2 = -2.0*lnprob([exp_M_200])
    print("Chi Squared = {0}".format(Chi_2))
    return exp_M_200, var_M_200
