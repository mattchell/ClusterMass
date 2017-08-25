# Dealing with fitting and signal to noise information
# (c) mattchelldavis

import numpy as np
import treecorr
from colossus.cosmology import cosmology as csmg
from colossus.halo import concentration as concen
from colossus.halo.profile_nfw import NFWProfile
#csmg.setCosmology('WMAP9')
import error as err
import cosmology as cosmos
import matplotlib.pyplot as plt
import scipy.optimize as opt
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.special import chdtr, erfinv, gamma


class FittingSignal:

    def __init__(self, z_c, z_g, lamb=0., clust_z_min=0., clust_z_max=1., lamb_min=20., lamb_max=200., gal_z_min=0., gal_z_max=1., n_bins_c=35, n_bins_g = 35):
        # finding galaxy redshift distribution
        p_g = np.zeros(n_bins_g)
        z_bounds_g, dz_g = np.linspace(gal_z_min, gal_z_max, n_bins_g+1, endpoint=True, retstep=True)
        z_step_g = np.linspace(gal_z_min + 0.5*dz_g, gal_z_max - 0.5*dz_g, n_bins_g, endpoint=True)
            
        for i in range(n_bins_g):
            p_g[i] = np.where(np.logical_and(z_g > z_bounds_g[i], z_g < z_bounds_g[i+1]))[0].size
            p_g[i] += 0.5*np.where(np.logical_or(z_g == z_bounds_g[i], z_g == z_bounds_g[i+1]))[0].size
        p_g = p_g/mid_integral(dz_g,p_g)

        self.p_g = p_g
        self.dz_g = dz_g
        self.z_step_g = z_step_g

        # finding the cluster redshift and richness distribution
        p_c = np.zeros((n_bins_c,n_bins_c))
            
        lamb_bounds, dlamb = np.linspace(lamb_min, lamb_max, n_bins_c+1, endpoint=True, retstep=True)
        lamb_step = np.linspace(lamb_min + 0.5*dlamb, lamb_max - 0.5*dlamb, n_bins_c, endpoint=True)
            
        z_bounds_c, dz_c = np.linspace(clust_z_min, clust_z_max, n_bins_c+1, endpoint=True, retstep=True)
        z_step_c = np.linspace(clust_z_min + 0.5*dz_c, clust_z_max - 0.5*dz_c, n_bins_c, endpoint=True)

        # itterate over the redshift bins
        for i in range(n_bins_c):
            # richness data for clusters that lie 
            # inside the redshift bin boundaries
            inside_lamb = lamb[np.where(np.logical_and(z_c > z_bounds_c[i], z_c < z_bounds_c[i+1]))]
            # richness data clusters that lie 
            # on the edges of the redshift bin boundaries 
            edge_lamb = lamb[np.where(np.logical_or(z_c == z_bounds_c[i], z_c == z_bounds_c[i+1]))]
            
            # itterate over the richness bins
            for j in range(n_bins_c):
                p_c[i, j] = np.where(np.logical_and(inside_lamb > lamb_bounds[j], inside_lamb < lamb_bounds[j+1]))[0].size
                p_c[i, j] += 0.5*np.where(np.logical_or(inside_lamb == lamb_bounds[j], inside_lamb == lamb_bounds[j+1]))[0].size
                p_c[i, j] += 0.5*np.where(np.logical_and(edge_lamb > lamb_bounds[j], edge_lamb < lamb_bounds[j+1]))[0].size
                p_c[i, j] += 0.25*np.where(np.logical_or(edge_lamb == lamb_bounds[j], edge_lamb == lamb_bounds[j+1]))[0].size
        p_c = p_c/mid_integral(np.array([dz_c, dlamb]),p_c)
        
        self.p_c = p_c
        self.dlamb = dlamb
        self.dz_c = dz_c
        self.z_step_c = z_step_c
        self.lamb_step = lamb_step
        self.clust_z_min = clust_z_min
        self.clust_z_max = clust_z_max

        return None


    # chi squared for multiple profiles
    def chi_sqrd(self, A, del_z_bar_flat, seps, n, Cov_mat_I):
        if A > 0:
            model_d_z_b = self.flat_model_delta_z_bar(A, seps, n)
        elif A == 0.:
            model_d_z_b = np.zeros(del_z_bar_flat.size)
        c_s = float(np.matrix(model_d_z_b-del_z_bar_flat)*Cov_mat_I*np.matrix(model_d_z_b-del_z_bar_flat).T)
        #print(np.matrix(model_d_z_b-del_z_bar_flat), Cov_mat_I, np.matrix(model_d_z_b-del_z_bar_flat).T, c_s)
        return c_s


    # fitting the data to the model using chi-squared
    def fit_for_A(self, seps, del_z_bar, N, b_g_rel, c_rel, n, Cov_mat, covar_type, nbins):
        # This is to isolate the n for debugging
        i, j = np.indices(Cov_mat.shape)
        mod_matrix = np.zeros(Cov_mat.shape)
        mod_matrix[np.logical_and(i == j, i >= nbins*(n-1))] = 1
        Cov_mat_I = np.matrix(mod_matrix)*Cov_mat.I*np.matrix(mod_matrix)
        # end debugging block
        self.c_rel = c_rel
        self.get_quad_coeff_for_alpha(b_g_rel, n)
        del_z_bar_flat = del_z_bar.flatten()
        print(del_z_bar_flat)
        A_step, dA = np.linspace(0., 10.**(2-n/2.), 199, endpoint=True, retstep=True)
        prob_step = np.asarray([np.exp(-0.5*self.chi_sqrd(A_i, del_z_bar_flat, seps, n, Cov_mat_I)) for A_i in A_step])
        prob_step *= 1./simp_integral(dA, prob_step)
        exp_A = simp_integral(dA, prob_step*A_step)
        var_A = simp_integral(dA, prob_step*np.power(A_step, 2)) - np.power(exp_A, 2)
        # find max probability A
        bi = np.array([0, 10**(2.-n/2.)])
        for i in range(20):
            left_chi = self.chi_sqrd(bi[0], del_z_bar_flat, seps, n, Cov_mat)
            right_chi = self.chi_sqrd(bi[1], del_z_bar_flat, seps, n, Cov_mat)
            if left_chi < right_chi:
                bi[1] = np.average(bi)
            else:
                bi[0] = np.average(bi)
        A_min_chi = np.average(bi)

        plt.plot(A_step, prob_step)
        plt.axvline(x=exp_A, ymin=0, ymax=1, color='k')
        plt.axvline(x=exp_A-np.sqrt(var_A), ymin=0, ymax=1, color='k', linestyle='--')
        plt.axvline(x=exp_A+np.sqrt(var_A), ymin=0, ymax=1, color='k', linestyle='--')
        plt.axvline(x=A_min_chi, ymin=0, ymax=1, color='r')
        plt.xlabel('A')
        plt.ylabel('Probability of fit')
        plt.savefig('auto/probability/combined/Prob_A_{0}_{1}.png'.format(n, covar_type))
        plt.close()
    
        CP_A = chdtr(nbins*n-1, self.chi_sqrd(0, del_z_bar_flat, seps, n, Cov_mat))
        signif_A = np.sqrt(2)*erfinv(CP_A)
        j = (np.where(np.array([ simp_integral(dA, prob_step[:i]) for i in range(3, A_step.size+1, 2) ]) > 0.95)[0][0] + 1)*2
        diff = 0.95 - simp_integral(dA, prob_step[:j-1])
        if 0.5*dA*(prob_step[j-2]+prob_step[j-1]) > diff:
            A_95 = A_step[j-2] + 0.5*dA
        else:
            A_95 = A_step[j-1] + 0.5*dA
            
            
        print("   Best fit A: {0}".format(exp_A))
        print("   A with max Prob: {0}".format(A_min_chi))
        print("   error bars: {0}".format(np.sqrt(var_A)))
        print("   significance: {0}".format(signif_A))
        print("   Chi^2 of fit: {0}".format(self.chi_sqrd(exp_A, del_z_bar_flat, seps, n, Cov_mat)))
        print("   95% confidence: {0}".format(A_95))
        return exp_A, var_A


    def get_quad_coeff_for_alpha(self, b_g_rel, n):
        quad_coeff = np.zeros((n, 3))
        for i in range(n):
            try:
                quad_coeff[i, :] = err.read_data("data/quadradic_fit_to_alpha_{0}.txt".format(n))
            except IOError:
                fa = FittingAlpha(b_g_rel, n=n, clust_z_min=self.clust_z_min, clust_z_max=self.clust_z_max)
                fa.find_alpha()
                fa.fit_alpha()
                fa.graph_alpha()
                quad_coeff[i, :] = err.read_data("data/quadradic_fit_to_alpha_{0}.txt".format(n))
        self.quad_coeff = quad_coeff
        return quad_coeff


    # alpha from model
    def alpha(self, z_c, i):
        return self.quad_coeff[i, 0]*np.power(z_c-0.215,2.)+self.quad_coeff[i, 1]*(z_c-0.215)+self.quad_coeff[i, 2]


    # flattened version of composite model
    def flat_model_delta_z_bar(self, A, seps, n):
        delta_z_bars = np.zeros(seps.shape)
        for i in range(n):
            delta_z_bars[i, :] = self.model_delta_z_bar(A, seps[i], i)
        return delta_z_bars.flatten()

    # calculate expectation value of signal over the sample using
    def model_delta_z_bar(self, A, Rs, i):
        delta_z_bars_i = np.asarray([self.alpha(clust[2], i)*self.Sigma_of_cluster(clust[2], clust[4], A, Rs) for clust in self.c_rel.T ])
        return np.average(delta_z_bars_i, axis=0)


    # calcualte signal of one cluster
    def Sigma_of_cluster(self, z_c, lamb, A, Rs):
        M_200 = self.mass_richness(lamb, A)
        c_200 = concen.concentration(M_200, '200c', z_c, model='diemer15')
        #NFW = NFWProfile(M=M_200, c=c_200, z=z_c, mdef='200c')
        return self.Sigma_NFW(M_200, c_200, Rs)


    # result from Simet with scaling constant 
    # calculate M_200 from lambda
    def mass_richness(self, lamb, A):
        return A*10.**14.344*np.power(lamb/40.,1.33)


    # The Sigma of a cluster given its NFW parameters
    def Sigma_NFW(self, M_200, c_200, Rs): # density(r) = A/(r/r_s)/(1+r/r_s)^2
        rho_bar = 5.997e28 # critical density of universe [M_sol/Mpc^3]
        Amp = (200*rho_bar/3.)*c_200/(np.log(1+c_200)-c_200/(1+c_200))
        r_s = np.power(3*M_200/(4*np.pi*200*rho_bar),1./3.)/c_200
        Sigma = Rs.copy()
        for i in range(Rs.size):
            if Rs[i] > r_s:
                sqrt_term = np.power(np.power(Rs[i]/r_s,2.)-1,0.5)
                Sigma[i] = 2.*r_s*Amp*(sqrt_term-np.arctan(sqrt_term))/np.power(sqrt_term,3.)
            elif Rs[i] == r_s:
                Sigma[i] = 2*Amp*r_s/3
            else: 
                sqrt_term = np.power(1-np.power(Rs[i]/r_s,2.),0.5)
                Sigma[i] = 2.*r_s*Amp*(np.arctanh(sqrt_term)-sqrt_term)/np.power(sqrt_term,3.)
        if np.all(Sigma == 0.):
            print('      Sigma profile = {0}, c_200 = {1}'.format(Sigma, c_200))
        return Sigma



class FittingAlpha:

    def __init__(self, b_g_rel, n=1, clust_z_min=0., clust_z_max=1., z_bins=30, cuts='all cuts'):
        self.z_bins = z_bins
        self.z_step, self.dz_c = np.linspace(clust_z_min, clust_z_max, z_bins, endpoint=True, retstep=True)
        self.b_g_rel = b_g_rel
        self.n = n
        self.cuts = cuts

        return None


    # calculating alpha at different cluster redshifts from the sample
    def find_alpha(self):
        a_s = np.zeros(self.z_bins) # preparing output
        a_vars = np.zeros(self.z_bins) # varience in a_s
        print("finding delta z bar dependence on Sigma...")
        self.n_point = 1 #number of Sigma values to fit
        Sigmas = np.linspace(-0.025*cosmos.Sig_cr(0.2,0.5), 0, self.n_point, endpoint=False)

        # find the signal on artificially lensed surveys 
        # over Simga and redshift values
        z_bar_of_Sigma, var, Ns = self.calc_delta_z_bar_of_Sig(Sigmas)

        # calcuating best fit line y = a(Sigma) at each redshift
        # this computes the values of alpha
        for i in range(self.z_bins):
            Y = np.matrix(z_bar_of_Sigma.T[i]).T
            A = np.matrix(Sigmas).T
            C = np.matrix([[var[min(j,k),i]/Ns[max(j,k),i] for j in range(self.n_point)] for k in range(self.n_point)])
            print(C)
            cov_B = (A.T*C.I*A).I
            B = cov_B*(A.T*C.I*Y)
            a = B[0,0]
            print("fitting a line to signa from de-lensed galaxy source\nAt cluster redshift z = {0:.4e}".format(self.z_step[i]))
            print("   slope: {0:.4e}".format(a))
            print("   standard deviation: {0:.4e}".format(np.sqrt(cov_B[0,0])))
            a_vars[i] = cov_B[0,0]
            a_s[i] = a
            Sigma_g = np.append(Sigmas,0.)
            z_bar_Sigma_g = np.append(z_bar_of_Sigma.T[i],0.)
            
            plt.plot(Sigma_g, z_bar_Sigma_g, "ro")
            plt.plot(Sigma_g, a*Sigma_g, "b")
            plt.title(r'z_c = {0:.4f}'.format(self.z_step[i]))
            plt.xlabel(r'$\Sigma$',size=20)
            plt.ylabel(r'$\Delta \bar z^{0}$'.format(self.n),size=20)
            plt.subplots_adjust(left=0.15)
            plt.savefig("auto/linear_fits/Delta_z^{0}_bar_Sig_fit_at_z_c={1:.4f}.png".format(self.n, self.z_step[i]))
            plt.close()
    
        self.a_s, self.a_vars = a_s, a_vars
        return a_s, a_vars


    # fit quadradic using the values and errors of alpha 
    def fit_alpha(self):
        Y = np.matrix(self.a_s).T
        A = np.matrix(np.asarray([np.power(self.z_step-0.215,2.), self.z_step-0.215, np.ones(self.z_step.size)])).T
        C = np.matrix(np.diag(self.a_vars))
        cov_B = (A.T*C.I*A).I
        B = cov_B*(A.T*C.I*Y)
        print(cov_B)
        print(np.array(B))
        self.a2, self.a1, self.a0 = B[0,0], B[1,0], B[2,0]
        self.cov_alpha_fit = cov_B
        
        return self.a2, self.a1, self.a0
    

    # graph alpha and its quadradit fit 
    def graph_alpha(self):   
        alpha_step = self.a2*np.power(self.z_step-0.215,2.) + self.a1*(self.z_step-0.215) + self.a0
        err_bars = np.power(self.a_vars, 0.5)
        
        fig = plt.figure(figsize=(7, 7))
        ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=4)
        ax2 = plt.subplot2grid((5, 1), (4, 0), sharex=ax1)
        
        alpha = ax1.errorbar(self.z_step, self.a_s, yerr=err_bars, fmt='o', color='b')
        model, = ax1.plot(self.z_step, alpha_step, color='k')
        ax1.set_ylabel(r'$ \alpha(z_c) \ \left( \frac{Mcp^2}{M_\odot} \right) $', fontsize=18)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.legend([alpha, model], [r'$\alpha(z_c)$', 'Quadradic Fit'], bbox_to_anchor=(0.38, 1.), bbox_transform=ax1.transAxes, numpoints=3)
        abs_err = 100*(self.a_s/alpha_step - 1.)
        abs_err_bars = 100*err_bars/alpha_step
        
        ax2.errorbar(self.z_step, abs_err, yerr=abs_err_bars, fmt='o', color='b')
        ax2.axhline(y=0, xmin=0, xmax=1, color='k')
        ax2.set_ylabel('Percent Difference (%)', fontsize=10)
        ax2.set_ylim([-5, 5])
        
        plt.xlabel(r'$z_c$', fontsize=18)
        plt.subplots_adjust(left=0.17, top=0.95)
        plt.savefig('auto/alpha/alpha_{0}_of_z_c.png'.format(self.n))
        plt.close()
    
        err.write_data(np.array([self.a2, self.a1, self.a0]), "data/quadradic_fit_to_alpha_{0}.txt".format(self.n))
        err.write_data(np.array(self.cov_alpha_fit), "data/quadradic_fit_to_alpha_{0}_variences.txt".format(self.n))
    
        return None


# gal[3]+del_mag < 19.9 and
# gal[3]+del_mag < 19.86+1.6*((gal[8]-gal[6])-(gal[7]-gal[8])/8.-0.8) and
# i_fib2_lensing(gal[11],gal[2],del_mag) < 21.5 and
# i_psf_lensing(gal[9],gal[2],del_mag)-(gal[10]+del_mag) > 0.2+0.2*(20.-(gal[10]+del_mag)) and
# z_psf_lensing(gal[13],gal[2],del_mag)-(gal[12]+del_mag) > 9.125-0.46*(gal[12]+del_mag)


    def del_mag(self, z_cl, z_gs, Sigma):
        return -2.5*np.log10(1+2.*Sigma/cosmos.Sig_cr(z_cl,z_gs))

    
    def calc_delta_z_bar_of_Sig(self, Sigmas):
        avg_z_lens = np.zeros((self.n_point, self.z_bins))
        z_var = np.zeros((self.n_point, self.z_bins))
        Ns = np.zeros((self.n_point, self.z_bins))
        mean_z = np.average(np.power(self.b_g_rel[2], self.n))
        
        # itterate over lens redshift
        for i in range(self.z_bins):
            print("z_c = {0:.4f}".format(self.z_step[i]))
            
            # itterate over Sigma values
            for j in range(self.n_point):
                print("   Sigma = {0:.4e}".format(Sigmas[j]))
                start = time.time()
                del_m = self.del_mag(self.z_step[i], self.b_g_rel[2], Sigmas[j])
                b_g_lens = self.artf_lens(del_m)
                '''plt.plot(b_g_lens[3],b_g_lens[9],"k,",alpha=0.2)
                plt.xlim((19.,19.9))
                plt.show()'''
                z_values = np.power(b_g_lens[2], self.n)
                avg_z_lens[j, i] = np.average(z_values)
                Ns[j, i] = z_values.size
                print("      number of galaxies: {0}".format(Ns[j, i]))
                z_var[j, i] = np.average(np.power(z_values, 2.))-np.power(mean_z, 2.)
                print("      time per delensing calcualtion: {0:.6f}".format(time.time()-start))
        return avg_z_lens-mean_z, z_var, Ns


    # SELECTION CUTS FOR SDSS
    # Cuts 1, 2, and 4 are irrelivant to lensing
    def artf_lens(self, del_m):
        # lensed data
        i_mags = self.b_g_rel[3] + del_m
        d_perps = (self.b_g_rel[8] - self.b_g_rel[6]) - (self.b_g_rel[7] - self.b_g_rel[8])/8.
        i_mods = self.b_g_rel[10] + del_m
        z_mods = self.b_g_rel[12] + del_m
        i_fib2s = i_fib2_lensing(self.b_g_rel[11], self.b_g_rel[2], del_m)
        i_psfs = i_psf_lensing(self.b_g_rel[9], self.b_g_rel[2], del_m)
        z_psfs = z_psf_lensing(self.b_g_rel[13], self.b_g_rel[2], del_m)
        
        if self.cuts == 'all cuts':
            # boolian data of satisfying cuts
            cut_2 = (i_mags < 19.86 + 1.6*(d_perps - 0.8))
            cut_3 = np.logical_and(i_mags > 17.5, i_mags < 19.9)
            cut_5 = (i_fib2s < 21.5)
            cut_6 = (i_psfs - i_mods > 0.2 + 0.2*(20.0 - i_mods))
            cut_7 = (z_psfs - z_mods > 9.125 - 0.46*z_mods)
            
            # all together now
            all_cuts = np.logical_and(np.logical_and(np.logical_and(cut_2, cut_3), np.logical_and(cut_5, cut_6)), cut_7)

        elif self.cuts == 'only i':
            all_cuts = np.logical_and(i_mags > 17.5, i_mags < 19.9)

        else:
            raise ValueError("I don't know what cuts to apply. I don't understand '{0}'".format(self.cuts))
        
        return self.b_g_rel.T[np.where(all_cuts)].T



# calculates the mid-remann sum assuming data is taken at 
# constant width dx and y is an array of function values at the mid-points
# data can also be given as multidimentional arrays 
def mid_integral(dxs, ys):
    """
    Computes mid-remann sum for 1-dimensional or n-dimensional data
    1-dimensional: provide dxs as a float and ys as an array
    n-dimensional: provide dxs as an array with n values and ys 
    as an multidimentional array with n or more axies
    """
    if (type(dxs) == float or type(dxs) == np.float64) and len(ys.shape) >= 1:
        return dxs*np.sum(ys, axis=0)
    elif type(dxs) == np.ndarray and len(ys.shape) >= dxs.size:
        return np.prod(dxs)*np.sum(ys, axis=tuple(range(len(dxs))))
    else:
        raise TypeError("Types inputed were not compatable or values did not match \ndxs: {0}\nys: {1}".format(dxs, ys))

# Simpson's method for approximating an integral
def simp_integral(dx,y):
    tot_y = y.size
    if tot_y%2 == 1:
        coeffs = np.asarray([simp_int_coeff(i,tot_y) for i in range(tot_y)])
        return dx/3.*np.sum(y*coeffs)
    else:
        raise ValueError("For simpson's rule you need an odd number of y values")

# Simpson's coefficents
def simp_int_coeff(i,n):
    if i == 0 or i == n-i:
        return 1.
    elif i%2 == 0:
        return 2.
    else:
        return 4.

# How i_psf is affected by magnification
def i_psf_lensing(i_psf,z,del_mag):
    # b_a = -4.4072352882 +/- 21.4745528772
    # b_b = 2.31175641404 +/- 1.38873092046
    # b_c = 20.3842561863 +/- 0.150368797443
    a_a = -3.57472592782 # +/- 21.4836316382
    a_b = -1.50680587953 # +/- 1.38657131458
    a_c = 0.41050188253  # +/- 0.150461969855
    return i_psf+del_mag*(a_a*np.power(z-0.575,2.)+a_b*(z-0.575)+a_c)

# calculating i_psf relationship to i in order to determin the how 
# magnification effects i_psf
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

# How i_fib2 is affected by magnification
def i_fib2_lensing(i_fib2,z,del_mag):
    # b_a is -4.38195964126 +/- 21.4764771664
    # b_b is 2.25821363505 +/- 1.38951789428
    # b_c is 20.9840750622 +/- 0.150163657267
    a_a = -3.5487537242  # +/- 21.4765329904
    a_b = -1.38710326979 # +/- 1.38589704853
    a_c = 0.466890911038 # +/- 0.150525523687
    return i_fib2+del_mag*(a_a*np.power(z-0.575,2.)+a_b*(z-0.575)+a_c)

# calculating i_fib2 relationship to i in order to determin the how 
# magnification effects i_fib2
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

# How z_psf is affected by magnification
def z_psf_lensing(z_psf,z,del_mag):
    # b_a is -4.95649648383 +/- 21.4716326096
    # b_b is 1.8902895995 +/- 1.38617389848
    # b_c is 19.8969551834 +/- 0.150418087437
    a_a = -0.359690643443 # +/- 21.4738333702
    a_b = -1.08224600757  # +/- 1.38824216295
    a_c = 0.409516957636  # +/- 0.150052959636
    return z_psf+del_mag*(a_a*np.power(z-0.575,2.)+a_b*(z-0.575)+a_c)

# calculating z_psf relationship to z in order to determin the how 
# magnification effects z_psf
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
