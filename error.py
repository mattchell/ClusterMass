# This handles different errors for the corrilation function
# (c) mattchelldavis

import numpy as np
import treecorr
import sys
import kmeans_radec as kmrd
import matplotlib.pyplot as plt
from astropy.io import fits
import cosmology as cosmos


# write data to a text file
def write_data(array,path):
    dim = len(array.shape)
    if dim == 1:
        lines = ["{0:.12e}".format(val)+"\n" for val in array]

    elif dim == 2:
        lines = [" ".join(["{0:.12e}".format(val) for val in row])+"\n" for row in array]
    
    else:
        print("Writting data to a file only works for an array \nOf dimention 1 or 2.")
        exit()
    
    fo = open(path,'w')
    fo.writelines(lines)
    fo.close()


# read data from a text file, returns an array
def read_data(path):
    fo = open(path, 'r')
    lines = [line[:-1] for line in fo.readlines()]
    fo.close()
    array = np.array([[float(val) for val in line.split()] for line in lines])
    row_sizes = np.array([row.size for row in array])
    if np.all(row_sizes == 1):
        array = array[:,0]
    return array


# Calculates Covarience given two data sets which are the realizations of the
# two random variables x_i and x_j
def Cov(x_i,x_j):
    return np.average(x_i*x_j)-np.average(x_i)*np.average(x_j)


# Calcultes Covarience Matrix given a data set of realizations of random vector
# Realizations are along the 0th axis
def Covarience_Matrix(x_vector):
    nbins = x_vector[0].size
    Cov_Mat = np.zeros((nbins,nbins))
    for i in range(nbins):
        for j in range(i+1):
            Covarience = Cov(x_vector[:,i],x_vector[:,j])
            Cov_Mat[i][j] = Covarience
            if i != j:
                Cov_Mat[j][i] = Covarience
    return Cov_Mat


# Calculates the Corrilation matrix given a covarience matrix
def Corrilation_Matrix(Cov_Mat):
    nbins = Cov_Mat[0].size
    Corr_Mat = Cov_Mat.copy()
    for i in range(nbins):
        for j in range(nbins):
            Corr_Mat[i,j] = Cov_Mat[i,j]/np.power(Cov_Mat[i,i]*Cov_Mat[j,j],0.5)
    return Corr_Mat


# This displays a matrix in a aesthetically pleasing manner
def plot_matrix(Mat, sep, R_min, R_max, min_val=None, max_val=None,title=None,save=False,save_path='auto/corr.png'):
    fig, ax = plt.subplots()
    edges = np.logspace(np.log10(R_min), np.log10(R_max), sep.size+1)
    plt.pcolor(edges, edges, Mat, vmin=min_val, vmax=max_val)
    plt.title(title)
    plt.set_cmap('coolwarm')
    plt.colorbar()

    plt.xlabel(r'R $(Mpc \ h^{-1})$')
    plt.ylabel(r'R $(Mpc \ h^{-1})$')
    plt.ylim([R_min, R_max])
    plt.xlim([R_min, R_max])
    plt.yscale('log')
    plt.xscale('log')
    plt.subplots_adjust(bottom=0.15,right=0.9,top=0.88,left=0.2)
    labels = [str(label) for label in ax.get_xticks().tolist()]
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    if save:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


# Calcualtes the solid angle of a data set in square degrees where x is an array
# with ra, dec along the zeroth axis and objects along the first
def survey_Omega(x,nbins,min_dec,max_dec):
    dec_bounds, d_dec = np.linspace(min_dec, max_dec, nbins+1, endpoint=True, retstep=True)
    Omega = 0.
    for i in range(nbins):
        # isolate a single horizontal strip of the galaxy data which only
        # contains the ra data
        strip = x[0][np.where((dec_bounds[i] < x[1]) & (x[1] <= dec_bounds[i+1]))]
        if strip.size != 0:
            # add the solid angle of this this strip to the total
            d_Omega = (np.max(strip)-np.min(strip))*d_dec*np.cos(np.pi*(dec_bounds[i+1]+dec_bounds[i])/360.)
            Omega += d_Omega
    print("Solid Angle = {0}".format(Omega))
    return Omega


# Calculate the k-means centers of the survey given xsamp, an array with 
# ra, dec along the zeroth axis and the objects along the first axis
def find_centers(x_samp,ncen,RA_bounds,Dec_bounds,maxiter=100):
    for i in range(10):
        RA = RA_bounds[0]+(RA_bounds[1]-RA_bounds[0])*np.random.rand(ncen)
        Dec = Dec_bounds[0]+(Dec_bounds[1]-Dec_bounds[0])*np.random.rand(ncen)
        cen_guess = np.array([RA,Dec]).T
        #print(cen_guess)
        km = kmrd.KMeans(cen_guess,verbose=0)
        km.run(X=x_samp.T,maxiter=maxiter)
        sys.stdout.flush()
        if (not np.any(np.bincount(km.labels) == 0)):
            return km.centers
    print("Did not find a good set of centers")


# Find the labels for a data set given the centers.
def get_center_labels(x,cent):
    return kmrd.find_nearest(x.T,cent)


# This calcualtes jack knife errors from a kmeans algorythm or a txt file called centers.txt in the same directory / outputs np array
def JackKnife(nn, cat_g, b_g_rel, c_rel, mean_z, sep, nbins, R_min, R_max, n=1, ncen=0, read_from_file='False'):
    print("finding Jack Knife covarience matrix...")
    
    if not read_from_file:
        # split data into North and South caltalogues, 
        # and make a continuous one
        b_g_rel_solid = b_g_rel.copy()
        b_g_rel_solid[2] = np.power(b_g_rel_solid[2], n)
        b_g_rel_solid[0,(b_g_rel_solid[0] > 300.)] -= 360.
        b_g_rel_N_solid = np.array([gal for gal in b_g_rel_solid.T if 100. < gal[0]]).T
        b_g_rel_S_solid = np.array([gal for gal in b_g_rel_solid.T if gal[0] < 100.]).T
        c_rel_solid = c_rel.copy()
        c_rel_solid[0,(c_rel_solid[0] > 300.)] -= 360.

        if ncen == 0:
            print("The number of centers needs to be defined for the calculate method")
            exit()

        # Compute number of centers in the north and the south by ratio of
        # solid angles
        dec_max_N, dec_min_N = np.max(b_g_rel_N_solid[1]), np.min(b_g_rel_N_solid[1])
        dec_max_S, dec_min_S = np.max(b_g_rel_S_solid[1]), np.min(b_g_rel_S_solid[1])
        Omega_N = survey_Omega(b_g_rel_N_solid[0:2],100,dec_min_N,dec_max_N)
        Omega_S = survey_Omega(b_g_rel_S_solid[0:2],100,dec_min_S,dec_max_S)
        ncen_N = int(round(ncen*Omega_N/(Omega_N+Omega_S)))
        ncen_S = ncen-ncen_N
        #print(Omega_N/Omega_S, float(ncen_N)/ncen_S)
        del dec_max_N, dec_min_N, dec_max_S, dec_min_S, Omega_N, Omega_S
        
        # Compute the centers in the north and in the south
        cent_N = find_centers(b_g_rel_N_solid[0:2],ncen_N,[105.,260.],[-5.,70.])
        cent_S = find_centers(b_g_rel_S_solid[0:2],ncen_S,[-40.,50.],[-10.,25.])
        cent = np.append(cent_N,cent_S,axis=0)
        #print('centers found:')
        #print(cent)
        
        # Use the centers to get the labels for every galaxy and cluster
        glabels = get_center_labels(b_g_rel_solid[0:2],cent)
        clabels = get_center_labels(c_rel_solid[0:2],cent)
        b_g_rel_solid = np.append(b_g_rel_solid,[glabels],axis=0)
        c_rel_solid = np.append(c_rel_solid,[clabels],axis=0)
        for i in range(ncen):
            b_g_rel_i = np.array([gal for gal in b_g_rel_solid.T if gal[-1] == i]).T
            plt.plot(b_g_rel_i[0],b_g_rel_i[1],',',color=(1,float(i)/ncen,float(i)/ncen))
        plt.savefig("auto/corrilation/centers_{0}.png".format(n))
        plt.close()

        # Compute jack knife signals
        z_bar_J = np.zeros((ncen,nbins))
        for i in range(ncen):
            c_i = np.asarray([clust for clust in c_rel_solid.T if clust[-1] != i]).T
            cat_c_i = treecorr.Catalog(ra=c_i[0], dec=c_i[1], r=c_i[3], ra_units='deg', dec_units='deg')
            nn.process(cat_c_i,cat_g)
            z_bar_J[i] = (nn.weight/nn.npairs)-mean_z
        
        # Compute covariences from these jack knife signals
        Cov_mat = (ncen-1)*Covarience_Matrix(z_bar_J)
        
        # Store this Covarience matrix in the file data/JK_Covar.txt
        write_data(Cov_mat,'data/JK_covar_n={0}.txt'.format(n))
        
    else:
        # Read Covarience Matrix from the file
        Cov_mat = read_data('data/JK_covar_n={0}.txt'.format(n))
        
        # Checking for file errors
        if not Cov_mat.shape[0] == Cov_mat.shape[1]:
            print("Corrupted file: correct or run with method = 'calculate'")
            exit()
        if not Cov_mat.shape[0] == nbins:
            print("Dimentions of the Covarience Matrix doesn't match the number of bins you supplied")
            exit()

    Corr_mat = Corrilation_Matrix(Cov_mat)
    plot_matrix(Corr_mat, sep, R_min, R_max, min_val=-1., max_val=1., title='Jack Knife Corrilation Matrix', save=True, save_path='auto/corrilation/JK_corr_{0}.png'.format(n))
    return np.matrix(Cov_mat)


# Errors from multiple random cluster data sets
def Randoms(n_clust, clust_z_min, clust_z_max, nn, cat_g, mean_z, sep, nbins, R_min, R_max, n=1, read_from_file=False):
    print('finding Randoms covariance matrix...')

    if not read_from_file:
        # Get the random clusters and apply the same cuts as the sample analysis
        rand_clusters = fits.getdata('/calvin1/mattchell200/redmapper_catalogs/sdss/v5.10/dr8_rand_zmask2_redmapper_v5.10_randcat_z0.05-0.60_lgt020.fit')
        rand_c = np.array([rand_clusters['ra'].copy(),rand_clusters['dec'].copy(),rand_clusters['ztrue'].copy()])
        rand_c_rel = np.array([np.append(clust,cosmos.easy_D_A(clust[2])) for clust in rand_c.T if clust_z_min < clust[2] < clust_z_max]).T
        
        # split the random clusters randomly into 'trials' that are the same 
        # size as the cluster sample
        n_rand_clust = rand_c_rel[0].size
        n_trials = int(n_rand_clust/n_clust)
        rand_c_rel = np.array([np.append(clust,np.random.random_integers(0,n_trials-1)) for clust in rand_c_rel.T]).T
        
        # Run the corrilation on each trial and calculate the covarience
        z_bar_R = np.zeros((n_trials,nbins))
        for j in range(n_trials):
            trial_j = np.array([clust for clust in rand_c_rel.T if int(clust[4]) == j]).T
            cat_c_j = treecorr.Catalog(ra=trial_j[0].copy(), dec=trial_j[1].copy(),  r=trial_j[3].copy(), ra_units='deg', dec_units='deg')
            nn.process(cat_c_j,cat_g)
            z_bar_R[j] = nn.weight.copy()/nn.npairs.copy()-mean_z
        Cov_mat = Covarience_Matrix(z_bar_R)

        # Store this Covarience matrix in the file data/R_Covar.txt
        write_data(Cov_mat, 'data/R_covar_{0}.txt'.format(n))
        
    else:
        # Read Covarience Matrix from the file
        Cov_mat = read_data('data/R_covar_{0}.txt'.format(n))
    
    Corr_mat = Corrilation_Matrix(Cov_mat)
    plot_matrix(Corr_mat, sep, R_min, R_max, min_val=-1., max_val=1., title='Randoms Corrilation Matrix', save=True, save_path='auto/corrilation/R_corr_{0}.png'.format(n))
    return np.matrix(Cov_mat)

def Analytic(z_s, n, N, nbins, seps, R_min, R_max):
    seps = seps.flatten()
    Cov_mat = np.zeros((nbins*n, nbins*n))
    for i in range(nbins*n):
        for j in range(nbins*n):
            if i%nbins == j%nbins:
                k = i//nbins+1
                l = j//nbins+1
                Cov_mat[i, j] = (np.average(np.power(z_s, k+l)) - np.average(np.power(z_s, k))*np.average(np.power(z_s, l)))/N[i%nbins]
    Corr_mat = Corrilation_Matrix(Cov_mat)
    plot_matrix(Corr_mat, seps, R_min, R_max, min_val=-1., max_val=1., title='Analytic Corrilation Matrix', save=True, save_path='auto/corrilation/A_corr_{0}.png'.format(n))
    return np.matrix(Cov_mat)

def get_rand_z(IP_z):
    IP = np.random.rand()
    check = [i for i in range(np.sum(IP_z).size) if IP < IP_z[1][i]]
    j = check[0]-1
    return (IP_z[j+1][0]-IP_z[j][0])/(IP_z[j+1][1]-IP_z[j][1])*(IP-IP_z[j][1])+IP_z[j][0]

def get_rand_RA():
    return (RA_hi-RA_lo)*np.random.rand()+RA_lo

def get_rand_D(D_norm,D_off):
    IP = np.random.rand()
    D = np.arcsin(IP*D_norm+D_off)
    return 180.*D/np.pi

def paper_method(nn,cat_g,RA_hi,RA_lo,D_hi,D_lo,z_hi,z_lo):
    # pre-calculations for random z values
    z = np.linspace(z_lo,z_hi,num=100,endpoint=True)
    IP_max = IP(z_hi)
    IP_z = np.asarray([z,[IP(z_i)/IP_max for z_i in z]])
    del z,IP_max

    # pre-calculations for random Dec
    D_norm = np.sin(np.pi*D_hi/180.)-np.sin(np.pi*D_lo/180)
    D_off = np.sin(np.pi*D_lo/180.)
    
    # sig from random data
    rand_z_bar = []
    for i in range(100):
        random_clust = np.asarray([[get_rand_RA(),get_rand_D(D_norm,D_off),cosmo.LOSD(get_rand_z(IP_z))] for j in range(20000)]).T
        cat_c_i = treecorr.Catalog(ra=random_clust[0], dec=random_clust[1], r=random_clust[2], ra_units='deg', dec_units='deg')
        nn.process(cat_c_i,cat_g)
        rand_z_bar.append(nn_i.weight/nn_i.npairs)
    rand_z_bar = np.asarray(rand_z_bar)
    mean_rand_z_bar = np.average(rand_z_bar,0)
    sig = np.sqrt(20000./6000.*np.average(np.power(rand_z_bar-mean_rand_z_bar,2.0),0))
    return sig
