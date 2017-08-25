# handeling the cosmology of the situation
# Units are Mpc, c, and M_Sol
# (c) mattchelldavis

import numpy as np
from scipy import integrate
from colossus.cosmology import cosmology as csmg
import matplotlib.pyplot as plt

# Cosmology Params
Om_M = 0.8          # Omega M
Om_L = 0.2          # Omega Lambda
D_H = 3000./.7      # Hubble Distance [Mpc]
G = 4.7856e-20      # Gravitational Constant [Mpc*c^2/M_Sol]


def E(z):
    return np.sqrt(Om_M*np.power(1+z,3.0)+Om_L)

def D_c(z):
    if type(z) == np.ndarray:
        results = np.zeros(z.size)
        for i in range(z.size):
            result = integrate.quad(lambda z_p: 1/E(z_p),0.0,z[i])
            results[i] = D_H*result[0]
        return results
    else:
        result = integrate.quad(lambda z_p: 1/E(z_p),0.0,z)
        return D_H*result[0]

def D_A(z):
    return D_c(z)/(z+1)

def D_L(z):
    return D_c(z)*(z+1)

Cosmology = csmg.setCosmology('WMAP9')
print(Cosmology.Om0)
Cosmology.Om0 = 0.8
Cosmology.OL0 = 0.2

# non-intesnive aproximation for Angular Diamerter Distance
z_step = np.linspace(0.0,2.0,num=100,endpoint=False)
#z_2 = np.linspace(2.0,8.0,num=12,endpoint=True)
#z = np.append(z_1,z_2)
z_tot = z_step.size
#DA_me = np.asarray([D_A(z_i) for z_i in z_step])
DA_r = np.asarray([Cosmology.angularDiameterDistance(z_i) for z_i in z_step])
#plt.plot(z_step, DA_me/DA_col, 'b')
#plt.plot(z_step, DA_col, 'r')
#plt.show()

def easy_D_A(z):
    if type(z) == np.ndarray:
        out_put = np.zeros(z.size)
        for i in range(z.size):
            j = np.where(z[i] < z_step)[0][0]
            out_put[i] = (DA_r[j]-DA_r[j-1])/(z_step[j]-z_step[j-1])*(z[i]-z_step[j])+DA_r[j]
        return out_put
    else:
        j = np.where(z < z_step)[0][0]
        return (DA_r[j]-DA_r[j-1])/(z_step[j]-z_step[j-1])*(z-z_step[j])+DA_r[j]

def Sig_cr(z_L,z_S): # Sigma Critical for redshift of the lens and source
    D_L = easy_D_A(z_L)
    D_S = easy_D_A(z_S)
    D_LS = D_S-(1+z_L)*D_L/(1+z_S)
    return D_S/(4.*np.pi*G*D_L*D_LS)
