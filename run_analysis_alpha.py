# This is the script to run the actual data
from delta_z_bar_analysis import Delta_z_bar
from fitting import FittingSignal, FittingAlpha
import matplotlib.pyplot as plt
import numpy as np

DZB = Delta_z_bar()
FS = FittingSignal(DZB.c_rel[2].copy(), DZB.b_g_rel[2].copy(), DZB.c_rel[4], clust_z_min=DZB.clust_z_max, clust_z_max=DZB.clust_z_max, gal_z_min=DZB.gal_z_min)
FA = FittingAlpha(DZB.b_g_rel, n=1, clust_z_min=DZB.clust_z_min, clust_z_max=DZB.clust_z_max)
z_step = FA.z_step
FS.get_quad_coeff_for_alpha(DZB.b_g_rel, 1)
alpha_step = FS.alpha(z_step)

# get all cuts data
a_s_all, a_vars_all = FA.find_alpha()

# get only i cuts data
FA.cuts = 'only i'
a_s_i, a_vars_i = FA.find_alpha()

# graph it
fig = plt.figure(figsize=(7, 8.5))
ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=4)
ax2 = plt.subplot2grid((5, 1), (4, 0), sharex=ax1)

alpha_all = ax1.errorbar(z_step, a_s_all, yerr=np.sqrt(a_vars_all), fmt='o', color='b')
alpha_i = ax1.errorbar(z_step, a_s_i, yerr=np.sqrt(a_vars_i), fmt='o', color='k')
model, = ax1.plot(z_step, alpha_step, color='k')
ax1.set_ylabel(r'$ \alpha(z_c) \ \left( \frac{Mpc^2}{M_\odot \ h} \right) $', fontsize=18)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.legend([alpha_i, alpha_all, model], ['Only i-magnitude Cut', 'All Cuts', r'Quadradic Fit to $\alpha(z_c)$'], bbox_to_anchor=(1, 0.18), bbox_transform=ax1.transAxes, numpoints=3)
abs_err = 100*(a_s_all/alpha_step - 1.)
abs_err_bars = 100*np.sqrt(a_vars_all)/alpha_step

ax2.errorbar(z_step, abs_err, yerr=abs_err_bars, fmt='o', color='b')
ax2.axhline(y=0, xmin=0, xmax=1, color='k')
ax2.set_ylabel('Percent Difference (%)\nBetween All Cuts and Fit', fontsize=10)
ax2.set_ylim([-5, 5])

plt.xlabel(r'$z_c$', fontsize=18)
plt.subplots_adjust(left=0.17, top=0.95)
plt.savefig('auto/alpha/alpha_comparison.png')
plt.close()

