# making it easier to plot
# (c) mattchelldavis

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def histogram(values,num_bin,range=None,color=None,alpha=1.,xlabel='',ylabel='',title='',grid=False,yscale='linear'):
    n, bins, patches = plt.hist(values,bins=num_bin,range=range,color=color,alpha=alpha)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale(yscale)
    plt.title(title)
    plt.grid(grid)
    plt.show()

def plot_posit(RA,DEC,color=None,marker='o'):
    plt.plot(RA,DEC,color=color,marker=marker)
    plt.xlabel('Right Ascention')
    plt.ylabel('Declination')
    plt.title('Clusters')
    plt.axis([0,360,-90,90])
    plt.show()
