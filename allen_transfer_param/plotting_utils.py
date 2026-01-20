import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.func import vmap, jacrev
import os
lam = 0.01

font = {'size'   : 16}
import matplotlib
matplotlib.rc('font', **font)

from matplotlib.colors import Normalize

xlim = 1.
ylim = 1.
plot_downsample = 2

class MidpointNormalize(Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))
    

def comparison_plotting(errors, X, Y, model_names, path, compare_to='DERL'):
    num_models = len(model_names)
    # Get the min and max values for the colorbar
    vmin, vmax = (np.min(errors), np.max(errors))
        
    levels = np.linspace(vmin,vmax,50)
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    
    # Define subplots
    fig, ax = plt.subplots(nrows=1, ncols=num_models, figsize=(4*num_models,4), layout='compressed', sharex=True, sharey=True)

    #fig.suptitle('Phase space field error comparison')
    
    for i in range(num_models):
        ax[i].set_xlim((-xlim,xlim))
        ax[i].set_ylim((-ylim,ylim))
        ax[i].set_title(f'{model_names[i]} - {compare_to}')
        ax[i].set_aspect('equal')
        contour = ax[i].contourf(X,Y,errors[i].reshape(X.shape), 50,cmap='seismic_r', vmin=vmin, vmax=vmax, levels=levels, norm=norm)
    
    fig.colorbar(contour, ax=ax, orientation='vertical', fraction=0.02, pad=0.01)
    plt.savefig(path, format='pdf')
    plt.close()

def error_plotting(errors, X, Y, model_names, path):
    num_models = len(model_names)
    # Get the min and max values for the colorbar
    vmin, vmax = (np.min(errors), np.max(errors))
        
    levels = np.linspace(vmin,vmax,50)
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    
    # Define subplots
    fig, ax = plt.subplots(nrows=1, ncols=num_models, figsize=(4*num_models,4), layout='compressed', sharex=True, sharey=True)

    #fig.suptitle('Phase space field error comparison')
    
    for i in range(num_models):
        ax[i].set_xlim((-xlim,xlim))
        ax[i].set_ylim((-ylim,ylim))
        ax[i].set_title(model_names[i])
        ax[i].set_aspect('equal')
        contour = ax[i].contourf(X,Y,errors[i].reshape(X.shape),50,cmap='jet', vmin=vmin, vmax=vmax, levels=levels)
    

    fig.colorbar(contour, ax=ax, orientation='vertical', fraction=0.02, pad=0.05)
    plt.savefig(path, format='pdf')
    plt.close()
