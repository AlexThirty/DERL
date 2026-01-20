import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.func import vmap, jacrev
import os
lam = 0.01

font = {'size'   : 16}
import matplotlib
matplotlib.rc('font', **font)

def allen_cahn_true(x: np.array):
    return np.sin(np.pi*x[:,0])*np.sin(np.pi*x[:,1])

def allen_cahn_forcing(x: np.array):
    return -2*lam*np.pi**2*allen_cahn_true(x) + allen_cahn_true(x)**3 - allen_cahn_true(x)

def allen_cahn_pdv(x: np.array):
    ux = np.pi*np.cos(np.pi*x[:,0])*np.sin(np.pi*x[:,1])
    uy = np.pi*np.sin(np.pi*x[:,0])*np.cos(np.pi*x[:,1])
    return np.column_stack((ux, uy))
 
from matplotlib.colors import Normalize

xlim = 1.
ylim = 1.
plot_downsample = 5

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
        
    errors = np.array(errors)
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
        contour = ax[i].contourf(X[::plot_downsample,::plot_downsample],Y[::plot_downsample,::plot_downsample],errors[i,::plot_downsample,::plot_downsample], 50,cmap='seismic_r', vmin=vmin, vmax=vmax, levels=levels, norm=norm)
    
    fig.colorbar(contour, ax=ax, orientation='vertical', fraction=0.02, pad=0.01)
    plt.savefig(path, format='pdf')
    plt.close()

def error_plotting(errors, X, Y, model_names, path):
    num_models = len(model_names)
    # Get the min and max values for the colorbar
    vmin, vmax = (np.min(errors), np.max(errors))
    
    errors = np.array(errors)
    
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
        contour = ax[i].contourf(X[::plot_downsample,::plot_downsample],Y[::plot_downsample,::plot_downsample],errors[i,::plot_downsample,::plot_downsample],50,cmap='jet', vmin=vmin, vmax=vmax, levels=levels)
    

    fig.colorbar(contour, ax=ax, orientation='vertical', fraction=0.02, pad=0.05)
    plt.savefig(path, format='pdf')
    plt.close()
