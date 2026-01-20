import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.func import vmap, jacrev
import os
from model import u_vec
from model import DensityNet
font = {'size'   : 16}
import matplotlib
matplotlib.rc('font', **font)
from matplotlib.colors import Normalize

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



def plotting_errors(model_list:list[DensityNet], t:float, model_names:list[str], path:str, rho_true_use:np.array, apx:str=''):
    num_models = len(model_list)
    
    x_min = -1.5
    x_max = 1.5
    dx = 0.01
    t_max = 10.0
    dt = 0.001
    plot_downsample = 5
    # Calculate the time
    t_plot = t/10.*t_max
    print(f'Plotting time: {t_plot}')
    # Index of the time step
    t_ind = int(t_plot/dt)
    #N = 500
    points_x = np.arange(x_min,x_max+dx,dx)
    points_y = np.arange(x_min,x_max+dx,dx)
    rho_true_plot = rho_true_use[t_ind].T
    
    # Generate the points grid
    X,Y = np.meshgrid(points_x,points_y)
    T = t_plot*np.ones_like(X.reshape((-1)))
    pts = np.vstack([T,X.reshape(-1),Y.reshape(-1)]).T
    # True velocity field
    vel = u_vec(torch.tensor(pts[:,1:])).numpy()
    #print(vel)
    U = np.array(vel[:,0].reshape(X.shape))
    V = np.array(vel[:,1].reshape(Y.shape))
    
    
    rhos = []
    for i, model in enumerate(model_list):
        rho = model.forward(torch.tensor(pts).to(model.device).float()).detach().cpu().numpy().reshape(X.shape)
        rhos.append(rho)
    rhos = np.array(rhos)
    
    cons = []
    for i, model in enumerate(model_list):
        con = model.evaluate_consistency(torch.tensor(pts).to(model.device).float()).detach().cpu().numpy().reshape(X.shape)
        cons.append(con)
    
    errors = []
    der_errors = []
    
    for i, model in enumerate(model_list):
        #error = np.sqrt((rhos[i]-rho_true_plot)**2)
        error = np.abs(rhos[i]-rho_true_plot)
        errors.append(error)
        
        
    errors = np.array(errors)
    vmin, vmax = (np.min(errors), np.max(errors))
    
    print(f'vmin: {vmin}, vmax: {vmax}')
    
    levels = np.linspace(vmin,vmax,50)
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    
    # Define subplots
    fig, ax = plt.subplots(nrows=1, ncols=num_models, figsize=(4*num_models,4), layout='compressed', sharex=True, sharey=True)

    #fig.suptitle('Phase space field error comparison')
    
    for i, model in enumerate(model_list):
        ax[i].set_xlim((x_min,x_max))
        ax[i].set_ylim((x_min,x_max))
        ax[i].set_title(model_names[i])
        ax[i].streamplot(X[::plot_downsample,::plot_downsample],Y[::plot_downsample,::plot_downsample],U[::plot_downsample,::plot_downsample],V[::plot_downsample,::plot_downsample],density=0.4,color='black', linewidth=0.05)
        ax[i].set_aspect('equal')
        contour = ax[i].contourf(X[::plot_downsample,::plot_downsample],Y[::plot_downsample,::plot_downsample],errors[i,::plot_downsample,::plot_downsample],50,cmap='jet', vmin=vmin, vmax=vmax, levels=levels)
        
    
    fig.colorbar(contour, ax=ax, orientation='vertical', fraction=0.02, pad=0.05)
    plt.savefig(os.path.join(path,f'errors_{apx}.pdf'))
    plt.close()
    
    
    
    # Now the comparison
    fig, ax = plt.subplots(nrows=1, ncols=num_models-1, figsize=(num_models*4-4,4), layout='compressed', sharex=True, sharey=True)
    
    comp = []
    for i in range(1,num_models):
        comp.append(errors[i] - errors[0])
        
    comp = np.array(comp)
    vmin = np.min(comp)
    vmax = np.max(comp)
    levels = np.linspace(vmin,vmax,50)
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    
    for i in range(1,num_models):
        ax[i-1].set_xlim((x_min,x_max))
        ax[i-1].set_ylim((x_min,x_max))
        ax[i-1].set_title(model_names[i]+' - DERL')
        ax[i-1].streamplot(X[::plot_downsample,::plot_downsample],Y[::plot_downsample,::plot_downsample],U[::plot_downsample,::plot_downsample],V[::plot_downsample,::plot_downsample],density=0.4,color='black', linewidth=0.05)
        ax[i-1].set_aspect('equal')
        contour = ax[i-1].contourf(X[::plot_downsample,::plot_downsample],Y[::plot_downsample,::plot_downsample],comp[i-1,::plot_downsample,::plot_downsample],50,cmap='seismic_r', vmin=vmin, vmax=vmax, levels=levels, norm=norm)
        
    fig.colorbar(contour, ax=ax, orientation='vertical', fraction=0.02, pad=0.05)
    
    plt.savefig(os.path.join(path,f'errors_comparison_{apx}.pdf'))
    
    
    
    
    # Now the same but with the consistencies
    fig, ax = plt.subplots(nrows=1, ncols=num_models, figsize=(4*num_models,4), layout='compressed', sharex=True, sharey=True)
    vmin, vmax = (np.min(cons), np.max(cons))
    print(f'vmin: {vmin}, vmax: {vmax}')
    cons = np.array(cons)
    
    levels = np.linspace(vmin,vmax,50)
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    for i, model in enumerate(model_list):
        ax[i].set_xlim((x_min,x_max))
        ax[i].set_ylim((x_min,x_max))
        ax[i].set_title(model_names[i])
        ax[i].streamplot(X[::plot_downsample,::plot_downsample],Y[::plot_downsample,::plot_downsample],U[::plot_downsample,::plot_downsample],V[::plot_downsample,::plot_downsample],density=0.4,color='black', linewidth=0.05)
        ax[i].set_aspect('equal')
        contour = ax[i].contourf(X[::plot_downsample,::plot_downsample],Y[::plot_downsample,::plot_downsample],cons[i,::plot_downsample,::plot_downsample],50,cmap='jet', vmin=vmin, vmax=vmax, levels=levels)
        
    
    fig.colorbar(contour, ax=ax, orientation='vertical', fraction=0.02, pad=0.05)
    plt.savefig(os.path.join(path,f'consistencies_{apx}.pdf'))
    plt.close()
    
    # Now the comparison
    fig, ax = plt.subplots(nrows=1, ncols=num_models-1, figsize=(num_models*4-4,4), layout='compressed', sharex=True, sharey=True)
    
    comp = []
    for i in range(1,num_models):
        comp.append(cons[i] - cons[0])
        
    comp = np.array(comp)
    vmin = np.min(comp)
    vmax = np.max(comp)
    levels = np.linspace(vmin,vmax,50)
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)

    for i in range(1,num_models):
        ax[i-1].set_xlim((x_min,x_max))
        ax[i-1].set_ylim((x_min,x_max))
        ax[i-1].set_title(model_names[i]+' - DERL')
        ax[i-1].streamplot(X[::plot_downsample,::plot_downsample],Y[::plot_downsample,::plot_downsample],U[::plot_downsample,::plot_downsample],V[::plot_downsample,::plot_downsample],density=0.4,color='black', linewidth=0.05)
        ax[i-1].set_aspect('equal')
        contour = ax[i-1].contourf(X[::plot_downsample,::plot_downsample],Y[::plot_downsample,::plot_downsample],comp[i-1,::plot_downsample,::plot_downsample],50,cmap='seismic_r', vmin=vmin, vmax=vmax, levels=levels, norm=norm)
    
    fig.colorbar(contour, ax=ax, orientation='vertical', fraction=0.02, pad=0.05)
    plt.savefig(os.path.join(path,f'consistencies_comparison_{apx}.pdf'))
    plt.close()
    