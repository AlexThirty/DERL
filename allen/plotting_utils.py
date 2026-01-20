import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.func import vmap, jacrev
import os
from model import AllenNet
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

 
def plotting_errors(model_list:list[AllenNet], model_names:list[str], path:str):
    # Define the grid
    xmin = -1.
    xmax = 1.
    dx = 0.01
    grid_size = dx**2
    x = np.arange(xmin, xmax+dx, dx)
    y = np.arange(xmin, xmax+dx, dx)
    x_plot, y_plot = np.meshgrid(x, y)
    x_pts = x_plot.reshape((-1,1))
    y_pts = y_plot.reshape((-1,1))
    pts = np.column_stack((x_pts, y_pts))
    u_grid = allen_cahn_true(pts).reshape((-1,1))
    forcing_grid = allen_cahn_forcing(pts)
    pdv_grid = allen_cahn_pdv(pts)
    
    # For each model, obtain the output, the forcing term and the pde residual
    u_models = []
    error_u_models = []
    forcing_models = []
    error_forcing_models = []
    pde_models = []
    pdv_models = []
    error_pdv_models = []
    
    for i, model in enumerate(model_list):
        print(f'Processing model {i}')
        # Function output part
        u = model.forward(torch.tensor(pts, dtype=torch.float32).to(model.device)).detach().cpu().numpy()
        u_models.append(u)
        error_u_models.append(np.abs(u - u_grid).reshape(x_plot.shape))
        print('Error u computed')
        # Forcing part
        forcing = model.evaluate_forcing(torch.tensor(pts, dtype=torch.float32).to(model.device)).detach().cpu().numpy()
        forcing_models.append(forcing)
        error_forcing_models.append(np.abs(forcing - forcing_grid).reshape(x_plot.shape))
        print('Error forcing computed')
        # Partial derivative part
        pdv = vmap(jacrev(model.forward_single))(torch.tensor(pts, dtype=torch.float32).to(model.device)).detach().cpu().numpy()[:,0,:]
        pdv_models.append(pdv)
        error_pdv_models.append(np.linalg.norm(pdv - pdv_grid, axis=1).reshape(x_plot.shape))
        print('Error pdv computed')
        # PDE residual part
        pde = model.evaluate_consistency(torch.tensor(pts, dtype=torch.float32).to(model.device)).detach().cpu().numpy()
        print(pde.shape)
        pde_models.append(pde.reshape(x_plot.shape))
        print('PDE residual computed')
    # Transform the 
    u_models = np.array(u_models)
    error_u_models = np.array(error_u_models)
    forcing_models = np.array(forcing_models)
    error_forcing_models = np.array(error_forcing_models)
    pdv_models = np.array(pdv_models)
    error_pdv_models = np.array(error_pdv_models)
    pde_models = np.array(pde_models)
    
    # Print the shapes
    print(f'u_models.shape: {u_models.shape}')
    print(f'error_u_models.shape: {error_u_models.shape}')
    print(f'forcing_models.shape: {forcing_models.shape}')
    print(f'error_forcing_models.shape: {error_forcing_models.shape}')
    print(f'pdv_models.shape: {pdv_models.shape}')
    print(f'error_pdv_models.shape: {error_pdv_models.shape}')
    print(f'pde_models.shape: {pde_models.shape}')
    
    
    # Error plotting
    error_plotting(error_u_models, x_plot, y_plot, model_names, path=f'{path}/errors.pdf')
    error_plotting(error_forcing_models, x_plot, y_plot, model_names, path=f'{path}/forcing_errors.pdf')
    error_plotting(error_pdv_models, x_plot, y_plot, model_names, path=f'{path}/pdv_errors.pdf')
    error_plotting(pde_models, x_plot, y_plot, model_names, path=f'{path}/pde_residuals.pdf')
    
    comparison_u_models = error_u_models[1:] - error_u_models[0]
    comparison_forcing_models = error_forcing_models[1:] - error_forcing_models[0]
    comparison_pdv_models = error_pdv_models[1:] - error_pdv_models[0]
    comparison_pde_models = pde_models[1:] - pde_models[0]
    
    
    comparison_plotting(comparison_u_models, x_plot, y_plot, model_names[1:], path=f'{path}/u_comparison.pdf')
    comparison_plotting(comparison_forcing_models, x_plot, y_plot, model_names[1:], path=f'{path}/forcing_comparison.pdf')
    comparison_plotting(comparison_pdv_models, x_plot, y_plot, model_names[1:], path=f'{path}/pdv_comparison.pdf')
    comparison_plotting(comparison_pde_models, x_plot, y_plot, model_names[1:], path=f'{path}/pde_comparison.pdf')
    
    # Now print the errors
    with open(f'{path}/errors.txt', 'w') as f:
        f.write('Errors for the Allen-Cahn problem\n')
        for i in range(len(model_names)):
            f.write(f'Model {model_names[i]}\n')
            f.write(f'Max error u: {np.max(error_u_models[i])}\n')
            f.write(f'Error u: {np.mean(error_u_models[i])} pm {np.std(error_u_models[i])}\n')
            f.write(f'Error u L2: {np.sqrt(grid_size*np.sum(error_u_models[i]**2))}\n')
            f.write(f'Error u L2 normalized: {np.sqrt(np.sum(error_u_models[i]**2)/np.sum(u_grid**2))}\n')
            f.write(f'Error forcing: {np.mean(error_forcing_models[i])} pm {np.std(error_forcing_models[i])}\n')
            f.write(f'Error forcing L2: {np.sqrt(grid_size*np.sum(error_forcing_models[i]**2))}\n')
            f.write(f'Error forcing L2 normalized: {np.sqrt(np.sum(error_forcing_models[i]**2)/np.sum(forcing_grid**2))}\n')
            f.write(f'Error pdv: {np.mean(error_pdv_models[i])} pm {np.std(error_pdv_models[i])}\n')
            f.write(f'Error pdv L2: {np.sqrt(grid_size*np.sum(error_pdv_models[i]**2))}\n')
            f.write(f'Error pde: {np.mean(pde_models[i])} pm {np.std(pde_models[i])}\n')
            f.write(f'Error pde L2: {np.sqrt(grid_size*np.sum(pde_models[i]**2))}\n')
            
    
    