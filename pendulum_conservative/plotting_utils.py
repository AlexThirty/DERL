import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.func import vmap, jacrev
import os
from model import u_vec
from model import PendulumNet

font = {'size'   : 16}
import matplotlib
matplotlib.rc('font', **font)

from matplotlib.colors import Normalize

xlim = np.pi/2
ylim = 1.5

plot_downsample = 4

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

def comparison_plotting(errors, X, Y, us, vs, U_true, V_true, model_names, path, compare_to='DERL'):
    num_models = len(model_names)
    # Get the min and max values for the colorbar
    vmin, vmax = (np.min(errors), np.max(errors))
    errors = np.array(errors)    
    
    levels = np.linspace(vmin,vmax,50)
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    
    # Define subplots
    fig, ax = plt.subplots(nrows=1, ncols=num_models, figsize=(4*num_models,4), layout='compressed')

    #fig.suptitle('Phase space field error comparison')
    
    for i in range(num_models):
        ax[i].set_xlim((-xlim,xlim))
        ax[i].set_ylim((-ylim,ylim))
        ax[i].set_title(f'{model_names[i]} - {compare_to}')
        ax[i].streamplot(X,Y,us[i],vs[i],density=0.5,color='black', linewidth=0.05)
        ax[i].set_aspect('equal')
        contour = ax[i].contourf(X[::plot_downsample,::plot_downsample],Y[::plot_downsample,::plot_downsample],errors[i,::plot_downsample,::plot_downsample], 50,cmap='seismic_r', vmin=vmin, vmax=vmax, levels=levels, norm=norm)
    
    fig.colorbar(contour, ax=ax, orientation='vertical', fraction=0.02, pad=0.05)
    plt.savefig(path, format='pdf')
    plt.close()

def error_plotting(errors, X, Y, us, vs, U_true, V_true, model_names, path):
    num_models = len(model_names)
    # Get the min and max values for the colorbar
    vmin, vmax = (np.min(errors), np.max(errors))
        
    errors = np.array(errors)
    levels = np.linspace(vmin,vmax,50)
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    
    # Define subplots
    fig, ax = plt.subplots(nrows=1, ncols=num_models, figsize=(4*num_models,4), layout='compressed')

    #fig.suptitle('Phase space field error comparison')
    
    for i in range(num_models):
        ax[i].set_xlim((-xlim,xlim))
        ax[i].set_ylim((-ylim,ylim))
        ax[i].set_title(model_names[i])
        ax[i].streamplot(X,Y,us[i],vs[i],density=0.5,color='black', linewidth=0.05)
        ax[i].set_aspect('equal')
        contour = ax[i].contourf(X[::plot_downsample,::plot_downsample],Y[::plot_downsample,::plot_downsample],errors[i,::plot_downsample,::plot_downsample],50,cmap='jet', vmin=vmin, vmax=vmax, levels=levels)

    fig.colorbar(contour, ax=ax, orientation='vertical', fraction=0.02, pad=0.05)
    plt.savefig(path, format='pdf')
    plt.close()

def plotting_errors(model_list:list[PendulumNet], b:float, model_names:list[str], name:str, path:str, consistency:bool=False, apx:str=''):
    
    # Number of points for the field
    N = 250
    
    X,Y = np.meshgrid(np.linspace(-xlim,xlim,N),np.linspace(-ylim,ylim,N))
    pts = np.vstack([X.reshape(-1),Y.reshape(-1)]).T
    grid_size = (2*xlim)*(2*ylim)/N**2
    # Get the true field
    vel_true = u_vec(torch.from_numpy(pts), b)
    #print(vel)
    U_true = np.array(vel_true[:,0].reshape(X.shape))
    V_true = np.array(vel_true[:,1].reshape(Y.shape))
    from scipy.integrate import solve_ivp
    t_max = 10.
    # Number of models
    num_models = len(model_list)
    t=0.
    errors = []
    cons = []
    init_cons = []
    us = []
    vs = []
    for i, model in enumerate(model_list):
        #vel = vmap(jacrev(model.forward_single))(torch.column_stack((t*torch.ones((pts.shape[0],1)),torch.from_numpy(pts).float())).to(model.device)).detach().cpu().numpy()[:,:,0]
        vel = model.evaluate_field(torch.column_stack((t*torch.ones((pts.shape[0],1)),torch.from_numpy(pts).float())).to(model.device)).detach().cpu().numpy()
        U = np.array(vel[:,0].reshape(X.shape))
        V = np.array(vel[:,1].reshape(Y.shape))
        # Get the error
        con = model.evaluate_consistency(torch.column_stack((t*torch.ones((pts.shape[0],1)),torch.from_numpy(pts).float())).to(model.device)).detach().cpu().numpy().reshape(X.shape)
        init_con = model.evaluate_init_consistency(torch.column_stack((t*torch.ones((pts.shape[0],1)),torch.from_numpy(pts).float())).to(model.device)).detach().cpu().numpy().reshape(X.shape)
        error = np.sqrt((U-U_true)**2 + (V-V_true)**2)
        errors.append(error)
        cons.append(con)
        init_cons.append(init_con)
        
        us.append(U)
        vs.append(V)
    
    # Convert to numpy arrays
    us = np.array(us)
    vs = np.array(vs)
    errors = np.array(errors)
    cons = np.array(cons)
    init_cons = np.array(init_cons)
    
    error_plotting(errors, X, Y, us, vs, U_true, V_true, model_names, path=f'{path}/error_field{apx}.pdf')
    error_plotting(cons, X, Y, us, vs, U_true, V_true, model_names, path=f'{path}/consistency{apx}.pdf')
    error_plotting(init_cons, X, Y, us, vs, U_true, V_true, model_names, path=f'{path}/init_consistency{apx}.pdf')
    


    # Now the comparison
    
    error_comp = []
    cons_comp = []
    init_cons_comp = []
    for i in range(1,num_models):
        error_comp.append(errors[i] - errors[0])
        cons_comp.append(cons[i] - cons[0])
        init_cons_comp.append(init_cons[i] - init_cons[0])
    
    
    comparison_plotting(error_comp, X, Y, us, vs, U_true, V_true, model_names[1:], path=f'{path}/error_comparison{apx}.pdf')
    comparison_plotting(cons_comp, X, Y, us, vs, U_true, V_true, model_names[1:], path=f'{path}/consistency_comparison{apx}.pdf')
    comparison_plotting(init_cons_comp, X, Y, us, vs, U_true, V_true, model_names[1:], path=f'{path}/init_consistency_comparison{apx}.pdf')
    
    
    
    with open(f'{path}/errors{apx}.txt', 'w') as f:
        print('Field error averaged over the domain', file=f)
        for i in range(num_models):
            print(f'Mode {model_names[i]}: RMSE {np.sqrt(np.mean(errors[i]**2))}', file=f)
            print(f'Mode {model_names[i]}: L2 loss: {np.sqrt(grid_size*np.sum(errors[i]**2))}', file=f)
            print(f'Mode {model_names[i]}: L2 normalized loss: {np.sqrt(np.sum(errors[i]**2))/np.sum(U_true**2 + V_true**2)}', file=f)
            print(f'Mode {model_names[i]}: {np.mean(errors[i])}, std {np.std(errors[i])}', file=f)
        print('\nConsistency error averaged over the domain', file=f)
        for i in range(num_models):
            print(f'Mode {model_names[i]}: RMSE {np.sqrt(np.mean(cons[i]**2))}', file=f)
            print(f'Mode {model_names[i]}: L2 loss: {np.sqrt(grid_size*np.sum(cons[i]**2))}', file=f)
            print(f'Mode {model_names[i]}: L2 normalized loss: {np.sqrt(np.sum(cons[i]**2))/np.sum(U_true**2 + V_true**2)}', file=f)
            print(f'Mode {model_names[i]}: {np.mean(cons[i])}, std {np.std(cons[i])}', file=f)
        print('\nInit consistency error averaged over the domain', file=f)
        for i in range(num_models):
            print(f'Mode {model_names[i]}: RMSE {np.sqrt(np.mean(init_cons[i]**2))}', file=f)
            print(f'Mode {model_names[i]}: L2 loss: {np.sqrt(grid_size*np.sum(init_cons[i]**2))}', file=f)
            print(f'Mode {model_names[i]}: L2 normalized loss: {np.sqrt(np.sum(init_cons[i]**2))/np.sum(U_true**2 + V_true**2)}', file=f)
            print(f'Mode {model_names[i]}: {np.mean(init_cons[i])}, std {np.std(init_cons[i])}', file=f)

