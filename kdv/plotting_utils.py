import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.func import vmap, jacrev, hessian
import os
from model import KdVPINN
 
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




def plot_errors(model_list:list[KdVPINN], model_names:list[str], path:str, kdv_data:np.array, model_pinn:KdVPINN, apx:str=''):
    n_models = len(model_list)
    model_list = np.array(model_list)
    print(f'Number of models: {n_models}')
    model_names = np.array(model_names)
    #if n_models > 4:
    #    model_list = model_list.reshape((-1,2))
    #    model_names = model_names.reshape((-1,2))
    
    xmin = -1.
    xmax = 1.
    tmin = 0.
    tmax = 1.   
    dt = 0.005
    dx = 0.005
    grid_size = dx*dt

    points_x = np.arange(start=xmin, stop=xmax+dx, step=dx)
    points_t = np.arange(start=tmin, stop=tmax+dt, step=dt)

    X,Y = np.meshgrid(points_t, points_x)
    pts = np.vstack([X.reshape(-1),Y.reshape(-1)]).T
    
    errors = []
    ders = []
    ders_true = []
    der_errors = []
    hes_errors = []
    for i, model in enumerate(model_list):
        out = model.forward(torch.from_numpy(pts).to(model.device).float()).detach().cpu().numpy().reshape(X.shape)
        der = vmap(jacrev(model.forward_single))(torch.from_numpy(pts).to(model.device).float()).detach().cpu().numpy().reshape((X.shape[0],X.shape[1],2))
        der_true = vmap(jacrev(model_pinn.forward_single))(torch.from_numpy(pts).to(model.device).float()).detach().cpu().numpy().reshape((X.shape[0],X.shape[1],2))
        
        out_pinn = model_pinn.forward(torch.from_numpy(pts).to(model.device).float()).detach().cpu().numpy().reshape(X.shape)
        
        hes = vmap(hessian(model.forward_single))(torch.from_numpy(pts).to(model.device).float()).detach().cpu().numpy().reshape((X.shape[0],X.shape[1],2,2))
        hes_true = vmap(hessian(model_pinn.forward_single))(torch.from_numpy(pts).to(model.device).float()).detach().cpu().numpy().reshape((X.shape[0],X.shape[1],2,2))
        
        ders.append(der)
        ders_true.append(der_true)
        
        der_errors.append(np.linalg.norm(der - der_true, axis=2))
        hes_errors.append(np.linalg.norm(hes - hes_true, axis=(2,3)))
        
        errors.append(np.sqrt((out - kdv_data.T)**2))
    
    pinn_error = np.sqrt((out_pinn - kdv_data.T)**2)
    errors = np.array(errors)
    der_errors = np.array(der_errors)
    hes_errors = np.array(hes_errors)
    
    cons = []
    for i, model in enumerate(model_list):
        con = model.evaluate_consistency(torch.from_numpy(pts).to(model.device).float()).detach().cpu().numpy().reshape(X.shape)
        #con[con>2] = 2
        cons.append(con)
    
    pinn_cons = model_pinn.evaluate_consistency(torch.from_numpy(pts).to(model.device).float()).detach().cpu().numpy().reshape(X.shape)
    pinn_cons = np.array(pinn_cons)
    cons = np.array(cons)
    
    print(model_names)
    
    with open(f'{path}/losses{apx}.txt', 'w') as f:
        print('Output error averaged over the domain', file=f)
        for i, model in enumerate(model_list):
            print(f'{model_names[i]}: mean {np.mean(errors[i])}, std {np.std(errors[i])}', file=f)
            print(f'{model_names[i]}: RMSE {np.sqrt(np.mean(errors[i]**2))}', file=f)
            print(f'{model_names[i]}: L2 loss {np.sqrt(grid_size*np.sum(errors[i]**2))}', file=f)
            print(f'{model_names[i]}: Normalized L2 loss {np.sqrt(np.sum(errors[i]**2)/np.sum((kdv_data.T)**2))}', file=f)
        print('PINN: mean', np.mean(pinn_error), 'std', np.std(pinn_error[0]), file=f)
        print('PINN: RMSE', np.sqrt(np.mean(pinn_error**2)), file=f)
        print(f'Pinn: L2 loss {np.sqrt(grid_size*np.sum(pinn_error**2))}', file=f)
        print(f'Pinn: Normalized L2 loss {np.sqrt(np.sum(pinn_error**2)/np.sum((kdv_data.T)**2))}', file=f)
            
        print('\nDerivative error averaged over the domain', file=f)
        for i, model in enumerate(model_list):
            print(f'{model_names[i]}: mean {np.mean(der_errors[i])}, std {np.std(der_errors[i])}', file=f)
            print(f'{model_names[i]}: RMSE {np.sqrt(np.mean(der_errors[i]**2))}', file=f)
            print(f'{model_names[i]}: L2 loss {np.sqrt(grid_size*np.sum(der_errors[i]**2))}', file=f)
            print(f'{model_names[i]}: Normalized L2 loss {np.sqrt(np.sum(der_errors[i]**2)/np.sum((ders_true[i])**2))}', file=f)

            
        print('\nHessian error averaged over the domain', file=f)
        for i, model in enumerate(model_list):
            print(f'{model_names[i]}: mean {np.mean(hes_errors[i])}, std {np.std(hes_errors[i])}', file=f)
            print(f'{model_names[i]}: RMSE {np.sqrt(np.mean(hes_errors[i]**2))}', file=f)
            print(f'{model_names[i]}: L2 loss {np.sqrt(grid_size*np.sum(hes_errors[i]**2))}', file=f)
            print(f'{model_names[i]}: Normalized L2 loss {np.sqrt(np.sum(hes_errors[i]**2)/np.sum((hes_true[i])**2))}', file=f)
        
        print('\nConsistency error averaged over the domain', file=f)
        for i, model in enumerate(model_list):
            print(f'{model_names[i]}: mean {np.mean(cons[i])}, std {np.std(cons[i])}', file=f)
            print(f'{model_names[i]}: RMSE {np.sqrt(np.mean(cons[i]**2))}\n', file=f)
            print(f'{model_names[i]}: L2 loss {np.sqrt(grid_size*np.sum(cons[i]**2))}', file=f)
            print(f'{model_names[i]}: Normalized L2 loss {np.sqrt(np.sum(cons[i]**2)/np.sum((cons[0])**2))}', file=f)
        print('PINN: mean', np.mean(pinn_cons), 'std', np.std(pinn_cons), file=f)
        print('PINN: RMSE', np.sqrt(np.mean(pinn_cons**2)), file=f)
        print(f'Pinn: L2 loss {np.sqrt(grid_size*np.sum(pinn_cons**2))}', file=f)
        print(f'Pinn: Normalized L2 loss {np.sqrt(np.sum(pinn_cons**2)/np.sum((cons[0])**2))}', file=f)
        
    plot_downsample = 5
    
    for i in range(n_models):
        cons[i][cons[i]>2] = 2
    
    
    if n_models > 4:
        nrows = 2
        ncols = int(n_models/2)
        
        ### ERROR PLOTS
        vmin = np.min(errors)
        vmax = np.max(errors)
        
        levels = np.linspace(vmin,vmax,50)
        
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows), sharex=True, sharey=True)
        for i in range(nrows):
            for j in range(ncols):
                ax[i,j].set_xlim((tmin,tmax))
                ax[i,j].set_ylim((xmin,xmax))
                ax[i,j].set_title(model_names[2*i+j])
                contour = ax[i,j].contourf(X[::plot_downsample,::plot_downsample],Y[::plot_downsample,::plot_downsample],errors[i*2+j,::plot_downsample,::plot_downsample],50,cmap='jet', vmin=vmin, vmax=vmax, levels=levels)
                for c in contour.collections:
                    c.set_rasterized(True)
        fig.colorbar(contour, ax=ax, orientation='vertical', fraction=0.02, pad=0.1)
        plt.savefig(f'{path}/errors{apx}.pdf', format='pdf')
        plt.close()        
        
        
        ### CONSISTENCY PLOTS
        vmin = np.min(cons)
        vmax = np.max(cons)
        
        levels = np.linspace(vmin,vmax,50)
                
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows), sharex=True, sharey=True)
        for i in range(nrows):
            for j in range(ncols):
                ax[i,j].set_xlim((tmin,tmax))
                ax[i,j].set_ylim((xmin,xmax))
                ax[i,j].set_title(model_names[2*i+j])
                contour = ax[i,j].contourf(X[::plot_downsample,::plot_downsample],Y[::plot_downsample,::plot_downsample],cons[i*2+j, ::plot_downsample,::plot_downsample],50,cmap='jet', vmin=vmin, vmax=vmax, levels=levels)
                #ax[i,j].sharex(ax[0,0])
                for c in contour.collections:
                    c.set_rasterized(True)
        fig.colorbar(contour, ax=ax, orientation='vertical', fraction=0.02, pad=0.1)
        plt.savefig(f'{path}/consistency{apx}.pdf', format='pdf')
        plt.close()
        
    else:
        ### ERROR PLOTS
        vmin = np.min(errors)
        vmax = np.max(errors)
        
        levels = np.linspace(vmin,vmax,50)
        
        fig, ax = plt.subplots(nrows=1, ncols=n_models, figsize=(4*n_models,4), sharex=True, sharey=True)
        for i in range(n_models):
            ax[i].set_xlim((tmin,tmax))
            ax[i].set_ylim((xmin,xmax))
            ax[i].set_title(model_names[i])
            contour = ax[i].contourf(X[::plot_downsample,::plot_downsample],Y[::plot_downsample,::plot_downsample],errors[i,::plot_downsample,::plot_downsample],50,cmap='jet', vmin=vmin, vmax=vmax, levels=levels)
            ax[i].sharex(ax[0])
            for c in contour.collections:
                c.set_rasterized(True)
        fig.colorbar(contour, ax=ax, orientation='vertical', fraction=0.02, pad=0.1)
        plt.savefig(f'{path}/errors{apx}.pdf', format='pdf')
        plt.close()        
        
        
        ### CONSISTENCY PLOTS
        vmin = np.min(cons)
        vmax = np.max(cons)
        
        levels = np.linspace(vmin,vmax,50)
                
        fig, ax = plt.subplots(nrows=1, ncols=n_models, figsize=(4*n_models,4), sharex=True, sharey=True)
        for i in range(n_models):
            ax[i].set_xlim((tmin,tmax))
            ax[i].set_ylim((xmin,xmax))
            ax[i].set_title(model_names[i])
            contour = ax[i].contourf(X[::plot_downsample,::plot_downsample],Y[::plot_downsample,::plot_downsample],cons[i,::plot_downsample,::plot_downsample],50,cmap='jet', vmin=vmin, vmax=vmax, levels=levels)
            for c in contour.collections:
                c.set_rasterized(True)
        #ax.set_rasterization_zorder(0)
        fig.colorbar(contour, ax=ax, orientation='vertical', fraction=0.02, pad=0.1)
        plt.savefig(f'{path}/consistency{apx}.pdf', format='pdf')
        plt.close()
        
    # ERROR COMPARISON PLOTS
    
    error_comp = []
    for i in range(1,n_models):
        error_comp.append(errors[i] - errors[0])
    
    error_comp = np.array(error_comp)
    vmin = np.min(error_comp)
    vmax = np.max(error_comp)
    levels = np.linspace(vmin,vmax,50)
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    nrows = 1
    ncols = n_models-1
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*nrows*ncols,4), sharex=True, sharey=True)
    for i in range(1,n_models):
        ax[i-1].set_xlim((tmin,tmax))
        ax[i-1].set_ylim((xmin,xmax))
        ax[i-1].set_title(model_names[i])
        contour = ax[i-1].contourf(X[::plot_downsample,::plot_downsample],Y[::plot_downsample,::plot_downsample],error_comp[i-1,::plot_downsample,::plot_downsample],50,cmap='seismic_r', vmin=vmin, vmax=vmax, levels=levels, norm=norm)
        for c in contour.collections:
            c.set_rasterized(True)
    fig.colorbar(contour, ax=ax, orientation='vertical', fraction=0.02, pad=0.1)
    plt.savefig(f'{path}/errors_comparison{apx}.pdf', format='pdf')
    plt.close()
    
    # CONSISTENCY COMPARISON PLOTS
    
    cons_comp = []
    for i in range(1,n_models):
        cons_comp.append(cons[i] - cons[0])
    
    cons_comp = np.array(cons_comp)
    vmin = np.min(cons_comp)
    vmax = np.max(cons_comp)
    levels = np.linspace(vmin,vmax,50)
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    
    nrows = 1
    ncols = n_models-1
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*nrows*ncols,4), sharex=True, sharey=True)
    
    for i in range(1,n_models):
        ax[i-1].set_xlim((tmin,tmax))
        ax[i-1].set_ylim((xmin,xmax))
        ax[i-1].set_title(model_names[i])
        contour = ax[i-1].contourf(X[::plot_downsample,::plot_downsample],Y[::plot_downsample,::plot_downsample],cons_comp[i-1,::plot_downsample,::plot_downsample],50,cmap='seismic_r', vmin=vmin, vmax=vmax, levels=levels, norm=norm)
    fig.colorbar(contour, ax=ax, orientation='vertical', fraction=0.02, pad=0.1)
    plt.savefig(f'{path}/consistency_comparison{apx}.pdf', format='pdf')
    plt.close()