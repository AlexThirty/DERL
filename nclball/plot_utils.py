### GENERAL SECTION
import numpy as np
import torch
from matplotlib import pyplot as plt
 
from matplotlib.colors import Normalize
from models.ncl import BallNCL, ball_uniform
from torch.func import vmap, jacrev
import os

#font = {'size'   : 16}
#import matplotlib
#matplotlib.rc('font', **font)

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

def ncl_errorplot(to_plot:np.array, model_names:list[str], X,Y, T:list[float], path:str, name:str, t:float, apx:str='', Z=0):
    
    # Plotting section
    fig, ax = plt.subplots(nrows=to_plot.shape[0], ncols=len(T), figsize=(4*len(T)+3,4*to_plot.shape[0]))
    
    vmin = np.nanmin(to_plot)
    vmax = np.nanmax(to_plot)
    levels = np.linspace(vmin, vmax, 50)
    
    
    for i in range(to_plot.shape[0]):
        for j in range(len(T)):
            # Plot a single thing
            a = 1.1
            exterior = X**2 + Y**2 + Z**2 >= 1
            rho = to_plot[i,j].reshape(X.shape)
            rho[exterior] = np.nan
            plt_dens = ax[i,j].contourf(X,Y,rho,50,cmap='jet', vmin=vmin, vmax=vmax, levels=levels)
            circle = plt.Circle((0, 0), 1.0, fill=False, lw=3,color='k')
            ax[i,j].add_patch(circle)
            
            ax[i,j].set_xlim(-a,a)
            ax[i,j].set_ylim(-a,a)
            
            #ax[i,j].axis('off')
            
            
            #ax[i,j].contourf(to_plot[i,j])
            for c in plt_dens.collections:
                c.set_rasterized(True)
            
            ax[0,j].set_title(f'T = {T[j]}', fontsize=16)
        ax[i,0].set_ylabel(model_names[i], fontsize=16)
    fig.tight_layout()
    fig.colorbar(plt_dens, ax=ax, orientation='vertical')
    plt.savefig(f'{path}/{name}{apx}.pdf')

def ncl_compareplot(to_plot:np.array, model_names:list[str], X,Y, T:list[float], path:str, name:str, t:float, apx:str='', Z=0):
    
    # Plotting section
    fig, ax = plt.subplots(nrows=to_plot.shape[0], ncols=len(T), figsize=(4*len(T)+3,4*to_plot.shape[0]))
    
    vmin = np.nanmin(to_plot)
    vmax = np.nanmax(to_plot)
    levels = np.linspace(vmin, vmax, 50)
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    print(vmin, vmax)
    
    for i in range(to_plot.shape[0]):
        for j in range(len(T)):
            # Plot a single thing
            a = 1.1
            exterior = X**2 + Y**2 + Z**2 >= 1
            rho = to_plot[i,j].reshape(X.shape)
            rho[exterior] = np.nan
            plt_dens = ax[i,j].contourf(X,Y,rho,50,cmap='seismic_r', vmin=vmin, vmax=vmax, norm=norm, levels=levels)
            circle = plt.Circle((0, 0), 1.0, fill=False, lw=3,color='k')
            ax[i,j].add_patch(circle)
            
            ax[i,j].set_xlim(-a,a)
            ax[i,j].set_ylim(-a,a)
            
            #ax[i,j].axis('off')
            
            
            #ax[i,j].contourf(to_plot[i,j])
            for c in plt_dens.collections:
                c.set_rasterized(True)
            ax[0,j].set_title(f'T = {T[j]}', fontsize=16)
        ax[i,0].set_ylabel(model_names[i], fontsize=16)
    fig.tight_layout()
    fig.colorbar(plt_dens, ax=ax, orientation='vertical')
    plt.savefig(f'{path}/{name}comparison{apx}.pdf')

def plot_errors(model_list:list[BallNCL], model_true:BallNCL, model_names:list, path:str, apx:str='', T = [0,0.25,0.5]):
    # Box size
    box = 4
    # Value of Z
    Z = 0
    # Points generation
    N = 250
    a = 1.1
    X,Y = np.meshgrid(np.linspace(-a,a,N),np.linspace(-a,a,N))
    
    errors = []
    outs = []
    ders = []
    us = []
    rhos =  []
    trues = []
    der_trues = []
    
    for k, model in enumerate(model_list):
        us_mode = []
        outs_mode = []
        ders_mode = []
        rhos_mode = []
        true_mode = []
        der_true_mode = []
        for t in T:
            # do the following but in bathces
            pts = np.vstack([np.ones(X.reshape(-1).shape)*t,X.reshape(-1),Y.reshape(-1),np.ones(X.reshape(-1).shape)*Z]).T
            # Use batches to avoid memory issues
            batch_size = 1000
            n_batches = len(pts)//batch_size
            
            u_t = []
            rho_t = []
            out_t = []
            der_t = []
            true_t = []
            der_true_t = []
            for i in range(n_batches):
                out = model.forward(torch.tensor(pts[i*batch_size:(i+1)*batch_size], dtype=torch.float32).to(model.device)).detach().cpu().numpy()[:,:4]
                der = vmap(jacrev(model.forward_single))(torch.tensor(pts[i*batch_size:(i+1)*batch_size], dtype=torch.float32).to(model.device)).detach().cpu().numpy()
                u_t.append(out[:,1:4]/out[:,0:1])
                rho_t.append(out[:,0])
                out_t.append(out)
                der_t.append(der)
                
                true_t.append(model_true.forward(torch.tensor(pts[i*batch_size:(i+1)*batch_size], dtype=torch.float32).to(model.device)).detach().cpu().numpy()[:,:4])
                der_true_t.append(vmap(jacrev(model_true.forward_single))(torch.tensor(pts[i*batch_size:(i+1)*batch_size], dtype=torch.float32).to(model.device)).detach().cpu().numpy())
            
            out = model.forward(torch.tensor(pts[(i+1)*batch_size:], dtype=torch.float32).to(model.device)).detach().cpu().numpy()[:,:4]
            der = vmap(jacrev(model.forward_single))(torch.tensor(pts[(i+1)*batch_size:], dtype=torch.float32).to(model.device)).detach().cpu().numpy()

            true_t.append(model_true.forward(torch.tensor(pts[(i+1)*batch_size:], dtype=torch.float32).to(model.device)).detach().cpu().numpy()[:,:4])
            der_true_t.append(vmap(jacrev(model_true.forward_single))(torch.tensor(pts[(i+1)*batch_size:], dtype=torch.float32).to(model.device)).detach().cpu().numpy())
            
            u_t.append(out[:,1:4]/out[:,0:1])
            rho_t.append(out[:,0])
            out_t.append(out)
            der_t.append(der)
        
            us_mode.append(np.concatenate(u_t))
            rhos_mode.append(np.concatenate(rho_t))
            outs_mode.append(np.concatenate(out_t))
            ders_mode.append(np.concatenate(der_t))
            true_mode.append(np.concatenate(true_t)) 
            der_true_mode.append(np.concatenate(der_true_t))   
        
        
            
        us.append(us_mode)
        rhos.append(rhos_mode)
        outs.append(outs_mode)
        ders.append(ders_mode)
        trues.append(true_mode)
        der_trues.append(der_true_mode)
    
    us = np.array(us)
    rhos = np.array(rhos)
    outs = np.array(outs)
    ders = np.array(ders)
    trues = np.array(trues)
    der_trues = np.array(der_trues)
    
    errors = np.linalg.norm(outs - trues, axis=3, ord=2)
    der_errors = np.linalg.norm(ders - der_trues, axis=(3,4), ord=2)
    comps = errors[1:]-np.repeat(errors[:1,:,:],len(model_list)-1,axis=0)

    us[:,:,np.logical_not(X**2 + Y**2 <= 1).reshape((-1))] = np.nan
    rhos[:,:,np.logical_not(X**2 + Y**2 <= 1).reshape((-1))] = np.nan
    outs[:,:,np.logical_not(X**2 + Y**2 <= 1).reshape((-1))] = np.nan
    trues[:,:,np.logical_not(X**2 + Y**2 <= 1).reshape((-1))] = np.nan
    errors[:,:,np.logical_not(X**2 + Y**2 <= 1).reshape((-1))] = np.nan
    der_errors[:,:,np.logical_not(X**2 + Y**2 <= 1).reshape((-1))] = np.nan

    #print(np.tile(trues, (3,1)).shape
    
    if os.path.exists(f'{path}/plotsjoint') == False:
        os.makedirs(f'{path}/plotsjoint')
    ncl_errorplot(errors, model_names, X,Y, T, f'{path}/plotsjoint', 'errors', t, apx)

    comps = errors[1:]-np.repeat(errors[:1,:,:],len(model_list)-1,axis=0)
    
    ncl_compareplot(comps, model_names[1:], X,Y, T, f'{path}/plotsjoint', 'comparisons', t, apx)
    
    ncl_errorplot(der_errors, model_names, X,Y, T, f'{path}/plotsjoint', 'der_errors', t, apx)
    der_comp = der_errors[1:] - np.repeat(der_errors[:1,:,:],len(model_list)-1,axis=0)
    ncl_compareplot(der_comp, model_names[1:], X,Y, T, f'{path}/plotsjoint', 'der_comp', t, apx)
    
    print('Plotted errors')
    
    F_cons = []
    div_cons = []
    for k, model in enumerate(model_list):
        F_cons_mode = []
        div_cons_mode = []
        for t in T:
            F_cons_t = []
            div_cons_t = []
            pts = np.vstack([np.ones(X.reshape(-1).shape)*t,X.reshape(-1),Y.reshape(-1),np.ones(X.reshape(-1).shape)*Z]).T
            
            for i in range(n_batches):
                F, div = model.evaluate_consistency(torch.tensor(pts[i*batch_size:(i+1)*batch_size], dtype=torch.float32).to(model.device))
                F_cons_t.append(F.detach().cpu().numpy())
                div_cons_t.append(div.detach().cpu().numpy())
            
            F, div = model.evaluate_consistency(torch.tensor(pts[(i+1)*batch_size:], dtype=torch.float32).to(model.device))
            F_cons_t.append(F.detach().cpu().numpy())
            div_cons_t.append(div.detach().cpu().numpy())
            
            F_cons_mode.append(np.concatenate(F_cons_t))
            div_cons_mode.append(np.concatenate(div_cons_t))
            
        F_cons.append(F_cons_mode)
        div_cons.append(div_cons_mode)
        
        
    F_cons = np.array(F_cons)
    div_cons = np.array(div_cons)
    
    F_cons[:,:,np.logical_not(X**2 + Y**2 <= 1).reshape((-1))] = np.nan
    div_cons[:,:,np.logical_not(X**2 + Y**2 <= 1).reshape((-1))] = np.nan
    
    ncl_errorplot(F_cons, model_names, X,Y, T, f'{path}/plotsjoint', 'F_cons', t, apx)
    ncl_errorplot(div_cons, model_names, X,Y, T, f'{path}/plotsjoint', 'div_cons', t, apx)
    
    F_cons_comp = F_cons[1:] - np.repeat(F_cons[:1,:,:],len(model_list)-1,axis=0)
    div_cons_comp = div_cons[1:] - np.repeat(div_cons[:1,:,:],len(model_list)-1,axis=0)
    
    ncl_compareplot(F_cons_comp, model_names[1:], X,Y, T, f'{path}/plotsjoint', 'F_cons_comp', t, apx)
    ncl_compareplot(div_cons_comp, model_names[1:], X,Y, T, f'{path}/plotsjoint', 'div_cons_comp', t, apx)
    
    
    
    
    pts = ball_uniform(100000, 1, 3)
    ts = np.random.uniform(0,0.5,100000)
    pts = np.hstack([ts.reshape(-1,1), pts])
    errors = []
    F_cons = []
    div_cons = []
    der_errors = []
    
    for k, model in enumerate(model_list):
        n_batches = len(pts)//batch_size
        errors_mode = []
        F_cons_mode = []
        div_cons_mode = []
        der_errors_mode = []
        for i in range(n_batches):
            out = model.forward(torch.tensor(pts[i*batch_size:(i+1)*batch_size], dtype=torch.float32).to(model.device)).detach().cpu().numpy()[:,:4]
            true = model_true.forward(torch.tensor(pts[i*batch_size:(i+1)*batch_size], dtype=torch.float32).to(model.device)).detach().cpu().numpy()[:,:4]
            der = vmap(jacrev(model.forward_single))(torch.tensor(pts[i*batch_size:(i+1)*batch_size], dtype=torch.float32).to(model.device)).detach().cpu().numpy()
            der_true = vmap(jacrev(model_true.forward_single))(torch.tensor(pts[i*batch_size:(i+1)*batch_size], dtype=torch.float32).to(model.device)).detach().cpu().numpy()
            F, div = model.evaluate_consistency(torch.tensor(pts[i*batch_size:(i+1)*batch_size], dtype=torch.float32).to(model.device))
            errors_mode.append(np.linalg.norm(out - true, axis=1, ord=2))
            der_errors_mode.append(np.linalg.norm(der - der_true, axis=(1,2), ord=2))   
            F_cons_mode.append(F.detach().cpu().numpy())
            div_cons_mode.append(div.detach().cpu().numpy())
            
        #out = model.forward(torch.tensor(pts[(i+1)*batch_size:], dtype=torch.float32).to(model.device)).detach().cpu().numpy()[:,:4]
        #true = model_true.forward(torch.tensor(pts[(i+1)*batch_size:], dtype=torch.float32).to(model.device)).detach().cpu().numpy()[:,:4]
        #F, div = model.evaluate_consistency(torch.tensor(pts[(i+1)*batch_size:], dtype=torch.float32).to(model.device))
        #errors_mode.append(np.linalg.norm(out - true, axis=1, ord=2))
        #der_errors_mode.append(np.linalg.norm(der - der_true, axis=(1,2), ord=2))
        #F_cons_mode.append(F.detach().cpu().numpy())
        #div_cons_mode.append(div.detach().cpu().numpy())
        
        errors.append(np.concatenate(errors_mode))
        der_errors.append(np.concatenate(der_errors_mode))
        F_cons.append(np.concatenate(F_cons_mode))
        div_cons.append(np.concatenate(div_cons_mode))
    
    errors = np.array(errors)
    der_errors = np.array(der_errors)
    F_cons = np.array(F_cons)
    div_cons = np.array(div_cons)

    
    
    with open(f'{path}/losses{apx}.txt', 'w') as f:
        print('Output error averaged over the domain', file=f)
        for i, model in enumerate(model_list):
            print(f'{model_names[i]}: mean {np.nanmean(errors[i])}, std {np.nanstd(errors[i])}', file=f)
            print(f'{model_names[i]}: L2 loss {np.sqrt(np.nanmean(errors[i]**2))}', file=f)
            
        print('Derivative error averaged over the domain', file=f)
        for i, model in enumerate(model_list):
            print(f'{model_names[i]}: mean {np.nanmean(der_errors[i])}, std {np.nanstd(der_errors[i])}', file=f)
            print(f'{model_names[i]}: L2 loss {np.sqrt(np.nanmean(der_errors[i]**2))}', file=f)
        
        print('F_cons error averaged over the domain', file=f)
        for i, model in enumerate(model_list):
            print(f'{model_names[i]}: mean {np.nanmean(F_cons[i])}, std {np.nanstd(F_cons[i])}', file=f)
            print(f'{model_names[i]}: L2 loss {np.sqrt(np.nanmean(F_cons[i]**2))}', file=f)
        
        print('div_cons error averaged over the domain', file=f)
        for i, model in enumerate(model_list):
            print(f'{model_names[i]}: mean {np.nanmean(div_cons[i])}, std {np.nanstd(div_cons[i])}', file=f)
            print(f'{model_names[i]}: L2 loss {np.sqrt(np.nanmean(div_cons[i]**2))}\n', file=f)
        
    print('Output error averaged over the domain')
    for i, model in enumerate(model_list):
        print(f'{model_names[i]}: mean {np.nanmean(errors[i])}, std {np.nanstd(errors[i])}')
        
    print('Derivative error averaged over the domain')
    for i, model in enumerate(model_list):
        print(f'{model_names[i]}: mean {np.nanmean(der_errors[i])}, std {np.nanstd(der_errors[i])}')
        
    print('F_cons error averaged over the domain')
    for i, model in enumerate(model_list):
        print(f'{model_names[i]}: mean {np.nanmean(F_cons[i])}, std {np.nanstd(F_cons[i])}')
        
    print('div_cons error averaged over the domain')
    for i, model in enumerate(model_list):
        print(f'{model_names[i]}: mean {np.nanmean(div_cons[i])}, std {np.nanstd(div_cons[i])}')
        