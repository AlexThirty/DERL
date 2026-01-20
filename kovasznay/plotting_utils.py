from generate import u_true, p_true, vorticity_true
import torch
from models.params import x_min, x_max, y_min, y_max
import numpy as np
from torch.func import vmap, jacrev, jacfwd, hessian
from models.pinnsformer import PINNsformer
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        if self.midpoint == self.vmin:
            normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
            normalized_max = 1
        elif self.midpoint == self.vmax:
            normalized_min = 0
            normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        else:
            normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
            normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))


def generate_grid_data(nx: int, ny: int, return_mesh=False):
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    x = np.linspace(x_min, x_max+dx, nx)
    y = np.linspace(y_min, y_max+dy, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    pts = np.column_stack([X.reshape((-1,1)), Y.reshape((-1,1))])
    u = u_true(torch.from_numpy(pts))
    p = p_true(torch.from_numpy(pts)).reshape((-1,1))
    out = torch.column_stack([u, p])
    out = torch.column_stack([u, p])
    vortex = vorticity_true(torch.from_numpy(pts))
    
    vortex_np = vortex.detach().numpy()
    out_np = out.detach().numpy()
    if return_mesh:
        return pts, out_np, vortex_np, X, Y
    
    return pts, out_np, vortex_np


def plot_models_errors(models, model_names, pts, out, vortex, plot_path, nx, ny, dv=1e-3):
    
    pts, out, vortex, X, Y = generate_grid_data(nx, ny, return_mesh=True)
    
    errors_models = {}
    values_models = {}
    for model, name in zip(models, model_names):
        # Calculate model error in batches
        batch_size = 1024
        model_out = []
        model_vortex = []
        for i in range(0, len(pts), batch_size):
            if isinstance(model, PINNsformer):
                x_train = pts[i:i+batch_size][:, 0:1]
                y_train = pts[i:i+batch_size][:, 1:2]
                x_train = np.expand_dims(np.tile(x_train[:], (5)) ,-1)
                y_train = np.expand_dims(np.tile(y_train[:], (5)) ,-1)
                x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True).to(model.device)
                y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=True).to(model.device)   
                batch_out = model.forward(x_train, y_train, return_final=True).cpu().detach().numpy()
            else:
                batch_pts = torch.from_numpy(pts[i:i+batch_size]).float().to(model.device)
                # Forward pass for the batch
                batch_out = model.forward(batch_pts, return_final=True).cpu().detach().numpy()
            model_out.append(batch_out)
            # Compute Jacobian for the batch
            if isinstance(model, PINNsformer):
                x_train = pts[i:i+batch_size][:, 0:1]
                y_train = pts[i:i+batch_size][:, 1:2]
                x_train = np.expand_dims(np.tile(x_train[:], (5)) ,-1)
                y_train = np.expand_dims(np.tile(y_train[:], (5)) ,-1)
                x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True).to(model.device)
                y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=True).to(model.device) 
                batch_vortex = model.forward_single_final(x_train, y_train).cpu().detach().numpy().reshape((-1,1))
            else:
                batch_Dy = vmap(jacrev(model.forward_single_final))(batch_pts).cpu().detach().numpy()
                batch_vortex = batch_Dy[:, 1, 0] - batch_Dy[:, 0, 1]
            model_vortex.append(batch_vortex.reshape((-1,1)))
        
        model_out = np.vstack(model_out)
        model_vortex = np.vstack(model_vortex).reshape((-1))
        
        print('Model:', name)
        print('Model out shape:', model_out.shape)
        print('Model vortex shape:', model_vortex.shape)
        
        # Calculate errors
        out_err = np.abs(out - model_out)
        vortex_err = np.abs(vortex - model_vortex)
        # Save in a dictionary
        errors = {
            'out_err': out_err,
            'vortex_err': vortex_err
        }
        values = {
            'out': model_out,
            'vortex': model_vortex
        }
        
        errors_models[name] = errors
        values_models[name] = values
    
    with open('model_errors.txt', 'w') as f:
        for name, errors in errors_models.items():
            out_err_norm = np.linalg.norm(errors['out_err'], ord=2) * dv
            vortex_err_norm = np.linalg.norm(errors['vortex_err'], ord=2) * dv
            out_err_max = np.max(errors['out_err'])
            vortex_err_max = np.max(errors['vortex_err'])
            f.write(f"{name}:\n")
            f.write(f"Output Error Norm: {out_err_norm}\n")
            f.write(f"Vortex Error Norm: {vortex_err_norm}\n")
            f.write(f"Maximum Output Error: {out_err_max}\n")
            f.write(f"Maximum Vortex Error: {vortex_err_max}\n\n")
    
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, len(models), figsize=(25, 10))
    
    # Determine the min and max values for the color scales
    out_err_min = np.inf
    out_err_max = -np.inf
    vortex_err_min = np.inf
    vortex_err_max = -np.inf
    
    for name in model_names:
        out_err = errors_models[name]['out_err']
        vortex_err = errors_models[name]['vortex_err']
        out_err_reshaped = np.linalg.norm(out_err.reshape(nx, ny, 3), axis=2)
        vortex_err_reshaped = vortex_err.reshape(nx, ny)
        
        out_err_min = min(out_err_min, out_err_reshaped.min())
        out_err_max = max(out_err_max, out_err_reshaped.max())
        vortex_err_min = min(vortex_err_min, vortex_err_reshaped.min())
        vortex_err_max = max(vortex_err_max, vortex_err_reshaped.max())
    
    norm_out_err = plt.Normalize(vmin=out_err_min, vmax=out_err_max)
    norm_vortex_err = plt.Normalize(vmin=vortex_err_min, vmax=vortex_err_max)
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        out_err = errors_models[name]['out_err']
        vortex_err = errors_models[name]['vortex_err']
        out_err_reshaped = np.linalg.norm(out_err.reshape(nx, ny, 3), axis=2)
        vortex_err_reshaped = vortex_err.reshape(nx, ny)
        
        # Plot output error
        out_err_plot = axs[0, i].contourf(X, Y, out_err_reshaped, cmap='jet', levels=50, norm=norm_out_err, vmin=out_err_min, vmax=out_err_max)
        axs[0, i].set_title(f'{name} Error')
        
        # Plot vortex error
        vortex_err_plot = axs[1, i].contourf(X, Y, vortex_err_reshaped, cmap='jet', levels=50, norm=norm_vortex_err, vmin=vortex_err_min, vmax=vortex_err_max)
        axs[1, i].set_title(f'{name} Vorticity Error')
    
    # Add a single colorbar for all subplots
    from matplotlib import cm
    fig.colorbar(cm.ScalarMappable(norm=norm_out_err, cmap='jet'), ax=axs[0, :], orientation='vertical', fraction=0.02, pad=0.04)
    fig.colorbar(cm.ScalarMappable(norm=norm_vortex_err, cmap='jet'), ax=axs[1, :], orientation='vertical', fraction=0.02, pad=0.04)
    
    plt.savefig(f'{plot_path}/errors.png')
    plt.close()
    fig, axs = plt.subplots(2, len(models) + 1, figsize=(30, 10))
    
    # Determine the min and max values for the color scales for model outputs and vortices
    model_out_min = np.inf
    model_out_max = -np.inf
    model_vortex_min = np.inf
    model_vortex_max = -np.inf
    
    for name in model_names:
        model_out = values_models[name]['out']
        model_vortex = values_models[name]['vortex']
        model_out_reshaped = np.linalg.norm(model_out.reshape(nx, ny, 3), axis=2)
        model_vortex_reshaped = model_vortex.reshape(nx, ny)
        
        model_out_min = min(model_out_min, model_out_reshaped.min())
        model_out_max = max(model_out_max, model_out_reshaped.max())
        model_vortex_min = min(model_vortex_min, model_vortex_reshaped.min())
        model_vortex_max = max(model_vortex_max, model_vortex_reshaped.max())
    
    norm_model_out = plt.Normalize(vmin=model_out_min, vmax=model_out_max)
    norm_model_vortex = plt.Normalize(vmin=model_vortex_min, vmax=model_vortex_max)
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        model_out = values_models[name]['out']
        model_vortex = values_models[name]['vortex']
        model_out_reshaped = np.linalg.norm(model_out.reshape(nx, ny, 3), axis=2)
        model_vortex_reshaped = model_vortex.reshape(nx, ny)
        
        # Plot output values
        axs[0, i].contourf(X, Y, model_out_reshaped, cmap='jet', levels=50, norm=norm_model_out, vmin=model_out_min, vmax=model_out_max)
        axs[0, i].set_title(f'{name} Prediction')
        
        # Plot vortex values
        axs[1, i].contourf(X, Y, model_vortex_reshaped, cmap='jet', levels=50, norm=norm_model_vortex, vmin=model_vortex_min, vmax=model_vortex_max)
        axs[1, i].set_title(f'{name} Vorticity pred.')
    
    # Plot true values
    true_out_reshaped = np.linalg.norm(out.reshape(nx, ny, 3), axis=2)
    true_vortex_reshaped = vortex.reshape(nx, ny)
    axs[0, -1].contourf(X, Y, true_out_reshaped, cmap='jet', levels=50, norm=norm_model_out, vmin=model_out_min, vmax=model_out_max)
    axs[0, -1].set_title('True Solution')
    axs[1, -1].contourf(X, Y, true_vortex_reshaped, cmap='jet', levels=50, norm=norm_model_vortex, vmin=model_vortex_min, vmax=model_vortex_max)
    axs[1, -1].set_title('True Vorticity')
    
    fig.colorbar(cm.ScalarMappable(norm=norm_model_out, cmap='jet'), ax=axs[0, :], orientation='vertical', fraction=0.02, pad=0.04)
    fig.colorbar(cm.ScalarMappable(norm=norm_model_vortex, cmap='jet'), ax=axs[1, :], orientation='vertical', fraction=0.02, pad=0.04)
    
    plt.savefig(f'{plot_path}/values.png')
    plt.close()

    # Plot error differences between models and the first model
    fig, axs = plt.subplots(2, len(models) - 1, figsize=(20, 10))
    
    base_out_err = errors_models[model_names[0]]['out_err']
    base_vortex_err = errors_models[model_names[0]]['vortex_err']
    
    
    
    # Determine the min and max values for the color scales for error differences
    diff_out_err_min = np.inf
    diff_out_err_max = -np.inf
    diff_vortex_err_min = np.inf
    diff_vortex_err_max = -np.inf

    for name in model_names[1:]:
        diff_out_err = np.abs(errors_models[name]['out_err'] - base_out_err)
        diff_vortex_err = np.abs(errors_models[name]['vortex_err'] - base_vortex_err)
        
        diff_out_err_reshaped = np.linalg.norm(diff_out_err.reshape(nx, ny, 3), axis=2)
        diff_vortex_err_reshaped = diff_vortex_err.reshape(nx, ny)
        
        diff_out_err_min = min(diff_out_err_min, diff_out_err_reshaped.min())
        diff_out_err_max = max(diff_out_err_max, diff_out_err_reshaped.max())
        diff_vortex_err_min = min(diff_vortex_err_min, diff_vortex_err_reshaped.min())
        diff_vortex_err_max = max(diff_vortex_err_max, diff_vortex_err_reshaped.max())

    norm_diff_out_err = MidpointNormalize(vmin=diff_out_err_min, vmax=diff_out_err_max, midpoint=0)
    norm_diff_vortex_err = MidpointNormalize(vmin=diff_vortex_err_min, vmax=diff_vortex_err_max, midpoint=0)

    for i, name in enumerate(model_names[1:]):
        diff_out_err = np.abs(errors_models[name]['out_err'] - base_out_err)
        diff_vortex_err = np.abs(errors_models[name]['vortex_err'] - base_vortex_err)
        
        diff_out_err_reshaped = np.linalg.norm(diff_out_err.reshape(nx, ny, 3), axis=2)
        diff_vortex_err_reshaped = diff_vortex_err.reshape(nx, ny)
        
        # Plot output error difference
        axs[0, i].contourf(X, Y, diff_out_err_reshaped, cmap='seismic_r', levels=50, norm=norm_diff_out_err, vmin=diff_out_err_min, vmax=diff_out_err_max)
        axs[0, i].set_title(f'{name} - {model_names[0]}')
        
        # Plot vortex error difference
        axs[1, i].contourf(X, Y, diff_vortex_err_reshaped, cmap='seismic_r', levels=50, norm=norm_diff_vortex_err, vmin=diff_vortex_err_min, vmax=diff_vortex_err_max)
        axs[1, i].set_title(f'{name} - {model_names[0]} vorticity')
    
    fig.colorbar(cm.ScalarMappable(norm=norm_diff_out_err, cmap='seismic_r'), ax=axs[0, :], orientation='vertical', fraction=0.02, pad=0.04)
    fig.colorbar(cm.ScalarMappable(norm=norm_diff_vortex_err, cmap='seismic_r'), ax=axs[1, :], orientation='vertical', fraction=0.02, pad=0.04)
    
    plt.savefig(f'{plot_path}/error_differences.png')
    plt.close()



def plot_models_consistencies(models, model_names, pts, out, vortex, plot_path, nx, ny, dv=1e-3):
    consistencies_models = {}
    
    pts, out, vortex, X, Y = generate_grid_data(nx, ny, return_mesh=True)
    
    for model, name in zip(models, model_names):
        if isinstance(model, PINNsformer):
            batch_size = 1024
            mom_cons_list = []
            div_cons_list = []
            for i in range(0, len(pts), batch_size):
                x_train = pts[i:i+batch_size, 0:1]
                y_train = pts[i:i+batch_size, 1:2]
                x_train = np.expand_dims(np.tile(x_train[:], (5)) ,-1)
                y_train = np.expand_dims(np.tile(y_train[:], (5)) ,-1)
                x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True).to(model.device)
                y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=True).to(model.device)
                # Forward pass for the batch
                mom_cons_batch, div_cons_batch = model.evaluate_consistency(x_train, y_train)
                mom_cons_list.append(mom_cons_batch.cpu().detach().numpy())
                div_cons_list.append(div_cons_batch.cpu().detach().numpy())
            mom_cons = np.vstack(mom_cons_list)
            div_cons = np.vstack(div_cons_list)
        else:
        # Calculate model consistency
            mom_cons, div_cons = model.evaluate_consistency(torch.from_numpy(pts).float().to(model.device))
            mom_cons = mom_cons.cpu().detach().numpy()
            div_cons = div_cons.cpu().detach().numpy()
        # Save in a dictionary
        consistencies = {
            'mom_cons': mom_cons,
            'div_cons': div_cons
        }
        
        consistencies_models[name] = consistencies
    
    with open('model_consistencies.txt', 'w') as f:
        for name, consistencies in consistencies_models.items():
            mom_cons_norm = np.linalg.norm(consistencies['mom_cons'], ord=2)*dv
            div_cons_norm = np.linalg.norm(consistencies['div_cons'], ord=2)*dv
            f.write(f"{name}:\n")
            f.write(f"Momentum Consistency Norm: {mom_cons_norm}\n")
            f.write(f"Divergence Consistency Norm: {div_cons_norm}\n\n")
    
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, len(models), figsize=(25, 10))
    
    # Determine the min and max values for the color scales
    mom_cons_min = np.inf
    mom_cons_max = -np.inf
    div_cons_min = np.inf
    div_cons_max = -np.inf
    
    for name in model_names:
        mom_cons = consistencies_models[name]['mom_cons']
        div_cons = consistencies_models[name]['div_cons']
        mom_cons_reshaped = np.linalg.norm(mom_cons.reshape(nx, ny, 2), axis=2)
        div_cons_reshaped = div_cons.reshape(nx, ny)
        
        mom_cons_min = min(mom_cons_min, mom_cons_reshaped.min())
        mom_cons_max = max(mom_cons_max, mom_cons_reshaped.max())
        if name != 'NCL':
            div_cons_min = min(div_cons_min, div_cons_reshaped.min())
            div_cons_max = max(div_cons_max, div_cons_reshaped.max())
    
    norm_mom_cons = plt.Normalize(vmin=mom_cons_min, vmax=mom_cons_max)
    norm_div_cons = plt.Normalize(vmin=div_cons_min, vmax=div_cons_max)
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        mom_cons = consistencies_models[name]['mom_cons']
        div_cons = consistencies_models[name]['div_cons']
        mom_cons_reshaped = np.linalg.norm(mom_cons.reshape(nx, ny, 2), axis=2)
        div_cons_reshaped = div_cons.reshape(nx, ny)
        
        # Plot momentum consistency
        mom_cons_plot = axs[0, i].contourf(X, Y, mom_cons_reshaped, cmap='jet', levels=50, norm=norm_mom_cons, vmin=mom_cons_min, vmax=mom_cons_max)
        axs[0, i].set_title(f'{name} Momentum Consistency')
        
        # Plot divergence consistency
        if name == 'NCL':
            axs[1, i].axis('off')
            axs[1, i].set_title(f'{name} Divergence Consistency')
        else:
            div_cons_plot = axs[1, i].contourf(X, Y, div_cons_reshaped, cmap='jet', levels=50, norm=norm_div_cons, vmin=div_cons_min, vmax=div_cons_max)
            axs[1, i].set_title(f'{name} Divergence Consistency')
    
    # Add a single colorbar for all subplots
    from matplotlib import cm
    fig.colorbar(cm.ScalarMappable(norm=norm_mom_cons, cmap='jet'), ax=axs[0, :], orientation='vertical', fraction=0.02, pad=0.04)
    if 'div_cons_plot' in locals():
        fig.colorbar(cm.ScalarMappable(norm=norm_div_cons, cmap='jet'), ax=axs[1, :], orientation='vertical', fraction=0.02, pad=0.04)
    
    plt.savefig(f'{plot_path}/consistencies.png')
    plt.close()
