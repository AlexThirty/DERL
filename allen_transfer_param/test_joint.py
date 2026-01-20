import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np
import random
import os
seed = 30
from itertools import cycle

from model import AllenNet
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--bc_weight', default=1., type=float, help='Weight for the BC loss')
parser.add_argument('--sys_weight', default=1., type=float, help='Weight for the rho loss')
parser.add_argument('--pde_weight', default=1., type=float, help='Weight for the PDE loss')
parser.add_argument('--lr_init', default=1e-4, type=float, help='Starting learning rate')
parser.add_argument('--device', default='cuda', type=str, help='Device to use')
parser.add_argument('--name', default='new', type=str, help='multieriment name')
parser.add_argument('--train_steps', default=100000, type=int, help='Number of training steps')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('--mode', default='Derivative', type=str, help='Mode: -1 for PINN learning, 0 for derivative learning, 1 for output learning')
parser.add_argument('--layers', default=4, type=int, help='Number of layers in the network')
parser.add_argument('--units', default=50, type=int, help='Number of units per layer in the network')

args = parser.parse_args()
bc_weight = args.bc_weight  
device = args.device
name = args.name
exp_name = name
train_steps = args.train_steps
epochs = args.epochs
layers = args.layers
units = args.units
lr_init = args.lr_init
mode = args.mode
sys_weight = args.sys_weight
pde_weight = args.pde_weight
title_mode = mode

if not os.path.exists(f'{name}'):
    os.makedirs(f'{name}')

print('Data loaded!')

activation = torch.nn.Tanh()

model_derl = AllenNet(
    in_dim=4,
    bc_weight=bc_weight,
    pde_weight=pde_weight,
    sys_weight=sys_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)
model_derl.load_state_dict(torch.load(f'saved_models/pinn_{name}_Derivative'))
model_derl.eval()

model_outl = AllenNet(
    in_dim=4,
    bc_weight=bc_weight,
    pde_weight=pde_weight,
    sys_weight=sys_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)
model_outl.load_state_dict(torch.load(f'saved_models/pinn_{name}_Output'))
model_outl.eval()

model_sob = AllenNet(
    in_dim=4,
    bc_weight=bc_weight,
    pde_weight=pde_weight,
    sys_weight=sys_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)
model_sob.load_state_dict(torch.load(f'saved_models/pinn_{name}_Sobolev'))
model_sob.eval()

model_pinn = AllenNet(
    in_dim=4,
    bc_weight=bc_weight,
    pde_weight=pde_weight,
    sys_weight=sys_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)
model_pinn.load_state_dict(torch.load(f'saved_models/pinn_new_PINN'))
model_pinn.eval()

model_pinnextend = AllenNet(
    in_dim=4,
    bc_weight=bc_weight,
    pde_weight=pde_weight,
    sys_weight=sys_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)
model_pinnextend.load_state_dict(torch.load(f'saved_models/pinn_extend_PINN'))
model_pinnextend.eval()

model_forget = AllenNet(
    in_dim=4,
    bc_weight=bc_weight,
    pde_weight=pde_weight,
    sys_weight=sys_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)
if name == 'extend':
    model_forget.load_state_dict(torch.load(f'saved_models/pinn_extend_Forgetting'))
model_forget.eval()

model_names = ['DERL', 'OUTL', 'SOB', 'PINN full', 'PINN no dist.', 'PINN fine-tune'] 
model_list = [model_derl, model_outl, model_sob, model_pinn, model_forget, model_pinnextend]

if not os.path.exists(f'{name}/plotsjoint'):
    os.makedirs(f'{name}/plotsjoint')
    
if not os.path.exists(f'{name}/plotsjoint/val'):
    os.makedirs(f'{name}/plotsjoint/val')
    
if not os.path.exists(f'{name}/plotsjoint/train'):
    os.makedirs(f'{name}/plotsjoint/train')
    
# load allen_pts and make the grid
allen_pts = np.load('allen_pts.npy')
allen_x = np.load('allen_x.npy')
allen_y = np.load('allen_y.npy')
x, y = np.meshgrid(allen_x, allen_y)
allen_pts = np.column_stack((x.reshape(-1), y.reshape(-1)))



# Import the valing lambdas
lam_train = np.load('allen_lam_train.npy')
lam_val = np.load('allen_lam_val.npy')

from generate import allen_cahn_true_multi, allen_cahn_forcing_multi
from plotting_utils import error_plotting, comparison_plotting


def evaluate_models_on_val_data(models, model_names, lam_list, allen_pts, device, x, y, lam_type='val'):
    total_errors = {name: 0 for name in model_names}
    total_forcing_errors = {name: 0 for name in model_names}
    num_val = len(lam_val)

    for lam in lam_list:
        print(f'lambda: {lam}')
        lam_in = np.tile(lam, (allen_pts.shape[0], 1))
        x_in = np.column_stack((allen_pts, lam_in))
        x_in = torch.tensor(x_in, dtype=torch.float32).to(device)

        u_preds = {}
        for model, name in zip(models, model_names):
            u_pred = model(x_in).reshape(-1)
            u_preds[name] = u_pred.cpu().detach().numpy()

        u_true = allen_cahn_true_multi(allen_pts, lam)
        f_true = allen_cahn_forcing_multi(allen_pts, lam)
        print('Calculating error')

        for name in model_names:
            error = np.linalg.norm(u_true - u_preds[name], 2) / np.linalg.norm(u_true, 2)
            total_errors[name] += error

            f_pred = models[model_names.index(name)].evaluate_forcing(x_in)
            f_pred = f_pred.cpu().detach().numpy()
            forcing_error = np.linalg.norm(f_true - f_pred, 2) / np.linalg.norm(f_true, 2)
            total_forcing_errors[name] += forcing_error

    avg_errors = {name: total_errors[name] / num_val for name in model_names}
    avg_forcing_errors = {name: total_forcing_errors[name] / num_val for name in model_names}

    with open(f'{exp_name}/plotsjoint/{lam_type}_results.txt', 'w') as f:
        for name in model_names:
            f.write(f'Average Error {name}: {avg_errors[name]}\n')
        f.write('\n')
        for name in model_names:
            f.write(f'Average Error {name} Forcing: {avg_forcing_errors[name]}\n')
        f.write('\n')

    for lam in lam_list:
        print(f'lambda: {lam}')
        lam_in = np.tile(lam, (allen_pts.shape[0], 1))
        x_in = np.column_stack((allen_pts, lam_in))
        x_in = torch.tensor(x_in, dtype=torch.float32).to(device)

        u_preds = {}
        for model, name in zip(models, model_names):
            u_pred = model(x_in).reshape(-1)
            u_preds[name] = u_pred.cpu().detach().numpy()

        u_true = allen_cahn_true_multi(allen_pts, lam)
        f_true = allen_cahn_forcing_multi(allen_pts, lam)
        print('Calculating error')

        with open(f'{exp_name}/plotsjoint/{lam_type}_results.txt', 'a') as f:
            f.write(f'lambda: {lam}\n')
            for name in model_names:
                error = np.linalg.norm(u_true - u_preds[name], 2) / np.linalg.norm(u_true, 2)
                f.write(f'Error {name}: {error}\n')

                f_pred = models[model_names.index(name)].evaluate_forcing(x_in)
                f_pred = f_pred.cpu().detach().numpy()
                forcing_error = np.linalg.norm(f_true - f_pred, 2) / np.linalg.norm(f_true, 2)
                f.write(f'Error {name} Forcing: {forcing_error}\n')
            f.write('\n')

        cap = 100.
        error_plots = {name: np.clip(np.abs(u_true - u_preds[name]).reshape(x.shape), 0, cap) for name in model_names}
        forcing_error_plots = {name: np.clip(np.abs(f_true - models[model_names.index(name)].evaluate_forcing(x_in).cpu().detach().numpy()).reshape(x.shape), 0, cap) for name in model_names}

        error_plotting([error_plots[name] for name in model_names], x, y, model_names, path=f'{exp_name}/plotsjoint/{lam_type}/error_lambda_{lam}.pdf')
        error_plotting([forcing_error_plots[name] for name in model_names], x, y, model_names, path=f'{exp_name}/plotsjoint/val/forcing_error_lambda_{lam}.pdf')

        comparison_plot = [error_plots[name] - error_plots['DERL'] for name in model_names if name != 'DERL']
        comparison_plotting(comparison_plot, x, y, [name for name in model_names if name != 'DERL'], path=f'{exp_name}/plotsjoint/{lam_type}/comparison_error_lambda_{lam}.pdf')

        comparison_plot = [forcing_error_plots[name] - forcing_error_plots['DERL'] for name in model_names if name != 'DERL']
        comparison_plotting(comparison_plot, x, y, [name for name in model_names if name != 'DERL'], path=f'{exp_name}/plotsjoint/{lam_type}/comparison_forcing_error_lambda_{lam}.pdf')

from plotting_utils import error_plotting, comparison_plotting
num_lam = 2

test_lam = np.random.random((3,num_lam))

test_lam = np.load('allen_lam_test.npy')

if not os.path.exists(f'{exp_name}/plotsjoint/test'):
    os.makedirs(f'{exp_name}/plotsjoint/test')
    
if not os.path.exists(f'{exp_name}/plotsjoint/results_u.csv'):
    evaluate_models_on_val_data(
        model_list,
        model_names,
        lam_val, allen_pts, device, x, y, 'val')

    evaluate_models_on_val_data(
        model_list,
        model_names,
        lam_train, allen_pts, device, x, y, 'train')
    
    # Calculate the average error among the test lambdas
    evaluate_models_on_val_data(
        model_list,
        model_names,
        test_lam, allen_pts, device, x, y, 'test')


# Generate a grid of lambda_1, lambda_2 in [0,1]x[0,1]
dl = 0.05
lam1 = np.arange(dl, 1+dl, dl)
lam2 = np.arange(dl, 1+dl, dl)
lam_grid = np.array(np.meshgrid(lam1, lam2)).T.reshape(-1, 2)

import pandas as pd
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap

def compute_errors_for_lambda_grid(models, model_names, lam_grid, allen_pts, device, x, y, name):
    if os.path.exists(f'{exp_name}/plotsjoint/results_u.csv'):
        results_u = pd.read_csv(f'{exp_name}/plotsjoint/results_u.csv', index_col=0)
        results_f = pd.read_csv(f'{exp_name}/plotsjoint/results_f.csv', index_col=0)
    else:
        results_u = []
        results_f = []

        # For each couple of lambda_1, lambda_2, compute the solution
        for lam in lam_grid:
            print(f'lambda: {lam}')
            # Add the lambda column
            lam_in = np.tile(lam, (allen_pts.shape[0], 1))
            x_in = np.column_stack((allen_pts, lam_in))
            x_in = torch.tensor(x_in, dtype=torch.float32).to(device)
            
            u_preds = {}
            f_preds = {}
            errors_u = {}
            errors_f = {}

            for model, name in zip(models, model_names):
                u_pred = model(x_in).reshape(-1)
                u_preds[name] = u_pred.cpu().detach().numpy()
                f_pred = model.evaluate_forcing(x_in)
                f_preds[name] = f_pred.cpu().detach().numpy()

            # Compute the true solution
            u_true = allen_cahn_true_multi(allen_pts, lam)
            f_true = allen_cahn_forcing_multi(allen_pts, lam)

            # Compute the errors
            for name in model_names:
                errors_u[name] = np.linalg.norm(u_true - u_preds[name], 2) / np.linalg.norm(u_true, 2)
                errors_f[name] = np.linalg.norm(f_true - f_preds[name], 2) / np.linalg.norm(f_true, 2)

            # Add the results to the dataframe
            results_u.append(pd.DataFrame([[lam[0], lam[1]] + [errors_u[name] for name in model_names]], columns=['lambda_1', 'lambda_2'] + model_names))
            results_f.append(pd.DataFrame([[lam[0], lam[1]] + [errors_f[name] for name in model_names]], columns=['lambda_1', 'lambda_2'] + model_names))

        results_u = pd.concat(results_u)
        results_f = pd.concat(results_f)

        # Save the results
        results_u.to_csv(f'{exp_name}/plotsjoint/results_u.csv')
        results_f.to_csv(f'{exp_name}/plotsjoint/results_f.csv')

    return results_u, results_f

results_u, results_f = compute_errors_for_lambda_grid(
    model_list, model_names,
    lam_grid, allen_pts, device, x, y, name
)


avg_errors_u = results_u[model_names].mean()
avg_errors_f = results_f[model_names].mean()
std_errors_u = results_u[model_names].std()
std_errors_f = results_f[model_names].std()

print("Average Errors for u:")
print(avg_errors_u)

print("Standard Deviation for u:")
print(std_errors_u)

print("Average Errors for f:")
print(avg_errors_f)

print("Standard Deviation for f:")
print(std_errors_f)

# Add standard deviation to the dataframe
results_u.loc['mean'] = avg_errors_u
results_u.loc['std'] = std_errors_u
results_f.loc['mean'] = avg_errors_f
results_f.loc['std'] = std_errors_f

print(results_u)
import matplotlib.pyplot as plt

# Determine the model with the lowest error for each square in the lambda_1, lambda_2 domain
def get_best_model(df, model_names):
    df['best_model'] = df[model_names].idxmin(axis=1)
    return df

df_u = get_best_model(results_u, model_names)
df_f = get_best_model(results_f, model_names)

df_u.to_csv(f'{exp_name}/plotsjoint/results_u_best.csv')
df_f.to_csv(f'{exp_name}/plotsjoint/results_f_best.csv')

fig, ax = plt.subplots(figsize=(8, 8))
# Remove rows with nans from the dfs
df_u = df_u.dropna()
df_f = df_f.dropna()

# Create grid data for lambda_1 and lambda_2
lambda_1 = np.linspace(df_u['lambda_1'].min(), df_u['lambda_1'].max(), 100)
lambda_2 = np.linspace(df_u['lambda_2'].min(), df_u['lambda_2'].max(), 100)
lambda_1, lambda_2 = np.meshgrid(lambda_1, lambda_2)

# Interpolate best model data
# Define one hot encoding for the models
model_to_one_hot = {name: i for i, name in enumerate(model_names)}
df_u['best_model_one_hot'] = df_u['best_model'].map(model_to_one_hot)

# Interpolate best model data
best_model_u = griddata((df_u['lambda_1'], df_u['lambda_2']), df_u['best_model_one_hot'], (lambda_1, lambda_2), method='nearest')

# Plot the best model regions
c = ax.pcolormesh(lambda_1, lambda_2, best_model_u, cmap='viridis', shading='auto', vmin=0, vmax=len(model_names)-1)

# Create a legend with color and model name
cmap = plt.get_cmap('viridis')
colors = [cmap(model_to_one_hot[name] / (len(model_names) - 1)) for name in model_names]
patches = [plt.plot([], [], marker="s", ms=10, ls="", mec=None, color=colors[model_to_one_hot[name]], 
            label="{:s}".format(name))[0] for name in model_names]
plt.legend(handles=patches, loc='upper left', ncol=1)

ax.set_xlabel(r'$\lambda_1$')
ax.set_ylabel(r'$\lambda_2$')
ax.set_title('Best Model Regions for Error on u')

plt.savefig(f'{exp_name}/plotsjoint/best_model_u.pdf')

fig, ax = plt.subplots(figsize=(8, 8))
df_f['best_model_one_hot'] = df_f['best_model'].map(model_to_one_hot)
# Interpolate best model data
best_model_f = griddata((df_f['lambda_1'], df_f['lambda_2']), df_f['best_model_one_hot'], (lambda_1, lambda_2), method='nearest')

# Plot the best model regions
c = ax.pcolormesh(lambda_1, lambda_2, best_model_f, cmap='viridis', shading='auto', vmin=0, vmax=len(model_names)-1)

# Create a legend with color and model name
cmap = plt.get_cmap('viridis')
colors = [cmap(model_to_one_hot[name] / (len(model_names) - 1)) for name in model_names]
patches = [plt.plot([], [], marker="s", ms=10, ls="", mec=None, color=colors[model_to_one_hot[name]], 
            label="{:s}".format(name))[0] for name in model_names]
plt.legend(handles=patches, loc='upper left', ncol=1)

ax.set_xlabel(r'$\lambda_1$')
ax.set_ylabel(r'$\lambda_2$')
ax.set_title('Best Model Regions for Error on f')

plt.savefig(f'{exp_name}/plotsjoint/best_model_f.pdf')




plt.rcParams.update({'font.size': 18})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Create grid data for lambda_1 and lambda_2
lambda_1 = np.linspace(df_u['lambda_1'].min(), df_u['lambda_1'].max(), 100)
lambda_2 = np.linspace(df_u['lambda_2'].min(), df_u['lambda_2'].max(), 100)
lambda_1, lambda_2 = np.meshgrid(lambda_1, lambda_2)
colors = ['royalblue', 'tomato', 'limegreen', 'yellow', 'black', 'purple']

# Interpolate best model data for u
df_u['best_model_one_hot'] = df_u['best_model'].map(model_to_one_hot)
best_model_u = griddata((df_u['lambda_1'], df_u['lambda_2']), df_u['best_model_one_hot'], (lambda_1, lambda_2), method='nearest')

# Plot the best model regions for u
c1 = ax1.pcolormesh(lambda_1, lambda_2, best_model_u, cmap=ListedColormap(colors), shading='auto', vmin=0, vmax=len(model_names)-1)
ax1.set_xlabel(r'$\xi_1$')
ax1.set_ylabel(r'$\xi_2$')

# Interpolate best model data for f
df_f['best_model_one_hot'] = df_f['best_model'].map(model_to_one_hot)
best_model_f = griddata((df_f['lambda_1'], df_f['lambda_2']), df_f['best_model_one_hot'], (lambda_1, lambda_2), method='nearest')

# Plot the best model regions for f
# Use a list of colors
c2 = ax2.pcolormesh(lambda_1, lambda_2, best_model_f, cmap=ListedColormap(colors), shading='auto', vmin=0, vmax=len(model_names)-1)
ax2.set_xlabel(r'$\xi_1$')
ax2.set_ylabel(r'$\xi_2$')

# Create a legend with color and model name
patches = [plt.plot([], [], marker="s", ms=10, ls="", mec=None, color=colors[i], 
            label="{:s}".format(name))[0] for i, name in enumerate(model_names)]
fig.legend(handles=patches, loc='upper center', ncol=5)

plt.savefig(f'{exp_name}/plotsjoint/best_model_combined.pdf')
