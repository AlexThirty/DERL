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
parser.add_argument('--name', default='exp', type=str, help='Experiment name')
parser.add_argument('--train_steps', default=100000, type=int, help='Number of training steps')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('--mode', default='Derivative', type=str, help='Mode: -1 for PINN learning, 0 for derivative learning, 1 for output learning')
parser.add_argument('--layers', default=4, type=int, help='Number of layers in the network')
parser.add_argument('--units', default=50, type=int, help='Number of units per layer in the network')

args = parser.parse_args()
bc_weight = args.bc_weight  
device = args.device
name = args.name
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
    
    
batch_size = 32

print('Data loaded!')

activation = torch.nn.Tanh()

model_derl = AllenNet(
    bc_weight=bc_weight,
    pde_weight=pde_weight,
    sys_weight=sys_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)
model_derl.load_state_dict(torch.load(f'{name}/saved_models/allen_netDerivative'))
model_derl.eval()

model_outl = AllenNet(
    bc_weight=bc_weight,
    pde_weight=pde_weight,
    sys_weight=sys_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)
model_outl.load_state_dict(torch.load(f'{name}/saved_models/allen_netOutput'))
model_outl.eval()

model_sob = AllenNet(
    bc_weight=bc_weight,
    pde_weight=pde_weight,
    sys_weight=sys_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)
model_sob.load_state_dict(torch.load(f'{name}/saved_models/allen_netSobolev'))
model_sob.eval()

model_pinn = AllenNet(
    bc_weight=bc_weight,
    pde_weight=pde_weight,
    sys_weight=sys_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)
model_pinn.load_state_dict(torch.load(f'{name}/saved_models/allen_netPINN'))
model_pinn.eval()

model_pinnout = AllenNet(
    bc_weight=bc_weight,
    pde_weight=pde_weight,
    sys_weight=sys_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)
model_pinnout.load_state_dict(torch.load(f'{name}/saved_models/allen_netPINN+Output'))
model_pinnout.eval()

if not os.path.exists(f'exp/plotsjoint'):
    os.makedirs(f'exp/plotsjoint')
    
if not os.path.exists(f'exp/plotsjoint/val'):
    os.makedirs(f'exp/plotsjoint/val')
    
if not os.path.exists(f'exp/plotsjoint/train'):
    os.makedirs(f'exp/plotsjoint/train')
    
if not os.path.exists(f'exp/plotsjoint/val_nopinn'):
    os.makedirs(f'exp/plotsjoint/val_nopinn')
    
if not os.path.exists(f'exp/plotsjoint/train_nopinn'):
    os.makedirs(f'exp/plotsjoint/train_nopinn')
    
# load allen_pts and make the grid
allen_pts = np.load('allen_pts.npy')
allen_x = np.load('allen_x.npy')
allen_y = np.load('allen_y.npy')
x, y = np.meshgrid(allen_x, allen_y)
allen_pts = np.column_stack((x.reshape(-1), y.reshape(-1)))

# Import the valing lambdas
from generate_exp import allen_cahn_true_exp, allen_cahn_forcing_exp
from plotting_utils import error_plotting, comparison_plotting
import pandas as pd
lam_train = np.load('allen_lam_train_exp.npy')
lam_val = np.load('allen_lam_val_exp.npy')

test_lam = [0.6, 0.9, 1.4, 2.3, 2.6, 2.8]
if not os.path.exists(f'exp/plotsjoint/test'):
    os.makedirs(f'exp/plotsjoint/test')
    
if not os.path.exists(f'exp/plotsjoint/test/test_nopinn'):
    os.makedirs(f'exp/plotsjoint/test/test_nopinn') 

def evaluate_models_on_val_data(models: list[AllenNet], model_names, lam_list, allen_pts, device, x, y, lam_type='val'):
    total_errors = {name: 0 for name in model_names}
    total_forcing_errors = {name: 0 for name in model_names}
    num_val = len(lam_list)

    errors_data = []
    forcing_errors_data = []

    for lam in lam_list:
        print(f'lambda: {lam}')
        lam_in = np.tile(lam, (allen_pts.shape[0], 1))
        x_in = np.column_stack((allen_pts, lam_in))
        x_in = torch.tensor(x_in, dtype=torch.float32).to(device)

        u_preds = {}
        for model, name in zip(models, model_names):
            u_pred = model(x_in).reshape(-1)
            u_preds[name] = u_pred.cpu().detach().numpy()

        u_true = allen_cahn_true_exp(allen_pts, lam)
        f_true = allen_cahn_forcing_exp(allen_pts, lam)
        print('Calculating error')

        for name in model_names:
            error = np.linalg.norm(u_true - u_preds[name], 2) / np.linalg.norm(u_true, 2)
            total_errors[name] += error
            errors_data.append({'lambda': lam, 'model': name, 'error': error})

            f_pred = models[model_names.index(name)].evaluate_forcing(x_in)
            f_pred = f_pred.cpu().detach().numpy()
            forcing_error = np.linalg.norm(f_true - f_pred, 2) / np.linalg.norm(f_true, 2)
            total_forcing_errors[name] += forcing_error
            forcing_errors_data.append({'lambda': lam, 'model': name, 'forcing_error': forcing_error})

    avg_errors = {name: total_errors[name] / num_val for name in model_names}
    avg_forcing_errors = {name: total_forcing_errors[name] / num_val for name in model_names}

    with open(f'exp/plotsjoint/{lam_type}_results.txt', 'w') as f:
        for name in model_names:
            f.write(f'Average Error {name}: {avg_errors[name]}\n')
        f.write('\n')
        for name in model_names:
            f.write(f'Average Error {name} Forcing: {avg_forcing_errors[name]}\n')
        f.write('\n')

    results_u = []
    results_f = []
    
    for lam in lam_list:
        print(f'lambda: {lam}')
        lam_in = np.tile(lam, (allen_pts.shape[0], 1))
        x_in = np.column_stack((allen_pts, lam_in))
        x_in = torch.tensor(x_in, dtype=torch.float32).to(device)

        u_preds = {}
        u_errors = {}
        f_preds = {}
        f_errors = {}
        for model, name in zip(models, model_names):
            u_pred = model(x_in).reshape(-1)
            u_preds[name] = u_pred.cpu().detach().numpy()

        u_true = allen_cahn_true_exp(allen_pts, lam)
        f_true = allen_cahn_forcing_exp(allen_pts, lam)
        print('Calculating error')

        with open(f'exp/plotsjoint/{lam_type}_results.txt', 'a') as f:
            f.write(f'lambda: {lam}\n')
            for name in model_names:
                error = np.linalg.norm(u_true - u_preds[name], 2) / np.linalg.norm(u_true, 2)
                u_errors[name] = error
                f.write(f'Error {name}: {error}\n')

                f_pred = models[model_names.index(name)].evaluate_forcing(x_in)
                f_pred = f_pred.cpu().detach().numpy()
                f_preds[name] = f_pred
                forcing_error = np.linalg.norm(f_true - f_pred, 2) / np.linalg.norm(f_true, 2)
                f_errors[name] = forcing_error
                f.write(f'Error {name} Forcing: {forcing_error}\n')
            f.write('\n')
        normalizer_u = np.linalg.norm(u_true, 2)
        normalizer_f = np.linalg.norm(f_true, 2)
        results_u.append(pd.DataFrame([[lam] + [u_errors[name] for name in model_names]], columns=['lambda'] + model_names))
        results_f.append(pd.DataFrame([[lam] + [f_errors[name] for name in model_names]], columns=['lambda'] + model_names))
        
        cap = 100.
        error_plots = {name: np.clip(np.abs(u_true - u_preds[name]).reshape(x.shape)/normalizer_u, 0, cap) for name in model_names}
        forcing_error_plots = {name: np.clip(np.abs(f_true - models[model_names.index(name)].evaluate_forcing(x_in).cpu().detach().numpy()).reshape(x.shape)/normalizer_f, 0, cap) for name in model_names}

        error_plotting([error_plots[name] for name in model_names], x, y, model_names, path=f'exp/plotsjoint/{lam_type}/error_lambda_{lam}.pdf')
        error_plotting([forcing_error_plots[name] for name in model_names], x, y, model_names, path=f'exp/plotsjoint/val/forcing_error_lambda_{lam}.pdf')

        comparison_plot = [error_plots[name] - error_plots['DERL'] for name in model_names if name != 'DERL']
        comparison_plotting(comparison_plot, x, y, [name for name in model_names if name != 'DERL'], path=f'exp/plotsjoint/{lam_type}/comparison_error_lambda_{lam}.pdf')

        comparison_plot = [forcing_error_plots[name] - forcing_error_plots['DERL'] for name in model_names if name != 'DERL']
        comparison_plotting(comparison_plot, x, y, [name for name in model_names if name != 'DERL'], path=f'exp/plotsjoint/{lam_type}/comparison_forcing_error_lambda_{lam}.pdf')

    results_u = pd.concat(results_u)
    results_f = pd.concat(results_f)

    return results_u, results_f

model_list = [model_derl, model_outl, model_pinnout, model_sob, model_pinn]
model_names = ['DERL', 'OUTL', 'OUTL+PINN', 'SOB', 'PINN']

model_list_nopinn = [model_derl, model_outl, model_pinnout, model_sob]
model_names_nopinn = ['DERL', 'OUTL', 'OUTL+PINN', 'SOB']


if not os.path.exists('exp/plotsjoint/errors_u.csv'):
    val_df_u, val_df_f = evaluate_models_on_val_data(model_list, model_names, lam_val, allen_pts, device, x, y, lam_type='val')
    train_df_u, train_df_f = evaluate_models_on_val_data(model_list, model_names, lam_train, allen_pts, device, x, y, lam_type='train')
    
    _, _ = evaluate_models_on_val_data(model_list_nopinn, model_names_nopinn, lam_val, allen_pts, device, x, y, lam_type='val_nopinn')
    _, _ = evaluate_models_on_val_data(model_list_nopinn, model_names_nopinn, lam_train, allen_pts, device, x, y, lam_type='train_nopinn')

    test_df_u, test_df_f = evaluate_models_on_val_data(model_list, model_names, test_lam, allen_pts, device, x, y, lam_type='test')

    _, _ = evaluate_models_on_val_data(model_list_nopinn, model_names_nopinn, test_lam, allen_pts, device, x, y, lam_type='test_nopinn')

    train_df_u['type'] = 'train'
    val_df_u['type'] = 'val'
    test_df_u['type'] = 'test'

    train_df_f['type'] = 'train'
    val_df_f['type'] = 'val'
    test_df_f['type'] = 'test'


    df_u = pd.concat([train_df_u, val_df_u, test_df_u])
    df_f = pd.concat([train_df_f, val_df_f, test_df_f])
    df_u.sort_values(by='lambda', inplace=True)
    df_f.sort_values(by='lambda', inplace=True)

else:
    df_u = pd.read_csv('exp/plotsjoint/errors_u.csv')
    df_f = pd.read_csv('exp/plotsjoint/errors_f.csv')
    df_u = df_u[['lambda', 'type'] + model_names]
    df_f = df_f[['lambda', 'type'] + model_names]
    
    
    
df_u_nopinn = df_u.drop(columns=['PINN'])
df_f_nopinn = df_f.drop(columns=['PINN'])

df_u.to_csv('exp/plotsjoint/errors_u.csv')
df_f.to_csv('exp/plotsjoint/errors_f.csv')

import matplotlib.pyplot as plt

def plot_errors(df, title, filename):
    plt.figure(figsize=(8, 8))
    markers = {'train': 'o', 'val': 's', 'test': '^'}
    model_names = [col for col in df.columns if col not in ['lambda', 'type']]
    print(model_names)
    colors = {name: color for name, color in zip(model_names, ['r', 'g', 'b', 'orange', 'k'])}
    
    for model_name in model_names:
        for data_type in df['type'].unique():
            subset = df[df['type'] == data_type]
            plt.scatter(subset['lambda'], subset[model_name], marker=markers[data_type], color=colors[model_name])
    
    # Now add lines independent of the type
    for model_name in model_names:
        plt.plot(df['lambda'], df[model_name], label=model_name, color=colors[model_name])
        
    # Add legend for markers
    for data_type in df['type'].unique():
        plt.scatter([], [], marker=markers[data_type], color='k', label=data_type)

    plt.xlabel('Lambda')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

plot_errors(df_u, r'Normalized $L^2$ error', 'exp/plotsjoint/errors_u.pdf')
plot_errors(df_u_nopinn, r'Normalized $L^2$ error', 'exp/plotsjoint/errors_u_nopinn.pdf')
plot_errors(df_f, r'PDE residual', 'exp/plotsjoint/errors_f.pdf')
plot_errors(df_f_nopinn, r'PDE residual', 'exp/plotsjoint/errors_f_nopinn.pdf')
plt.rcParams.update({'font.size': 20})

def plot_combined_errors(df_u, df_f, title, filename):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    model_names_u = [col for col in df_u.columns if col not in ['lambda', 'type']]
    model_names_f = [col for col in df_f.columns if col not in ['lambda', 'type']]
    colors_u = {name: color for name, color in zip(model_names_u, ['royalblue', 'tomato', 'orange', 'limegreen', 'yellow'])}
    colors_f = {name: color for name, color in zip(model_names_f, ['royalblue', 'tomato', 'orange', 'limegreen', 'yellow'])}
    
    for model_name in model_names_u:
        for data_type in df_u['type'].unique():
            subset = df_u[df_u['type'] == data_type]
            axs[0].scatter(subset['lambda'], subset[model_name], color=colors_u[model_name])
    
    for model_name in model_names_f:
        for data_type in df_f['type'].unique():
            subset = df_f[df_f['type'] == data_type]
            axs[1].scatter(subset['lambda'], subset[model_name], color=colors_f[model_name])
    
    for model_name in model_names_u:
        axs[0].plot(df_u['lambda'], df_u[model_name], label=model_name, color=colors_u[model_name])
        
    for model_name in model_names_f:
        axs[1].plot(df_f['lambda'], df_f[model_name], label=model_name, color=colors_f[model_name])
        
    #for data_type in df_u['type'].unique():
    #    axs[0].scatter([], [], color='k', label=data_type)
    #    axs[1].scatter([], [], color='k', label=data_type)
    
    axs[0].set_xlabel(r'$\xi$')
    axs[0].set_ylabel(r'$\|u-\hat{u}\|_2$')
    axs[0].set_yscale('log')
    axs[0].grid(True)

    axs[1].set_xlabel(r'$\xi$')
    axs[1].set_ylabel(r'PDE residual')
    axs[1].set_yscale('log')
    axs[1].grid(True)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename)
    plt.close()

plot_combined_errors(df_u, df_f, r'Normalized $L^2$ error and PDE residual', 'exp/plotsjoint/combined_errors.pdf')
plot_combined_errors(df_u_nopinn, df_f_nopinn, r'Normalized $L^2$ error and PDE residual', 'exp/plotsjoint/combined_errors_nopinn.pdf')