seeds = [30, 42, 2025]


import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np
import random
import os
seed = 30
from torch.func import vmap, jacrev
from itertools import cycle
from numpy import ma
from matplotlib import cbook

from generate import allen_cahn_true, allen_cahn_forcing, allen_cahn_pdv
from model import AllenNet


font = {'size'   : 16}
import matplotlib
matplotlib.rc('font', **font)

parser = argparse.ArgumentParser()
parser.add_argument('--bc_weight', default=1., type=float, help='Weight for the BC loss')
parser.add_argument('--sys_weight', default=1., type=float, help='Weight for the rho loss')
parser.add_argument('--pde_weight', default=1., type=float, help='Weight for the PDE loss')
parser.add_argument('--lr_init', default=1e-4, type=float, help='Starting learning rate')
parser.add_argument('--device', default='cuda', type=str, help='Device to use')
parser.add_argument('--name', default='grid', type=str, help='Experiment name')
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

# Export path and the type of pendulum
EXP_PATH = '.'
    
print(f'Working on {EXP_PATH}')

if not os.path.exists(f'{EXP_PATH}/{name}'):
    os.mkdir(f'{EXP_PATH}/{name}')





activation = torch.nn.Tanh()


model_0 = AllenNet(
    sys_weight=sys_weight,
    pde_weight=pde_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    bc_weight=bc_weight,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)


model_1 = AllenNet(
    sys_weight=sys_weight,
    pde_weight=pde_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    bc_weight=bc_weight,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)


model_neg = AllenNet(
    sys_weight=sys_weight,
    pde_weight= pde_weight,
    bc_weight=bc_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)



model_sob = AllenNet(
    sys_weight=sys_weight,
    pde_weight= pde_weight,
    bc_weight=bc_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)


model_sobhes = AllenNet(
    sys_weight=sys_weight,
    pde_weight= pde_weight,
    bc_weight=bc_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)

model_hes = AllenNet(
    sys_weight=sys_weight,
    pde_weight= pde_weight,
    bc_weight=bc_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)

model_0hes = AllenNet(
    sys_weight=sys_weight,
    pde_weight= pde_weight,
    bc_weight=bc_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)

model_pinnout = AllenNet(
    sys_weight=sys_weight,
    pde_weight= pde_weight,
    bc_weight=bc_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)

xmin = -1.
xmax = 1.
dx = 0.01


torch.cuda.empty_cache()
model_1.eval()
model_0.eval()
model_neg.eval()
model_sob.eval()
model_hes.eval()
model_sobhes.eval()
model_0hes.eval()
model_pinnout.eval()
# %%
import os


errors_dict = {}

for seed in seeds:
    print(f'Processing seed {seed}')
    model_0.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/allen_netDerivative_seed_{seed}'))
    model_1.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/allen_netOutput_seed_{seed}'))
    model_sob.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/allen_netSobolev_seed_{seed}'))
    model_pinnout.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/allen_netPINN+Output_seed_{seed}'))

    torch.cuda.empty_cache()
    model_1.eval()
    model_0.eval()
    model_neg.eval()
    model_sob.eval()

    model_list = [model_0, model_1, model_sob, model_pinnout]
    model_names = ['DERL', 'OUTL', 'SOB', 'OUTL+PINN']

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
    pdv_models = []
    error_pdv_models = []
    pde_models = []

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
        pde_models.append(pde.reshape(x_plot.shape))
        print('PDE residual computed')

    # Transform the arrays
    u_models = np.array(u_models)
    error_u_models = np.array(error_u_models)
    forcing_models = np.array(forcing_models)
    error_forcing_models = np.array(error_forcing_models)
    pdv_models = np.array(pdv_models)
    error_pdv_models = np.array(error_pdv_models)
    pde_models = np.array(pde_models)
    # Store errors in dictionary
    errors_dict[seed] = {
        'error_u_L2': np.sqrt(grid_size * np.sum(error_u_models**2, axis=(1, 2))),
        'error_forcing_L2': np.sqrt(grid_size * np.sum(error_forcing_models**2, axis=(1, 2))),
        'error_pdv_L2': np.sqrt(grid_size * np.sum(error_pdv_models**2, axis=(1, 2))),
        'error_pde_L2': np.sqrt(grid_size * np.sum(pde_models**2, axis=(1, 2))),
        'error_u_max': np.max(np.abs(error_u_models), axis=(1, 2)),
        'error_forcing_max': np.max(np.abs(error_forcing_models), axis=(1, 2)),
        'error_pdv_max': np.max(np.abs(error_pdv_models), axis=(1, 2)),
        'error_pde_max': np.max(np.abs(pde_models), axis=(1, 2))
    }

    # Print the shapes
    print(f'error_u_L2.shape: {errors_dict[seed]["error_u_L2"].shape}')
    print(f'error_forcing_L2.shape: {errors_dict[seed]["error_forcing_L2"].shape}')
    print(f'error_pdv_L2.shape: {errors_dict[seed]["error_pdv_L2"].shape}')
    print(f'error_pde_L2.shape: {errors_dict[seed]["error_pde_L2"].shape}')
    print(f'error_u_max.shape: {errors_dict[seed]["error_u_max"].shape}')
    print(f'error_forcing_max.shape: {errors_dict[seed]["error_forcing_max"].shape}')
    print(f'error_pdv_max.shape: {errors_dict[seed]["error_pdv_max"].shape}')
    print(f'error_pde_max.shape: {errors_dict[seed]["error_pde_max"].shape}')

    comparison_u_models = errors_dict[seed]['error_u_L2'][1:] - errors_dict[seed]['error_u_L2'][0]
    comparison_forcing_models = errors_dict[seed]['error_forcing_L2'][1:] - errors_dict[seed]['error_forcing_L2'][0]
    comparison_pdv_models = errors_dict[seed]['error_pdv_L2'][1:] - errors_dict[seed]['error_pdv_L2'][0]
    comparison_pde_models = errors_dict[seed]['error_pde_L2'][1:] - errors_dict[seed]['error_pde_L2'][0]

    # Now print the errors
    with open(f'{name}/errors_seed_{seed}.txt', 'w') as f:
        f.write('Errors for the Allen-Cahn problem\n')
        for i in range(len(model_names)):
            f.write(f'Model {model_names[i]}\n')
            f.write(f'Error u L2: {errors_dict[seed]["error_u_L2"][i]}\n')
            f.write(f'Error u L2 normalized: {errors_dict[seed]["error_u_L2"][i] / np.sqrt(np.sum(u_grid**2))}\n')
            f.write(f'Error u max: {errors_dict[seed]["error_u_max"][i]}\n')
            f.write(f'Error forcing L2: {errors_dict[seed]["error_forcing_L2"][i]}\n')
            f.write(f'Error forcing L2 normalized: {errors_dict[seed]["error_forcing_L2"][i] / np.sqrt(np.sum(forcing_grid**2))}\n')
            f.write(f'Error forcing max: {errors_dict[seed]["error_forcing_max"][i]}\n')
            f.write(f'Error pdv L2: {errors_dict[seed]["error_pdv_L2"][i]}\n')
            f.write(f'Error pdv max: {errors_dict[seed]["error_pdv_max"][i]}\n')
            f.write(f'Error pde L2: {errors_dict[seed]["error_pde_L2"][i]}\n')
            f.write(f'Error pde max: {errors_dict[seed]["error_pde_max"][i]}\n')

# Calculate averages and standard deviations across seeds
avg_errors = {}
std_errors = {}

for key in errors_dict[seeds[0]].keys():
    avg_errors[key] = np.mean([errors_dict[seed][key] for seed in seeds], axis=0)
    std_errors[key] = np.std([errors_dict[seed][key] for seed in seeds], axis=0)

# Save the averages and standard deviations
with open(f'{name}/avg_std_errors.txt', 'w') as f:
    f.write('Averages and standard deviations for the Allen-Cahn problem across seeds\n')
    for i in range(len(model_names)):
        f.write(f'Model {model_names[i]}\n')
        f.write(f'Average Error u L2: {avg_errors["error_u_L2"][i]} pm {std_errors["error_u_L2"][i]}\n')
        f.write(f'Average Error u L2 normalized: {avg_errors["error_u_L2"][i] / np.sqrt(np.sum(u_grid**2))}\n')
        f.write(f'Average Error u max: {avg_errors["error_u_max"][i]} pm {std_errors["error_u_max"][i]}\n')
        f.write(f'Average Error forcing L2: {avg_errors["error_forcing_L2"][i]} pm {std_errors["error_forcing_L2"][i]}\n')
        f.write(f'Average Error forcing L2 normalized: {avg_errors["error_forcing_L2"][i] / np.sqrt(np.sum(forcing_grid**2))}\n')
        f.write(f'Average Error forcing max: {avg_errors["error_forcing_max"][i]} pm {std_errors["error_forcing_max"][i]}\n')
        f.write(f'Average Error pdv L2: {avg_errors["error_pdv_L2"][i]} pm {std_errors["error_pdv_L2"][i]}\n')
        f.write(f'Average Error pdv max: {avg_errors["error_pdv_max"][i]} pm {std_errors["error_pdv_max"][i]}\n')
        f.write(f'Average Error pde L2: {avg_errors["error_pde_L2"][i]} pm {std_errors["error_pde_L2"][i]}\n')
        f.write(f'Average Error pde max: {avg_errors["error_pde_max"][i]} pm {std_errors["error_pde_max"][i]}\n')
