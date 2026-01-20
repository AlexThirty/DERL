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
model_derl.load_state_dict(torch.load(f'saved_models/pinn_{name}_Derivative_phase2'))
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
model_outl.load_state_dict(torch.load(f'saved_models/pinn_{name}_Output_phase2'))
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
model_sob.load_state_dict(torch.load(f'saved_models/pinn_{name}_Sobolev_phase2'))
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
model_pinn.load_state_dict(torch.load(f'saved_models/pinn_full'))
model_pinn.eval()

model_pinnextend = AllenNet(
    bc_weight=bc_weight,
    pde_weight=pde_weight,
    sys_weight=sys_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,
    last_activation=False,
).to(device)
model_pinnextend.load_state_dict(torch.load(f'saved_models/pinn_extend_PINN_phase2'))
model_pinnextend.eval()

model_forget = AllenNet(
    bc_weight=bc_weight,
    pde_weight=pde_weight,
    sys_weight=sys_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,
    last_activation=False,
).to(device)
if os.path.exists(f'saved_models/pinn_extend_Forgetting'):
    model_forget.load_state_dict(torch.load(f'saved_models/pinn_extend_Forgetting_phase2'))
model_forget.eval()

if not os.path.exists(f'{name}/plotsjoint'):
    os.makedirs(f'{name}/plotsjoint')
    
import numpy as np
from matplotlib import pyplot as plt
from generate import allen_cahn_true
# Generate the grid for the true solution
xmin = -1.
xmax = 1.
ymin = -1.
ymax = 1.
dx = 0.01
x = np.arange(xmin, xmax+dx, dx)
y = np.arange(ymin, ymax+dx, dx)
x_pts, y_pts = np.meshgrid(x, y)
pts = np.column_stack((x_pts.reshape((-1,1)), y_pts.reshape((-1,1))))

pts = torch.from_numpy(pts).to(device).float()


models = [model_derl, model_outl, model_sob, model_pinn, model_forget, model_pinnextend]
model_names = ['DERL', 'OUTL', 'SOB', 'PINN full', 'PINN no dist.', 'PINN fine-tune']
u_preds = []
f_preds = []
errors = []
errors_f = []

for model in models:
    u_pred = model(pts).reshape(-1).cpu().detach().numpy()
    u_preds.append(u_pred)
    
    f_pred = model.evaluate_forcing(pts).cpu().detach().numpy()
    f_preds.append(f_pred)

# Compute the true solution
from generate import allen_cahn_forcing, allen_cahn_true
u_true = allen_cahn_true(pts.cpu().detach().numpy())
f_true = allen_cahn_forcing(pts.cpu().detach().numpy())

print('Calculating error')
# Compute the error
for u_pred, f_pred in zip(u_preds, f_preds):
    error = np.linalg.norm(u_true - u_pred, 2) / np.linalg.norm(u_true, 2)
    errors.append(error)
    
    error_f = np.linalg.norm(f_true - f_pred, 2) / np.linalg.norm(f_true, 2)
    errors_f.append(error_f)

with open(f'{name}/plotsjoint/test_results.txt', 'w') as f:
    for model_name, error, error_f in zip(model_names, errors, errors_f):
        f.write(f'Error {model_name}: {error}\n')
        f.write(f'Error {model_name} Forcing: {error_f}\n')
    f.write('\n')

from plotting_utils import error_plotting, comparison_plotting
cap = 1.

error_plots = [np.clip(np.abs(u_true - u_pred).reshape(x_pts.shape), 0, cap) for u_pred in u_preds]
error_f_plots = [np.clip(np.abs(f_true - f_pred).reshape(x_pts.shape), 0, cap) for f_pred in f_preds]

error_plotting(error_plots, x_pts, y_pts, model_names, path=f'{name}/plotsjoint/error.pdf')
error_plotting(error_f_plots, x_pts, y_pts, model_names, path=f'{name}/plotsjoint/forcing_error.pdf')

comparison_plot = [error_plots[i] - error_plots[0] for i in range(1, len(error_plots))]
comparison_plotting(comparison_plot, x_pts, y_pts, model_names[1:], path=f'{name}/plotsjoint/comparison_error.pdf')

comparison_plot = [error_f_plots[i] - error_f_plots[0] for i in range(1, len(error_f_plots))]
comparison_plotting(comparison_plot, x_pts, y_pts, model_names[1:], path=f'{name}/plotsjoint/comparison_forcing_error.pdf')


error_plotting(error_plots[:-1], x_pts, y_pts, model_names[:-1], path=f'{name}/plotsjoint/error_noforget.pdf')
error_plotting(error_f_plots[:-1], x_pts, y_pts, model_names[:-1], path=f'{name}/plotsjoint/forcing_error_noforget.pdf')

comparison_plot = [error_plots[i] - error_plots[0] for i in range(1, len(error_plots))]
comparison_plotting(comparison_plot[:-1], x_pts, y_pts, model_names[1:-1], path=f'{name}/plotsjoint/comparison_error_noforget.pdf')
comparison_plot = [error_f_plots[i] - error_f_plots[0] for i in range(1, len(error_f_plots))]
comparison_plotting(comparison_plot[:-1], x_pts, y_pts, model_names[1:-1], path=f'{name}/plotsjoint/comparison_forcing_error_noforget.pdf')