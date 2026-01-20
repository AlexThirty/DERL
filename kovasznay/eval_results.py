import torch
import numpy as np

from models.params import x_min, x_max, y_min, y_max, nx, ny
from models.pinn import PINN
from models.pinnsformer import PINNsformer, init_weights
device = 'cuda:3'

import matplotlib.pyplot as plt

# Set the font size for matplotlib
plt.rcParams.update({'font.size': 16})

pinnsformer_model = PINNsformer(d_out=3, d_hidden=32, d_model=32, N=1, heads=2).to(device)
pinnsformer_model.apply(init_weights)
pinnsformer_model.device = device

# Define the models
derl_model = PINN(
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
outl_model = PINN(
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
sobl_model = PINN(
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
pinnout_model = PINN(
    hidden_units=[64 for i in range(4)],
    device=device).to(device)

# Load the weights
derl_model.load_state_dict(torch.load('saved_models/Derivative.pt'))
outl_model.load_state_dict(torch.load('saved_models/Output.pt'))
sobl_model.load_state_dict(torch.load('saved_models/Sobolev.pt'))
pinnout_model.load_state_dict(torch.load('saved_models/PINN+Output.pt'))
pinnsformer_model.load_state_dict(torch.load('saved_models/PINNsformer.pt'))

# Set the models to evaluation mode
derl_model.eval()
outl_model.eval()
sobl_model.eval()
pinnout_model.eval()

from plotting_utils import plot_models_errors, generate_grid_data, plot_models_consistencies

# Define the models and their names
models = [derl_model, outl_model, pinnout_model, sobl_model, pinnsformer_model]
model_names = ['DERL', 'OUTL', 'OUTL+PINN', 'SOB', 'PINNsformer']

pts, out, vortex = generate_grid_data(nx, ny)
import os
if not os.path.exists('plots/'):
    os.makedirs('plots/')
    

dx = (x_max - x_min) / nx
dy = (y_max - y_min) / ny
dv = dx * dy

print('nx:', nx)
print('ny:', ny)
print('dv:', dv)

plot_models_errors(models, model_names, pts, out, vortex, 'plots/', nx, ny, dv=dv)
plot_models_consistencies(models, model_names, pts, out, vortex, 'plots/', nx, ny, dv=dv)


if not os.path.exists('plots_no_pinnsformer/'):
    os.makedirs('plots_no_pinnsformer/')

# Remove PINNsformer from the models and model_names
models = [derl_model, outl_model, pinnout_model, sobl_model]
model_names = ['DERL', 'OUTL', 'OUTL+PINN', 'SOB']

# Generate plots without PINNsformer
plot_models_errors(models, model_names, pts, out, vortex, 'plots_no_pinnsformer/', nx, ny, dv=dv)
plot_models_consistencies(models, model_names, pts, out, vortex, 'plots_no_pinnsformer/', nx, ny, dv=dv)