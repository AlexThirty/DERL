import torch
from model import KdVPINN, SinActivation
import argparse
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import os
from torch.func import vmap, jacrev, hessian
from itertools import cycle

font = {'size'   : 16}
import matplotlib
matplotlib.rc('font', **font)

seed = 30
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--init_weight', default=5., type=float, help='Weight for the init loss')
parser.add_argument('--pde_weight', default=0.1, type=float, help='Weight for the F loss')
parser.add_argument('--sys_weight', default=0.1, type=float, help='Weight for the system loss')
parser.add_argument('--bc_weight', default=1., type=float, help='Weight for the F loss')
parser.add_argument('--lr_init', default=1e-4, type=float, help='Starting learning rate')
parser.add_argument('--device', default='cuda:2', type=str, help='Device to use')
parser.add_argument('--name', default='kdv', type=str, help='Experiment name')
parser.add_argument('--train_steps', default=100000, type=int, help='Number of training steps')
parser.add_argument('--epochs', default=500, type=int, help='Number of epochs')
parser.add_argument('--batch_size', default=64, type=int, help='Number of samples per step')
parser.add_argument('--layers', default=9, type=int, help='Number of layers in the network')
parser.add_argument('--units', default=50, type=int, help='Number of units per layer in the network')

args = parser.parse_args()
init_weight = args.init_weight
pde_weight = args.pde_weight
bc_weight = args.bc_weight
device = args.device
name = args.name
train_steps = args.train_steps
epochs = args.epochs
batch_size = args.batch_size
layers = args.layers
units = args.units
lr_init = args.lr_init
sys_weight = args.sys_weight

pde_dataset = torch.load(os.path.join('data', f'kdv_pde_dataset.pth'), weights_only=False)
init_dataset = torch.load(os.path.join('data', f'kdv_init_dataset.pth'), weights_only=False)
bc1_dataset = torch.load(os.path.join('data', f'kdv_bc1_dataset.pth'), weights_only=False)
bc2_dataset = torch.load(os.path.join('data', f'kdv_bc2_dataset.pth'), weights_only=False)

# Generate the dataloaders
pde_dataloader = DataLoader(pde_dataset, batch_size, generator=gen, shuffle=True)
init_dataloader = DataLoader(init_dataset, batch_size, generator=gen, shuffle=True)
bc1_dataloader = DataLoader(bc1_dataset, batch_size, generator=gen, shuffle=True)
bc2_dataloader = DataLoader(bc2_dataset, batch_size, generator=gen, shuffle=True)
test_dataloader = DataLoader(pde_dataset, 256, generator=gen, shuffle=True)
    
# Create PINN models for different phases
model_pinn = KdVPINN(hidden_units=[units for _ in range(layers)],
                pde_weight=pde_weight,
                distillation_weight=1,
                sys_weight=sys_weight,
                bc_weight=bc_weight,
                init_weight=init_weight,
                device=device,
                activation=torch.nn.Tanh(),
                lr=lr_init)
model_pinn.eval()

# Phase 1 PINN
model_pinn_sob_new = KdVPINN(hidden_units=[units for _ in range(layers)],
                pde_weight=pde_weight,
                sys_weight=sys_weight,
                bc_weight=bc_weight,
                init_weight=init_weight,
                distillation_weight=1.,
                device=device,
                activation=torch.nn.Tanh(),
                lr=lr_init)
model_pinn_sob_new.eval()

model_pinn_sob_new.load_state_dict(torch.load(f'./saved_models/pinn_phase2_Sobolev_new'))

model_pinn_sob_extend = KdVPINN(hidden_units=[units for _ in range(layers)],
                pde_weight=pde_weight,
                sys_weight=sys_weight,
                bc_weight=bc_weight,
                init_weight=init_weight,
                distillation_weight=1.,
                device=device,
                activation=torch.nn.Tanh(),
                lr=lr_init)
model_pinn_sob_extend.eval()

model_pinn_sob_extend.load_state_dict(torch.load(f'./saved_models/pinn_phase2_Sobolev_extend'))

# Phase 2 PINN (can extend from phase 1)
model_pinn_derl_new = KdVPINN(hidden_units=[units for _ in range(layers)],
                pde_weight=pde_weight,
                sys_weight=sys_weight,
                bc_weight=bc_weight,
                distillation_weight=1.,
                init_weight=init_weight,
                device=device,
                activation=torch.nn.Tanh(),
                lr=lr_init)
model_pinn_derl_new.eval()

model_pinn_derl_new.load_state_dict(torch.load(f'./saved_models/pinn_phase2_Derivative_new'))

model_pinn_derl_extend = KdVPINN(hidden_units=[units for _ in range(layers)],
                pde_weight=pde_weight,
                sys_weight=sys_weight,
                bc_weight=bc_weight,
                distillation_weight=1.,
                init_weight=init_weight,
                device=device,
                activation=torch.nn.Tanh(),
                lr=lr_init)
model_pinn_derl_extend.eval()
model_pinn_derl_extend.load_state_dict(torch.load(f'./saved_models/pinn_phase2_Derivative_extend'))

# Output mode models
model_pinn_output_new = KdVPINN(hidden_units=[units for _ in range(layers)],
                pde_weight=pde_weight,
                sys_weight=sys_weight,
                bc_weight=bc_weight,
                distillation_weight=1.,
                init_weight=init_weight,
                device=device,
                activation=torch.nn.Tanh(),
                lr=lr_init)
model_pinn_output_new.eval()
model_pinn_output_new.load_state_dict(torch.load(f'./saved_models/pinn_phase2_Output_new'))

model_pinn_output_extend = KdVPINN(hidden_units=[units for _ in range(layers)],
                pde_weight=pde_weight,
                sys_weight=sys_weight,
                bc_weight=bc_weight,
                distillation_weight=1.,
                init_weight=init_weight,
                device=device,
                activation=torch.nn.Tanh(),
                lr=lr_init)
model_pinn_output_extend.eval()
model_pinn_output_extend.load_state_dict(torch.load(f'./saved_models/pinn_phase2_Output_extend'))


model_pinn_forget = KdVPINN(hidden_units=[units for _ in range(layers)],
                pde_weight=pde_weight,
                sys_weight=sys_weight,
                bc_weight=bc_weight,
                distillation_weight=1.,
                init_weight=init_weight,
                device=device,
                activation=torch.nn.Tanh(),
                lr=lr_init)
model_pinn_forget.eval()
model_pinn_forget.load_state_dict(torch.load(f'./saved_models/pinn_phase2_Forgetting_extend'))

model_pinn_replay = KdVPINN(hidden_units=[units for _ in range(layers)],
                pde_weight=pde_weight,
                sys_weight=sys_weight,
                bc_weight=bc_weight,
                distillation_weight=1.,
                init_weight=init_weight,
                device=device,
                activation=torch.nn.Tanh(),
                lr=lr_init)
model_pinn_replay.eval()
model_pinn_replay.load_state_dict(torch.load(f'./saved_models/pinn_phase2_PINN_extend'))

# Load model weights
model_pinn.load_state_dict(torch.load(f'saved_models/pinn'))

import numpy as np
from matplotlib import pyplot as plt
#plotting function to generate the figures for the ball problem
from matplotlib.colors import TwoSlopeNorm

if not os.path.exists(f'plotsjoint'):
    os.makedirs(f'plotsjoint')

xlim = 1.
ylim = 1.

# Import the true function
with open(f'data/kdv_data.npy', 'rb') as f:
    kdv_data = np.load(f)
    print(f'kdv_data.shape: {kdv_data.shape}')

from plotting_utils import plot_errors
from scipy.ndimage import uniform_filter1d
from matplotlib.lines import Line2D

plot_errors(
    [
        model_pinn_derl_new, model_pinn_derl_extend,
        model_pinn_sob_new, model_pinn_sob_extend,
        model_pinn_output_new, model_pinn_output_extend,
        model_pinn, model_pinn_forget, model_pinn_replay
    ],
    [
        'DERL from-scratch', 'DERL continual',
        'SOB from-scratch', 'SOB continual',
        'OUTL from-scratch', 'OUTL continual',
        'PINN', 'PINN no dist.', 'PINN with replay'
    ],
    f'plotsjoint', kdv_data, model_pinn,
)


with open(f'pinn/traindata.npy', 'rb') as f:
    pinn_losses = np.load(f)
    print(f'pinn_losses.shape: {pinn_losses.shape}')

with open(f'plots_phase0/traindata.npy', 'rb') as f:
    phase0_losses = np.load(f)
    print(f'phase0_losses.shape: {phase0_losses.shape}')
    
with open(f'plots_phase1_Sobolev_new/traindata.npy', 'rb') as f:
    phase1_sob_new_losses = np.load(f)
    print(f'phase1_sob_new_losses.shape: {phase1_sob_new_losses.shape}')
    
with open(f'plots_phase1_Sobolev_extend/traindata.npy', 'rb') as f:
    phase1_sob_extend_losses = np.load(f)
    print(f'phase1_sob_extend_losses.shape: {phase1_sob_extend_losses.shape}')
    
with open(f'plots_phase1_Derivative_new/traindata.npy', 'rb') as f:
    phase1_derl_new_losses = np.load(f)
    print(f'phase1_derl_new_losses.shape: {phase1_derl_new_losses.shape}')
    
with open(f'plots_phase1_Derivative_extend/traindata.npy', 'rb') as f:
    phase1_derl_extend_losses = np.load(f)
    print(f'phase1_derl_extend_losses.shape: {phase1_derl_extend_losses.shape}')
    
with open(f'plots_phase1_Output_new/traindata.npy', 'rb') as f:
    phase1_output_new_losses = np.load(f)
    print(f'phase1_output_new_losses.shape: {phase1_output_new_losses.shape}')
    
with open(f'plots_phase1_Output_extend/traindata.npy', 'rb') as f:
    phase1_output_extend_losses = np.load(f)
    print(f'phase1_output_extend_losses.shape: {phase1_output_extend_losses.shape}')

with open(f'plots_phase2_Sobolev_new/traindata.npy', 'rb') as f:
    phase2_sob_new_losses = np.load(f)
    print(f'phase2_sob_new_losses.shape: {phase2_sob_new_losses.shape}')
    
with open(f'plots_phase2_Sobolev_extend/traindata.npy', 'rb') as f:
    phase2_sob_extend_losses = np.load(f)
    print(f'phase2_sob_extend_losses.shape: {phase2_sob_extend_losses.shape}')
    
with open(f'plots_phase2_Derivative_new/traindata.npy', 'rb') as f:
    phase2_derl_new_losses = np.load(f)
    print(f'phase2_derl_new_losses.shape: {phase2_derl_new_losses.shape}')

with open(f'plots_phase2_Derivative_extend/traindata.npy', 'rb') as f:
    phase2_derl_extend_losses = np.load(f)
    print(f'phase2_derl_extend_losses.shape: {phase2_derl_extend_losses.shape}')

with open(f'plots_phase2_Output_new/traindata.npy', 'rb') as f:
    phase2_output_new_losses = np.load(f)
    print(f'phase2_output_new_losses.shape: {phase2_output_new_losses.shape}')

with open(f'plots_phase2_Output_extend/traindata.npy', 'rb') as f:
    phase2_output_extend_losses = np.load(f)
    print(f'phase2_output_extend_losses.shape: {phase2_output_extend_losses.shape}')
    
with open(f'plots_phase1_Forgetting_extend/traindata.npy', 'rb') as f: 
    phase1_forget_losses = np.load(f)
    print(f'phase1_forget_losses.shape: {phase1_forget_losses.shape}')   

with open(f'plots_phase2_Forgetting_extend/traindata.npy', 'rb') as f:
    phase2_forget_losses = np.load(f)
    print(f'phase2_forget_losses.shape: {phase2_forget_losses.shape}')
    
with open(f'plots_phase1_PINN_extend/traindata.npy', 'rb') as f:
    phase1_replay_losses = np.load(f)
    print(f'phase1_replay_losses.shape: {phase1_replay_losses.shape}')

with open(f'plots_phase2_PINN_extend/traindata.npy', 'rb') as f:
    phase2_replay_losses = np.load(f)
    print(f'phase2_replay_losses.shape: {phase2_replay_losses.shape}')

    
plt.figure(figsize=(15, 6))
# Define colors for each method
colors = {
    'PINN': '#1f77b4',
    'DERL_new': "#fea954",
    'DERL_extend': '#ff7f0e',
    'SOB_new': '#98fb98',
    'SOB_extend': '#2ca02c',
    'OUTL_new': '#ff6666',
    'OUTL_extend': '#d62728',
    'Forgetting': '#9467bd',
    'Replay': '#8c564b'
}

def smooth(y, window=7):
    return uniform_filter1d(y, size=window, mode='nearest')

# PINN standalone
plt.plot(
    pinn_losses[:,0], 
    smooth(pinn_losses[:,-1]), 
    label='PINN full', 
    linewidth=2, 
    color=colors['PINN']
)

# DERL new: phase1 + phase2
derl_new_losses = np.concatenate([phase0_losses, phase1_derl_new_losses, phase2_derl_new_losses])
derl_new_losses[:,0] = np.arange(len(derl_new_losses))*100
plt.plot(
    derl_new_losses[:,0], 
    smooth(derl_new_losses[:,1]), 
    label='DERL from-scratch', 
    linewidth=2, 
    color=colors['DERL_new']
)

# DERL extend: phase1 + phase2
derl_extend_losses = np.concatenate([phase0_losses, phase1_derl_extend_losses, phase2_derl_extend_losses])
derl_extend_losses[:,0] = np.arange(len(derl_extend_losses))*100
plt.plot(
    derl_extend_losses[:,0], 
    smooth(derl_extend_losses[:,1]), 
    label='DERL continual', 
    linewidth=2, 
    color=colors['DERL_extend']
)

# SOB new: phase1 + phase2
sob_new_losses = np.concatenate([phase0_losses, phase1_sob_new_losses, phase2_sob_new_losses])
sob_new_losses[:,0] = np.arange(len(sob_new_losses))*100
plt.plot(
    sob_new_losses[:,0], 
    smooth(sob_new_losses[:,1]), 
    label='SOB from-scratch', 
    linewidth=2, 
    color=colors['SOB_new']
)

# SOB extend: phase1 + phase2
sob_extend_losses = np.concatenate([phase0_losses, phase1_sob_extend_losses, phase2_sob_extend_losses])
sob_extend_losses[:,0] = np.arange(len(sob_extend_losses))*100
plt.plot(
    sob_extend_losses[:,0], 
    smooth(sob_extend_losses[:,1]), 
    label='SOB continual', 
    linewidth=2, 
    color=colors['SOB_extend']
)

# Output new: phase1 + phase2
output_new_losses = np.concatenate([phase0_losses, phase1_output_new_losses, phase2_output_new_losses])
output_new_losses[:,0] = np.arange(len(output_new_losses))*100
plt.plot(
    output_new_losses[:,0], 
    smooth(output_new_losses[:,1]), 
    label='Output from-scratch', 
    linewidth=2, 
    color=colors['OUTL_new']
)

# Output extend: phase1 + phase2
output_extend_losses = np.concatenate([phase0_losses, phase1_output_extend_losses, phase2_output_extend_losses])
output_extend_losses[:,0] = np.arange(len(output_extend_losses))*100
plt.plot(
    output_extend_losses[:,0], 
    smooth(output_extend_losses[:,1]), 
    label='Output continual', 
    linewidth=2, 
    color=colors['OUTL_extend']
)

# Forgetting extend: phase1 + phase2
forget_losses = np.concatenate([phase0_losses, phase1_forget_losses, phase2_forget_losses])
forget_losses[:,0] = np.arange(len(forget_losses))*100
plt.plot(
    forget_losses[:,0], 
    smooth(forget_losses[:,1]), 
    label='PINN no dist.', 
    linewidth=2, 
    color=colors['Forgetting']
)

# Replay extend: phase1 + phase2
replay_losses = np.concatenate([phase0_losses, phase1_replay_losses, phase2_replay_losses])
replay_losses[:,0] = np.arange(len(replay_losses))*100
plt.plot(
    replay_losses[:,0], 
    smooth(replay_losses[:,1]), 
    label='PINN with replay', 
    linewidth=2, 
    color=colors['Replay']
)

# Add vertical lines at specified x values
vertical_lines = [30000, 60000]
for x in vertical_lines:
    plt.axvline(x=x, color='gray', linestyle='--', linewidth=1)

# Custom legend: color for method
method_handles = [
    Line2D([0], [0], color=colors['DERL_new'], lw=3, label='DERL from-scratch'),
    Line2D([0], [0], color=colors['DERL_extend'], lw=3, label='DERL continual'),
    Line2D([0], [0], color=colors['SOB_new'], lw=3, label='SOB from-scratch'),
    Line2D([0], [0], color=colors['SOB_extend'], lw=3, label='SOB continual'),
    Line2D([0], [0], color=colors['OUTL_new'], lw=3, label='OUTL from-scratch'),
    Line2D([0], [0], color=colors['OUTL_extend'], lw=3, label='OUTL continual'),
    Line2D([0], [0], color=colors['PINN'], lw=3, label='PINN full'),
    Line2D([0], [0], color=colors['Forgetting'], lw=3, label='PINN no dist.'),
    Line2D([0], [0], color=colors['Replay'], lw=3, label='PINN with replay'),
]
plt.legend(handles=method_handles, loc='upper center', ncol=4)
plt.yscale('log')
plt.ylim(2e-4,2)
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Prediction Error')
plt.tight_layout()
plt.savefig('plotsjoint/loss_curves.pdf')

plt.figure(figsize=(15, 6))
# Define colors for each method
colors = {
    'PINN': '#1f77b4',
    'DERL_extend': '#ff7f0e',
    'SOB_extend': '#2ca02c',
    'OUTL_extend': '#d62728',
    'Forgetting': '#9467bd',
    'Replay': '#8c564b'
}
def smooth(y, window=7):
    return uniform_filter1d(y, size=window, mode='nearest')

# PINN standalone
plt.plot(
    pinn_losses[:,0], 
    smooth(pinn_losses[:,-1]), 
    label='PINN full', 
    linewidth=2, 
    color=colors['PINN']
)

# DERL extend: phase1 + phase2
derl_extend_losses = np.concatenate([phase0_losses, phase1_derl_extend_losses, phase2_derl_extend_losses])
derl_extend_losses[:,0] = np.arange(len(derl_extend_losses))*100
plt.plot(
    derl_extend_losses[:,0], 
    smooth(derl_extend_losses[:,1]), 
    label='DERL continual', 
    linewidth=2, 
    color=colors['DERL_extend']
)

# SOB extend: phase1 + phase2
sob_extend_losses = np.concatenate([phase0_losses, phase1_sob_extend_losses, phase2_sob_extend_losses])
sob_extend_losses[:,0] = np.arange(len(sob_extend_losses))*100
plt.plot(
    sob_extend_losses[:,0], 
    smooth(sob_extend_losses[:,1]), 
    label='SOB continual', 
    linewidth=2, 
    color=colors['SOB_extend']
)

# Output extend: phase1 + phase2
output_extend_losses = np.concatenate([phase0_losses, phase1_output_extend_losses, phase2_output_extend_losses])
output_extend_losses[:,0] = np.arange(len(output_extend_losses))*100
plt.plot(
    output_extend_losses[:,0], 
    smooth(output_extend_losses[:,1]), 
    label='Output continual', 
    linewidth=2, 
    color=colors['OUTL_extend']
)

# Forgetting extend: phase1 + phase2
forget_losses = np.concatenate([phase0_losses, phase1_forget_losses, phase2_forget_losses])
forget_losses[:,0] = np.arange(len(forget_losses))*100
plt.plot(
    forget_losses[:,0], 
    smooth(forget_losses[:,1]), 
    label='PINN no dist.', 
    linewidth=2, 
    color=colors['Forgetting']
)

# Replay extend: phase1 + phase2
replay_losses = np.concatenate([phase0_losses, phase1_replay_losses, phase2_replay_losses])
replay_losses[:,0] = np.arange(len(replay_losses))*100
plt.plot(
    replay_losses[:,0], 
    smooth(replay_losses[:,1]), 
    label='PINN with replay', 
    linewidth=2, 
    color=colors['Replay']
) 

# Add vertical lines at specified x values
vertical_lines = [30000, 60000]
for x in vertical_lines:
    plt.axvline(x=x, color='gray', linestyle='--', linewidth=1)

# Custom legend: color for method
method_handles = [
    Line2D([0], [0], color=colors['DERL_extend'], lw=3, label='DERL continual'),
    Line2D([0], [0], color=colors['SOB_extend'], lw=3, label='SOB continual'),
    Line2D([0], [0], color=colors['OUTL_extend'], lw=3, label='OUTL continual'),
    Line2D([0], [0], color=colors['PINN'], lw=3, label='PINN full'),
    Line2D([0], [0], color=colors['Forgetting'], lw=3, label='PINN no dist.'),
    Line2D([0], [0], color=colors['Replay'], lw=3, label='PINN with replay'),
]
plt.legend(handles=method_handles, loc='upper right')
plt.yscale('log')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Prediction Error')
plt.tight_layout()
plt.savefig('plotsjoint/loss_curves_continual.pdf')
