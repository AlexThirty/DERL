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
parser.add_argument('--device', default='cuda:2', type=str, help='Device to use')
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

true_dataset = torch.load('val_data/val_data.pt')
boundary_dataset = torch.load('boundary_dataset.pt')
bc_loader = DataLoader(boundary_dataset, batch_size=batch_size, shuffle=True)
true_loader = DataLoader(true_dataset, batch_size=2048, shuffle=True)

print('Data loaded!')

activation = torch.nn.Tanh()

model = AllenNet(
    bc_weight=bc_weight,
    pde_weight=pde_weight,
    sys_weight=sys_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)

# %%
model.load_state_dict(torch.load(f'{name}/saved_models/allen_net{title_mode}'))

model.eval()

if not os.path.exists(f'exp/plots{mode}'):
    os.makedirs(f'exp/plots{mode}')
if not os.path.exists(f'exp/plots{mode}/val'):
    os.makedirs(f'exp/plots{mode}/val')
if not os.path.exists(f'exp/plots{mode}/train'):
    os.makedirs(f'exp/plots{mode}/train')

# load allen_pts
allen_pts = np.load('allen_pts.npy')
allen_x = np.load('allen_x.npy')
allen_y = np.load('allen_y.npy')
x, y = np.meshgrid(allen_x, allen_y)
allen_pts = np.column_stack((x.reshape(-1), y.reshape(-1)))

print(allen_pts.shape)
# Now load the valing lambdas
lam_train = np.load('allen_lam_train_exp.npy')
lam_val = np.load('allen_lam_val_exp.npy')

for lam in lam_val:
    print(f'Lambda: {lam}')
    x_in = np.column_stack((allen_pts, lam*np.ones(allen_pts.shape[0])))
    x_in = torch.tensor(x_in, dtype=torch.float32).to(device)
    u_pred = model(x_in).reshape(-1)
    u_pred = u_pred.cpu().detach().numpy()
    
    # Compute the true solution
    from generate_exp import allen_cahn_true_exp, allen_cahn_forcing_exp
    u_true = allen_cahn_true_exp(allen_pts, lam)
    f_true = allen_cahn_forcing_exp(allen_pts, lam)
    print('Calculating error')
    # Compute the error
    print(u_true.shape)
    print(u_pred.shape)
    error = np.linalg.norm(u_true - u_pred, 2)/np.linalg.norm(u_true, 2)
    print(f'Error for lambda={lam}: {error}')
    
    f_pred = model.evaluate_forcing(x_in)
    f_pred = f_pred.cpu().detach().numpy()
    
    error_f = np.linalg.norm(f_true - f_pred, 2)/np.linalg.norm(f_true, 2)
    print(f'Error for forcing lambda={lam}: {error_f}')
    
    import matplotlib.pyplot as plt
    # Plot the true and predicted solutions
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    cs1 = axs[0].contourf(x, y, u_true.reshape(x.shape), cmap='viridis', levels=50)
    fig.colorbar(cs1, ax=axs[0])
    axs[0].set_title(f'True Solution for lambda={lam}')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('u')
    axs[0].set_aspect('equal')

    cs2 = axs[1].contourf(x, y, u_pred.reshape(x.shape), cmap='viridis', levels=50)
    fig.colorbar(cs2, ax=axs[1])
    axs[1].set_title(f'Predicted Solution for lambda={lam}')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('u')
    axs[1].set_aspect('equal')

    cs3 = axs[2].contourf(x, y, np.abs(u_true - u_pred).reshape(x.shape), cmap='viridis', levels=50)
    fig.colorbar(cs3, ax=axs[2])
    axs[2].set_title(f'Error for lambda={lam}')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('Error')
    axs[2].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'exp/plots{mode}/val/solution_lambda_{lam}.pdf')
    plt.close(fig)

    # Plot the true and predicted forcing
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    cs1 = axs[0].contourf(x, y, f_true.reshape(x.shape), cmap='viridis', levels=50)
    fig.colorbar(cs1, ax=axs[0])
    axs[0].set_title(f'True Forcing for lambda={lam}')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('f')
    axs[0].set_aspect('equal')

    cs2 = axs[1].contourf(x, y, f_pred.reshape(x.shape), cmap='viridis', levels=50)
    fig.colorbar(cs2, ax=axs[1])
    axs[1].set_title(f'Predicted Forcing for lambda={lam}')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('f')
    axs[1].set_aspect('equal')

    cs3 = axs[2].contourf(x, y, np.abs(f_true - f_pred).reshape(x.shape), cmap='viridis', levels=50)
    fig.colorbar(cs3, ax=axs[2])
    axs[2].set_title(f'Error in Forcing for lambda={lam}')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('Error')
    axs[2].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'exp/plots{mode}/val/forcing_lambda_{lam}.pdf')
    plt.close(fig)
    
from generate_exp import allen_cahn_true_exp, allen_cahn_forcing_exp
for lam in lam_train:
    x_in = np.column_stack((allen_pts, lam*np.ones(allen_pts.shape[0])))
    x_in = torch.tensor(x_in, dtype=torch.float32).to(device)
    u_pred = model(x_in).reshape(-1)
    u_pred = u_pred.cpu().detach().numpy()
    
    # Compute the true solution
    u_true = allen_cahn_true_exp(allen_pts, lam)
    f_true = allen_cahn_forcing_exp(allen_pts, lam)
    
    # Compute the error
    error = np.linalg.norm(u_true - u_pred, 2)/np.linalg.norm(u_true, 2)
    print(f'Error for lambda={lam}: {error}')
    
    f_pred = model.evaluate_forcing(x_in)
    f_pred = f_pred.cpu().detach().numpy()
    
    error_f = np.linalg.norm(f_true - f_pred, 2)/np.linalg.norm(f_true, 2)
    print(f'Error for forcing lambda={lam}: {error_f}')
    
    import matplotlib.pyplot as plt

    # Plot the true and predicted solutions
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    cs1 = axs[0].contourf(x, y, u_true.reshape(x.shape), cmap='viridis', levels=50)
    fig.colorbar(cs1, ax=axs[0])
    axs[0].set_title(f'True Solution for lambda={lam}')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('u')
    axs[0].set_aspect('equal')

    cs2 = axs[1].contourf(x, y, u_pred.reshape(x.shape), cmap='viridis', levels=50)
    fig.colorbar(cs2, ax=axs[1])
    axs[1].set_title(f'Predicted Solution for lambda={lam}')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('u')
    axs[1].set_aspect('equal')

    cs3 = axs[2].contourf(x, y, np.abs(u_true - u_pred).reshape(x.shape), cmap='viridis', levels=50)
    fig.colorbar(cs3, ax=axs[2])
    axs[2].set_title(f'Error for lambda={lam}')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('Error')
    axs[2].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'exp/plots{mode}/train/solution_lambda_{lam}_train.pdf')
    plt.close(fig)
    
    # Plot the true and predicted forcing
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    cs1 = axs[0].contourf(x, y, f_true.reshape(x.shape), cmap='viridis', levels=50)
    fig.colorbar(cs1, ax=axs[0])
    axs[0].set_title(f'True Forcing for lambda={lam}')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('f')
    axs[0].set_aspect('equal')

    cs2 = axs[1].contourf(x, y, f_pred.reshape(x.shape), cmap='viridis', levels=50)
    fig.colorbar(cs2, ax=axs[1])
    axs[1].set_title(f'Predicted Forcing for lambda={lam}')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('f')
    axs[1].set_aspect('equal')

    cs3 = axs[2].contourf(x, y, np.abs(f_true - f_pred).reshape(x.shape), cmap='viridis', levels=50)
    fig.colorbar(cs3, ax=axs[2])
    axs[2].set_title(f'Error in Forcing for lambda={lam}')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('Error')
    axs[2].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'exp/plots{mode}/train/forcing_lambda_{lam}_train.pdf')
    plt.close(fig)
    
    
test_lam = [0.3, 1.5, 2.8]

for lam in test_lam:
    print(f'Lambda: {lam}')
    x_in = np.column_stack((allen_pts, lam*np.ones(allen_pts.shape[0])))
    x_in = torch.tensor(x_in, dtype=torch.float32).to(device)
    u_pred = model(x_in).reshape(-1)
    u_pred = u_pred.cpu().detach().numpy()
    
    # Compute the true solution
    u_true = allen_cahn_true_exp(allen_pts, lam)
    f_true = allen_cahn_forcing_exp(allen_pts, lam)
    print('Calculating error')
    # Compute the error
    print(u_true.shape)
    print(u_pred.shape)
    error = np.linalg.norm(u_true - u_pred, 2)/np.linalg.norm(u_true, 2)
    print(f'Error for lambda={lam}: {error}')
    
    f_pred = model.evaluate_forcing(x_in)
    f_pred = f_pred.cpu().detach().numpy()
    
    error_f = np.linalg.norm(f_true - f_pred, 2)/np.linalg.norm(f_true, 2)
    print(f'Error for forcing lambda={lam}: {error_f}')
    
    import matplotlib.pyplot as plt
    # Plot the true and predicted solutions
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    cs1 = axs[0].contourf(x, y, u_true.reshape(x.shape), cmap='viridis', levels=50)
    fig.colorbar(cs1, ax=axs[0])
    axs[0].set_title(f'True Solution for lambda={lam}')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('u')
    axs[0].set_aspect('equal')

    cs2 = axs[1].contourf(x, y, u_pred.reshape(x.shape), cmap='viridis', levels=50)
    fig.colorbar(cs2, ax=axs[1])
    axs[1].set_title(f'Predicted Solution for lambda={lam}')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('u')
    axs[1].set_aspect('equal')

    cs3 = axs[2].contourf(x, y, np.abs(u_true - u_pred).reshape(x.shape), cmap='viridis', levels=50)
    fig.colorbar(cs3, ax=axs[2])
    axs[2].set_title(f'Error for lambda={lam}')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('Error')
    axs[2].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'exp/plots{mode}/val/solution_lambda_{lam}_test.pdf')
    plt.close(fig)

    # Plot the true and predicted forcing
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    cs1 = axs[0].contourf(x, y, f_true.reshape(x.shape), cmap='viridis', levels=50)
    fig.colorbar(cs1, ax=axs[0])
    axs[0].set_title(f'True Forcing for lambda={lam}')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('f')
    axs[0].set_aspect('equal')

    cs2 = axs[1].contourf(x, y, f_pred.reshape(x.shape), cmap='viridis', levels=50)
    fig.colorbar(cs2, ax=axs[1])
    axs[1].set_title(f'Predicted Forcing for lambda={lam}')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('f')
    axs[1].set_aspect('equal')

    cs3 = axs[2].contourf(x, y, np.abs(f_true - f_pred).reshape(x.shape), cmap='viridis', levels=50)
    fig.colorbar(cs3, ax=axs[2])
    axs[2].set_title(f'Error in Forcing for lambda={lam}')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('Error')
    axs[2].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'exp/plots{mode}/val/forcing_lambda_{lam}_test.pdf')
    plt.close(fig)
