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
parser.add_argument('--bc_weight', default=10., type=float, help='Weight for the BC loss')
parser.add_argument('--sys_weight', default=1., type=float, help='Weight for the rho loss')
parser.add_argument('--pde_weight', default=10., type=float, help='Weight for the PDE loss')
parser.add_argument('--lr_init', default=1e-4, type=float, help='Starting learning rate')
parser.add_argument('--device', default='cuda', type=str, help='Device to use')
parser.add_argument('--name', default='grid', type=str, help='Experiment name')
parser.add_argument('--train_steps', default=100000, type=int, help='Number of training steps')
parser.add_argument('--epochs', default=200, type=int, help='Number of epochs')
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

prefix = name
title_mode = mode  


batch_size = 32

#batch_size = 32
print('Loading the data...')    


train_dataset = torch.load(os.path.join('data', 'dataset_full_rand_down.pth'))
eval_dataset = torch.load(os.path.join('data', 'dataset_down.pth'))
bc_dataset = torch.load(os.path.join('data', f'boundary_full_down.pth'))
#else:
#    bc_dataset = None
# Generate the dataloader
train_dataloader = DataLoader(train_dataset, batch_size, generator=gen, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, 2048, generator=gen, shuffle=True)
bc_dataloader = DataLoader(bc_dataset, batch_size, generator=gen, shuffle=True)

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
model.load_state_dict(torch.load(f'saved_models/pinn_phase1'))

# %%

if not os.path.exists(f'plots_phase1'):
    os.mkdir(f'plots_phase1')

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
u_grid = allen_cahn_true(torch.tensor(pts)).reshape((-1,1)).reshape(x_pts.shape)

u_pred = model.forward(torch.tensor(pts).to(device).float()).detach().cpu().numpy().reshape(x_pts.shape)

u_err = np.abs(u_grid - u_pred)

from matplotlib import cm



# Plot the predicted solution for y < 0
fig, ax = plt.subplots()
contour = ax.tricontourf(x_pts, y_pts, u_pred, cmap=cm.jet, levels=50)
fig.colorbar(contour, ax=ax, orientation='vertical')
plt.savefig(f'plots_phase1/pred_solution.png')
plt.close()

# Filter points where y < 0
mask = y_pts < 0
x_pts_filtered = x_pts[mask]
y_pts_filtered = y_pts[mask]
u_pred_filtered = u_pred[mask]
u_err_filtered = u_err[mask]

# Plot the predicted solution for y < 0
fig, ax = plt.subplots()
contour = ax.tricontourf(x_pts_filtered, y_pts_filtered, u_pred_filtered, cmap=cm.jet, levels=50)
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.colorbar(contour, ax=ax, orientation='vertical')
fig.suptitle('Predicted solution (y < 0)')
plt.savefig(f'plots_phase1/pred_solution_y_lt_0.png')
plt.close()

# Plot the error wrt the true solution for y < 0
fig, ax = plt.subplots()
contour = ax.tricontourf(x_pts_filtered, y_pts_filtered, u_err_filtered, cmap=cm.jet, levels=50)
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.colorbar(contour, ax=ax, orientation='vertical')
fig.suptitle('Error of the predicted solution (y < 0)')
plt.savefig(f'plots_phase1/error_y_lt_0.png')
plt.close()


loss_combination = np.load(f'plots_phase1/traindata.npy')

epoch_list = loss_combination[:,0]
out_losses_train = loss_combination[:,1]
der_losses_train = loss_combination[:,2]
pde_losses_train = loss_combination[:,3]
bc_losses_train = loss_combination[:,4]
tot_losses_train = loss_combination[:,5]
hes_losses_train = loss_combination[:,6]

N = 10
l = len(np.convolve(out_losses_train, np.ones(N)/N, mode='valid'))
plt.figure()
plt.plot(epoch_list[:l], np.convolve(pde_losses_train, np.ones(N)/N, mode='valid'), label='pde_loss', color='red')
plt.plot(epoch_list[:l], np.convolve(out_losses_train, np.ones(N)/N, mode='valid'), label='out_loss', color='green')
plt.plot(epoch_list[:l], np.convolve(der_losses_train, np.ones(N)/N, mode='valid'), label='der_loss', color='blue')
plt.plot(epoch_list[:l], np.convolve(bc_losses_train, np.ones(N)/N, mode='valid'), label='bc_loss', color='purple')
plt.legend()
plt.yscale('log')
plt.title('Losses of the student model')
plt.xlabel('Training steps')
plt.ylabel('Loss')
plt.savefig(f'plots_phase1/train_losses.png')
plt.close()


loss_combination = np.load(f'plots_phase1/testdata.npy')
epoch_list = loss_combination[:,0]
out_losses_test = loss_combination[:,1]
der_losses_test = loss_combination[:,2]
pde_losses_test = loss_combination[:,3]
bc_losses_test = loss_combination[:,4]
tot_losses_test = loss_combination[:,5]
times_test = loss_combination[:,6]
hes_losses_test = loss_combination[:,7]

    
plt.figure()
plt.plot(epoch_list, pde_losses_test, label='pde_loss', color='red')
plt.plot(epoch_list, out_losses_test, label='out_loss', color='green')
plt.plot(epoch_list, der_losses_test, label='der_loss', color='blue')
plt.plot(epoch_list, bc_losses_test, label='bc_loss', color='purple')
plt.legend()
plt.yscale('log')
plt.title('Losses of the student model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(f'plots_phase1/test_losses.png')



from torch.utils.data import TensorDataset
from torch.func import vmap, jacrev, hessian

# Empty cache just to be sure
torch.cuda.empty_cache()
# Create tensors for I/O
pde_x = []
pde_y = []
pde_Dy = []
pde_Hy = []

for train_data in train_dataloader:
    x = train_data[0]
    pde_x.append(x)
    pde_y.append(model.forward(x.to(device).float()).detach().cpu())
    pde_Dy.append(vmap(jacrev(model.forward_single))(x.to(device).float()).detach().cpu())
    pde_Hy.append(vmap(hessian(model.forward_single))(x.to(device).float()).detach().cpu())
    torch.cuda.empty_cache()

pde_x = torch.cat(pde_x)
pde_y = torch.cat(pde_y)
pde_Dy = torch.cat(pde_Dy)
pde_Hy = torch.cat(pde_Hy)

print(f'pde_x.shape: {pde_x.shape}')
print(f'pde_y.shape: {pde_y.shape}')
print(f'pde_Dy.shape: {pde_Dy.shape}')
print(f'pde_Hy.shape: {pde_Hy.shape}')
pde_distillation_dataset = TensorDataset(pde_x, pde_y, pde_Dy, pde_Hy)
torch.save(pde_distillation_dataset, os.path.join('data', f'train_distillation_dataset.pth'))
print('Created PDE distillation dataset')