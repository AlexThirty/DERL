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
from torch.optim import LBFGS

parser = argparse.ArgumentParser()
parser.add_argument('--bc_weight', default=1., type=float, help='Weight for the BC loss')
parser.add_argument('--sys_weight', default=1., type=float, help='Weight for the rho loss')
parser.add_argument('--pde_weight', default=1., type=float, help='Weight for the PDE loss')
parser.add_argument('--lr_init', default=5e-4, type=float, help='Starting learning rate')
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

prefix = name
title_mode = mode  


batch_size = 100

#batch_size = 32
print('Loading the data...')    


train_dataset = torch.load(os.path.join('train_data', 'train_data_multi_rand.pt'))
eval_dataset = torch.load(os.path.join('train_data', 'train_data_multi.pt'))
bc_dataset = torch.load(os.path.join('train_data', f'boundary_train_data.pt'))
#else:
#    bc_dataset = None
# Generate the dataloader
train_dataloader = DataLoader(train_dataset, batch_size, generator=gen, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, 10000, generator=gen, shuffle=True)
bc_dataloader = DataLoader(bc_dataset, batch_size, generator=gen, shuffle=True)

print('Data loaded!')
activation = torch.nn.Tanh()

model = AllenNet(
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

# %%
model.load_state_dict(torch.load(f'saved_models/pinn_phase1'))

# %%

if not os.path.exists(f'plots_phase1'):
    os.mkdir(f'plots_phase1')

import numpy as np
from matplotlib import pyplot as plt


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


loss_combination = np.load(f'plots_phase1/valdata.npy')
epoch_list = loss_combination[:,0]
out_losses_val = loss_combination[:,1]
der_losses_val = loss_combination[:,2]
pde_losses_val = loss_combination[:,3]
bc_losses_val = loss_combination[:,4]
tot_losses_val = loss_combination[:,5]
times_val = loss_combination[:,6]
hes_losses_val = loss_combination[:,7]

    
plt.figure()
plt.plot(epoch_list, pde_losses_val, label='pde_loss', color='red')
plt.plot(epoch_list, out_losses_val, label='out_loss', color='green')
plt.plot(epoch_list, der_losses_val, label='der_loss', color='blue')
plt.plot(epoch_list, bc_losses_val, label='bc_loss', color='purple')
plt.legend()
plt.yscale('log')
plt.title('Losses of the student model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(f'plots_phase1/val_losses.png')



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
torch.save(pde_distillation_dataset, os.path.join('val_data', f'train_distillation_dataset.pt'))
print('Created PDE distillation dataset')