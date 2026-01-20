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
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)
#from tuner_results import allen_best_params

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

prefix = name
if name not in ['rand', 'grid']:
    raise ValueError('Name must be either rand or grid')
title_mode = mode

#best_params = continuity_best_params[name][str(mode)]
#init_weight = best_params['init_weight']
#if mode != 'PINN':
#    sys_weight = best_params['sys_weight']
#else:
#    sys_weight = 0.
#lr_init = best_params['lr_init']
#bc_weight = best_params['bc_weight']
#if mode == 'PINN'
#    pde_weight = best_params['pde_weight'] 
#else:
#    pde_weight = 0.  


batch_size = 32

#batch_size = 32
print('Loading the data...')    


train_dataset = test_dataset = torch.load(os.path.join('data', f'{prefix}_dataset.pth'))
bc_dataset = torch.load(os.path.join('data', f'bc_dataset.pth'))
#else:
#    bc_dataset = None
# Generate the dataloader
train_dataloader = DataLoader(train_dataset, batch_size, generator=gen, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, 256, generator=gen, shuffle=True, num_workers=4)
bc_dataloader = DataLoader(bc_dataset, batch_size, generator=gen, shuffle=True, num_workers=4)

print('Data loaded!')

print(train_dataset[:][0].shape)

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
# %%
model_0.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/allen_netDerivative'))
model_1.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/allen_netOutput'))
model_neg.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/allen_netPINN'))
model_sob.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/allen_netSobolev'))
model_hes.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/allen_netHessian'))
model_sobhes.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/allen_netSobolev+Hessian'))
model_0hes.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/allen_netDerivative+Hessian'))
model_pinnout.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/allen_netPINN+Output'))


torch.cuda.empty_cache()
model_1.eval()
model_0.eval()
model_neg.eval()
model_sob.eval()


if not os.path.exists(f'{EXP_PATH}/{name}/plotsjoint'):
    os.mkdir(f'{EXP_PATH}/{name}/plotsjoint')
    
if not os.path.exists(f'{EXP_PATH}/{name}/plotsjoint_hes'):
    os.mkdir(f'{EXP_PATH}/{name}/plotsjoint_hes')
    
    
from plotting_utils import plotting_errors

model_list_hes = [model_0, model_1, model_pinnout, model_sob, model_hes, model_sobhes, model_0hes, model_neg]
name_list_hes = ['DERL', 'OUTL', 'OUTL+PINN', 'SOB', 'HESL', 'DER+HESL', 'SOB+HES', 'PINN']
plotting_errors(model_list_hes, name_list_hes, f'{EXP_PATH}/{name}/plotsjoint_hes')

model_list = [model_0, model_1, model_pinnout, model_sob, model_neg]
name_list = ['DERL', 'OUTL', 'OUTL+PINN', 'SOB', 'PINN']
plotting_errors(model_list, name_list, f'{EXP_PATH}/{name}/plotsjoint')

if not os.path.exists(f'{EXP_PATH}/{name}/plotsjoint_nopinn'):
    os.mkdir(f'{EXP_PATH}/{name}/plotsjoint_nopinn')

model_list_nopinn = [model_0, model_1, model_pinnout, model_sob]
name_list_nopinn = ['DERL', 'OUTL', 'OUTL+PINN', 'SOB']
plotting_errors(model_list_nopinn, name_list_nopinn, f'{EXP_PATH}/{name}/plotsjoint_nopinn')

# Now plot very model

print('Plots saved!')
from matplotlib import pyplot as plt

def plot_model(model:AllenNet, title_mode:str, file_mode:str = 'a', ):
    
    with open(f'{EXP_PATH}/{name}/plots{title_mode}/testdata.npy', 'rb') as f:
        loss_combination_test = np.load(f)
    
    epoch_list = loss_combination_test[:,0]
    out_losses_test = loss_combination_test[:,1]
    der_losses_test = loss_combination_test[:,2]
    pde_losses_test = loss_combination_test[:,3]
    bc_losses_test = loss_combination_test[:,4]
    tot_losses_test = loss_combination_test[:,5]
    times = loss_combination_test[:,6]
    
    with open(f'{EXP_PATH}/{name}/plotsjoint/losses.txt', file_mode) as f:
        print(f'Losses for {title_mode}', file=f)
        print(f'Out_loss: {np.mean(np.sqrt(out_losses_test[-10:]))}', file=f)
        print(f'Der_loss: {np.mean(np.sqrt(der_losses_test[-10:]))}', file=f)
        print(f'PDE_loss: {np.mean(np.sqrt(pde_losses_test[-10:]))}', file=f)
        print(f'BC_loss: {np.mean(np.sqrt(bc_losses_test[-10:]))}', file=f)
        print(f'Total_loss: {np.mean(np.sqrt(tot_losses_test[-10:]))}', file=f)
        print(f'Time: {np.mean(times[-10:])}', file=f)
        
    plt.figure()
    plt.plot(epoch_list, out_losses_test, label='Out loss')
    plt.plot(epoch_list, der_losses_test, label='Der loss')
    plt.plot(epoch_list, pde_losses_test, label='PDE loss')
    plt.plot(epoch_list, bc_losses_test, label='BC loss')
    plt.plot(epoch_list, tot_losses_test, label='Total loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{EXP_PATH}/{name}/plots{title_mode}/test_losses.pdf')
    
    
    model.eval()
    x_plot = np.arange(xmin, xmax+dx, dx)
    y_plot = np.arange(xmin, xmax+dx, dx)
    X,Y = np.meshgrid(x_plot,y_plot)
    pts = np.vstack([X.reshape(-1),Y.reshape(-1)]).T
    u = model.forward(torch.tensor(pts, dtype=torch.float32).to(model.device)).detach().cpu().numpy().reshape(X.shape)
    pdv = vmap(jacrev(model.forward_single))(torch.tensor(pts, dtype=torch.float32).to(model.device)).detach().cpu().numpy()[:,0,:]
    pdv_err = np.linalg.norm(pdv - allen_cahn_pdv(pts), axis=1).reshape(X.shape)/np.linalg.norm(allen_cahn_pdv(pts), axis=1).reshape(X.shape)
    
    pde = model.evaluate_consistency(torch.tensor(pts, dtype=torch.float32).to(model.device)).detach().cpu().numpy().reshape(X.shape)
    forcing = model.evaluate_forcing(torch.tensor(pts, dtype=torch.float32).to(model.device)).detach().cpu().numpy().reshape(X.shape)
    forcing_err = np.abs(forcing - allen_cahn_forcing(pts).reshape(X.shape))
    
    plt.figure()
    plt.contourf(X,Y,u, levels=50, cmap='jet')
    plt.colorbar()
    plt.savefig(f'{EXP_PATH}/{name}/plots{title_mode}/pred_solution.pdf')
    
    error = np.abs(u - allen_cahn_true(pts).reshape(X.shape))
    
    plt.figure()
    plt.contourf(X,Y,error, levels=50, cmap='jet')
    plt.colorbar()
    plt.savefig(f'{EXP_PATH}/{name}/plots{title_mode}/error.pdf')
    plt.close()
    
    plt.figure()
    plt.contourf(X,Y,pdv_err, levels=50, cmap='jet')
    plt.colorbar()
    plt.savefig(f'{EXP_PATH}/{name}/plots{title_mode}/pdv_error.pdf')
    plt.close()
    
    plt.figure()
    plt.contourf(X,Y,pde, levels=50, cmap='jet')
    plt.colorbar()
    plt.savefig(f'{EXP_PATH}/{name}/plots{title_mode}/pde.pdf')
    plt.close()
    
    plt.figure()
    plt.contourf(X,Y,forcing, levels=50, cmap='jet')
    plt.colorbar()
    plt.savefig(f'{EXP_PATH}/{name}/plots{title_mode}/forcing.pdf')
    plt.close()
    
    plt.figure()
    plt.contourf(X,Y,forcing_err, levels=50, cmap='jet')
    plt.colorbar()
    plt.savefig(f'{EXP_PATH}/{name}/plots{title_mode}/forcing_error.pdf')
    plt.close()


plot_model(model_0, 'Derivative', file_mode='w')
print('Derivative done')
plot_model(model_1, 'Output')
print('Output done')
plot_model(model_neg, 'PINN')
print('PINN done')
plot_model(model_sob, 'Sobolev')
print('Sobolev done')
plot_model(model_hes, 'Hessian')
print('Hessian done')
plot_model(model_sobhes, 'Sobolev+Hessian')
print('Sobolev+Hessian done')
plot_model(model_0hes, 'Derivative+Hessian')
print('Derivative+Hessian done')
plot_model(model_pinnout, 'PINN+Output')


N = 5

from scipy import ndimage
def plot_loss_curves(to_plot, step_list, names, path, title, colors):
    plt.figure()
    for i in range(len(to_plot)):
        #y2 = np.convolve(to_plot[i], np.ones((N,))/N, mode='same')#[:-(N-1)]
        #plt.plot(step_list[:-(N-1)], plot_y, label=names[i], color=colors[i])
        y2 = ndimage.median_filter(to_plot[i], size=3)
        #y2 = to_plot[i]
        plt.plot(step_list, y2, label=names[i], color = colors[i])
    plt.legend()
    plt.yscale('log')
    plt.title(title)
    plt.savefig(path, format='pdf')
    plt.close()
    
with open(f'{EXP_PATH}/{name}/plotsDerivative/testdata.npy', 'rb') as f:
    derivative_losses = np.load(f)

with open(f'{EXP_PATH}/{name}/plotsOutput/testdata.npy', 'rb') as f:
    output_losses = np.load(f)

with open(f'{EXP_PATH}/{name}/plotsSobolev/testdata.npy', 'rb') as f:
    sobolev_losses = np.load(f)
    
with open(f'{EXP_PATH}/{name}/plotsPINN/testdata.npy', 'rb') as f:
    negdata_losses = np.load(f)
    
step_list = derivative_losses[:,0]

plot_loss_curves([derivative_losses[:,1], output_losses[:,1], sobolev_losses[:,1], negdata_losses[:,1]], step_list, ['Derivative', 'Output', 'Sobolev', 'PINN'], f'{EXP_PATH}/{name}/plotsjoint/out_losses.pdf', 'Out losses', ['blue', 'red', 'purple', 'green'])
plot_loss_curves([derivative_losses[:,2], output_losses[:,2], sobolev_losses[:,2], negdata_losses[:,2]], step_list, ['Derivative', 'Output', 'Sobolev', 'PINN'], f'{EXP_PATH}/{name}/plotsjoint/der_losses.pdf', 'Der losses', ['blue', 'red', 'purple', 'green'])
plot_loss_curves([derivative_losses[:,3], output_losses[:,3], sobolev_losses[:,3], negdata_losses[:,3]], step_list, ['Derivative', 'Output', 'Sobolev', 'PINN'], f'{EXP_PATH}/{name}/plotsjoint/pde_losses.pdf', 'PDE losses', ['blue', 'red', 'purple', 'green'])
plot_loss_curves([derivative_losses[:,4], output_losses[:,4], sobolev_losses[:,4], negdata_losses[:,4]], step_list, ['Derivative', 'Output', 'Sobolev', 'PINN'], f'{EXP_PATH}/{name}/plotsjoint/bc_losses.pdf', 'BC losses', ['blue', 'red', 'purple', 'green'])