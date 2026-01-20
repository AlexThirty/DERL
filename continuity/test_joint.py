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
font = {'size'   : 16}
import matplotlib
matplotlib.rc('font', **font)
from matplotlib.colors import Normalize

class MidPointNorm(Normalize):    
    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self,vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")       
        elif vmin == vmax:
            result.fill(0) # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            #First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint            
            resdat[resdat>0] /= abs(vmax - midpoint)            
            resdat[resdat<0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = ma.array(resdat, mask=result.mask, copy=False)                

        if is_scalar:
            result = result[0]            
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if cbook.iterable(value):
            val = ma.asarray(value)
            val = 2 * (val-0.5)  
            val[val>0]  *= abs(vmax - midpoint)
            val[val<0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (value - 0.5)
            if val < 0: 
                return  val*abs(vmin-midpoint) + midpoint
            else:
                return  val*abs(vmax-midpoint) + midpoint

class MidpointNormalize(Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))

from model import DensityNet, u_vec, u
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)
#from tuner_results import density_best_params

parser = argparse.ArgumentParser()
parser.add_argument('--init_weight', default=1., type=float, help='Weight for the init loss')
parser.add_argument('--sys_weight', default=1., type=float, help='Weight for the rho loss')
parser.add_argument('--bc_weight', default=1., type=float, help='Weight for the boundary loss')
parser.add_argument('--pde_weight', default=1., type=float, help='Weight for the boundary loss')

parser.add_argument('--lr_init', default=1e-4, type=float, help='Starting learning rate')
parser.add_argument('--device', default='cuda:2', type=str, help='Device to use')
parser.add_argument('--name', default='grid', type=str, help='Experiment name')
parser.add_argument('--train_steps', default=100000, type=int, help='Number of training steps')
parser.add_argument('--epochs', default=500, type=int, help='Number of epochs')
parser.add_argument('--mode', default=0, type=int, help='Mode: -1 for PINN learning, 0 for derivative learning, 1 for output learning')
parser.add_argument('--batch_size', default=64, type=int, help='Number of samples per step')
parser.add_argument('--layers', default=4, type=int, help='Number of layers in the network')
parser.add_argument('--units', default=50, type=int, help='Number of units per layer in the network')

args = parser.parse_args()
init_weight = args.init_weight
device = args.device
name = args.name
train_steps = args.train_steps
epochs = args.epochs
batch_size = args.batch_size
pde_weight = args.pde_weight
layers = args.layers
units = args.units
lr_init = args.lr_init
mode = args.mode
sys_weight = args.sys_weight
bc_weight = args.bc_weight

# Export path and the type of density
EXP_PATH = '.'
    
print(f'Working on {EXP_PATH}')

if not os.path.exists(f'{EXP_PATH}/{name}'):
    os.mkdir(f'{EXP_PATH}/{name}')

if name == 'full':
    prefix = 'rho_full'
elif name == 'grid':
    prefix = 'rho_grid'
elif name == 'extrapolate':
    prefix = 'rho_extrapolate'
elif name == 'interpolate':
    prefix = 'rho_interpolate'
elif name == 'adapt':
    prefix = 'rho_adapt'
else:
    raise ValueError(f'name value is not in the options')



#best_params = density_best_params[density_type][name][str(mode)]
#batch_size = best_params['batch_size']
#init_weight = best_params['init_weight']
#sys_weight = best_params['sys_weight']
#lr_init = best_params['lr_init']
    
print('Loading the data...')    

# Load the data
train_dataset = test_dataset = torch.load(os.path.join(EXP_PATH, f'data/{prefix}_dataset.pth'))

init_dataset = torch.load(os.path.join(EXP_PATH, f'data/rho_init_dataset.pth'))

bc_dataset = torch.load(os.path.join(EXP_PATH, f'data/rho_bc_dataset.pth'))
# Generate the dataloader
train_dataloader = DataLoader(train_dataset, batch_size, generator=gen, shuffle=True, num_workers=12)
test_dataloader = DataLoader(test_dataset, batch_size, generator=gen, shuffle=True, num_workers=12)
init_dataloader = DataLoader(init_dataset, int(batch_size/10), generator=gen, shuffle=True, num_workers=12)
print('Data loaded!')

print(train_dataset[:][0].shape)

activation = torch.nn.Tanh()


model_0 = DensityNet(
    init_weight=init_weight,
    sys_weight=sys_weight,
    pde_weight=pde_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    bc_weight=bc_weight,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)


model_1 = DensityNet(
    init_weight=init_weight,
    sys_weight=sys_weight,
    pde_weight=pde_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    bc_weight=bc_weight,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)


model_neg = DensityNet(
    init_weight=init_weight,
    sys_weight=sys_weight,
    pde_weight= pde_weight,
    bc_weight=bc_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)



model_sob = DensityNet(
    init_weight=init_weight,
    sys_weight=sys_weight,
    pde_weight= pde_weight,
    bc_weight=bc_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)

model_pinnout = DensityNet(
    init_weight=init_weight,
    sys_weight=sys_weight,
    pde_weight= pde_weight,
    bc_weight=bc_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)

torch.cuda.empty_cache()
model_1.eval()
model_0.eval()
model_neg.eval()
model_sob.eval()
model_pinnout.eval()
# %%
import os
# %%
model_0.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/density_netDerivative'))
model_1.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/density_netOutput'))
model_neg.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/density_netPINN'))
model_sob.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/density_netSobolev'))
model_pinnout.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/density_netPINN+Output'))

if not os.path.exists(f'{EXP_PATH}/{name}/plotsjoint'):
    os.mkdir(f'{EXP_PATH}/{name}/plotsjoint')

import numpy as np
from matplotlib import pyplot as plt
#plotting function to generate the figures for the ball problem

with open(f'{EXP_PATH}/data/rho.npy', 'rb') as f:
    rho_true = np.load(f)

from scipy.interpolate import RegularGridInterpolator
#print(rho_true)
#from dynsys_simulator import dt, dx, x_vec, y_vec, t_vec, t_max
dt = 0.001
dx = 0.01
t_max = 10.
x_max_interp = 1.5
x_min_interp = -1.5
x_steps = int((x_max_interp-x_min_interp)/dx)+1
x_vec = x_min_interp + np.arange(x_steps)*dx
y_vec = x_min_interp + np.arange(x_steps)*dx
t_vec = np.arange(0.,t_max+dt,dt)

x_min = -1.5
x_min_idx = int((x_min-x_min_interp)/dx)
x_max = 1.5
x_max_idx = int((x_max-x_min_interp)/dx)
t_max = 10.
rho_true_use = rho_true
#rho_interp = RegularGridInterpolator((t_vec, x_vec, y_vec), rho_true, method='cubic')
from matplotlib.colors import TwoSlopeNorm
print(rho_true_use.shape)
vmax = np.max(rho_true)
vmin = 0
levels = np.linspace(vmin,vmax,100)

# Limit for plotting
xlim = x_max +0.01
ylim = x_max +0.01
grid_size = dx**2*dt

from plotting_utils import plotting_errors




T,X,Y = np.meshgrid(t_vec,x_vec,y_vec, indexing='ij')
pts = np.vstack([T.reshape(-1),X.reshape(-1),Y.reshape(-1)]).T



batch_size_eval = 100000  # Define a batch size for evaluation

def batched_forward(model:DensityNet, pts, batch_size):
    outputs = []
    for i in range(0, len(pts), batch_size):
        print(f'Batch {i} out of {len(pts)}')
        batch_pts = pts[i:i+batch_size]
        batch_out = model.forward(torch.tensor(batch_pts).to(device).float()).detach().cpu().numpy()
        outputs.append(batch_out)
    return np.concatenate(outputs, axis=0)

def batched_consistency(model: DensityNet, pts, batch_size):
    consistencies = []
    for i in range(0, len(pts), batch_size):
        batch_pts = pts[i:i+batch_size]
        batch_consistency = model.evaluate_consistency(torch.tensor(batch_pts).to(device).float()).detach().cpu().numpy()
        consistencies.append(batch_consistency)
    return np.concatenate(consistencies, axis=0)

models = [model_0, model_1, model_pinnout, model_sob, model_neg]
models_no_pinn = [model_0, model_1, model_pinnout, model_sob]
model_names = ['DERL', 'OUTL', 'OUTL+PINN', 'SOB', 'PINN']
model_names_no_pinn = ['DERL', 'OUTL', 'OUTL+PINN', 'SOB']

outputs = []
errors = []
consistencies = []

for model in models:
    out = batched_forward(model, pts, batch_size_eval).reshape(X.shape)
    outputs.append(out)
    errors.append(np.abs(out - rho_true))
    consistencies.append(batched_consistency(model, pts, batch_size_eval).reshape(X.shape))

print('Output error averaged over the domain')

with open(f'{EXP_PATH}/{name}/plotsjoint/losses.txt', 'w') as f:
    print('Output error averaged over the domain', file=f)
    for i, model_name in enumerate(model_names):
        error = errors[i]
        consistency = consistencies[i]
        print(f'{model_name}: mean {np.mean(error)}, std {np.std(error)}', file=f)
        print(f'{model_name} RMSE: {np.sqrt(np.mean(error**2))}', file=f)
        print(f'{model_name} max loss {np.max(error)}', file=f)
        print(f'{model_name} L2 loss: {np.sqrt(grid_size*np.sum(error**2))}', file=f)
        print(f'{model_name} L2 normalized loss: {np.sqrt(np.sum(error**2)/np.sum(rho_true**2))}', file=f)
        
    print('Physical consistency error averaged over the domain', file=f)
    for i, model_name in enumerate(model_names):
        consistency = consistencies[i]
        print(f'{model_name}: mean {np.mean(np.nan_to_num(consistency))}, std {np.std(np.nan_to_num(consistency))}', file=f)
        print(f'{model_name} RMSE: {np.sqrt(np.mean(consistency**2))}', file=f)
        print(f'{model_name} max loss {np.max(consistency)}', file=f)
        print(f'{model_name} L2 loss: {np.sqrt(grid_size*np.sum(consistency**2))}', file=f)

print('Field error averaged over the domain')
for i, model_name in enumerate(model_names):
    error = errors[i]
    print(f'{model_name}: mean {np.mean(error)}, std {np.std(error)}')

print('Physical consistency error averaged over the domain')
for i, model_name in enumerate(model_names):
    consistency = consistencies[i]
    print(f'{model_name}: mean {np.mean(np.nan_to_num(consistency))}, std {np.std(np.nan_to_num(consistency))}')

model_masses = {name: [] for name in model_names}

if not os.path.exists(f'{EXP_PATH}/{name}/plotsjoint_nopinn'):
    os.mkdir(f'{EXP_PATH}/{name}/plotsjoint_nopinn')

for t in [0, 5, 9]:
    plotting_errors(models, t=t, rho_true_use=rho_true_use, model_names=model_names, path=f'{EXP_PATH}/{name}/plotsjoint/', apx=t)
    plotting_errors(models_no_pinn, t=t, rho_true_use=rho_true_use, model_names=model_names_no_pinn, path=f'{EXP_PATH}/{name}/plotsjoint_nopinn/', apx=t)

for t in np.arange(0, 10, 0.1):
    t_plot = t / 10. * t_max
    t_ind = int(t_plot / dt)

    X, Y = np.meshgrid(x_vec, y_vec)
    T = t_plot * np.ones_like(X.reshape((-1)))
    pts = np.vstack([T, X.reshape(-1), Y.reshape(-1)]).T

    for model, model_name in zip(models, model_names):
        rho_pred = model.forward(torch.tensor(pts).to(device).float()).detach().cpu().numpy().reshape(X.shape)
        mass = np.sum(rho_pred) * dx ** 2
        model_masses[model_name].append(mass)

tot_mass = np.sum(rho_true_use[0]) * dx ** 2

model_masses_err = {name: np.sqrt(np.mean((np.array(masses) - tot_mass) ** 2)) for name, masses in model_masses.items()}

with open(f'{EXP_PATH}/{name}/plotsjoint/losses.txt', 'a') as f:
    print('Mass conservation', file=f)
    for model_name, mass_err in model_masses_err.items():
        print(f'{model_name}: {mass_err}', file=f)

print('Mass conservation')
for model_name, mass_err in model_masses_err.items():
    print(f'{model_name}: {mass_err}')

plt.figure(figsize=(6, 5))
time_points = np.arange(0, 10, 0.1)
for model_name in model_names:
    plt.plot(time_points, model_masses[model_name], label=model_name)
plt.plot(time_points, np.ones_like(time_points) * tot_mass, label='True Mass')
plt.xlabel('Time')
plt.ylabel('Mass')
plt.legend()
plt.title('Mass Conservation')
plt.savefig(f'{EXP_PATH}/{name}/plotsjoint/mass_conservation.pdf', format='pdf')

vmin, vmax = (0, np.max(rho_true_use))
levels = np.linspace(vmin,vmax,100)


def model_plots(model:DensityNet, title_mode):
    
    
    
    
    with open(f'{EXP_PATH}/{name}/plots{title_mode}/testdata.npy', 'rb') as f:
        loss_combination_test = np.load(f)
    
    epoch_list = loss_combination_test[:,0]
    out_losses_test = loss_combination_test[:,1]
    der_losses_test = loss_combination_test[:,2]
    pde_losses_test = loss_combination_test[:,3]
    init_losses_test = loss_combination_test[:,4]
    tot_losses_test = loss_combination_test[:,5]
    bc_losses_test = loss_combination_test[:,6]
    time_test = loss_combination_test[:,7]
    # TODO add time
    
    
    
    with open(f'{EXP_PATH}/{name}/plotsjoint/losses.txt', 'a') as f:
        print(f'Losses for {title_mode}', file=f)
        print(f'Out_loss: {np.mean(np.sqrt(out_losses_test[-10:]))}', file=f)
        print(f'Der_loss: {np.mean(np.sqrt(der_losses_test[-10:]))}', file=f)
        print(f'PDE_loss: {np.mean(np.sqrt(pde_losses_test[-10:]))}', file=f)
        print(f'Init_loss: {np.mean(np.sqrt(init_losses_test[-10:]))}', file=f)
        print(f'BC_loss: {np.mean(np.sqrt(bc_losses_test[-10:]))}', file=f)
        print(f'Total_loss: {np.mean(np.sqrt(tot_losses_test[-10:]))}', file=f)
        print(f'Time: {np.mean(time_test[-10:])}', file=f)
            
            
    vmin, vmax = (0, np.max(rho_true_use))
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(13,4), layout='compressed')
    for i, t in enumerate([0,5,9]):
        # Calculate the grid for the density
        t_plot = t/10.*t_max
        print(f'Plotting time: {t_plot}')
        # Index of the time step
        t_ind = int(t_plot/dt)
        
        N = 250
        X,Y = np.meshgrid(np.linspace(x_min,x_max,N),np.linspace(x_min,x_max,N))
        T = t_plot*np.ones_like(X.reshape((-1)))
        pts = np.vstack([T,X.reshape(-1),Y.reshape(-1)]).T
        rho_plot = model.forward(torch.tensor(pts).to(device).float()).detach().cpu().numpy().reshape(X.shape)
        # Calculate the density on the grid
        #print(rho_plot)
        #plots the streamplot for the velocity field
        # Velocity field
        vel = u_vec(torch.from_numpy(pts))
        U = np.array(vel[:,0].reshape(X.shape))
        V = np.array(vel[:,1].reshape(Y.shape))
        # Streamplot of the velocity field
        ax[i].streamplot(X,Y,U,V,density=0.4,color='grey',linewidth=0.05)
        # Limits of the plot
        ax[i].set_xlim((x_min - 0.01,x_max + 0.01))
        ax[i].set_ylim((x_min - 0.01,x_max + 0.01))
        ax[i].set_title(f'Time: {t}')
        ax[i].set_aspect('equal')
        # Density plot
        contour = ax[i].contourf(X,Y,rho_plot,50,cmap='Greys', levels=levels, vmin=vmin, vmax=vmax)
    fig.colorbar(contour, ax=ax, orientation='vertical', fraction=0.02, pad=0.05)
    plt.savefig(f'{EXP_PATH}/{name}/plots{title_mode}/dynsys_traj_multi.pdf', format='pdf')
    
    for t in range(10):
        # Calculate the time
        t_plot = t/10.*t_max
        print(f'Plotting time: {t_plot}')
        # Index of the time step
        t_ind = int(t_plot/dt)
        #N = 500
        points_x = np.arange(x_min,x_max+dx,dx)
        points_y = np.arange(x_min,x_max+dx,dx)
        
        # Generate the points grid
        X,Y = np.meshgrid(points_x,points_y)
        T = t_plot*np.ones_like(X.reshape((-1)))
        pts = np.vstack([T,X.reshape(-1),Y.reshape(-1)]).T

        # Now obtain the true density 
        rho_true_plot = rho_true_use[t_ind].reshape(X.T.shape).T
        
        rho_plot = model.forward(torch.tensor(pts).to(device).float()).detach().cpu().numpy().reshape(X.shape)
        #plots the streamplot for the velocity field
        plt.figure(figsize=(6,5))

        pcm = plt.contourf(X,Y,rho_true_plot,50,cmap='Greys', vmin=vmin, vmax=vmax, levels=levels)
        #print(pts)
        vel = u_vec(torch.tensor(pts[:,1:])).numpy()
        #print(vel)
        U = np.array(vel[:,0].reshape(X.shape))
        V = np.array(vel[:,1].reshape(Y.shape))
        #mask the outside of the ball
        plt.streamplot(X,Y,U,V,density=0.4,linewidth=0.05,color='grey')
        plt.xlim((x_min-0.01,x_max + 0.01))
        plt.ylim((x_min-0.01,x_max + 0.01))
        plt.colorbar(pcm)
        #add outline for aesthetics

        plt.savefig(f'{EXP_PATH}/{name}/plots{title_mode}/density_traj{t}.pdf', format='pdf')
        plt.close()
        
        ### ERROR FIGURE
        plt.figure(figsize=(6,5))
        pcm = plt.contourf(X,Y,np.abs(rho_plot-rho_true_plot),50,cmap='jet')
        #print(pts)
        vel = u_vec(torch.tensor(pts[:,1:])).numpy()
        #print(vel)
        U = np.array(vel[:,0].reshape(X.shape))
        V = np.array(vel[:,1].reshape(Y.shape))
        #mask the outside of the ball
        plt.streamplot(X,Y,U,V,density=0.4,linewidth=0.05, color='black')
        plt.xlim((x_min-0.01,x_max + 0.01))
        plt.ylim((x_min-0.01,x_max + 0.01))
        #add outline for aesthetics
        plt.colorbar(pcm)
        plt.savefig(f'{EXP_PATH}/{name}/plots{title_mode}/error_traj{t}.pdf', format='pdf')
        plt.close()


    with open(f'{EXP_PATH}/{name}/plots{title_mode}/traindata.npy', 'rb') as f:
        loss_combination_train = np.load(f)
        
    epoch_list = loss_combination_train[:,0]
    out_losses_train = loss_combination_train[:,1]
    der_losses_train = loss_combination_train[:,2]
    pde_losses_train = loss_combination_train[:,3]
    init_losses_train = loss_combination_train[:,4]
    tot_losses_train = loss_combination_train[:,5]
    field_losses_train = loss_combination_train[:,6]

    N = 100
    l = len(np.convolve(out_losses_train, np.ones(N)/N, mode='valid'))
    plt.figure()
    plt.plot(epoch_list[:l], np.convolve(pde_losses_train, np.ones(N)/N, mode='valid'), label='pde_loss', color='red')
    plt.plot(epoch_list[:l], np.convolve(out_losses_train, np.ones(N)/N, mode='valid'), label='out_loss', color='green')
    plt.plot(epoch_list[:l], np.convolve(der_losses_train, np.ones(N)/N, mode='valid'), label='der_loss', color='blue')
    plt.plot(epoch_list[:l], np.convolve(init_losses_train, np.ones(N)/N, mode='valid'), label='init_loss', color='orange')
    #plt.plot(epoch_list[:l], np.convolve(field_losses_train, np.ones(N)/N, mode='valid'), label='field_loss', color='purple')
    plt.legend()
    plt.yscale('log')
    plt.title('Losses of the student model')
    plt.xlabel('Training steps')
    plt.ylabel('Loss')
    plt.savefig(f'{EXP_PATH}/{name}/plots{title_mode}/train_losses.pdf')
    plt.close()


    with open(f'{EXP_PATH}/{name}/plots{title_mode}/testdata.npy', 'rb') as f:
        loss_combination_test = np.load(f)
        
    epoch_list = loss_combination_test[:,0]
    out_losses_test = loss_combination_test[:,1]
    der_losses_test = loss_combination_test[:,2]
    pde_losses_test = loss_combination_test[:,3]
    init_losses_test = loss_combination_test[:,4]
    tot_losses_test = loss_combination_test[:,5]
    field_losses_test = loss_combination_test[:,6]
        
    plt.figure()
    plt.plot(epoch_list, pde_losses_test, label='pde_loss', color='red')
    plt.plot(epoch_list, out_losses_test, label='out_loss', color='green')
    plt.plot(epoch_list, der_losses_test, label='der_loss', color='blue')
    plt.plot(epoch_list, init_losses_test, label='init_loss', color='orange')
    #plt.plot(epoch_list, field_losses_test, label='field_loss', color='purple')
    plt.legend()
    plt.yscale('log')
    plt.title('Losses of the student model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(f'{EXP_PATH}/{name}/plots{title_mode}/test_losses.pdf')
    
    
model_plots(model_0, 'Derivative')
model_plots(model_1, 'Output')
model_plots(model_neg, 'PINN')
model_plots(model_sob, 'Sobolev')
model_plots(model_pinnout, 'PINN+Output')

N = 20

def plot_loss_curves(to_plot, step_list, names, path, title, colors):
    plt.figure()
    for i in range(len(to_plot)):
        plot_y = np.convolve(to_plot[i], np.ones(N)/N, mode='valid')
        plt.plot(step_list[:-(N-1)], plot_y, label=names[i], color = colors[i])
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

with open(f'{EXP_PATH}/{name}/plotsPINN+Output/testdata.npy', 'rb') as f:
    pinnout_losses = np.load(f)

step_list = derivative_losses[:,0]

plot_loss_curves([derivative_losses[:,1], output_losses[:,1], negdata_losses[:,1], sobolev_losses[:,1], pinnout_losses[:,1]], step_list, ['Derivative', 'Output', 'PINN', 'Sobolev', 'PINN+Output'], f'{EXP_PATH}/{name}/plotsjoint/out_losses.pdf', 'Out Losses', ['blue', 'red', 'green', 'purple', 'orange'])
plot_loss_curves([derivative_losses[:,2], output_losses[:,2], negdata_losses[:,2], sobolev_losses[:,2], pinnout_losses[:,2]], step_list, ['Derivative', 'Output', 'PINN', 'Sobolev', 'PINN+Output'], f'{EXP_PATH}/{name}/plotsjoint/der_losses.pdf', 'Der Losses', ['blue', 'red', 'green', 'purple', 'orange'])
plot_loss_curves([derivative_losses[:,3], output_losses[:,3], negdata_losses[:,3], sobolev_losses[:,3], pinnout_losses[:,3]], step_list, ['Derivative', 'Output', 'PINN', 'Sobolev', 'PINN+Output'], f'{EXP_PATH}/{name}/plotsjoint/pde_losses.pdf', 'PDE Losses', ['blue', 'red', 'green', 'purple', 'orange'])
plot_loss_curves([derivative_losses[:,4], output_losses[:,4], negdata_losses[:,4], sobolev_losses[:,4], pinnout_losses[:,4]], step_list, ['Derivative', 'Output', 'PINN', 'Sobolev', 'PINN+Output'], f'{EXP_PATH}/{name}/plotsjoint/init_losses.pdf', 'Init Losses', ['blue', 'red', 'green', 'purple', 'orange'])
plot_loss_curves([derivative_losses[:,6], output_losses[:,6], negdata_losses[:,6], sobolev_losses[:,6], pinnout_losses[:,6]], step_list, ['Derivative', 'Output', 'PINN', 'Sobolev', 'PINN+Output'], f'{EXP_PATH}/{name}/plotsjoint/bc_losses.pdf', 'BC Losses', ['blue', 'red', 'green', 'purple', 'orange'])