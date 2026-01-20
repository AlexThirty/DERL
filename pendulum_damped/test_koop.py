import torch
import argparse
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import random
from torch import nn
from collections import OrderedDict
import os
seed = 30
from torch.func import vmap, jacrev, hessian
from itertools import cycle
from external_models.koop_net import PendulumNet
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)
from model import u_vec
font = {'size'   : 16}
import matplotlib
matplotlib.rc('font', **font)

parser = argparse.ArgumentParser()
parser.add_argument('--init_weight', default=1., type=float, help='Weight for the init loss')
parser.add_argument('--sys_weight', default=1., type=float, help='Weight for the rho loss')
parser.add_argument('--lr_init', default=5e-4, type=float, help='Starting learning rate')
parser.add_argument('--device', default='cuda:2', type=str, help='Device to use')
parser.add_argument('--name', default='koop', type=str, help='Experiment name')
parser.add_argument('--train_steps', default=10000, type=int, help='Number of training steps')
parser.add_argument('--epochs', default=500, type=int, help='Number of epochs')
parser.add_argument('--mode', default=0, type=int, help='Mode: -1 for PINN learning, 0 for derivative learning, 1 for output learning')
parser.add_argument('--batch_size', default=256, type=int, help='Number of samples per step')
parser.add_argument('--layers', default=4, type=int, help='Number of layers in the network')
parser.add_argument('--units', default=50, type=int, help='Number of units per layer in the network')

b = 0.5
EXP_PATH = '.'


args = parser.parse_args()
init_weight = args.init_weight
device = args.device
name = args.name
train_steps = args.train_steps
epochs = args.epochs
batch_size = args.batch_size
layers = args.layers
units = args.units
lr_init = args.lr_init
mode = args.mode
sys_weight = args.sys_weight
dt = 1e-2

if not os.path.exists(f'{EXP_PATH}/{name}'):
    os.mkdir(f'{EXP_PATH}/{name}')


prefix = 'true'
new_data = torch.load(os.path.join('data', f'true_dataset_test.pth'), weights_only=False)
print('Loading the data...')    

# Load the data
train_dataset = torch.load(os.path.join('data', f'{prefix}_dataset_train.pth'), weights_only=False)
test_dataset = torch.load(os.path.join('data', f'{prefix}_dataset_test.pth'), weights_only=False)
val_dataset = torch.load(os.path.join('data', f'{prefix}_dataset_val.pth'), weights_only=False)

train_dataloader = DataLoader(train_dataset, batch_size, generator=gen, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size, generator=gen, shuffle=True)

print('Data loaded!')

activation = torch.nn.Tanh()

model = PendulumNet(
    pred_weight=1.,
    ae_weight=1.,
    lin_weight=1.,
    hidden_units=[50,50],
    lr_init=lr_init,
    device=device,
).to(device)


torch.cuda.empty_cache()
model.eval()
# %%
import os
# %%
model.load_state_dict(torch.load(os.path.join('koop/saved_models', f'pendulum_netkoop')))

# %%
import numpy as np
from matplotlib import pyplot as plt
#plotting function to generate the figures for the ball problem

if not os.path.exists(f'{EXP_PATH}/{name}/koop'):
    os.mkdir(f'{EXP_PATH}/{name}/koop')
    
if not os.path.exists(f'{EXP_PATH}/{name}/koop_nopinn'):
    os.mkdir(f'{EXP_PATH}/{name}/koop_nopinn')

from plotting_utils import plotting_errors
models = [model]
models_no_pinn = [model]
model_names = ['KOOP']
model_names_no_pinn = ['KOOP']

#plotting_errors(models, b, model_names, name, f'{EXP_PATH}/{name}/koop')
#plotting_errors(models, b, model_names, name, f'{EXP_PATH}/{name}/koop', consistency=True)
#plotting_errors(models_no_pinn, b, model_names_no_pinn, name, f'{EXP_PATH}/{name}/koop_nopinn')
#plotting_errors(models_no_pinn, b, model_names_no_pinn, name, f'{EXP_PATH}/{name}/koop_nopinn', consistency=True)

N = 250
xlim = np.pi/2
ylim = 1.5
X,Y = np.meshgrid(np.linspace(-xlim,xlim,N),np.linspace(-ylim,ylim,N))
pts = np.vstack([X.reshape(-1),Y.reshape(-1)]).T


import matplotlib.patches as patches
n_train = 40
n_test = 10

new_data_xv0 = new_data[:][0].reshape((n_test,-1,3))[:,0,1:]
train_data_xv0 = train_dataset[:][0].reshape((30,-1,3))[:,0,1:]
val_data_xv0 = val_dataset[:][0].reshape((10,-1,3))[:,0,1:]
train_data_xv0 = torch.concat((train_data_xv0, val_data_xv0), dim=0).numpy()
test_data_xv0 = test_dataset[:][0].reshape((n_test,-1,3))[:,0,1:].numpy()


points = n_test
t_max = 10
steps = int(t_max/dt)+1
xv0 = new_data_xv0.numpy()
x0 = xv0[:,0]
v0 = xv0[:,1]

xv = torch.zeros((points,steps,2))
u_save = torch.zeros((points,steps,2))
xv[:,0,0] = torch.from_numpy(x0)
xv[:,0,1] = torch.from_numpy(v0)

m = 1. # old was 0.1
l = 10. # old was 1.
g = 9.81
# Function for the velocity field
def u(xv, b=b):
    return np.array([xv[1],-(g/l)*np.sin(xv[0])-(b/m)*xv[1]])

# Solve the new trajectories data
from scipy.integrate import solve_ivp
t_base = np.arange(start=0, stop=t_max+dt, step=dt)
for i in range(points):
    xv[i,:,:] = torch.from_numpy(solve_ivp(lambda t, y: u(y), [0,t_max], xv0[i], t_eval=t_base).y.T)
    
# Get the trajectoreis for trainina and testing
xv_train = torch.zeros((n_train,steps,2))
xv_train[:,0,0] = torch.from_numpy(train_data_xv0[:,0])
xv_train[:,0,1] = torch.from_numpy(train_data_xv0[:,1])

for i in range(n_train):
    xv_train[i,:,:] = torch.from_numpy(solve_ivp(lambda t, y: u(y), [0,t_max], train_data_xv0[i], t_eval=t_base).y.T)

xv_test = torch.zeros((n_test,steps,2))
xv_test[:,0,0] = torch.from_numpy(test_data_xv0[:,0])
xv_test[:,0,1] = torch.from_numpy(test_data_xv0[:,1])

for i in range(n_test):
    xv_test[i,:,:] = torch.from_numpy(solve_ivp(lambda t, y: u(y), [0,t_max], test_data_xv0[i], t_eval=t_base).y.T)

from typing import List
def time_loss_curve(model_list: List[PendulumNet], names, xv_sol, title, file_mode='a'):
    # Get the list of times at wich we evaluate the loss
    times = np.arange(0,10+0.1,0.1)
    loss_lists = {names[i]:[] for i in range(len(model_list))}
    xv0 = xv_sol[:,0].numpy()
    for i in range(len(model_list)):
        # Get the current model
        model = model_list[i]
        # Prepare the prediction array
        xv_pred = torch.zeros((xv_sol.shape))
        # For each starting condition
        for j in range(xv_sol.shape[0]):
            # Evaluate the model on the trajectory
            xv_pred[j] = model.evaluate_trajectory(xv_sol[j].to(device), time_steps=len(times)).detach().cpu()
        loss_val = np.abs(xv_pred[:,:,0] - xv_sol[:,:,0])
        for t in times:
            loss_lists[names[i]].append(loss_val[:,:int(t*100)].mean(axis=1).mean())
    with open(f'{EXP_PATH}/{name}/koop/error_curves.txt', file_mode) as f:
        print(f'{title} learning', file=f)
        for key in loss_lists.keys():
            print(f'{key} loss: {loss_lists[key]}', file=f)
        print('\n', file=f)
    plt.figure(figsize=(8,5))
    for key in loss_lists.keys():
        plt.plot(times, loss_lists[key], label=key)
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Time')
    plt.title('Cumulative error over time')
    
    plt.savefig(f'{EXP_PATH}/{name}/koop/{title}.pdf', format='pdf')
    plt.close()
        
#time_loss_curve(models, model_names, xv, 'cumulative_loss', file_mode='w')
#time_loss_curve(models_no_pinn, model_names_no_pinn, xv, 'cumulative_loss_nopinn', file_mode='a')
    
    

def model_plots(model:PendulumNet, title:str, xv:torch.Tensor, title_mode:str, file_mode:str = 'a'):
    
    # Number of points for the field
    N = 250
    X,Y = np.meshgrid(np.linspace(-xlim,xlim,N),np.linspace(-ylim,ylim,N))
    pts = np.vstack([X.reshape(-1),Y.reshape(-1)]).T


    t_base = np.arange(start=0, stop=t_max+dt, step=dt)

    xv_pred = torch.zeros((xv.shape[0], xv.shape[1]))
    xv_cons = []
    xv_init_cons = []
    xv_energy = []
    xv_energy_pred = []
    xv_ders = []
    xv_true_ders = []
    xv_pred_train =  torch.zeros((xv_train.shape))
    xv_train_cons = []
    
    xv_pred = torch.zeros((xv.shape[0], xv.shape[1]))
    xv_cons = []
    xv_init_cons = []
    xv_energy = []
    xv_energy_pred = []
    xv_ders = []
    xv_true_ders = []
    xv_pred_train =  torch.zeros((xv_train.shape))
    xv_train_cons = []
    
    #xv_iter = torch.zeros((xv.shape))
    for i in range(xv.shape[0]):
        trajectories = model.evaluate_trajectory(x0=xv[i,0,:], time_steps=steps).detach().cpu()
        xv_pred[i,:] = trajectories[:,0]
        xv_ders.append(trajectories[:,1].reshape((-1,1)))
        xv_true_ders.append(u_vec(xv[i,:,:], b=b))
    for i in range(xv_train.shape[0]):
        trajectories = model.evaluate_trajectory(x0=xv_train[i,0,:], time_steps=steps).detach().cpu()
        print(xv_pred_train.shape, trajectories.shape)
        xv_pred_train[i,:,:] = trajectories
    #xv_iter = model.evaluate_trajectory(x0=torch.from_numpy(xv0), time_steps=steps).detach().cpu()
    xv_train_cons = np.array(xv_train_cons)
    xv_ders = np.array(xv_ders)
    xv = xv.numpy()
    for i in range(10):
        plt.plot(xv[i,:,0], xv[i,:,1], color='blue')
        plt.plot(xv_pred[i,:], xv_ders[i,:,0], color='red')
        #plt.plot(xv_iter[i,:,0], xv_iter[i,:,1], color='green')
        #plt.legend()
    blue_patch = patches.Patch(color='blue', label='True trajectories')
    red_patch = patches.Patch(color='red', label='Predicted trajectories')
    #green_patch = patches.Patch(color='green', label='Iterative trajectories')
    plt.legend(handles=[blue_patch,red_patch])
    plt.xlabel(r'Angle: $\theta$')
    plt.ylabel(r'Angular speed: $\omega$')

    plt.title(f'{title_mode} phase trajectories')
    plt.savefig(f'{EXP_PATH}/{name}/plots{title}/pendulum_phase_trajectory.pdf', format='pdf')
        
    plt.close()

    plt.figure(figsize=(8,5))
    for i in range(10):
        plt.plot(t_base, xv[i,:,0], color='blue')
        plt.plot(t_base, xv_pred[i,:], color='red')
        #plt.plot(t_base, xv_iter[i,:,0], color='green')
    blue_patch = patches.Patch(color='blue', label='True trajectories')
    red_patch = patches.Patch(color='red', label='Predicted trajectories')
    #green_patch = patches.Patch(color='green', label='Iterative trajectories')
    plt.legend(handles=[blue_patch,red_patch])
    plt.xlabel(r'Time: $t$')
    plt.ylabel(r'Angle: $\theta$')
    plt.title(f'{title_mode} time trajectories')
    plt.savefig(f'{EXP_PATH}/{name}/plots{title}/pendulum_trajectory.pdf', format='pdf')
    plt.close()
    
    
    
    plt.figure(figsize=(8,5))
    for i in range(10):
        plt.plot(t_base, xv[i,:,1], color='blue')
        plt.plot(t_base, xv_ders[i,:,0], color='red')
        #plt.plot(t_base, xv_iter[i,:,1], color='green')
    blue_patch = patches.Patch(color='blue', label='True velocities')
    red_patch = patches.Patch(color='red', label='Predicted velocities')
    #green_patch = patches.Patch(color='green', label='Iterative velocities')
    plt.legend(handles=[blue_patch,red_patch])
    plt.xlabel(r'Time: $t$')
    plt.ylabel(r'Angular velocity: $\omega$')
    plt.title(f'{title_mode} time velocities')
    plt.savefig(f'{EXP_PATH}/{name}/plots{title}/pendulum_velocities.pdf', format='pdf')
    plt.close()
    
    
    ### Calculate the loss between the predicted and true trajectories
    loss = torch.nn.MSELoss()
    xv_train_true = xv_train.float()
    loss_val = np.linalg.norm(xv_train_true - xv_pred_train, axis=2).sum(axis=1)*dt
    loss_mean = np.sqrt(loss_val.mean())
    loss_std = loss_val.std()
    cons_mean = np.sqrt(xv_train_cons.mean())
    cons_std = xv_train_cons.std()
    with open(f'{EXP_PATH}/{name}/plotskoop/losses.txt', 'w') as f:
        #f.write(f'{loss_val}')
        print(f'Training set RMSE for {title} learning: {loss_mean} +- {loss_std}', file=f)
        print(f'Training set RMSE for {title} learning: {loss_mean} +- {loss_std}')
        print(f'Training set consistency for {title} learning: {cons_mean} +- {cons_std}', file=f)
        print(f'Training set consistency for {title} learning: {cons_mean} +- {cons_std}')
    
    
    ### Calculate the loss between the predicted and true trajectories
    xv_true = torch.from_numpy(xv).float()
    xv_pred = xv_pred
    loss_val = (np.abs(xv_true[:,:,0]-xv_pred)**2).sum(axis=1)*dt
    loss_max = np.sqrt(loss_val).max()
    loss_mean = np.sqrt(loss_val).mean()
    loss_std = np.sqrt(loss_val).std()
    der_loss = (np.linalg.norm(np.array(xv_ders)-np.array(xv_true_ders), axis=2)**2).sum(axis=1)*dt
    der_mean = np.sqrt(der_loss).mean()
    der_std = np.sqrt(der_loss).std()
    with open(f'{EXP_PATH}/{name}/plotskoop/losses.txt', 'a') as f:
        #f.write(f'{loss_val}')
        print(f'New trajectories loss for {title} learning: {loss_mean} +- {loss_std}', file=f)
        print(f'New trajectories loss for {title} learning: {loss_mean} +- {loss_std}')
        print(f'New trajectories max loss for {title} learning: {loss_max}', file=f)
        print(f'New trajectories max loss for {title} learning: {loss_max}')
        print(f'New trajectories derivative loss for {title} learning: {der_mean} +- {der_std}', file=f)
        print(f'New trajectories derivative loss for {title} learning: {der_mean} +- {der_std}')
model_plots(model, 'koop', xv, 'KOOP', file_mode='w')