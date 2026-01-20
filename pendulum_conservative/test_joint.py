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
from model import PendulumNet
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
parser.add_argument('--name', default='true', type=str, help='Experiment name')
parser.add_argument('--train_steps', default=10000, type=int, help='Number of training steps')
parser.add_argument('--epochs', default=500, type=int, help='Number of epochs')
parser.add_argument('--mode', default=0, type=int, help='Mode: -1 for PINN learning, 0 for derivative learning, 1 for output learning')
parser.add_argument('--batch_size', default=256, type=int, help='Number of samples per step')
parser.add_argument('--layers', default=4, type=int, help='Number of layers in the network')
parser.add_argument('--units', default=50, type=int, help='Number of units per layer in the network')

b = 0
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

new_data = None
if name == 'true':
    prefix = 'true'
    new_data = torch.load(os.path.join('data', f'true_dataset_test.pth'))
elif name == 'extrapolate':
    prefix = 'true_extrapolate'
    new_data = torch.load(os.path.join('data', f'true_extrapolate_dataset_new.pth'))
elif name == 'interpolate':
    prefix = 'true_interpolate'
    new_data = torch.load(os.path.join('data', f'true_interpolate_dataset_new.pth'))
elif name == 'adapt':
    prefix = 'true_adapt'
    new_data = torch.load(os.path.join('data', f'true_adapt_dataset_new.pth'))
elif name == 'emp':
    prefix = 'emp'
    new_data = torch.load(os.path.join('data', f'emp_dataset_test.pth'))
else:
    raise ValueError(f'name value is not in the options')



if mode == 0:
    title_mode = 'Derivative'
elif mode == 1:
    title_mode = 'Output'
elif mode == -1:
    title_mode = 'PINN'
else:
    raise ValueError('Mode is not valid')
    
print('Loading the data...')    

# Load the data
train_dataset = torch.load(os.path.join('data', f'{prefix}_dataset_train.pth'))
test_dataset = torch.load(os.path.join('data', f'{prefix}_dataset_test.pth'))
val_dataset = torch.load(os.path.join('data', f'{prefix}_dataset_val.pth'))

if name in ['adapt', 'interpolate']:
    bc_dataset = torch.load(os.path.join('data', f'{prefix}_bc_train.pth'))
else:
    bc_dataset = None
# Generate the dataloaders
#train_dataset = ConcatDataset([train_dataset, val_dataset])

train_dataloader = DataLoader(train_dataset, batch_size, generator=gen, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size, generator=gen, shuffle=True)

print('Data loaded!')

activation = torch.nn.Tanh()

model_0 = PendulumNet(
    init_weight=init_weight,
    sys_weight=sys_weight,
    pde_weight=0.,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
    b=b,
).to(device)


model_1 = PendulumNet(
    init_weight=init_weight,
    sys_weight=sys_weight,
    pde_weight=0.,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
    b=b,
).to(device)


model_neg = PendulumNet(
    init_weight=init_weight,
    sys_weight=sys_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    pde_weight=0.,
    device=device,
    activation=activation,    
    last_activation=False,
    b=b,
).to(device)

model_sob = PendulumNet(
    init_weight=init_weight,
    sys_weight=sys_weight,
    pde_weight=0.,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
    b=b,
).to(device)

model_pinnout = PendulumNet(
    init_weight=init_weight,
    sys_weight=sys_weight,
    pde_weight=0.,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
    b=b,
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
model_0.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/pendulum_netDerivative'))
model_1.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/pendulum_netOutput'))
model_neg.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/pendulum_netPINN'))
model_sob.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/pendulum_netSobolev'))
model_pinnout.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/pendulum_netPINN+Output'))

# %%
import numpy as np
from matplotlib import pyplot as plt
#plotting function to generate the figures for the ball problem

if not os.path.exists(f'{EXP_PATH}/{name}/plotsjoint'):
    os.mkdir(f'{EXP_PATH}/{name}/plotsjoint')
    
if not os.path.exists(f'{EXP_PATH}/{name}/plotsjoint_nopinn'):
    os.mkdir(f'{EXP_PATH}/{name}/plotsjoint_nopinn')

from plotting_utils import plotting_errors
models = [model_0, model_1, model_pinnout, model_sob, model_neg]
models_no_pinn = [model_0, model_1, model_pinnout, model_sob]
model_names = ['DERL', 'OUTL', 'OUTL+PINN', 'SOB', 'PINN']
model_names_no_pinn = ['DERL', 'OUTL', 'OUTL+PINN', 'SOB']

plotting_errors(models, b, model_names, name, f'{EXP_PATH}/{name}/plotsjoint')
plotting_errors(models, b, model_names, name, f'{EXP_PATH}/{name}/plotsjoint', consistency=True)
plotting_errors(models_no_pinn, b, model_names_no_pinn, name, f'{EXP_PATH}/{name}/plotsjoint_nopinn')
plotting_errors(models_no_pinn, b, model_names_no_pinn, name, f'{EXP_PATH}/{name}/plotsjoint_nopinn', consistency=True)

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
    
def time_loss_curve(model_list, names, xv_sol, title, file_mode='a'):
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
            in_pts = torch.column_stack((torch.from_numpy(t_base).reshape((-1,1)), torch.tile(torch.from_numpy(xv0[j]), (steps,1)))).float()
            xv_pred[j] = model.forward(in_pts.to(device)).detach().cpu()
        loss_val = np.abs(xv_pred[:,:,0] - xv_sol[:,:,0])
        for t in times:
            loss_lists[names[i]].append(loss_val[:,:int(t*100)].mean(axis=1).mean())
    with open(f'{EXP_PATH}/{name}/plotsjoint/error_curves.txt', file_mode) as f:
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
    
    plt.savefig(f'{EXP_PATH}/{name}/plotsjoint/{title}.pdf', format='pdf')
    plt.close()
    
def time_loss_curve_derivatives(model_list, names, xv_sol, title, file_mode='a'):
    # Get the list of times at which we evaluate the loss
    times = np.arange(0, 10 + 0.1, 0.1)
    loss_lists = {names[i]: [] for i in range(len(model_list))}
    xv0 = xv_sol[:, 0].numpy()
    for i in range(len(model_list)):
        # Get the current model
        model = model_list[i]
        # Prepare the prediction array
        xv_pred = torch.zeros((xv_sol.shape))
        xv_ders = torch.zeros((xv_sol.shape))
        xv_true_ders = torch.zeros((xv_sol.shape))
        # For each starting condition
        for j in range(xv.shape[0]):
            # Evaluate the model on the trajectory
            in_pts = torch.column_stack((torch.from_numpy(t_base).reshape((-1, 1)), torch.tile(torch.from_numpy(xv0[j]), (steps, 1)))).float()
            xv_pred[j] = model.forward(in_pts.to(device)).detach().cpu()
            # Calculate the derivatives
            xv_ders[j] = model.evaluate_field(in_pts.to(device)).detach().cpu()
            xv_true_ders[j] = u_vec(xv_sol[j], b=b)
            loss_val = np.linalg.norm(xv_ders - xv_true_ders, axis=2)
        for t in times:
            loss_lists[names[i]].append(loss_val[:, :int(t*100)].mean(axis=1).mean())
    with open(f'{EXP_PATH}/{name}/plotsjoint/error_curves_derivatives.txt', file_mode) as f:
        print(f'{title} learning', file=f)
        for key in loss_lists.keys():
            print(f'{key} loss: {loss_lists[key]}', file=f)
        print('\n', file=f)
    plt.figure(figsize=(8, 5))
    for key in loss_lists.keys():
        plt.plot(times, loss_lists[key], label=key)
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Time')
    plt.title('Cumulative derivative error over time')
    
    plt.savefig(f'{EXP_PATH}/{name}/plotsjoint/{title}_derivatives.pdf', format='pdf')
    plt.close()

time_loss_curve_derivatives(models, model_names, xv, 'cumulative_loss_derivatives', file_mode='w')
time_loss_curve_derivatives(models_no_pinn, model_names_no_pinn, xv, 'cumulative_loss_nopinn_derivatives', file_mode='a')
    
time_loss_curve(models, model_names, xv, 'cumulative_loss', file_mode='w')
time_loss_curve(models_no_pinn, model_names_no_pinn, xv, 'cumulative_loss_nopinn', file_mode='a')
    
    

def model_plots(model:PendulumNet, title:str, xv:torch.Tensor, title_mode:str, file_mode:str = 'a'):
    
    # Number of points for the field
    N = 250
    X,Y = np.meshgrid(np.linspace(-xlim,xlim,N),np.linspace(-ylim,ylim,N))
    pts = np.vstack([X.reshape(-1),Y.reshape(-1)]).T

    #plots the streamplot for the velocity field
    plt.figure(figsize=(5,5))
    #print(pts)
    
    #print(vel)
    vel = model.evaluate_field(torch.column_stack((0*torch.ones((pts.shape[0],1)),torch.from_numpy(pts).float())).to(device)).detach().cpu().numpy()
    U = np.array(vel[:,0].reshape(X.shape))
    V = np.array(vel[:,1].reshape(Y.shape))
    #mask the outside of the ball

    plt.streamplot(X,Y,U,V,density=0.5,color='black', linewidth=0.05)

    plt.xlim((-xlim,xlim))
    plt.ylim((-ylim,ylim))
    #add outline for aesthetics
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
    
    #xv_iter = torch.zeros((xv.shape))
    for i in range(xv.shape[0]):
        in_pts = torch.column_stack((torch.from_numpy(t_base).reshape((-1,1)), torch.tile(torch.from_numpy(xv0[i]), (steps,1)))).float()
        xv_pred[i,:] = model.forward(in_pts.to(device)).detach().cpu().reshape((-1))
        der = model.evaluate_field(in_pts.to(device)).detach().cpu().numpy()
        xv_ders.append(model.evaluate_field(in_pts.to(device)).detach().cpu().numpy())
        xv_true_ders.append(u_vec(xv[i,:,:], b=b))
        xv_cons.append(model.evaluate_consistency(in_pts.to(device)).detach().cpu().numpy())
        xv_energy.append(0.5*m*l**2*xv[i,:,1]**2 + m*g*l*(1-np.cos(xv[i,:,0])))
        xv_energy_pred.append(0.5*m*l**2*xv_pred[i,:]**2 + m*g*l*(1-np.cos(der[i,0])))
        xv_init_cons.append(model.evaluate_init_consistency(in_pts.to(device)).detach().cpu().numpy())
    for i in range(xv_train.shape[0]):
        in_pts = torch.column_stack((torch.from_numpy(t_base).reshape((-1,1)), torch.tile(torch.from_numpy(train_data_xv0[i]), (steps,1)))).float()
        xv_pred_train[i,:,:] = model.forward(in_pts.to(device)).detach().cpu()
        xv_train_cons.append(model.evaluate_consistency(in_pts.to(device)).detach().cpu().numpy())
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
        plt.plot(t_base, xv_energy[i], color='blue')
        plt.plot(t_base, xv_energy_pred[i], color='red')
    blue_patch = patches.Patch(color='blue', label='True energy')
    red_patch = patches.Patch(color='red', label='Predicted energy')
    plt.legend(handles=[blue_patch,red_patch])
    plt.xlabel(r'Time: $t$')
    plt.ylabel(r'Energy')
    plt.yscale('log')
    plt.title(f'{title_mode} time energy')
    plt.savefig(f'{EXP_PATH}/{name}/plots{title}/pendulum_energy.pdf', format='pdf')
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

    #plots the streamplot for the velocity field
    plt.figure(figsize=(5,5))
    #print(pts)
    vel = model.evaluate_field(torch.column_stack((0*torch.ones((pts.shape[0],1)),torch.from_numpy(pts).float())).to(device)).detach().cpu().numpy()
    #print(vel)
    U = np.array(vel[:,0].reshape(X.shape))
    V = np.array(vel[:,1].reshape(Y.shape))
    #mask the outside of the ball



    plt.streamplot(X,Y,U,V,density=0.5,color='black', linewidth=0.05)
    for i in range(10):
        plt.plot(xv_pred[i,:], xv_ders[i,:,0], label=f'trajectory{i}', color='red')
    plt.xlim((-xlim,xlim))
    plt.ylim((-ylim,ylim))

    plt.xlabel(r'Angle: $\theta$')
    plt.ylabel(r'Angular speed: $\omega$')

    plt.title(f'{title_mode} predicted field')
    plt.savefig(f'{EXP_PATH}/{name}/plots{title}/predicted_field.pdf', format='pdf')
    plt.close()
    
    plt.figure(figsize=(6,5))
    vel_true = u_vec(torch.from_numpy(pts), b=b)
    #print(vel)
    U_true = np.array(vel_true[:,0].reshape(X.shape))
    V_true = np.array(vel_true[:,1].reshape(Y.shape))
    plt.contourf(X,Y,np.sqrt((U-U_true)**2+(V-V_true)**2),100,cmap='jet')
    plt.title('Error in predicted fields')
    plt.colorbar()
    plt.xlim((-xlim,xlim))
    plt.ylim((-ylim,ylim))
    plt.xlabel(r'Angle: $\theta$')
    plt.ylabel(r'Angular speed: $\omega$')

    plt.title(f'{title_mode} field error')
    plt.savefig(f'{EXP_PATH}/{name}/plots{title}/error_field.pdf', format='pdf')
    plt.close()
    
    # Plot the consistency of the PDE
    plt.figure(figsize=(6,5))
    vel_true = u_vec(torch.from_numpy(pts), b=b)
    #print(vel)
    U_true = np.array(vel_true[:,0].reshape(X.shape))
    V_true = np.array(vel_true[:,1].reshape(Y.shape))
    consistencty = model.evaluate_consistency(torch.column_stack((0*torch.ones((pts.shape[0],1)),torch.from_numpy(pts).float())).to(device)).detach().cpu().numpy().reshape(X.shape)
    plt.contourf(X,Y,consistencty,100,cmap='jet')
    plt.colorbar()
    plt.xlim((-xlim,xlim))
    plt.ylim((-ylim,ylim))
    plt.xlabel(r'Angle: $\theta$')
    plt.ylabel(r'Angular speed: $\omega$')
    
    plt.title(f'{title_mode} PDE consistency')
    plt.savefig(f'{EXP_PATH}/{name}/plots{title}/pde_consistency.pdf', format='pdf')
    #print(vel)    
    
    with open(f'{EXP_PATH}/{name}/plots{title}/testdata.npy', 'rb') as f:
        loss_combination = np.load(f)
    
    epoch_list = loss_combination[:,0]
    out_losses = np.sqrt(loss_combination[:,1])
    der_losses = np.sqrt(loss_combination[:,2])
    pde_losses = np.sqrt(loss_combination[:,3])
    init_losses = np.sqrt(loss_combination[:,4])
    tot_losses = np.sqrt(loss_combination[:,5])
    init_pde_losses = np.sqrt(loss_combination[:,6])
    if loss_combination.shape[1] > 7:
        times = loss_combination[:,7]
    else:
        times = np.zeros_like(epoch_list)
    
    with open(f'{EXP_PATH}/{name}/plotsjoint/losses.txt', file_mode) as f:
        print(f'Model {title} learning', file=f)
        print(f'Out losses: {np.sqrt(out_losses[-1])}', file=f)
        print(f'Der losses: {np.sqrt(der_losses[-1])}', file=f)
        print(f'PDE losses: {np.sqrt(pde_losses[-1])}', file=f)
        print(f'Init losses: {np.sqrt(init_losses[-1])}', file=f)
        print(f'Total losses: {np.sqrt(tot_losses[-1])}', file=f)
        print(f'Init PDE losses: {np.sqrt(init_pde_losses[-1])}', file=f)
        print(f'Times: {times[-1]}', file=f)
        print('\n', file=f)

    
    ### Calculate the loss between the predicted and true trajectories
    loss = torch.nn.MSELoss()
    xv_train_true = xv_train.float()
    loss_val = np.linalg.norm(xv_train_true - xv_pred_train, axis=2).sum(axis=1)*dt
    loss_mean = np.sqrt(loss_val.mean())
    loss_std = loss_val.std()
    cons_mean = np.sqrt(xv_train_cons.mean())
    cons_std = xv_train_cons.std()
    with open(f'{EXP_PATH}/{name}/plotsjoint/losses.txt', 'a') as f:
        #f.write(f'{loss_val}')
        print(f'Training set RMSE for {title} learning: {loss_mean} +- {loss_std}', file=f)
        print(f'Training set RMSE for {title} learning: {loss_mean} +- {loss_std}')
        print(f'Training set consistency for {title} learning: {cons_mean} +- {cons_std}', file=f)
        print(f'Training set consistency for {title} learning: {cons_mean} +- {cons_std}')
    
    
    ### Calculate the loss between the predicted and true trajectories
    xv_true = torch.from_numpy(xv).float()
    xv_pred = xv_pred
    loss_val = (np.abs(xv_true[:,:,0]-xv_pred)**2).sum(axis=1)*dt
    loss_mean = np.sqrt(loss_val).mean()
    loss_std = np.sqrt(loss_val).std()
    cons_val = (np.array(xv_cons)**2).sum(axis=1)*dt
    cons_mean = np.sqrt(cons_val).mean()
    cons_std = np.sqrt(cons_val).std()
    init_cons_val = (np.array(xv_init_cons)**2).sum(axis=1)*dt
    init_cons_mean = np.sqrt(init_cons_val).mean()
    init_cons_std = np.sqrt(init_cons_val).std()
    der_loss = (np.linalg.norm(np.array(xv_ders)-np.array(xv_true_ders), axis=2)**2).sum(axis=1)*dt
    der_mean = np.sqrt(der_loss).mean()
    der_std = np.sqrt(der_loss).std()
    with open(f'{EXP_PATH}/{name}/plotsjoint/losses.txt', 'a') as f:
        #f.write(f'{loss_val}')
        print(f'New trajectories loss for {title} learning: {loss_mean} +- {loss_std}', file=f)
        print(f'New trajectories loss for {title} learning: {loss_mean} +- {loss_std}')
        print(f'New trajectories derivative loss for {title} learning: {der_mean} +- {der_std}', file=f)
        print(f'New trajectories derivative loss for {title} learning: {der_mean} +- {der_std}')
        print(f'New trajectories consistency for {title} learning: {cons_mean} +- {cons_std}', file=f)
        print(f'New trajectories consistency for {title} learning: {cons_mean} +- {cons_std}')
        print(f'New trajectories init consistency for {title} learning: {init_cons_mean} +- {init_cons_std}', file=f)
        print(f'New trajectories init consistency for {title} learning: {init_cons_mean} +- {init_cons_std}')
        
N = 20
        
def plot_loss_curves(to_plot, step_list, names, path, title, colors):
    plt.figure()
    for i in range(len(to_plot)):
        plot_y = np.convolve(to_plot[i], np.ones((N,))/N, mode='valid')
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
plot_loss_curves([derivative_losses[:,1], output_losses[:,1], sobolev_losses[:,1], negdata_losses[:,1], pinnout_losses[:,1]], step_list, ['Derivative', 'Output', 'Sobolev', 'PINN', 'PINN+Output'], f'{EXP_PATH}/{name}/plotsjoint/out_losses.pdf', 'Trajectory losses', colors=['blue', 'red', 'purple', 'green', 'orange'])
plot_loss_curves([derivative_losses[:,2], output_losses[:,2], sobolev_losses[:,2], negdata_losses[:,2], pinnout_losses[:,2]], step_list, ['Derivative', 'Output', 'Sobolev', 'PINN', 'PINN+Output'], f'{EXP_PATH}/{name}/plotsjoint/der_losses.pdf', 'Derivative losses', colors=['blue', 'red', 'purple', 'green', 'orange'])
plot_loss_curves([derivative_losses[:,3], output_losses[:,3], sobolev_losses[:,3], negdata_losses[:,3], pinnout_losses[:,3]], step_list, ['Derivative', 'Output', 'Sobolev', 'PINN', 'PINN+Output'], f'{EXP_PATH}/{name}/plotsjoint/pde_losses.pdf', 'PDE losses', colors=['blue', 'red', 'purple', 'green', 'orange'])
plot_loss_curves([derivative_losses[:,4], output_losses[:,4], sobolev_losses[:,4], negdata_losses[:,4], pinnout_losses[:,4]], step_list, ['Derivative', 'Output', 'Sobolev', 'PINN', 'PINN+Output'], f'{EXP_PATH}/{name}/plotsjoint/init_losses.pdf', 'Init losses', colors=['blue', 'red', 'purple', 'green', 'orange'])
plot_loss_curves([derivative_losses[:,6], output_losses[:,6], sobolev_losses[:,6], negdata_losses[:,6], pinnout_losses[:,6]], step_list, ['Derivative', 'Output', 'Sobolev', 'PINN', 'PINN+Output'], f'{EXP_PATH}/{name}/plotsjoint/init_pde_losses.pdf', 'Init PDE losses', colors=['blue', 'red', 'purple', 'green', 'orange'])

model_plots(model_0, 'Derivative', xv, 'DERL', file_mode = 'w')
model_plots(model_1, 'Output', xv, 'OUTL')
model_plots(model_neg, 'PINN', xv, 'PINN')
model_plots(model_sob, 'Sobolev', xv, 'SOB')
model_plots(model_pinnout, 'PINN+Output', xv, 'OUTL+PINN')
