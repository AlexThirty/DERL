import jax
import jax.numpy as jnp
import numpy as np # get rid of this eventually
import argparse
from jax import jit
from jax.experimental.ode import odeint
from functools import partial # reduces arguments to function by making some subset implicit

from jax.example_libraries import stax
from jax.example_libraries import optimizers

import os, sys, time

from external_models.lnn import lagrangian_eom_rk4, lagrangian_eom, unconstrained_eom, raw_lagrangian_eom, raw_lagrangian_eom_damped
from external_models.lnn_models import mlp as make_mlp
from external_models.lnn_utils import wrap_coords

from external_models.lnn_hps import learned_dynamics, extended_mlp


class ObjectView(object):
    def __init__(self, d): self.__dict__ = d
    
from external_models.lnn_physics import analytical_fn

vfnc = jax.jit(jax.vmap(analytical_fn))
b = 0.        # damping coefficient

EXP_PATH = '.'

m = 1. # old was 0.1
l = 10. # old was 1.
g = 9.81

def u_vec(xv):
    return torch.column_stack((xv[:,1],-(g/l)*np.sin(xv[:,0])-(b/m)*xv[:,1]))

args = ObjectView({'dataset_size': 200,
 'fps': 10,
 'samples': 100,
 'num_epochs': 80000,
 'seed': 30,
 'loss': 'l1',
 'act': 'softplus',
 'hidden_dim': 600,
 'output_dim': 1,
 'layers': 3,
 'n_updates': 1,
 'lr': 0.001,
 'lr2': 2e-05,
 'dt': 0.1,
 'model': 'gln',
 'batch_size': 512,
 'l2reg': 5.7e-07,
})
# args = loaded['args']
rng = jax.random.PRNGKey(args.seed)

from matplotlib import pyplot as plt

best_params = None
best_loss = np.inf

init_random_params, nn_forward_fn = extended_mlp(args)
from external_models import lnn_hps
lnn_hps.nn_forward_fn = nn_forward_fn
_, init_params = init_random_params(rng+1, (-1, 2))
rng += 1
model = (nn_forward_fn, init_params)
opt_init, opt_update, get_params = optimizers.adam(args.lr)



(nn_forward_fn, init_params) = model


import torch
train_dataset = torch.load(f'data/true_dataset_train.pth')
val_dataset = torch.load(f'data/true_dataset_val.pth')
test_dataset = torch.load(f'data/true_dataset_test.pth')
        
import flax
from flax import linen as nn
from flax.training import checkpoints, train_state

checkpoint_dir = os.path.abspath(f'{EXP_PATH}/true/saved_models')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, 'pendulum_lnn')
# Load the model parameters
params = checkpoints.restore_checkpoint(checkpoint_path, init_params)
# Save the model

#from models.pendulum import u_vec
name = 'true'
title = 'LNN'
# --------------- Plotting ----------------
new_data = torch.load(os.path.join('data', f'true_dataset_test.pth'))
new_data_xv0 = new_data[:][0].reshape((10,-1,3))[:,0,1:]
xv0 = new_data_xv0.numpy()
x0 = xv0[:,0]
v0 = xv0[:,1]
if not os.path.exists(f'{EXP_PATH}/{name}/plots{title}'):
    os.makedirs(f'{EXP_PATH}/{name}/plots{title}')

from matplotlib import patches
dt = 0.001
points = 10
t_max = 10
steps = int(t_max/dt)+1
xv = torch.zeros((points,steps,2))
u_save = torch.zeros((points,steps,2))
xv[:,0,0] = torch.from_numpy(x0)
xv[:,0,1] = torch.from_numpy(v0)
for i in range(1,steps):
# v[i] = v[i-1] + dt*u(x[i-1])
    xv[:,i,0] = xv[:,i-1,0] + dt*u_vec(xv[:,i-1,:])[:,0]
    xv[:,i,1] = xv[:,i-1,1] + dt*u_vec(xv[:,i-1,:])[:,1]
t_base = np.arange(start=0, stop=t_max+dt, step=dt)

# Number of points for the field
N = 500
xlim = np.pi/2
ylim = 2.
X,Y = np.meshgrid(np.linspace(-xlim,xlim,N),np.linspace(-ylim,ylim,N))
pts = np.vstack([X.reshape(-1),Y.reshape(-1)]).T

#plots the streamplot for the velocity field
plt.figure(figsize=(5,5))
#print(pts)
vel = u_vec(torch.from_numpy(pts))
#print(vel)
U = np.array(vel[:,0].reshape(X.shape))
V = np.array(vel[:,1].reshape(Y.shape))
#mask the outside of the ball

plt.streamplot(X,Y,U,V,density=1,color=U**2 + V**2, linewidth=0.15)
lag = raw_lagrangian_eom_damped if b!=0. else raw_lagrangian_eom

plt.xlim((-xlim,xlim))
plt.ylim((-ylim,ylim))
#add outline for aesthetics
t_base = np.arange(start=0, stop=t_max+dt, step=dt)
#xv_pred = model.evaluate_trajectory(x0=xv[:,0,:].float(), time_steps=steps).detach().cpu().numpy()
xv_pred = torch.zeros((xv.shape))
xv_iter = torch.zeros((xv.shape))
for i in range(xv.shape[0]):
    xv_pred[i,0,:] = xv[i,0,:]
    in_pts = torch.column_stack((torch.from_numpy(t_base).reshape((-1,1)), torch.tile(torch.from_numpy(xv0[i]), (steps,1)))).float()
    print('Gathering trajectory for point', i) 
    xv_pred[i,:,:] = torch.from_numpy(np.array(jax.device_get(odeint(partial(lag, learned_dynamics(params)), xv0[i], t_base))))

xv = xv.numpy()
for i in range(10):
    plt.plot(xv[i,:,0], xv[i,:,1], color='blue')
    plt.plot(xv_pred[i,:,0], xv_pred[i,:,1], color='red')
    #plt.plot(xv_iter[i,:,0], xv_iter[i,:,1], color='green')
    #plt.legend()
blue_patch = patches.Patch(color='blue', label='True trajectories')
red_patch = patches.Patch(color='red', label='Predicted trajectories')
#green_patch = patches.Patch(color='green', label='Iterative trajectories')
plt.legend(handles=[blue_patch,red_patch])
plt.xlabel(r'Angle: $\theta$')
plt.ylabel(r'Angular speed: $\omega$')

plt.title(f'LNN learning phase trajectories')
plt.savefig(f'{EXP_PATH}/{name}/plots{title}/pendulum_phase_trajectory.png', dpi=300)
    
plt.close()

plt.figure(figsize=(8,5))
t_base = np.arange(start=0, stop=t_max+dt, step=dt)
for i in range(10):
    plt.plot(t_base, xv[i,:,0], color='blue')
    plt.plot(t_base, xv_pred[i,:,0], color='red')
blue_patch = patches.Patch(color='blue', label='True trajectories')
red_patch = patches.Patch(color='red', label='Predicted trajectories')
plt.legend(handles=[blue_patch,red_patch])
plt.xlabel(r'Time: $t$')
plt.ylabel(r'Angle: $\theta$')
plt.title(f'LNN learning time trajectories')
plt.savefig(f'{EXP_PATH}/{name}/plots{title}/pendulum_trajectory.png', dpi=300)
plt.close()

#plots the streamplot for the velocity field
plt.figure(figsize=(5,5))
#print(pts)
vel = jax.vmap(partial(lag, learned_dynamics(params)))(pts)
print(np.max(vel))
print(np.min(vel))
U = np.array(vel[:,0].reshape(X.shape))
V = np.array(vel[:,1].reshape(Y.shape))
#mask the outside of the ball


plt.streamplot(X,Y,U,V,density=1,color=U**2 + V**2, linewidth=0.15)
for i in range(10):
    plt.plot(xv_pred[i,:,0], xv_pred[i,:,1], label=f'trajectory{i}', color='red')
plt.xlim((-xlim,xlim))
plt.ylim((-ylim,ylim))

plt.xlabel(r'Angle: $\theta$')
plt.ylabel(r'Angular speed: $\omega$')

plt.title(f'LNN learning predicted field')
plt.savefig(f'{EXP_PATH}/{name}/plots{title}/predicted_field.png')
plt.close()

plt.figure(figsize=(6,5))
vel_true = u_vec(torch.from_numpy(pts))
#print(vel)
U_true = np.array(vel_true[:,0].reshape(X.shape))
V_true = np.array(vel_true[:,1].reshape(Y.shape))
err = np.sqrt((U-U_true)**2+(V-V_true)**2)
err[np.argwhere(np.abs(err>1))] = 1
plt.contourf(X,Y,err,100,cmap='jet')
plt.title('Error in predicted fields')
plt.colorbar()
plt.xlim((-xlim,xlim))
plt.ylim((-ylim,ylim))
plt.xlabel(r'Angle: $\theta$')
plt.ylabel(r'Angular speed: $\omega$')

plt.title(f'LNN learning field error')
plt.savefig(f'{EXP_PATH}/{name}/plots{title}/error_field.png')
plt.close()