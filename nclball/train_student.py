import torch
from torch import nn
from model import BallNCL
import argparse
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import os
from itertools import cycle
from tuner_results import ncl_best_params

seed = 30
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--lr_init', default=1e-3, type=float, help='Initial learning rate')
parser.add_argument('--div_weight', default=1e-1, type=float, help='Weight for the div loss')
parser.add_argument('--init_weight', default=3e1, type=float, help='Weight for the init loss')
parser.add_argument('--f_weight', default=1e-1, type=float, help='Weight for the F loss')
parser.add_argument('--bc_weight', default=1e-1, type=float, help='Weight for the F loss')
parser.add_argument('--sys_weight', default=0., type=float, help='Weight for the system loss')
parser.add_argument('--device', default='cuda:2', type=str, help='Device to use')
parser.add_argument('--name', default='student', type=str, help='Experiment name')
parser.add_argument('--train_steps', default=10000, type=int, help='Number of training steps')
parser.add_argument('--epochs', default=1, type=int, help='Number of epochs')
parser.add_argument('--batch_size', default=1000, type=int, help='Number of samples per step')
parser.add_argument('--mode', default='Derivative', type=str, help='Radius of the ball')

args = parser.parse_args()
div_weight = args.div_weight
init_weight = args.init_weight
f_weight = args.f_weight
bc_weight = args.bc_weight
device = args.device
name = args.name
train_steps = args.train_steps
epochs = args.epochs
lr_init = args.lr_init
batch_size = args.batch_size
sys_weight = args.sys_weight
mode = args.mode


EXP_PATH = 'nclball'
if not os.path.exists(EXP_PATH):
    os.mkdir(EXP_PATH)
    
lr_init = ncl_best_params[mode]['lr_init']
batch_size = 100
init_weight = ncl_best_params[mode]['init_weight']
sys_weight = ncl_best_params[mode]['sys_weight']
bc_weight = ncl_best_params[mode]['bc_weight']


print(f'Running experiment with name {name}')
print(f'sys_weight {sys_weight}, bc_weight {bc_weight}, init_weight {init_weight}')
print(f"Using {device} device")

print('Generating data')
num_boundary = batch_size
num_domain = batch_size

EXP_PATH = '.'
if not os.path.exists(EXP_PATH):
    os.mkdir(EXP_PATH)


print('Loading data')
pde_dataset = torch.load(os.path.join(EXP_PATH, 'data', 'pdedistillation_dataset.pth'))
init_dataset = torch.load(os.path.join(EXP_PATH, 'data', 'initdistillation_dataset.pth'))
bc_dataset = torch.load(os.path.join(EXP_PATH, 'data', 'bcdistillation_dataset.pth'))

# Generate the dataloaders
pde_dataloader = DataLoader(pde_dataset, batch_size, generator=gen, shuffle=True)
init_dataloader = DataLoader(init_dataset, batch_size, generator=gen, shuffle=True)
bc_dataloader = DataLoader(bc_dataset, batch_size, generator=gen, shuffle=True)

print('Data loaded!')



model = BallNCL(hidden_units=[128 for _ in range(4)],
                sys_weight=sys_weight,
                div_weight=0,
                F_weight=0,
                init_weight=init_weight,
                bc_weight=bc_weight,
                radius=1.,
                lr=lr_init,
                activation=nn.Softplus(beta=25.),
                device=device).to(device=device)

model.train()

# Prepare the lists

step_list = []
out_losses = []
der_losses = []
F_losses = []
div_losses = []
init_losses = []
bc_losses = []
tot_losses = []
hes_losses = []
times_list = []

import time

def train_loop(
        pde_dataloader:DataLoader,
        init_dataloader:DataLoader,
        bc_dataloader:DataLoader,
        print_every:int=100):
    
    for epoch in range(epochs):
        model.train()
        step_prefix = epoch*min(len(pde_dataloader),train_steps)
        print(f'Epoch: {epoch}, step_prefix: {step_prefix}')
        # Training mode for the network
        start_time = time.time()
        for step, (pde_data, init_data, bc_data) in enumerate(zip(pde_dataloader, init_dataloader, bc_dataloader)):
            if step > train_steps*1000/100:
                break
            # Load batches from dataloaders
            x_pde = pde_data[0].to(device).float().requires_grad_(True)
            y_pde = pde_data[1].to(device).float()
            Dy_pde = pde_data[2].to(device).float()
            D2y_pde = pde_data[3].to(device).float()
                    
            x_init = init_data[0].to(device).float()
            y_init = init_data[1].to(device).float()
            
            x_bc = bc_data[0].to(device).float().requires_grad_(True)
            y_bc = bc_data[1].to(device).float()
            
            
            # Call zero grad on optimizer
            model.opt.zero_grad()
            
            loss = model.student_loss_fn(mode=mode,
                x_pde=x_pde, y_pde=y_pde, Dy_pde=Dy_pde,
                x_bc=x_bc, 
                x_init=x_init, y_init=y_init, D2y_pde=D2y_pde
            )
            # Backward the loss, calculate gradients
            loss.backward()
            # Optimizer step
            model.opt.step()
            # Update the learning rate scheduling
            #model.lr_scheduler.step()
            
            # Printing
            if step % print_every == 0 and step>0:
                time_elapsed = time.time() - start_time
                start_time = time.time()
                #print('Train losses')
                with torch.no_grad():
                    step_val, out_loss_val, der_loss_val, hes_loss_val, F_loss_val, div_loss_val, bc_loss_val, init_loss_val, tot_loss_val = model.eval_student_loss_fn(
                        step=step+step_prefix, x_pde=x_pde, x_bc=x_bc, x_init=x_init, y_init=y_init, y_pde=y_pde, Dy_pde=Dy_pde, mode=mode, D2y_pde=D2y_pde
                    )
                    step_list.append(step_val)
                    out_losses.append(out_loss_val)
                    der_losses.append(der_loss_val)
                    F_losses.append(F_loss_val)
                    div_losses.append(div_loss_val)
                    bc_losses.append(bc_loss_val)
                    init_losses.append(init_loss_val)
                    tot_losses.append(tot_loss_val)
                    hes_losses.append(hes_loss_val)
                    times_list.append(time_elapsed)
                    print(f'Time elapsed: {time_elapsed}')
                    print(f'Step: {step+step_prefix}, out_loss: {out_loss_val}, der_loss:{der_loss_val}, hes_loss:{hes_loss_val} F_loss: {F_loss_val}, div_loss: {div_loss_val}, bc_loss: {bc_loss_val}, init_loss: {init_loss_val}, tot_loss: {tot_loss_val}')        
        # Calculate and average the loss over the test dataloader
        
train_loop(pde_dataloader=pde_dataloader, init_dataloader=init_dataloader, bc_dataloader=bc_dataloader, print_every=100)
print('Training done!')

if not os.path.exists(f'{EXP_PATH}/student{mode}'):
    os.mkdir(f'{EXP_PATH}/student{mode}')

# Save the model
if not os.path.exists(f'{EXP_PATH}/student{mode}/saved_models'):
    os.mkdir(f'{EXP_PATH}/student{mode}/saved_models')
torch.save(model.state_dict(), f'{EXP_PATH}/student{mode}/saved_models/nclball')

# Load it to be sure it works
model.load_state_dict(torch.load(f'{EXP_PATH}/student{mode}/saved_models/nclball'))

if not os.path.exists(f'{EXP_PATH}/student{mode}/plots'):
    os.mkdir(f'{EXP_PATH}/student{mode}/plots')

from matplotlib import pyplot as plt
#plotting function to generate the figures for the ball problem


def plotVelDensBall(u,rho,T=[0,0.25,0.5],apx=""):
    box= 8
    #our plots
    fig1,ax1 = plt.subplots(1,3,figsize=(3*box,box))
    fig2,ax2 = plt.subplots(1,3,figsize=(3*box,box))
    
    for i,t in enumerate(T): 
        plotDensBall(t,rho,Z=0,ax=ax1[i])
        plotVelBall(t,u,Z=0,ax=ax2[i])
    
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(f'{EXP_PATH}/student{mode}/plots/density.png')
    fig2.savefig(f'{EXP_PATH}/student{mode}/plots/velocity.png')
    

def plotVelBall(T,u,ax,Z=0):
    N = 250
    a = 1.1
    X,Y = np.meshgrid(np.linspace(-a,a,N),np.linspace(-a,a,N))
    exterior = X**2 + Y**2 + Z**2 >= 1
    pts = np.vstack([np.ones(X.reshape(-1).shape)*T,X.reshape(-1),Y.reshape(-1),np.ones(X.reshape(-1).shape)*Z]).T

    #plots the streamplot for the velocity field
    if ax is None:
        fig,ax = plt.subplots(1,2,figsize=(14,7))
    ax.set_xlim(-a,a)
    ax.set_ylim(-a,a)
    
    vel = u(torch.tensor(pts, dtype=torch.float32).to(device)).cpu().detach()
    U = np.array(vel[:,0].reshape(X.shape))
    V = np.array(vel[:,1].reshape(Y.shape))
    #mask the outside of the ball
    U[exterior] = np.nan
    V[exterior] = np.nan
    plt_str = ax.streamplot(X,Y,U,V,density=0.35,color=U**2 + V**2, arrowsize=5,linewidth=3)
    
    #add outline for aesthetics
    circle = plt.Circle((0, 0), 1.05, fill=False, lw=3,color='k')
    ax.add_patch(circle)
    ax.axis('off')
    

def plotDensBall(T,rho,ax,Z=0):
    N = 250
    a = 1.1
    X,Y = np.meshgrid(np.linspace(-a,a,N),np.linspace(-a,a,N))
    exterior = X**2 + Y**2 + Z**2 >= 1
    pts = np.vstack([np.ones(X.reshape(-1).shape)*T,X.reshape(-1),Y.reshape(-1),np.ones(X.reshape(-1).shape)*Z]).T

    density = rho(torch.tensor(pts, dtype=torch.float32).to(device)).cpu().detach().reshape(X.shape)
    density = np.array(density)
    density[exterior] = np.nan
    plt_dens = ax.contourf(X,Y,density,20)
    circle = plt.Circle((0, 0), 1.0, fill=False, lw=3,color='k')
    ax.add_patch(circle)
    
    ax.set_xlim(-a,a)
    ax.set_ylim(-a,a)
    
    ax.axis('off')

# %%
rho = lambda x: model.forward(x)[:,0]
u = lambda x: model.forward(x)[:,1:4]/model.forward(x)[:,0].reshape((-1,1))

plotVelDensBall(u=u, rho=rho)
plt.close()

step_list = torch.tensor(step_list).cpu().numpy()
out_losses = torch.tensor(out_losses).cpu().numpy()
der_losses = torch.tensor(der_losses).cpu().numpy()
F_losses = torch.tensor(F_losses).cpu().numpy()
init_losses = torch.tensor(init_losses).cpu().numpy()
div_losses = torch.tensor(div_losses).cpu().numpy()
bc_losses = torch.tensor(bc_losses).cpu().numpy()
tot_losses = torch.tensor(tot_losses).cpu().numpy()
hes_losses = torch.tensor(hes_losses).cpu().numpy()
times_list = np.array(times_list)

loss_combination_list = np.stack([step_list, out_losses, der_losses, F_losses, div_losses, bc_losses, init_losses, tot_losses, times_list, hes_losses], axis=1)
with open(f'{EXP_PATH}/student{mode}/losses.npy', 'wb') as f:
    np.save(f, loss_combination_list)



plt.figure()
plt.plot(step_list, F_losses, label='F_loss', color='red')
plt.plot(step_list, out_losses, label='out_loss', color='yellow')
plt.plot(step_list, init_losses, label='init_loss', color='orange')
plt.plot(step_list, der_losses, label='der_loss', color='blue')
plt.plot(step_list, div_losses, label='div_loss', color='green')
plt.plot(step_list, bc_losses, label='bc_loss', color='purple')
plt.plot(step_list, hes_losses, label='hes_loss', color='brown')
plt.plot(step_list, tot_losses, label='tot_loss', color='black')
plt.legend()
plt.yscale('log')
plt.savefig(f'{EXP_PATH}/student{mode}/plots/losses.png')

print('Plotting done!')
print('Start saving I/O dataset')