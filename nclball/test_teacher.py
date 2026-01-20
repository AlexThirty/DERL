# %%
import torch
from torch import nn
from model import BallNCL
import argparse
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import os
from itertools import cycle
from torch.func import vmap, jacrev, hessian

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
parser.add_argument('--device', default='cuda:2', type=str, help='Device to use')
parser.add_argument('--name', default='ncl', type=str, help='Experiment name')
parser.add_argument('--train_steps', default=10000, type=int, help='Number of training steps')
parser.add_argument('--epochs', default=1, type=int, help='Number of epochs')
parser.add_argument('--batch_size', default=1000, type=int, help='Number of samples per step')

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


EXP_PATH = '.'
if not os.path.exists(EXP_PATH):
    os.mkdir(EXP_PATH)

print(f'Running experiment with name {name}')
print(f'F_weight {f_weight}, div_weight {div_weight}, bc_weight {bc_weight}, init_weight {init_weight}')
print(f"Using {device} device")

print('Generating data')
num_boundary = batch_size
num_domain = batch_size

EXP_PATH = 'nclball'
if not os.path.exists(EXP_PATH):
    os.mkdir(EXP_PATH)


print('Loading data')
pde_dataset = torch.load(os.path.join(EXP_PATH, 'data', 'pde_dataset.pth'))
init_dataset = torch.load(os.path.join(EXP_PATH, 'data', 'init_dataset.pth'))
bc_dataset = torch.load(os.path.join(EXP_PATH, 'data', 'bc_dataset.pth'))

# Generate the dataloaders
pde_dataloader = DataLoader(pde_dataset, batch_size, generator=gen, shuffle=True)
init_dataloader = DataLoader(init_dataset, batch_size, generator=gen, shuffle=True)
bc_dataloader = DataLoader(bc_dataset, batch_size, generator=gen, shuffle=True)

print('Data loaded!')



model = BallNCL(hidden_units=[128 for _ in range(4)],
                sys_weight=0.,
                div_weight=div_weight,
                F_weight=f_weight,
                init_weight=init_weight,
                bc_weight=bc_weight,
                radius=1.,
                lr=lr_init,
                activation=nn.Softplus(beta=25.),
                device=device).to(device=device)

model.train()


# Load it to be sure it works
model.load_state_dict(torch.load(f'{EXP_PATH}/teacher/saved_models/nclball_teacher'))

if not os.path.exists(f'{EXP_PATH}/teacher/plots'):
    os.mkdir(f'{EXP_PATH}/teacher/plots')

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
    fig1.savefig(f'{EXP_PATH}/teacher/plots/density.png')
    fig2.savefig(f'{EXP_PATH}/teacher/plots/velocity.png')
    

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

with open(f'{EXP_PATH}/teacher/losses.npy', 'rb') as f:
    loss_combination_list = np.load(f)
    

step_list = loss_combination_list[:,0]
F_losses = loss_combination_list[:,1]
div_losses = loss_combination_list[:,2]
bc_losses = loss_combination_list[:,3]
init_losses = loss_combination_list[:,4]
tot_losses = loss_combination_list[:,5]
#times = loss_combination_list[:,6]

print(f'BC loss avg: {np.mean(bc_losses[-50:])}') 

plt.figure()
plt.plot(step_list, F_losses, label='F_loss', color='red')
plt.plot(step_list, init_losses, label='init_loss', color='orange')
plt.plot(step_list, div_losses, label='div_loss', color='green')
plt.plot(step_list, bc_losses, label='bc_loss', color='purple')
plt.plot(step_list, tot_losses, label='tot_loss', color='black')
plt.legend()
plt.yscale('log')
plt.savefig(f'{EXP_PATH}/teacher/plots/losses.png')

print('Plotting done!')
print('Start saving I/O dataset')

with open(f'{EXP_PATH}/joint/losses.txt', 'a') as f:
    f.write('Teacher\n')
    f.write(f'BC loss avg: {np.mean(bc_losses[-10:])}\n')
    f.write(f'F loss avg: {np.mean(F_losses[-10:])}\n')
    f.write(f'Init loss avg: {np.mean(init_losses[-10:])}\n')
    f.write(f'Div loss avg: {np.mean(div_losses[-10:])}\n')
    f.write(f'Total loss avg: {np.mean(tot_losses[-10:])}\n')


# Empty cache just to be sure
torch.cuda.empty_cache()
# Create tensors for I/O
pde_x = []
pde_y = []
pde_ytrue = []
pde_Dy = []
pde_Hy = []



for (x,y) in pde_dataloader:
    pde_x.append(x)
    pde_y.append(model.forward(x.to(device).float()).detach().cpu())
    pde_ytrue.append(y)
    pde_Dy.append(vmap(jacrev(model.forward_single))(x.to(device).float()).detach().cpu())
    pde_Hy.append(vmap(hessian(model.forward_single))(x.to(device).float()).detach().cpu())
    torch.cuda.empty_cache()

pde_x = torch.cat(pde_x)
pde_y = torch.cat(pde_y)
pde_Dy = torch.cat(pde_Dy)
pde_ytrue = torch.cat(pde_ytrue)
pde_Hy = torch.cat(pde_Hy)

print(f'pde_x.shape: {pde_x.shape}')
print(f'pde_y.shape: {pde_y.shape}')
print(f'pde_Dy.shape: {pde_Dy.shape}')
print(f'pde_Hy.shape: {pde_Hy.shape}')
print(f'pde_ytrue.shape: {pde_ytrue.shape}')
pde_distillation_dataset = TensorDataset(pde_x, pde_y, pde_Dy, pde_Hy, pde_ytrue)
torch.save(pde_distillation_dataset, os.path.join(EXP_PATH, 'data', f'pdedistillation_dataset.pth'))
print('Created PDE distillation dataset')

# Empty cache just to be sure
torch.cuda.empty_cache()
# Create tensors for I/O
init_x = []
init_y = []
init_Dy = []
init_Hy = []
init_ytrue = []
for (x,y) in init_dataloader:
    init_x.append(x)
    init_ytrue.append(y)
    init_y.append(model.forward(x.to(device).float()).detach().cpu())
    init_Dy.append(torch.func.vmap(torch.func.jacrev(model.forward_single))(x.to(device).float()).detach().cpu())
    init_Hy.append(torch.func.vmap(torch.func.hessian(model.forward_single))(x.to(device).float()).detach().cpu())
    torch.cuda.empty_cache()

init_x = torch.cat(init_x)
init_y = torch.cat(init_y)
init_Dy = torch.cat(init_Dy)
init_Hy = torch.cat(init_Hy)
init_ytrue = torch.cat(init_ytrue)

print(f'init_x.shape: {init_x.shape}')
print(f'init_y.shape: {init_y.shape}')
print(f'init_Dy.shape: {init_Dy.shape}')
print(f'init_Hy.shape: {init_Hy.shape}')
print(f'init_ytrue.shape: {init_ytrue.shape}')
init_distillation_dataset = TensorDataset(init_x, init_y, init_Dy, init_Hy, init_ytrue)
torch.save(init_distillation_dataset, os.path.join(EXP_PATH, 'data', f'initdistillation_dataset.pth'))
print('Created init distillation dataset')



# Empty cache just to be sure
torch.cuda.empty_cache()
# Create tensors for I/O
bc_x = []
bc_y = []
bc_Dy = []
bc_Hy = []
bc_ytrue = []
for (x,y) in bc_dataloader:
    bc_x.append(x)
    bc_ytrue.append(y)
    bc_y.append(model.forward(x.to(device).float()).detach().cpu())
    bc_Dy.append(torch.func.vmap(torch.func.jacrev(model.forward_single))(x.to(device).float()).detach().cpu())
    bc_Hy.append(torch.func.vmap(torch.func.hessian(model.forward_single))(x.to(device).float()).detach().cpu())
    torch.cuda.empty_cache()

bc_x = torch.cat(bc_x)
bc_y = torch.cat(bc_y)
bc_Dy = torch.cat(bc_Dy)
bc_Hy = torch.cat(bc_Hy)
bc_ytrue = torch.cat(bc_ytrue)

print(f'bc_x.shape: {bc_x.shape}')
print(f'bc_y.shape: {bc_y.shape}')
print(f'bc_Dy.shape: {bc_Dy.shape}')
print(f'bc_Hy.shape: {bc_Hy.shape}')
print(f'bc_ytrue.shape: {bc_ytrue.shape}')
bc_distillation_dataset = TensorDataset(bc_x, bc_y, bc_Dy, bc_Hy, bc_ytrue)
torch.save(bc_distillation_dataset, os.path.join(EXP_PATH, 'data', f'bcdistillation_dataset.pth'))
print('Created BC distillation dataset')

