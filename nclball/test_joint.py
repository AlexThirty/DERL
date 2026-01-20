# %%
import torch
from torch import nn
from model import BallNCL
import argparse
from torch.utils.data import DataLoader
import numpy as np
import random
import os
from itertools import cycle

#font = {'size'   : 16}
#import matplotlib
#matplotlib.rc('font', **font)

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



model_0 = BallNCL(hidden_units=[128 for _ in range(4)],
                sys_weight=0.,
                div_weight=div_weight,
                F_weight=f_weight,
                init_weight=init_weight,
                bc_weight=bc_weight,
                radius=1.,
                lr=lr_init,
                activation=nn.Softplus(beta=25.),
                device=device).to(device=device)

model_0.eval()


model_1 = BallNCL(hidden_units=[128 for _ in range(4)],
                sys_weight=0.,
                div_weight=div_weight,
                F_weight=f_weight,
                init_weight=init_weight,
                bc_weight=bc_weight,
                radius=1.,
                lr=lr_init,
                activation=nn.Softplus(beta=25.),
                device=device).to(device=device)

model_1.eval()

model_sob = BallNCL(hidden_units=[128 for _ in range(4)],
                sys_weight=0.,
                div_weight=div_weight,
                F_weight=f_weight,
                init_weight=init_weight,
                bc_weight=bc_weight,
                radius=1.,
                lr=lr_init,
                activation=nn.Softplus(beta=25.),
                device=device).to(device=device)

model_sob.eval()

model_neg = BallNCL(hidden_units=[128 for _ in range(4)],
                sys_weight=0.,
                div_weight=div_weight,
                F_weight=f_weight,
                init_weight=init_weight,
                bc_weight=bc_weight,
                radius=1.,
                lr=lr_init,
                activation=nn.Softplus(beta=25.),
                device=device).to(device=device)

model_neg.eval()


# Load it to be sure it works
model_0.load_state_dict(torch.load(f'{EXP_PATH}/studentDerivative/saved_models/nclball'))
model_1.load_state_dict(torch.load(f'{EXP_PATH}/studentOutput/saved_models/nclball'))
model_sob.load_state_dict(torch.load(f'{EXP_PATH}/studentSobolev/saved_models/nclball'))
model_neg.load_state_dict(torch.load(f'{EXP_PATH}/teacher/saved_models/nclball_teacher'))

### GENERAL SECTION
import numpy as np
import torch
from matplotlib import pyplot as plt
 
from matplotlib.colors import Normalize

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

if not os.path.exists(f'{EXP_PATH}/joint/plots'):
    os.makedirs(f'{EXP_PATH}/joint/plots')

from matplotlib import pyplot as plt
#plotting function to generate the figures for the ball problem


from plot_utils import plot_errors

from scipy import ndimage
def plot_loss_curves(to_plot, step_list, names, path, title, colors):
    plt.figure()
    for i in range(len(to_plot)):
        #plot_y = np.convolve(to_plot[i], np.ones((N,))/N, mode='valid')
        #plt.plot(step_list[:-(N-1)], plot_y, label=names[i], color=colors[i])
        y2 = ndimage.median_filter(to_plot[i], size=15)
        plt.plot(step_list, y2, label=names[i], color = colors[i])
    plt.legend()
    plt.yscale('log')
    plt.title(title)
    plt.savefig(path, dpi=300)
    plt.close()
    
with open(f'{EXP_PATH}/studentDerivative/losses.npy', 'rb') as f:
    derivative_losses = np.load(f)

with open(f'{EXP_PATH}/studentOutput/losses.npy', 'rb') as f:
    output_losses = np.load(f)

with open(f'{EXP_PATH}/studentSobolev/losses.npy', 'rb') as f:
    sobolev_losses = np.load(f)

    
step_list = derivative_losses[:,0]

plot_loss_curves([derivative_losses[:,1], output_losses[:,1], sobolev_losses[:,1]], step_list, ['Derivative', 'Output', 'Sobolev'], f'{EXP_PATH}/joint/plotsjoint/output_loss.pdf', 'Output Loss', ['blue', 'red', 'purple'])
plot_loss_curves([derivative_losses[:,2], output_losses[:,2], sobolev_losses[:,2]], step_list, ['Derivative', 'Output', 'Sobolev'], f'{EXP_PATH}/joint/plotsjoint/derivative_loss.pdf', 'Derivative Loss',['blue', 'red', 'purple'])
plot_loss_curves([derivative_losses[:,3], output_losses[:,3], sobolev_losses[:,3]], step_list, ['Derivative', 'Output', 'Sobolev'], f'{EXP_PATH}/joint/plotsjoint/F_loss.pdf', 'F Loss',['blue', 'red', 'purple'])
plot_loss_curves([derivative_losses[:,4], output_losses[:,4], sobolev_losses[:,4]], step_list, ['Derivative', 'Output', 'Sobolev'], f'{EXP_PATH}/joint/plotsjoint/div_loss.pdf', 'Div Loss',['blue', 'red', 'purple'])
plot_loss_curves([derivative_losses[:,5], output_losses[:,5], sobolev_losses[:,5]], step_list, ['Derivative', 'Output', 'Sobolev'], f'{EXP_PATH}/joint/plotsjoint/bc_loss.pdf', 'BC Loss',['blue', 'red', 'purple'])
plot_loss_curves([derivative_losses[:,6], output_losses[:,6], sobolev_losses[:,6]], step_list, ['Derivative', 'Output', 'Sobolev'], f'{EXP_PATH}/joint/plotsjoint/init_loss.pdf', 'Init Loss', ['blue', 'red', 'purple'])


plot_errors([model_0, model_1, model_sob], model_neg, ['DERL', 'OUTL', 'SOB'], f'{EXP_PATH}/joint',)



def plotErrorVelDensBall(u_list:list,
                         rho_list:list,
                         T=[0,0.25,0.5],
                         model_titles=['Derivative', 'Output', 'Sobolev'],
                         apx="",
                         cmap='jet',
                         suptitles=['Error Plots'],
                         device='cpu',
                         save_paths=['']):
    # Box size
    box = 8
    # Value of Z
    Z = 0
    # Points generation
    N = 250
    a = 1.1
    X,Y = np.meshgrid(np.linspace(-a,a,N),np.linspace(-a,a,N))
    # List of errors
    us = []
    rhos = []
    
    # Loop over the models
    for u, rho in zip(u_list, rho_list):    
        us_mode = []
        rhos_mode = []
        # Loop over time
        for t in T:
            # do the following but in bathces
            pts = np.vstack([np.ones(X.reshape(-1).shape)*t,X.reshape(-1),Y.reshape(-1),np.ones(X.reshape(-1).shape)*Z]).T
            # Use batches to avoid memory issues
            batch_size = 1000
            n_batches = len(pts)//batch_size
            
            u_t = []
            rho_t = []
            for i in range(n_batches):
                u_t.append(u(torch.tensor(pts[i*batch_size:(i+1)*batch_size], dtype=torch.float32).to(device)).cpu().detach())
                rho_t.append(rho(torch.tensor(pts[i*batch_size:(i+1)*batch_size], dtype=torch.float32).to(device)).cpu().detach())
            u_t.append(u(torch.tensor(pts[(i+1)*batch_size:], dtype=torch.float32).to(device)).cpu().detach())
            rho_t.append(rho(torch.tensor(pts[(i+1)*batch_size:], dtype=torch.float32).to(device)).cpu().detach())
            
            # Concatenate the results
            us_mode.append(np.concatenate(u_t))
            rhos_mode.append(np.concatenate(rho_t))
        us.append(us_mode)
        rhos.append(rhos_mode)
    
    # Get the final array of results
    us = np.array(us)
    rhos = np.array(rhos)

    # Mask the exterior of the ball     
    us[:,:,np.logical_not(X**2 + Y**2 <= 1).reshape((-1))] = np.nan
    rhos[:,:,np.logical_not(X**2 + Y**2 <= 1).reshape((-1))] = np.nan
    
    # Get the max and min for the normalization
    us_vmin = np.nanmin(us)
    us_vmax = np.nanmax(us)

    us_levels = np.linspace(us_vmin, us_vmax, 50)
    if us_vmin < 0:
        us_norm = MidpointNormalize(vmin=us_vmin, midpoint=0, vmax=us_vmax)
    else:
        us_norm = None
    
    rhos_vmin = np.nanmin(rhos)
    rhos_vmax = np.nanmax(rhos)

    rhos_levels = np.linspace(rhos_vmin, rhos_vmax, 50)
    if rhos_vmin < 0:   
        rhos_norm = MidpointNormalize(vmin=rhos_vmin, midpoint=0, vmax=rhos_vmax)
    else:
        rhos_norm = None
    
    
    # Plotting section
    fig1, ax1 = plt.subplots(len(u_list),3,figsize=(3*box,len(u_list)*box))
    fig2, ax2 = plt.subplots(len(u_list),3,figsize=(3*box,len(u_list)*box))
    for i, (rho, u) in enumerate(zip(rho_list, u_list)):
        ax1[i,0].set_ylabel(model_titles[i], fontsize=16)
        ax2[i,0].set_ylabel(model_titles[i], fontsize=16)
        for j, t in enumerate(T):
            ax1[0,j].set_title(f'T = {t}', fontsize=16)
            ax2[0,j].set_title(f'T = {t}', fontsize=16)
            plotDensError(X,Y,t,rhos[i,j],ax=ax1[i,j],cmap=cmap,vmin=rhos_vmin, vmax=rhos_vmax, norm=rhos_norm)
            torch.cuda.empty_cache()
            plotDensError(X,Y,t,us[i,j],ax=ax2[i,j], cmap=cmap,vmin=us_vmin, vmax=us_vmax, norm=us_norm)
            torch.cuda.empty_cache()
    
    fig1.suptitle(suptitles[0], fontsize=16)
    fig2.suptitle(suptitles[1], fontsize=16)
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(save_paths[0])
    fig2.savefig(save_paths[1])


def plotDensError(X,Y,T,rho,ax,Z=0, cmap='jet', vmin=None, vmax=None, norm=None):
    a = 1.1
    exterior = X**2 + Y**2 + Z**2 >= 1
    rho = rho.reshape(X.shape)
    rho[exterior] = np.nan
    plt_dens = ax.contourf(X,Y,rho,50,cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    circle = plt.Circle((0, 0), 1.0, fill=False, lw=3,color='k')
    ax.add_patch(circle)
    
    ax.set_xlim(-a,a)
    ax.set_ylim(-a,a)
    
    ax.axis('off')


def plotVelBall(T,u,ax,Z=0,title=''):
    N = 100
    a = 1.1
    X,Y = np.meshgrid(np.linspace(-a,a,N),np.linspace(-a,a,N))
    exterior = X**2 + Y**2 + Z**2 >= 1
    pts = np.vstack([np.ones(X.reshape(-1).shape)*T,X.reshape(-1),Y.reshape(-1),np.ones(X.reshape(-1).shape)*Z]).T

    #plots the streamplot for the velocity field
    if ax is None:
        fig,ax = plt.subplots(1,2,figsize=(14,7))
    ax.set_xlim(-a,a)
    ax.set_title(title)
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




def plotDensBall(T,rho,ax,Z=0, title=''):
    N = 100
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
    ax.set_title(title)
    ax.set_ylim(-a,a)
    
    ax.axis('off')


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
    fig1.savefig(f'{EXP_PATH}/student{apx}/plots/density.pdf')
    fig2.savefig(f'{EXP_PATH}/student{apx}/plots/velocity.pdf')
    
def plotMultiVelDensBall(u_list:list,rho_list:list,T=[0,0.25,0.5],apx="", titles=['DERL', 'OUTL', 'SOB', 'NCL']):
    box = 8
    #our plots
    fig1, ax1 = plt.subplots(len(u_list),3,figsize=(3*box,len(u_list)*box))
    fig2, ax2 = plt.subplots(len(u_list),3,figsize=(3*box,len(u_list)*box))
    for i, (rho, u) in enumerate(zip(rho_list, u_list)):
        for j, t in enumerate(T):
            plotDensBall(t,rho,Z=0,ax=ax1[i,j],title=f'{titles[i]}, t={t}')
            torch.cuda.empty_cache()
            plotVelBall(t,u,Z=0,ax=ax2[i,j],title=f'{titles[i]}, t={t}')
            torch.cuda.empty_cache()
            
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(f'{EXP_PATH}/joint/plots/density{apx}.pdf')
    fig2.savefig(f'{EXP_PATH}/joint/plots/velocity{apx}.pdf')

from matplotlib.colors import TwoSlopeNorm




def plot_model(model, name):
    rho = lambda x: model.forward(x)[:,0]
    u = lambda x: model.forward(x)[:,1:4]/model.forward(x)[:,0].reshape((-1,1))
    plotVelDensBall(u=u, rho=rho, apx=name)
    plt.close()
    with open(f'{EXP_PATH}/student{name}/losses.npy', 'rb') as f:
        loss_combination_list = np.load(f)
    step_list = loss_combination_list[:,0]
    out_losses = loss_combination_list[:,1]
    der_losses = loss_combination_list[:,2]
    F_losses = loss_combination_list[:,3]
    div_losses = loss_combination_list[:,4]
    bc_losses = loss_combination_list[:,5]
    init_losses = loss_combination_list[:,6]
    tot_losses = loss_combination_list[:,7]
    times = loss_combination_list[:,8]
    # TODO add time
    with open(f'{EXP_PATH}/joint/losses.txt', 'a') as f:
        print(f'{name}', file=f)
        print(f'Output loss: {np.mean(out_losses[-10:])}', file=f)
        print(f'Derivative loss: {np.mean(der_losses[-10:])}', file=f)
        print(f'F loss: {np.mean(F_losses[-10:])}', file=f)
        print(f'Div loss: {np.mean(div_losses[-10:])}', file=f)
        print(f'BC loss: {np.mean(bc_losses[-10:])}', file=f)
        print(f'Init loss: {np.mean(init_losses[-10:])}', file=f)
        print(f'Total loss: {np.mean(tot_losses[-10:])}', file=f)
        print(f'Time: {np.mean(times[-10:])}', file=f)
        print('\n', file=f)

    plt.figure()
    plt.plot(step_list, F_losses, label='F_loss', color='red')
    plt.plot(step_list, out_losses, label='out_loss', color='green')
    plt.plot(step_list, der_losses, label='der_loss', color='blue')
    plt.plot(step_list, init_losses, label='init_loss', color='orange')
    plt.plot(step_list, div_losses, label='div_loss', color='green')
    plt.plot(step_list, bc_losses, label='bc_loss', color='purple')
    plt.plot(step_list, tot_losses, label='tot_loss', color='black')
    plt.legend()
    plt.yscale('log')
    plt.savefig(f'{EXP_PATH}/student{name}/plots/losses.pdf')


plt.close()

with open(f'{EXP_PATH}/teacher/losses.npy', 'rb') as f:
    loss_combination_list = np.load(f)
    
    
plot_model(model_0, 'Derivative')
plot_model(model_1, 'Output')
plot_model(model_sob, 'Sobolev')
