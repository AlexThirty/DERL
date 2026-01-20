import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np
import random
import os
seed = 30
from itertools import cycle

from model import DensityNet, u_vec
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)
#from tuner_results import pendulum_best_params
from tuner_results import continuity_best_params


parser = argparse.ArgumentParser()
parser.add_argument('--init_weight', default=1., type=float, help='Weight for the init loss')
parser.add_argument('--bc_weight', default=1., type=float, help='Weight for the init loss')
parser.add_argument('--sys_weight', default=1., type=float, help='Weight for the rho loss')
parser.add_argument('--lr_init', default=1e-4, type=float, help='Starting learning rate')
parser.add_argument('--device', default='cuda', type=str, help='Device to use')
parser.add_argument('--name', default='grid', type=str, help='Experiment name')
parser.add_argument('--train_steps', default=100000, type=int, help='Number of training steps')
parser.add_argument('--epochs', default=200, type=int, help='Number of epochs')
parser.add_argument('--mode', default='Derivative', type=str, help='Mode: -1 for PINN learning, 0 for derivative learning, 1 for output learning')
parser.add_argument('--batch_size', default=32, type=int, help='Number of samples per step')
parser.add_argument('--layers', default=4, type=int, help='Number of layers in the network')
parser.add_argument('--units', default=50, type=int, help='Number of units per layer in the network')


args = parser.parse_args()
bc_weight = args.bc_weight  
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

# Export path and the type of pendulum
EXP_PATH = '.'
    
print(f'Working on {EXP_PATH}')

if not os.path.exists(f'{EXP_PATH}/{name}'):
    os.mkdir(f'{EXP_PATH}/{name}')

if name == 'grid':
    prefix = 'rho_grid'
elif name == 'full':
    prefix = 'rho_full'
elif name == 'extrapolate':
    prefix = 'rho_extrapolate'
elif name == 'interpolate':
    prefix = 'rho_interpolate'
elif name == 'adapt':
    prefix = 'rho_adapt'
else:
    raise ValueError(f'name value is not in the options')

title_mode = mode


best_params = continuity_best_params[name][str(mode)]
init_weight = best_params['init_weight']
if mode != 'PINN':
    sys_weight = best_params['sys_weight']
else:
    sys_weight = 0.
lr_init = best_params['lr_init']
bc_weight = best_params['bc_weight']
if mode == 'PINN' or mode == 'PINN+Output':
    pde_weight = best_params['pde_weight'] 
else:
    pde_weight = 0.  

#pde_weight = 1.
#sys_weight = 1.
#bc_weight = 1.
#init_weight = 1.
#lr_init = 1e-4

if name == 'grid':
    batch_size = 128
elif name == 'full':
    batch_size = 128

#batch_size = 32
print('Loading the data...')    

# Load the data
if name in ['adapt', 'interpolate', 'extrapolate']:
    train_dataset = torch.load(os.path.join('data', f'{prefix}_dataset_train.pth'))
    test_dataset = torch.load(os.path.join('data', f'{prefix}_dataset_test.pth'))
# If on the full data, there is not test set
else:
    train_dataset = test_dataset = torch.load(os.path.join('data', f'{prefix}_dataset.pth'))

init_dataset = torch.load(os.path.join('data', f'rho_init_dataset.pth'))

bc_dataset = torch.load(os.path.join('data', f'rho_bc_dataset.pth'))
#else:
#    bc_dataset = None
# Generate the dataloader
train_dataloader = DataLoader(train_dataset, batch_size, generator=gen, shuffle=True, num_workers=12)
test_dataloader = DataLoader(test_dataset, 256, generator=gen, shuffle=True, num_workers=12)
init_dataloader = DataLoader(init_dataset, batch_size, generator=gen, shuffle=True, num_workers=12)
bc_dataloader = DataLoader(bc_dataset, batch_size, generator=gen, shuffle=True, num_workers=12)

print('Data loaded!')

print(train_dataset[:][0].shape)

activation = torch.nn.Tanh()

model = DensityNet(
    init_weight=init_weight,
    bc_weight=bc_weight,
    pde_weight=pde_weight,
    sys_weight=sys_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)


step_list= []
out_losses_train = []
der_losses_train = []
pde_losses_train = []
init_losses_train = []
tot_losses_train = []
bc_losses_train = []

step_list_test = []
out_losses_test = []
der_losses_test = []
pde_losses_test = []
init_losses_test = []
tot_losses_test = []
bc_losses_test = []
times_test = []

from torch.optim import LBFGS
optim = LBFGS(model.parameters(), lr=1e-1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)


import time
def train_loop(epochs:int,
        train_dataloader:DataLoader,
        test_dataloader:DataLoader,
        init_dataloader:DataLoader,
        bc_dataloader:DataLoader,
        print_every:int=100):
    
    # Training mode for the network
    
    for epoch in range(epochs):
        model.train()
        step_prefix = epoch*min(len(train_dataloader),train_steps)
        start_time = time.time()
        print(f'Epoch: {epoch}, step_prefix: {step_prefix}')
        for step, (train_data, init_data, bc_data) in enumerate(zip(train_dataloader, cycle(init_dataloader), cycle(bc_dataloader))):
            if step > train_steps:
                break
            # Load batches from dataloaders
            x_train = train_data[0].to(device).float().requires_grad_(True)
            
            y_train = train_data[1].to(device).float()
            Dy_train = train_data[2].to(device).float()
            
            x_init = init_data[0].to(device).float()
            y_init = init_data[1].to(device).float()
            
            x_bc = bc_data[0].to(device).float()
            y_bc = bc_data[1].to(device).float()
            
            #if bc_dataset is not None:
            #    x_bc = bc_dataset[:][0].to(device).float()
            #    y_bc = bc_dataset[:][1].to(device).float()
            #else:
            #    x_bc = None
            #    y_bc = None
            
            # Call zero grad on optimizer
            model.opt.zero_grad()
            
            loss = model.loss_fn(mode=mode,
                x=x_train, y=y_train, Dy=Dy_train, x_init=x_init, y_init=y_init, x_bc=x_bc, y_bc=y_bc
            )
            # Backward the loss, calculate gradients
            loss.backward()
            # Optimizer step
            model.opt.step()
            # Update the learning rate scheduling
            
            # Printing
            if (step_prefix+step) % print_every == 0 and step>0:
                #print('Train losses')
                with torch.no_grad():
                    step_val, out_loss_train, der_loss_train, pde_loss_train, init_loss_train, tot_loss_train, bc_loss_train = model.eval_losses(
                        step=step_prefix+step, mode=mode,
                        x=x_train, y=y_train, Dy=Dy_train, x_init=x_init, y_init=y_init, x_bc=x_bc, y_bc=y_bc, print_to_screen=True
                    )
                    step_list.append(step_val)
                    tot_losses_train.append(tot_loss_train)
                    out_losses_train.append(out_loss_train)
                    der_losses_train.append(der_loss_train)
                    init_losses_train.append(init_loss_train)
                    pde_losses_train.append(pde_loss_train)
                    bc_losses_train.append(bc_loss_train)
        
                    #print(f'Step: {step_prefix+step}, Out loss: {out_loss_train}, Der loss: {der_loss_train}, PDE loss: {pde_loss_train}, Init loss: {init_loss_train}, Tot loss: {tot_loss_train}, bc loss: {bc_loss_train}')
        # Calculate and average the loss over the test dataloader
        stop_time = time.time()
        print(f'Epoch time: {stop_time-start_time}')
        epoch_time = stop_time-start_time
        times_test.append(epoch_time)
        model.eval()
        test_loss = 0.0
        out_loss_test = 0.0
        der_loss_test = 0.0
        pde_loss_test = 0.0
        init_loss_test = 0.0
        tot_loss_test = 0.0
        bc_loss_test = 0.0
        
        with torch.no_grad():
            for (test_data, init_data, bc_data) in zip(test_dataloader, cycle(init_dataloader), cycle(bc_dataloader)):
                x_test = test_data[0].to(device).float().requires_grad_(True)
                y_test = test_data[1].to(device).float()
                Dy_test = test_data[2].to(device).float()
                
                x_init = init_data[0].to(device).float()
                y_init = init_data[1].to(device).float()

                x_bc = bc_data[0].to(device).float()
                y_bc = bc_data[1].to(device).float()
                
                step_test, out_loss, der_loss, pde_loss, init_loss, tot_loss, bc_loss = model.eval_losses(step=step_prefix+step, mode=mode,
                                                                                        x=x_test, y=y_test, Dy=Dy_test, x_bc=x_bc, y_bc=y_bc, x_init=x_init, y_init=y_init)
                
                out_loss_test += out_loss.item()
                der_loss_test += der_loss.item()
                pde_loss_test += pde_loss.item()
                init_loss_test += init_loss.item()
                tot_loss_test += tot_loss.item()
                bc_loss_test += bc_loss.item()
                
                test_loss += tot_loss.item()
                
            test_loss /= len(test_dataloader)
            out_loss_test /= len(test_dataloader)
            der_loss_test /= len(test_dataloader)
            pde_loss_test /= len(test_dataloader)
            init_loss_test /= len(test_dataloader)
            tot_loss_test /= len(test_dataloader)
            bc_loss_test /= len(test_dataloader)
        
        step_list_test.append(step_test)
        out_losses_test.append(out_loss_test)
        der_losses_test.append(der_loss_test)
        pde_losses_test.append(pde_loss_test)
        init_losses_test.append(init_loss_test)
        tot_losses_test.append(tot_loss_test)
        bc_losses_test.append(bc_loss_test)
            
        print(f"Average test loss: {test_loss}")
        print(f"Average output loss: {out_loss_test}")
        print(f"Average derivative loss: {der_loss_test}")
        print(f"Average PDE loss: {pde_loss_test}")
        print(f"Average initialization loss: {init_loss_test}")
        print(f"Average total loss: {tot_loss_test}")
        print(f"Average bc loss: {bc_loss_test}")
train_loop(epochs=epochs, train_dataloader=train_dataloader, test_dataloader=test_dataloader, init_dataloader=init_dataloader, bc_dataloader=bc_dataloader, print_every=100)


torch.cuda.empty_cache()
model.eval()
# %%
import os
if not os.path.exists(f'{EXP_PATH}/{name}/saved_models'):
    os.mkdir(f'{EXP_PATH}/{name}/saved_models')
torch.save(model.state_dict(), f'{EXP_PATH}/{name}/saved_models/density_net{title_mode}')
# %%
model.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/density_net{title_mode}'))

# %%

if not os.path.exists(f'{EXP_PATH}/{name}/plots{title_mode}'):
    os.mkdir(f'{EXP_PATH}/{name}/plots{title_mode}')

import numpy as np
from matplotlib import pyplot as plt
#plotting function to generate the figures for the ball problem

with open(f'{EXP_PATH}/data/rho.npy', 'rb') as f:
    rho_true = np.load(f)

from scipy.interpolate import RegularGridInterpolator
#print(rho_true)
#from dynsys_simulator_new import dt, dx
dt = 0.005
dx = 0.01
x_max_interp = 1.5
x_min_interp = -1.5
x_steps = int((x_max_interp-x_min_interp)/dx)+1
x_vec = x_min_interp + np.arange(x_steps)*dx
y_vec = x_min_interp + np.arange(x_steps)*dx
t_vec = np.arange(0.,1.+dt,dt)
'''
x_min = -1.5
x_max = 1.5
t_max = 1.
rho_interp = RegularGridInterpolator((t_vec, x_vec, y_vec), rho_true, method='cubic')


vmax = np.max(rho_true)
vmin = 0
levels = np.linspace(vmin,vmax,100)

for t in range(10):
    # Calculate the time
    t_plot = t/10.*t_max
    print(f'Plotting time: {t_plot}')
    # Index of the time step
    t_ind = int(t_plot/dt)
    N = 500
    points_x = np.linspace(x_min,x_max,N)
    points_y = np.linspace(x_min,x_max,N)
    
    # Generate the points grid
    X,Y = np.meshgrid(points_x,points_y)
    T = t_plot*np.ones_like(X.reshape((-1)))
    pts = np.vstack([T,X.reshape(-1),Y.reshape(-1)]).T

    # Now obtain the true density 
    rho_true_plot = rho_interp(pts).reshape(X.shape)
    rho_plot = model.forward(torch.tensor(pts).to(device).float()).detach().cpu().numpy().reshape(X.shape)
    #plots the streamplot for the velocity bc
    plt.figure(figsize=(6,5))
    pcm = plt.contourf(X,Y,rho_plot,100,cmap='Greens', vmin=vmin, vmax=vmax, levels=levels)
    #print(pts)
    vel = u_vec(torch.tensor(pts[:,1:])).numpy()
    #print(vel)
    U = np.array(vel[:,0].reshape(X.shape))
    V = np.array(vel[:,1].reshape(Y.shape))
    #mask the outside of the ball
    plt.streamplot(X,Y,U,V,density=1,linewidth=0.2)
    plt.xlim((x_min-0.01,x_max + 0.01))
    plt.ylim((x_min-0.01,x_max + 0.01))
    plt.colorbar(pcm)
    #add outline for aesthetics

    plt.savefig(f'{EXP_PATH}/{name}/plots{title_mode}/density_traj{t}.png', dpi=300)
    plt.close()
    
    ### ERROR FIGURE
    plt.figure(figsize=(6,5))
    pcm = plt.contourf(X,Y,np.abs(rho_plot-rho_true_plot),100,cmap='jet')
    #print(pts)
    vel = u_vec(torch.tensor(pts[:,1:])).numpy()
    #print(vel)
    U = np.array(vel[:,0].reshape(X.shape))
    V = np.array(vel[:,1].reshape(Y.shape))
    #mask the outside of the ball
    plt.streamplot(X,Y,U,V,density=1,linewidth=0.2)
    plt.xlim((x_min-0.01,x_max + 0.01))
    plt.ylim((x_min-0.01,x_max + 0.01))
    #add outline for aesthetics
    plt.colorbar(pcm)
    plt.savefig(f'{EXP_PATH}/{name}/plots{title_mode}/error_traj{t}.png', dpi=300)
    plt.close()
'''
# Convert the losses arrays
epoch_list = torch.tensor(step_list).cpu().numpy()
out_losses_train = torch.tensor(out_losses_train).cpu().numpy()
der_losses_train = torch.tensor(der_losses_train).cpu().numpy()
pde_losses_train = torch.tensor(pde_losses_train).cpu().numpy()
init_losses_train = torch.tensor(init_losses_train).cpu().numpy()
tot_losses_train = torch.tensor(tot_losses_train).cpu().numpy()
bc_losses_train = torch.tensor(bc_losses_train).cpu().numpy()    

loss_combination_train = np.column_stack([epoch_list, out_losses_train, der_losses_train, pde_losses_train, init_losses_train, tot_losses_train, bc_losses_train])
with open(f'{EXP_PATH}/{name}/plots{title_mode}/traindata.npy', 'wb') as f:
    np.save(f, loss_combination_train)

N = 100
l = len(np.convolve(out_losses_train, np.ones(N)/N, mode='valid'))
plt.figure()
plt.plot(epoch_list[:l], np.convolve(pde_losses_train, np.ones(N)/N, mode='valid'), label='pde_loss', color='red')
plt.plot(epoch_list[:l], np.convolve(out_losses_train, np.ones(N)/N, mode='valid'), label='out_loss', color='green')
plt.plot(epoch_list[:l], np.convolve(der_losses_train, np.ones(N)/N, mode='valid'), label='der_loss', color='blue')
plt.plot(epoch_list[:l], np.convolve(init_losses_train, np.ones(N)/N, mode='valid'), label='init_loss', color='orange')
plt.plot(epoch_list[:l], np.convolve(bc_losses_train, np.ones(N)/N, mode='valid'), label='bc_loss', color='purple')
plt.legend()
plt.yscale('log')
plt.title('Losses of the student model')
plt.xlabel('Training steps')
plt.ylabel('Loss')
plt.savefig(f'{EXP_PATH}/{name}/plots{title_mode}/train_losses.png')
plt.close()


# Convert the losses arrays
epoch_list = torch.tensor(step_list_test).cpu().numpy()
out_losses_test = torch.tensor(out_losses_test).cpu().numpy()
der_losses_test = torch.tensor(der_losses_test).cpu().numpy()
pde_losses_test = torch.tensor(pde_losses_test).cpu().numpy()
init_losses_test = torch.tensor(init_losses_test).cpu().numpy()
tot_losses_test = torch.tensor(tot_losses_test).cpu().numpy()
bc_losses_test = torch.tensor(bc_losses_test).cpu().numpy()
times_test = np.array(times_test)


loss_combination_test = np.column_stack([epoch_list, out_losses_test, der_losses_test, pde_losses_test, init_losses_test, tot_losses_test, bc_losses_test, times_test])
with open(f'{EXP_PATH}/{name}/plots{title_mode}/testdata.npy', 'wb') as f:
    np.save(f, loss_combination_test)
    
plt.figure()
plt.plot(epoch_list, pde_losses_test, label='pde_loss', color='red')
plt.plot(epoch_list, out_losses_test, label='out_loss', color='green')
plt.plot(epoch_list, der_losses_test, label='der_loss', color='blue')
plt.plot(epoch_list, init_losses_test, label='init_loss', color='orange')
plt.plot(epoch_list, bc_losses_test, label='bc_loss', color='purple')
plt.legend()
plt.yscale('log')
plt.title('Losses of the student model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(f'{EXP_PATH}/{name}/plots{title_mode}/test_losses.png')