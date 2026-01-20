import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np
import random
import os
from itertools import cycle

from model import AllenNet
from tuner_results import allen_best_params


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
parser.add_argument('--step', default=30, type=float, help='Seed for the experiment')


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
seed = 30

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)

# Export path and the type of pendulum
if not os.path.exists(f'{name}'):
    os.mkdir(f'{name}')

prefix = name
if name not in ['rand', 'grid']:
    raise ValueError('Name must be either rand or grid')
title_mode = mode + f'_step_{args.step}'

best_params = allen_best_params[name][str(mode)]
if mode != 'PINN':
    sys_weight = best_params['sys_weight']
else:
    sys_weight = 0.
lr_init = best_params['lr_init']
bc_weight = best_params['bc_weight']
if mode == 'PINN':
    pde_weight = best_params['pde_weight'] 
else:
    pde_weight = 0. 

if name == 'rand':
    bc_weight = 1.
    sys_weight = 1.
    pde_weight = 1.

epochs = 100
if name == 'rand':
    batch_size = 1000
else:
    batch_size = 32
#batch_size = 32
print('Loading the data...')    


train_dataset = test_dataset = torch.load(os.path.join('data', f'{prefix}_dataset_{args.step}.pth'))
bc_dataset = torch.load(os.path.join('data', f'bc_dataset.pth'))
#else:
#    bc_dataset = None
# Generate the dataloader
train_dataloader = DataLoader(train_dataset, batch_size, generator=gen, shuffle=True)
test_dataloader = DataLoader(test_dataset, 10000, generator=gen, shuffle=True)
bc_dataloader = DataLoader(bc_dataset, batch_size, generator=gen, shuffle=True)

print('Data loaded!')

print(train_dataset[:][0].shape)

activation = torch.nn.Tanh()

model = AllenNet(
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
tot_losses_train = []
bc_losses_train = []
hes_losses_train = []

step_list_test = []
out_losses_test = []
der_losses_test = []
pde_losses_test = []
tot_losses_test = []
bc_losses_test = []
hes_losses_test = []    
times_test = []

from torch.optim import LBFGS

optim = LBFGS(model.parameters(), lr=1e-1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)

import time
def train_loop(epochs:int,
        train_dataloader:DataLoader,
        test_dataloader:DataLoader,
        bc_dataloader:DataLoader,
        print_every:int=100):
    
    # Training mode for the network
    
    for epoch in range(epochs):
        model.train()
        step_prefix = epoch*min(len(train_dataloader),train_steps)
        start_time = time.time()
        print(f'Epoch: {epoch}, step_prefix: {step_prefix}')
        for step, (train_data, bc_data) in enumerate(zip(train_dataloader, cycle(bc_dataloader))):
            if step > train_steps:
                break
            # Load batches from dataloaders
            x_train = train_data[0].to(device).float().requires_grad_(True)
            
            y_train = train_data[1].to(device).float()
            Dy_train = train_data[2].to(device).float()
            D2y_train = train_data[3].to(device).float()
            
            x_bc = bc_data[0].to(device).float()
            y_bc = bc_data[1].to(device).float()
            
            #if name == 'grid':
            # Call zero grad on optimizer
            model.opt.zero_grad()
            
            loss = model.loss_fn(mode=mode,
                x=x_train, y=y_train, Dy=Dy_train, x_bc=x_bc, y_bc=y_bc, Hy=D2y_train
            )
            # Backward the loss, calculate gradients
            #loss.backward()
            # Optimizer step
            #model.opt.step()
            # Update the learning rate scheduling
            def closure():
                optim.zero_grad()
                loss = model.loss_fn(mode=mode,
                    x=x_train, y=y_train, Dy=Dy_train, x_bc=x_bc, y_bc=y_bc, Hy=D2y_train
                )
                loss.backward()
                return loss
            optim.step(closure)
            
            # Printing
            if (step_prefix+step) % print_every == 0:
                #print('Train losses')
                with torch.no_grad():
                    step_val, out_loss_train, der_loss_train, hes_loss_train, pde_loss_train, bc_loss_train, tot_loss_train = model.eval_losses(
                        step=step_prefix+step, mode=mode,
                        x=x_train, y=y_train, Dy=Dy_train, x_bc=x_bc, y_bc=y_bc, print_to_screen=True, Hy=D2y_train
                    )
                    step_list.append(step_val)
                    tot_losses_train.append(tot_loss_train)
                    out_losses_train.append(out_loss_train)
                    der_losses_train.append(der_loss_train)
                    hes_losses_train.append(hes_loss_train)
                    pde_losses_train.append(pde_loss_train)
                    bc_losses_train.append(bc_loss_train)
        
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
        tot_loss_test = 0.0
        bc_loss_test = 0.0
        hes_loss_test = 0.0
        
        with torch.no_grad():
            for (test_data, bc_data) in zip(test_dataloader, cycle(bc_dataloader)):
                x_test = test_data[0].to(device).float().requires_grad_(True)
                y_test = test_data[1].to(device).float()
                Dy_test = test_data[2].to(device).float()
                D2y_test = test_data[3].to(device).float()

                x_bc = bc_data[0].to(device).float()
                y_bc = bc_data[1].to(device).float()
                
                step_test, out_loss, der_loss, hes_loss, pde_loss, bc_loss, tot_loss = model.eval_losses(step=step_prefix+step, mode=mode,
                                                                                        x=x_test, y=y_test, Dy=Dy_test, x_bc=x_bc, y_bc=y_bc, Hy=D2y_test)
                
                out_loss_test += out_loss.item()
                der_loss_test += der_loss.item()
                pde_loss_test += pde_loss.item()
                tot_loss_test += tot_loss.item()
                bc_loss_test += bc_loss.item()
                hes_loss_test += hes_loss.item()
                
                test_loss += tot_loss.item()
                
            test_loss /= len(test_dataloader)
            out_loss_test /= len(test_dataloader)
            der_loss_test /= len(test_dataloader)
            pde_loss_test /= len(test_dataloader)
            tot_loss_test /= len(test_dataloader)
            bc_loss_test /= len(test_dataloader)
            hes_loss_test /= len(test_dataloader)
        
        step_list_test.append(step_test)
        out_losses_test.append(out_loss_test)
        der_losses_test.append(der_loss_test)
        pde_losses_test.append(pde_loss_test)
        tot_losses_test.append(tot_loss_test)
        bc_losses_test.append(bc_loss_test)
        hes_losses_test.append(hes_loss_test)
            
        print(f"Average test loss: {test_loss}")
        print(f"Average output loss: {out_loss_test}")
        print(f"Average derivative loss: {der_loss_test}")
        print(f"Average PDE loss: {pde_loss_test}")
        print(f"Average total loss: {tot_loss_test}")
        print(f"Average bc loss: {bc_loss_test}")
        print(f"Average hessian loss: {hes_loss_test}")
train_loop(epochs=epochs, train_dataloader=train_dataloader, test_dataloader=test_dataloader, bc_dataloader=bc_dataloader, print_every=1 if name == 'rand' else 100)


torch.cuda.empty_cache()
model.eval()
# %%
import os
if not os.path.exists(f'{name}/saved_models'):
    os.mkdir(f'{name}/saved_models')
torch.save(model.state_dict(), f'{name}/saved_models/allen_net{title_mode}')
# %%
model.load_state_dict(torch.load(f'{name}/saved_models/allen_net{title_mode}'))

# %%

if not os.path.exists(f'{name}/plots{title_mode}'):
    os.mkdir(f'{name}/plots{title_mode}')

import numpy as np
from matplotlib import pyplot as plt
from model import allen_cahn_true
# Generate the grid for the true solution
xmin = -1.
xmax = 1.
dx = 0.01
x = np.arange(xmin, xmax+dx, dx)
y = np.arange(xmin, xmax+dx, dx)
x_pts, y_pts = np.meshgrid(x, y)
pts = np.column_stack((x_pts.reshape((-1,1)), y_pts.reshape((-1,1))))
u_grid = allen_cahn_true(torch.tensor(pts)).reshape((-1,1)).reshape(x_pts.shape)

u_pred = model.forward(torch.tensor(pts).to(device).float()).detach().cpu().numpy().reshape(x_pts.shape)

u_err = np.abs(u_grid - u_pred)

from matplotlib import cm

# Plot the predicted solution
fig, ax = plt.subplots()
im = ax.imshow(u_pred, cmap=cm.jet)
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.colorbar(im, ax=ax, orientation='vertical')
fig.suptitle('Predicted solution')
plt.savefig(f'{name}/plots{title_mode}/pred_solution.png')
plt.close()

# Plot the error wrt the true solution
fig, ax = plt.subplots()
im = ax.imshow(u_err, cmap=cm.jet)
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.colorbar(im, ax=ax, orientation='vertical')
fig.suptitle('Error of the predicted solution')
plt.savefig(f'{name}/plots{title_mode}/error.png')
plt.close()


# Convert the losses arrays
epoch_list = torch.tensor(step_list).cpu().numpy()
out_losses_train = torch.tensor(out_losses_train).cpu().numpy()
der_losses_train = torch.tensor(der_losses_train).cpu().numpy()
pde_losses_train = torch.tensor(pde_losses_train).cpu().numpy()
tot_losses_train = torch.tensor(tot_losses_train).cpu().numpy()
bc_losses_train = torch.tensor(bc_losses_train).cpu().numpy()    
hes_losses_train = torch.tensor(hes_losses_train).cpu().numpy()

loss_combination_train = np.column_stack([epoch_list, out_losses_train, der_losses_train, pde_losses_train, bc_losses_train, tot_losses_train, hes_losses_train])
with open(f'{name}/plots{title_mode}/traindata.npy', 'wb') as f:
    np.save(f, loss_combination_train)

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
plt.savefig(f'{name}/plots{title_mode}/train_losses.png')
plt.close()


# Convert the losses arrays
epoch_list = torch.tensor(step_list_test).cpu().numpy()
out_losses_test = torch.tensor(out_losses_test).cpu().numpy()
der_losses_test = torch.tensor(der_losses_test).cpu().numpy()
pde_losses_test = torch.tensor(pde_losses_test).cpu().numpy()
tot_losses_test = torch.tensor(tot_losses_test).cpu().numpy()
bc_losses_test = torch.tensor(bc_losses_test).cpu().numpy()
hes_losses_test = torch.tensor(hes_losses_test).cpu().numpy()
times_test = np.array(times_test)


loss_combination_test = np.column_stack([epoch_list, out_losses_test, der_losses_test, pde_losses_test, bc_losses_test, tot_losses_test, times_test, hes_losses_test])
with open(f'{name}/plots{title_mode}/testdata.npy', 'wb') as f:
    np.save(f, loss_combination_test)
    
plt.figure()
plt.plot(epoch_list, pde_losses_test, label='pde_loss', color='red')
plt.plot(epoch_list, out_losses_test, label='out_loss', color='green')
plt.plot(epoch_list, der_losses_test, label='der_loss', color='blue')
plt.plot(epoch_list, bc_losses_test, label='bc_loss', color='purple')
plt.legend()
plt.yscale('log')
plt.title('Losses of the student model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(f'{name}/plots{title_mode}/test_losses.png')