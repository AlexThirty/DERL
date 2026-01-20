import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import time
import os
seed = 1234
from itertools import cycle
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)

from models.pinn import PINN, generate_random_points, generate_boundary_points

parser = argparse.ArgumentParser()
parser.add_argument('--div_weight', default=1., type=float, help='Weight for the system pde loss')
parser.add_argument('--bc_weight', default=1., type=float, help='Weight for the boundary condition loss')
parser.add_argument('--sys_weight', default=1., type=float, help='Weight for the system pde loss')
parser.add_argument('--mom_weight', default=1., type=float, help='Weight for the PINN loss')
parser.add_argument('--lr_init', default=1e-3, type=float, help='Starting learning rate')
parser.add_argument('--device', default='cuda:2', type=str, help='Device to use')
parser.add_argument('--name', default='pinn', type=str, help='Experiment name')
parser.add_argument('--train_steps', default=1000, type=int, help='Number of training steps')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('--mode', default='Derivative', type=str, help='Mode: -1 for PINN learning, 0 for derivative learning, 1 for output learning')
parser.add_argument('--batch_size', default=10000, type=int, help='Number of samples per step')
parser.add_argument('--layers', default=4, type=int, help='Number of layers in the network')
parser.add_argument('--units', default=64, type=int, help='Number of units per layer in the network')
parser.add_argument('--grid', default=False, type=bool, help='Use grid data', action=argparse.BooleanOptionalAction)

args = parser.parse_args()
div_weight = args.div_weight
mom_weight = args.mom_weight
bc_weight = args.bc_weight
lr_init = args.lr_init
device = args.device
name = args.name
train_steps = args.train_steps
epochs = args.epochs
mode = args.mode
batch_size = args.batch_size
sys_weight = args.sys_weight
layers = args.layers
units = args.units

import pandas as pd
tuner_results_path = 'tuner_results/pinn.csv'
if os.path.exists(tuner_results_path):
    tuner_results = pd.read_csv(tuner_results_path)

    print(tuner_results.head())
    # Get the first row of the tuner results
    first_row = tuner_results.iloc[0]

    # Extract the weights from the first row
    mom_weight = first_row['config/mom_weight']
    div_weight = first_row['config/div_weight']
    bc_weight = first_row['config/bc_weight']
    lr_init = first_row['config/lr_init']

# Load the datasets in the folder
data_path = 'data/random_data.pt' if not args.grid else 'data/grid_data.pt'
internal_dataset = torch.load(data_path)
boundary_dataset = torch.load('data/boundary_data.pt')

# Now prepare the dataloaders
internal_loader = DataLoader(internal_dataset, batch_size=batch_size, shuffle=True, generator=gen)
boundary_loader = DataLoader(boundary_dataset, batch_size=batch_size, shuffle=True, generator=gen)

# Initialize the model
model = PINN(
    hidden_units=[units for _ in range(layers)],
    mom_weight=mom_weight,
    sys_weight=sys_weight,
    div_weight=div_weight,
    bc_weight=bc_weight,
    device=device
)
model.to(device)

print(model)

step_list = []
div_losses = []
mom_losses = []
y_losses = []
bc_losses = []
div_losses = []
tot_losses = []

step_list_test = []
div_losses_test = []
mom_losses_test = []
y_losses_test = []
bc_losses_test = []
div_losses_test = []
tot_losses_test = []

time_test = []

optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=100, max_eval=20, history_size=100)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.5)
# Training loop
def train_loop(epochs:int,
        internal_dataloader:DataLoader,
        boundary_dataloader:DataLoader,
        print_every:int=100):
    
    # Training mode for the network
    model.train()

    
    for epoch in range(epochs):
        
        
        start_time = time.time()
        step_prefix = epoch*len(internal_loader)
        model.train()
        for step, (pde_data, bc_data) in enumerate(zip(internal_loader, cycle(boundary_dataloader))):
            if step > train_steps:
                break
            # Load batches from dataloaders
            x_pde = pde_data[0].to(device).float().requires_grad_(True)
            y_pde = pde_data[1].to(device).float()
            Dy_pde = pde_data[2].to(device).float()

            # Boundary conditions            
            x_bc = bc_data[0].to(device).float().requires_grad_(True)
            y_bc = bc_data[1].to(device).float()
            
            
            def closure():
                optimizer.zero_grad()
                loss = model.loss_fn(
                    x_pde=x_pde, y_pde=y_pde, Dy_pde=Dy_pde,
                    x_bc=x_bc, y_bc=y_bc, mode=mode
                )
                loss.backward()
                return loss
            
            optimizer.step(closure)
            '''
            # Call zero grad on optimizer
            optimizer.zero_grad()
        
            loss = model.loss_fn(
                x_pde=x_pde, y_pde=y_pde, Dy_pde=Dy_pde,
                x_bc=x_bc, y_bc=y_bc,
            )
            # Backward the loss, calculate gradients
            loss.backward()
            # Optimizer step
            optimizer.step()
            '''
            # Printing
            if (step_prefix+step) % print_every == 0 and step>0:
                with torch.no_grad():
                    _, mom_loss, div_loss, y_loss, bc_loss_val, Dy_loss, tot_loss_val = model.eval_losses(
                        step=step_prefix+step,
                        x_pde=x_pde, y_pde=y_pde, Dy_pde=Dy_pde,
                        x_bc=x_bc, y_bc=y_bc, mode=mode
                    )
                        
                    step_list.append(step_prefix+step)
                    div_losses.append(div_loss.item())
                    mom_losses.append(mom_loss.item())
                    y_losses.append(y_loss.item())
                    bc_losses.append(bc_loss_val.item())
                    tot_losses.append(tot_loss_val.item())
                    
                    #print('Step: ', step_prefix+step)
                    #print(f'Mom loss: {mom_loss}, Div loss: {div_loss}, y loss: {y_loss}')
                    #print(f'BC loss: {bc_loss_val}, Discrepancy loss: {discrepancy_loss}, Total loss: {tot_loss_val}')
        end_time = time.time()
        
        epoch_time = end_time - start_time
        print('\n')
        print(f'Epoch: {epoch}, time: {epoch_time}')
        time_test.append(epoch_time)
        
        # Testing the model
        model.eval()
        mom_loss_test = 0.
        div_loss_test = 0.
        y_loss_test = 0.
        Dy_loss_test = 0.
        bc_loss_test = 0.
        tot_loss_test = 0.
        
        with torch.no_grad():
            for (pde_data, bc_data) in zip(internal_dataloader, cycle(boundary_dataloader)):
                # Load batches from dataloaders
                x_pde = pde_data[0].to(device).float().requires_grad_(True)
                y_pde = pde_data[1].to(device).float()
                Dy_pde = pde_data[2].to(device).float()

                # Boundary conditions            
                x_bc = bc_data[0].to(device).float().requires_grad_(True)
                y_bc = bc_data[1].to(device).float()
                
                _, mom_loss, div_loss, y_loss, bc_loss_val, Dy_loss, tot_loss_val = model.eval_losses(
                    step=step_prefix+step,
                    x_pde=x_pde, y_pde=y_pde, Dy_pde=Dy_pde,
                    x_bc=x_bc, y_bc=y_bc
                )
                
                div_loss_test += div_loss.item()
                mom_loss_test += mom_loss.item()
                y_loss_test += y_loss.item()
                bc_loss_test += bc_loss_val.item()
                tot_loss_test += tot_loss_val.item()
                Dy_loss_test += Dy_loss.item()
                
            div_loss_test /= len(internal_dataloader)
            mom_loss_test /= len(internal_dataloader)
            y_loss_test /= len(internal_dataloader)
            bc_loss_test /= len(internal_dataloader)
            tot_loss_test /= len(internal_dataloader)
            Dy_loss_test /= len(internal_dataloader)
                
                
                
        step_list_test.append(step_prefix+step)
        div_losses_test.append(div_loss_test)
        mom_losses_test.append(mom_loss_test)
        y_losses_test.append(y_loss_test)
        bc_losses_test.append(bc_loss_test)
        tot_losses_test.append(tot_loss_test)
        
        
            
        print(f'Test Mom loss: {mom_loss_test}, Test Div loss: {div_loss_test}, Test y loss: {y_loss_test}')
        print(f'Test BC loss: {bc_loss_test}, Test Total loss: {tot_loss_test}, Test Dy loss: {Dy_loss_test}')
        print('------------------------------------')            
        
train_loop(epochs, internal_loader, boundary_loader, print_every=10)

# Save the model
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')
torch.save(model.state_dict(), f'saved_models/{mode}.pt')

if not os.path.exists(f'results_{mode}'):
    os.makedirs(f'results_{mode}')

import matplotlib.pyplot as plt

step_list = np.array(step_list)
div_losses = np.array(div_losses)
mom_losses = np.array(mom_losses)
y_losses = np.array(y_losses)
bc_losses = np.array(bc_losses)
tot_losses = np.array(tot_losses)

step_list_test = np.array(step_list_test)
div_losses_test = np.array(div_losses_test)
mom_losses_test = np.array(mom_losses_test)
y_losses_test = np.array(y_losses_test)
bc_losses_test = np.array(bc_losses_test)
tot_losses_test = np.array(tot_losses_test)


train_losses = np.vstack((step_list, div_losses, mom_losses, y_losses, bc_losses, tot_losses)).T
test_losses = np.vstack((step_list_test, div_losses_test, mom_losses_test, y_losses_test, bc_losses_test, tot_losses_test)).T

np.save(f'results_{mode}/{mode}_train_losses.npy', train_losses)
np.save(f'results_{mode}/{mode}_test_losses.npy', test_losses)


plt.figure()

plt.plot(step_list, div_losses, label='Divergence Loss')
plt.plot(step_list, mom_losses, label='Momentum Loss')
plt.plot(step_list, y_losses, label='Y Loss')
plt.plot(step_list, bc_losses, label='BC Loss')
plt.plot(step_list, tot_losses, label='Total Loss')

plt.yscale('log')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig(f'results_{mode}/{mode}_training_losses.png')
plt.figure(figsize=(10, 8))

plt.plot(step_list_test, div_losses_test, label='Divergence Loss Test')
plt.plot(step_list_test, mom_losses_test, label='Momentum Loss Test')
plt.plot(step_list_test, y_losses_test, label='Y Loss Test')
plt.plot(step_list_test, bc_losses_test, label='BC Loss Test')
plt.plot(step_list_test, tot_losses_test, label='Total Loss Test')

plt.yscale('log')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Testing Losses')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig(f'results_{mode}/{mode}_testing_losses.png')