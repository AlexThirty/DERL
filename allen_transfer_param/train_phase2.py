import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np
import random
import os
seed = 30
from itertools import cycle

from model import AllenNet
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)
from torch.optim import LBFGS

parser = argparse.ArgumentParser()
parser.add_argument('--bc_weight', default=1., type=float, help='Weight for the BC loss')
parser.add_argument('--sys_weight', default=1., type=float, help='Weight for the rho loss')
parser.add_argument('--pde_weight', default=1., type=float, help='Weight for the PDE loss')
parser.add_argument('--lr_init', default=1e-3, type=float, help='Starting learning rate')
parser.add_argument('--device', default='cuda', type=str, help='Device to use')
parser.add_argument('--name', default='grid', type=str, help='Experiment name')
parser.add_argument('--train_steps', default=100000, type=int, help='Number of training steps')
parser.add_argument('--epochs', default=200, type=int, help='Number of epochs')
parser.add_argument('--mode', default='Derivative', type=str, help='Mode: -1 for PINN learning, 0 for derivative learning, 1 for output learning')
parser.add_argument('--layers', default=4, type=int, help='Number of layers in the network')
parser.add_argument('--units', default=50, type=int, help='Number of units per layer in the network')


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

# Export path and the type of pendulum

prefix = name
title_mode = mode  


batch_size = 100
if mode == 'Forgetting':
    batch_size = 100
#batch_size = 32
print('Loading the data...')    


train_dataset = torch.load(os.path.join('val_data', 'val_data_multi_rand.pt'), weights_only=False)
distill_dataset = torch.load(os.path.join('val_data', 'train_distillation_dataset.pt'), weights_only=False)
eval_dataset = torch.load(os.path.join('val_data', 'joint_data_multi_rand.pt'), weights_only=False)
bc_dataset = torch.load(os.path.join('val_data', f'boundary_val_data.pt'), weights_only=False)

if mode == 'PINN' and name == 'extend':
    # Take only 20% of the eval_dataset and init_dataset in random order
    def subset_dataset(dataset, fraction=0.2):
        n = len(dataset)
        indices = torch.randperm(n)[:int(n * fraction)]
        return torch.utils.data.Subset(dataset, indices)
    distill_dataset = subset_dataset(distill_dataset, 0.2)
#else:
#    bc_dataset = None
# Generate the dataloader
train_dataloader = DataLoader(train_dataset, batch_size, generator=gen, shuffle=True)
distill_dataloader = DataLoader(distill_dataset, batch_size, generator=gen, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, 10000, generator=gen, shuffle=True)
bc_dataloader = DataLoader(bc_dataset, batch_size, generator=gen, shuffle=True)

print('Data loaded!')
activation = torch.nn.Tanh()

model = AllenNet(
    in_dim=4,
    bc_weight=bc_weight,
    pde_weight=pde_weight,
    sys_weight=sys_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)

optim = LBFGS(model.parameters(), lr=1, max_iter=20, max_eval=None, history_size=100)
print(train_dataset[:][0].shape)
print(distill_dataset[:][0].shape)

if name=='extend':
    model.load_state_dict(torch.load('saved_models/pinn_phase1'))
    epochs = int(epochs/2)

step_list= []
out_losses_train = []
der_losses_train = []
pde_losses_train = []
tot_losses_train = []
bc_losses_train = []
hes_losses_train = []

step_list_val = []
out_losses_val = []
der_losses_val = []
pde_losses_val = []
tot_losses_val = []
bc_losses_val = []
hes_losses_val = []    
times_val = []

epoch_lim = -1 if name == 'extend' else epochs/2
import time
def train_loop(epochs:int,
        train_dataloader:DataLoader,
        eval_dataloader:DataLoader,
        distill_dataloader:DataLoader,
        boundary_dataloader:DataLoader,
        print_every:int=100):
    
    # Training mode for the network
    
    for epoch in range(epochs):
        model.train()
        step_prefix = epoch*min(len(distill_dataloader),train_steps)
        start_time = time.time()
        print(f'Epoch: {epoch}, step_prefix: {step_prefix}')
        if epoch > epoch_lim:
            step = 0
            distill_data = distill_dataset[:]
            train_data = train_dataset[:]
            bc_data = bc_dataset[:]
            
            # Load batches from dataloaders
            x_train = train_data[0].to(device).float().requires_grad_(True)
            
            y_train = train_data[1].to(device).float()
            Dy_train = train_data[2].to(device).float()
            D2y_train = train_data[3].to(device).float()
            
            x_bc = bc_data[0].to(device).float()
            y_bc = bc_data[1].to(device).float()
            
            x_distill = distill_data[0].to(device).float().requires_grad_(True)
            y_distill = distill_data[1].to(device).float()
            Dy_distill = distill_data[2].to(device).float()[:,0,:2]
            D2y_distill = distill_data[3].to(device).float()[:,0,:2,:2]
            
            def closure():
                model.opt.zero_grad()
                loss = model.loss_fn_phase2(
                    x_new=x_train, x=x_distill, y=y_distill, Dy=Dy_distill, Hy=D2y_distill, x_bc=x_bc, y_bc=y_bc, mode=mode
                )
                loss.backward()
                return loss
            
            optim.step(closure)

            # Printing
            with torch.no_grad():
                step_val, out_loss_train, der_loss_train, hes_loss_train, pde_loss_train, bc_loss_train, tot_loss_train = model.eval_losses_phase2(
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
            
        else:
            for step, (distill_data, train_data, bc_data) in enumerate(zip(cycle(distill_dataloader), train_dataloader, cycle(boundary_dataloader))):
                if step > train_steps:
                    break
                # Load batches from dataloaders
                x_train = train_data[0].to(device).float().requires_grad_(True)
                
                y_train = train_data[1].to(device).float()
                Dy_train = train_data[2].to(device).float()
                D2y_train = train_data[3].to(device).float()
                
                x_bc = bc_data[0].to(device).float()
                y_bc = bc_data[1].to(device).float()
                
                x_distill = distill_data[0].to(device).float().requires_grad_(True)
                y_distill = distill_data[1].to(device).float()
                Dy_distill = distill_data[2].to(device).float()[:,0,:2]
                D2y_distill = distill_data[3].to(device).float()[:,0,:2,:2]

                # Call zero grad on optimizer
                model.opt.zero_grad()
                
                loss = model.loss_fn_phase2(
                    x_new=x_train, x=x_distill, y=y_distill, Dy=Dy_distill, Hy=D2y_distill, x_bc=x_bc, y_bc=y_bc, mode=mode
                )
                # Backward the loss, calculate gradients
                loss.backward()
                # Optimizer step
                model.opt.step()
                # Update the learning rate scheduling
                
                # Printing
                if (step_prefix+step) % print_every == 0:
                    #print('Train losses')
                    with torch.no_grad():
                        step_val, out_loss_train, der_loss_train, hes_loss_train, pde_loss_train, bc_loss_train, tot_loss_train = model.eval_losses_phase2(
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
            
        # Calculate and average the loss over the val dataloader
        stop_time = time.time()
        print(f'Epoch time: {stop_time-start_time}')
        epoch_time = stop_time-start_time
        times_val.append(epoch_time)
        model.eval()
        val_loss = 0.0
        out_loss_val = 0.0
        der_loss_val = 0.0
        pde_loss_val = 0.0
        tot_loss_val = 0.0
        bc_loss_val = 0.0
        hes_loss_val = 0.0
        
        with torch.no_grad():
            for (val_data, bc_data) in zip(eval_dataloader, cycle(bc_dataloader)):
                x_val = val_data[0].to(device).float().requires_grad_(True)
                y_val = val_data[1].to(device).float()
                Dy_val = val_data[2].to(device).float()
                D2y_val = val_data[3].to(device).float()

                x_bc = bc_data[0].to(device).float()
                y_bc = bc_data[1].to(device).float()
                
                step_val, out_loss, der_loss, hes_loss, pde_loss, bc_loss, tot_loss = model.eval_losses_phase1(step=step_prefix+step,
                                                                                        x=x_val, y=y_val, Dy=Dy_val, x_bc=x_bc, y_bc=y_bc, Hy=D2y_val)
                
                out_loss_val += out_loss.item()
                der_loss_val += der_loss.item()
                pde_loss_val += pde_loss.item()
                tot_loss_val += tot_loss.item()
                bc_loss_val += bc_loss.item()
                hes_loss_val += hes_loss.item()
                
                val_loss += tot_loss.item()
                
            val_loss /= len(eval_dataloader)
            out_loss_val /= len(eval_dataloader)
            der_loss_val /= len(eval_dataloader)
            pde_loss_val /= len(eval_dataloader)
            tot_loss_val /= len(eval_dataloader)
            bc_loss_val /= len(eval_dataloader)
            hes_loss_val /= len(eval_dataloader)
        
        step_list_val.append(step_val)
        out_losses_val.append(out_loss_val)
        der_losses_val.append(der_loss_val)
        pde_losses_val.append(pde_loss_val)
        tot_losses_val.append(tot_loss_val)
        bc_losses_val.append(bc_loss_val)
        hes_losses_val.append(hes_loss_val)
            
        print(f"Average val loss: {val_loss}")
        print(f"Average output loss: {out_loss_val}")
        print(f"Average derivative loss: {der_loss_val}")
        print(f"Average PDE loss: {pde_loss_val}")
        print(f"Average total loss: {tot_loss_val}")
        print(f"Average bc loss: {bc_loss_val}")
        print(f"Average hessian loss: {hes_loss_val}")
train_loop(epochs=epochs, train_dataloader=train_dataloader, distill_dataloader=distill_dataloader, eval_dataloader=eval_dataloader, boundary_dataloader=bc_dataloader, print_every=10)


torch.cuda.empty_cache()
model.eval()
# %%
import os
if not os.path.exists(f'saved_models'):
    os.mkdir(f'saved_models')
torch.save(model.state_dict(), f'saved_models/pinn_{name}_{mode}')
# %%
model.load_state_dict(torch.load(f'saved_models/pinn_{name}_{mode}'))

# %%

if not os.path.exists(f'{name}'):
    os.mkdir(f'{name}')
    
if not os.path.exists(f'{name}/plots_{mode}'):
    os.mkdir(f'{name}/plots_{mode}')


import numpy as np
from matplotlib import pyplot as plt

# Convert the losses arrays
epoch_list = torch.tensor(step_list).cpu().numpy()
out_losses_train = torch.tensor(out_losses_train).cpu().numpy()
der_losses_train = torch.tensor(der_losses_train).cpu().numpy()
pde_losses_train = torch.tensor(pde_losses_train).cpu().numpy()
tot_losses_train = torch.tensor(tot_losses_train).cpu().numpy()
bc_losses_train = torch.tensor(bc_losses_train).cpu().numpy()    
hes_losses_train = torch.tensor(hes_losses_train).cpu().numpy()

loss_combination_train = np.column_stack([epoch_list, out_losses_train, der_losses_train, pde_losses_train, bc_losses_train, tot_losses_train, hes_losses_train])
with open(f'{name}/plots_{mode}/traindata.npy', 'wb') as f:
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
plt.savefig(f'{name}/plots_{mode}/train_losses.png')
plt.close()


# Convert the losses arrays
epoch_list = torch.tensor(step_list_val).cpu().numpy()
out_losses_val = torch.tensor(out_losses_val).cpu().numpy()
der_losses_val = torch.tensor(der_losses_val).cpu().numpy()
pde_losses_val = torch.tensor(pde_losses_val).cpu().numpy()
tot_losses_val = torch.tensor(tot_losses_val).cpu().numpy()
bc_losses_val = torch.tensor(bc_losses_val).cpu().numpy()
hes_losses_val = torch.tensor(hes_losses_val).cpu().numpy()
times_val = np.array(times_val)


loss_combination_val = np.column_stack([epoch_list, out_losses_val, der_losses_val, pde_losses_val, bc_losses_val, tot_losses_val, times_val, hes_losses_val])
with open(f'{name}/plots_{mode}/valdata.npy', 'wb') as f:
    np.save(f, loss_combination_val)
    
plt.figure()
plt.plot(epoch_list, pde_losses_val, label='pde_loss', color='red')
plt.plot(epoch_list, out_losses_val, label='out_loss', color='green')
plt.plot(epoch_list, der_losses_val, label='der_loss', color='blue')
plt.plot(epoch_list, bc_losses_val, label='bc_loss', color='purple')
plt.legend()
plt.yscale('log')
plt.title('Losses of the student model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(f'{name}/plots_{mode}/val_losses.png')