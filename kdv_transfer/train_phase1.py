import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import os
seed = 30
from itertools import cycle

from model import KdVPINN
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)
from torch.optim import LBFGS

parser = argparse.ArgumentParser()
parser.add_argument('--bc_weight', default=0.001, type=float, help='Weight for the BC loss')
parser.add_argument('--sys_weight', default=1., type=float, help='Weight for the rho loss')
parser.add_argument('--pde_weight', default=1., type=float, help='Weight for the PDE loss')
parser.add_argument('--lr_init', default=1e-3, type=float, help='Starting learning rate')
parser.add_argument('--device', default='cuda', type=str, help='Device to use')
parser.add_argument('--name', default='new', type=str, help='Experiment name')
parser.add_argument('--train_steps', default=1000, type=int, help='Number of training steps')
parser.add_argument('--epochs', default=30, type=int, help='Number of epochs')
parser.add_argument('--mode', default='Derivative', type=str, help='Mode: -1 for PINN learning, 0 for derivative learning, 1 for output learning')
parser.add_argument('--layers', default=9, type=int, help='Number of layers in the network')
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


batch_size = 1000

#batch_size = 32
print('Loading the data...')    
save_dir = f'plots_phase1_{mode}_{name}'

# Generate the times
t = 0.35*torch.distributions.Uniform(0.,1.).sample((batch_size*train_steps,1)) + 0.35
# Generate the point
points = torch.distributions.Uniform(-1.,1.).sample((batch_size*train_steps, 1))
# Stack them
x_pde = torch.column_stack((t, points))

train_dataset = torch.utils.data.TensorDataset(x_pde) 

# Generate the times
#t =  t_max*torch.distributions.Uniform(0.,1.).sample((num_boundary,1))
# Generate the points
t = 0.35*torch.distributions.Uniform(0.,1.).sample((batch_size*train_steps,1)) + 0.35
#points_bc1 = -torch.ones((num_boundary,1))
points_bc1 = -torch.ones_like(t)
# Stack them
x_bc1 = torch.column_stack((t, points_bc1))
print(f'x_bc.shape: {x_bc1.shape}')
# Fake target
y_bc1 = torch.zeros_like(t).reshape((-1,1))
print(f'y_bc.shape: {y_bc1.shape}')
# Generate the dataset
bc1_dataset = TensorDataset(x_bc1, y_bc1)

# Generate the times
#t =  t_max*torch.distributions.Uniform(0.,1.).sample((num_boundary,1))
t = 0.35*torch.distributions.Uniform(0.,1.).sample((batch_size*train_steps,1)) + 0.35
# Generate the points
#points_bc2 = torch.ones((num_boundary, 1))
points_bc2 = torch.ones_like(t)
# Stack them
x_bc2 = torch.column_stack((t, points_bc2))
print(f'x_bc.shape: {x_bc2.shape}')
# Fake target
y_bc2 = torch.zeros_like(t).reshape((-1,1))
print(f'y_bc.shape: {y_bc2.shape}')
# Generate the dataset
bc2_dataset = TensorDataset(x_bc2, y_bc2)

eval_dataset = torch.load('data/kdv_pde_dataset.pth', weights_only=False)
init_dataset = torch.load('data/kdv_init_dataset.pth', weights_only=False)
distillation_dataset = torch.load('data/dataset_distillation_phase0.pth', weights_only=False)


if mode == 'PINN' and name == 'extend':
    # Take only 20% of the eval_dataset and init_dataset in random order
    def subset_dataset(dataset, fraction=0.2):
        n = len(dataset)
        indices = torch.randperm(n)[:int(n * fraction)]
        return torch.utils.data.Subset(dataset, indices)
    distillation_dataset = subset_dataset(distillation_dataset, 0.2)
#else:
#    bc_dataset = None
# Generate the dataloader
train_dataloader = DataLoader(train_dataset, batch_size, generator=gen, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, 2048, generator=gen, shuffle=True)
bc1_dataloader = DataLoader(bc1_dataset, batch_size, generator=gen, shuffle=True)
bc2_dataloader = DataLoader(bc2_dataset, batch_size, generator=gen, shuffle=True)
init_dataloader = DataLoader(init_dataset, batch_size, generator=gen, shuffle=True)
distillation_dataloader = DataLoader(distillation_dataset, batch_size, generator=gen, shuffle=True)
print('Data loaded!')
activation = torch.nn.Tanh()

model = KdVPINN(
    bc_weight=bc_weight,
    pde_weight=pde_weight,
    sys_weight=sys_weight,
    distillation_weight=1.,
    hidden_units=[units for _ in range(layers)],
    device=device,
    activation=activation,    
    last_activation=False,
    lr=lr_init,
    init_weight=10.
).to(device)

if name=='extend':
    model.load_state_dict(torch.load('saved_models/pinn_phase0'))


step_list= []
out_losses_train = []
pde_losses_train = []
tot_losses_train = []
init_losses_train = []
bc_losses_train = []
distillation_losses_train = []

step_list_test = []
out_losses_test = []
pde_losses_test = []
tot_losses_test = []
bc_losses_test = []
init_losses_test = []
distillation_losses_test = []
times_test = []

#optim = LBFGS(model.parameters(), lr=1e-1)

optim = torch.optim.Adam(model.parameters(), lr=lr_init)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)
import time
def train_loop(epochs:int,
        train_dataloader:DataLoader,
        eval_dataloader:DataLoader,
        init_dataloader:DataLoader,
        bc1_dataloader:DataLoader,
        bc2_dataloader:DataLoader,
        distillation_dataloader:DataLoader,
        print_every:int=100):
    
    # Training mode for the network
    
    for epoch in range(epochs):
        model.train()
        step_prefix = epoch*min(len(train_dataloader),train_steps)
        print(f'Epoch: {epoch}, step_prefix: {step_prefix}')
        for step, (train_data, eval_data, bc1_data, bc2_data, init_data, distill_data) in enumerate(zip(train_dataloader, cycle(eval_dataloader), cycle(bc1_dataloader), cycle(bc2_dataloader), cycle(init_dataloader), cycle(distillation_dataloader))):
            if step > train_steps:
                break
            start_time = time.time()
            # Load batches from dataloaders
            x_pde = train_data[0].to(device).float().requires_grad_(True)
            
            
            x_bc1 = bc1_data[0].to(device).float()
            y_bc1 = bc1_data[1].to(device).float()
            x_bc2 = bc2_data[0].to(device).float()
            y_bc2 = bc2_data[1].to(device).float()
            x_init = init_data[0].to(device).float()
            y_init = init_data[1].to(device).float()

            x_eval = eval_data[0].to(device).float()
            y_eval = eval_data[1].to(device).float()

            distill_x = distill_data[0].to(device).float()
            distill_y = distill_data[1].to(device).float()
            distill_Dy = distill_data[2].to(device).float()
            '''
            def closure():
                model.opt.zero_grad()
                loss = model.loss_fn(
                    x_pde=x_pde, x_bc1=x_bc1, x_bc2=x_bc2, x_init=x_init, y_init=y_init, x_distill=distill_x, y_distill=distill_y, Dy_distill=distill_Dy, mode=mode
                )
                loss.backward()
                return loss
            
            optim.step(closure)
            '''
            # Call zero grad on optimizer
            model.opt.zero_grad()
            
            loss = model.loss_fn(
                    x_pde=x_pde, x_bc1=x_bc1, x_bc2=x_bc2, x_init=x_init, y_init=y_init, x_distill=distill_x, y_distill=distill_y, Dy_distill=distill_Dy, mode=mode
            )
            # Backward the loss, calculate gradients
            loss.backward()
            # Optimizer step
            #model.opt.step()
            # Update the learning rate scheduling
            optim.step()
            # Printing
            if (step_prefix+step) % print_every == 0:                #print('Train losses')
                with torch.no_grad():
                    step_val, out_loss_train, pde_loss_train, init_loss_train, bc_loss_train, distillation_loss_train, tot_loss_train = model.eval_losses(
                        step=step_prefix+step, x_pde=x_eval, y_pde=y_eval, x_bc1=x_bc1, x_bc2=x_bc2, x_init=x_init, y_init=y_init, print_to_screen=True,
                        x_distill=distill_x, y_distill=distill_y, Dy_distill=distill_Dy, mode=mode
                    )
                    step_list.append(step_val)
                    tot_losses_train.append(tot_loss_train)
                    out_losses_train.append(out_loss_train)
                    pde_losses_train.append(pde_loss_train)
                    init_losses_train.append(init_loss_train)
                    bc_losses_train.append(bc_loss_train)
                    distillation_losses_train.append(distillation_loss_train)

            # Calculate and average the loss over the test dataloader
        stop_time = time.time()
        print(f'Epoch time: {stop_time-start_time}')
        epoch_time = stop_time-start_time
        times_test.append(epoch_time)
        model.eval()
        test_loss = 0.0
        out_loss_test = 0.0
        pde_loss_test = 0.0
        tot_loss_test = 0.0
        bc_loss_test = 0.0
        init_loss_test = 0.0
        distillation_loss_test = 0.0

        with torch.no_grad():
            for (eval_data, bc1_data, bc2_data, init_data, distill_data) in zip(eval_dataloader, cycle(bc1_dataloader), cycle(bc2_dataloader), cycle(init_dataloader), cycle(distillation_dataloader)):
                x_eval = eval_data[0].to(device).float()
                y_eval = eval_data[1].to(device).float()

                x_bc1 = bc1_data[0].to(device).float()
                y_bc1 = bc1_data[1].to(device).float()
                x_bc2 = bc2_data[0].to(device).float()
                y_bc2 = bc2_data[1].to(device).float()
                x_init = init_data[0].to(device).float()
                y_init = init_data[1].to(device).float()

                x_distill = distill_data[0].to(device).float()
                y_distill = distill_data[1].to(device).float()
                Dy_distill = distill_data[2].to(device).float()

                step_test, out_loss, pde_loss, init_loss, bc_loss, distillation_loss, tot_loss = model.eval_losses(
                    step=step_prefix+step,
                    x_pde=x_eval, y_pde=y_eval, x_bc1=x_bc1, x_bc2=x_bc2, x_init=x_init, y_init=y_init, print_to_screen=False,
                    x_distill=x_distill, y_distill=y_distill, Dy_distill=Dy_distill, mode=mode
                )

                out_loss_test += out_loss.item()
                pde_loss_test += pde_loss.item()
                tot_loss_test += tot_loss.item()
                bc_loss_test += bc_loss.item()
                init_loss_test += init_loss.item()
                distillation_loss_test += distillation_loss.item()
                test_loss += tot_loss.item()

            test_loss /= len(eval_dataloader)
            out_loss_test /= len(eval_dataloader)
            pde_loss_test /= len(eval_dataloader)
            tot_loss_test /= len(eval_dataloader)
            bc_loss_test /= len(eval_dataloader)
            init_loss_test /= len(eval_dataloader)
            distillation_loss_test /= len(eval_dataloader)

        step_list_test.append(step_test)
        out_losses_test.append(out_loss_test)
        pde_losses_test.append(pde_loss_test)
        tot_losses_test.append(tot_loss_test)
        bc_losses_test.append(bc_loss_test)
        init_losses_test.append(init_loss_test)
        distillation_losses_test.append(distillation_loss_test)

        print(f"Average test loss: {test_loss}")
        print(f"Average output loss: {out_loss_test}")
        print(f"Average PDE loss: {pde_loss_test}")
        print(f"Average total loss: {tot_loss_test}")
        print(f"Average bc loss: {bc_loss_test}")
        print(f"Average init loss: {init_loss_test}")
        print(f"Average distillation loss: {distillation_loss_test}")
            
        if (epoch+1) % 5 == 0:
            scheduler.step()
train_loop(epochs=epochs, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, bc1_dataloader=bc1_dataloader, bc2_dataloader=bc2_dataloader, init_dataloader=init_dataloader, distillation_dataloader=distillation_dataloader, print_every=100)


from plotting_utils import plot_errors

torch.cuda.empty_cache()
model.eval()
# %%
import os
if not os.path.exists(f'saved_models'):
    os.mkdir(f'saved_models')
torch.save(model.state_dict(), f'saved_models/pinn_phase1_{mode}_{name}')
# %%
model.load_state_dict(torch.load(f'saved_models/pinn_phase1_{mode}_{name}'))

# %%
if not os.path.exists(f'{save_dir}'):
    os.mkdir(f'{save_dir}')
kdv_data = np.load('data/kdv_data.npy')

if not os.path.exists(f'{save_dir}'):
    os.mkdir(f'{save_dir}')
kdv_data = np.load('data/kdv_data.npy')

model_phase0 = KdVPINN(
    bc_weight=bc_weight,
    pde_weight=pde_weight,
    sys_weight=sys_weight,
    distillation_weight=1.,
    hidden_units=[units for _ in range(layers)],
    device=device,
    activation=activation,    
    last_activation=False,
    lr=lr_init,
    init_weight=1.
).to(device)
model_phase0.load_state_dict(torch.load('saved_models/pinn_phase0'))

from plotting_utils import plot_errors
plot_errors([model, model_phase0], ['Phase 1', 'Phase 0'], f'{save_dir}', kdv_data, model, compare=False)

import numpy as np
from matplotlib import pyplot as plt

# Convert the losses arrays
epoch_list = torch.tensor(step_list).cpu().numpy()
out_losses_train = torch.tensor(out_losses_train).cpu().numpy()
pde_losses_train = torch.tensor(pde_losses_train).cpu().numpy()
tot_losses_train = torch.tensor(tot_losses_train).cpu().numpy()
bc_losses_train = torch.tensor(bc_losses_train).cpu().numpy()    
init_losses_train = torch.tensor(init_losses_train).cpu().numpy()
distillation_losses_train = torch.tensor(distillation_losses_train).cpu().numpy()

print(epoch_list.shape, out_losses_train.shape, pde_losses_train.shape, bc_losses_train.shape, tot_losses_train.shape, init_losses_train.shape, distillation_losses_train.shape)

loss_combination_train = np.column_stack([
    epoch_list, out_losses_train, pde_losses_train, bc_losses_train, tot_losses_train, init_losses_train, distillation_losses_train
])
with open(f'{save_dir}/traindata.npy', 'wb') as f:
    np.save(f, loss_combination_train)

N = 1
l = len(np.convolve(out_losses_train, np.ones(N)/N, mode='valid'))
plt.figure()
plt.plot(epoch_list[:l], np.convolve(pde_losses_train, np.ones(N)/N, mode='valid'), label='pde_loss', color='red')
plt.plot(epoch_list[:l], np.convolve(out_losses_train, np.ones(N)/N, mode='valid'), label='out_loss', color='green')
plt.plot(epoch_list[:l], np.convolve(bc_losses_train, np.ones(N)/N, mode='valid'), label='bc_loss', color='purple')
plt.plot(epoch_list[:l], np.convolve(init_losses_train, np.ones(N)/N, mode='valid'), label='init_loss', color='orange')
plt.plot(epoch_list[:l], np.convolve(distillation_losses_train, np.ones(N)/N, mode='valid'), label='distillation_loss', color='brown')
plt.legend()
plt.yscale('log')
plt.title('Losses of the student model')
plt.xlabel('Training steps')
plt.ylabel('Loss')
plt.savefig(f'{save_dir}/train_losses.png')
plt.close()

# Convert the losses arrays
epoch_list = torch.tensor(step_list_test).cpu().numpy()
out_losses_test = torch.tensor(out_losses_test).cpu().numpy()
pde_losses_test = torch.tensor(pde_losses_test).cpu().numpy()
tot_losses_test = torch.tensor(tot_losses_test).cpu().numpy()
bc_losses_test = torch.tensor(bc_losses_test).cpu().numpy()
init_losses_test = torch.tensor(init_losses_test).cpu().numpy()
distillation_losses_test = torch.tensor(distillation_losses_test).cpu().numpy()
times_test = np.array(times_test)


print(epoch_list.shape, out_losses_test.shape, pde_losses_test.shape, bc_losses_test.shape, tot_losses_test.shape, init_losses_test.shape, distillation_losses_test.shape, times_test.shape)
loss_combination_test = np.column_stack([
    epoch_list, out_losses_test, pde_losses_test, bc_losses_test, tot_losses_test, init_losses_test, distillation_losses_test, times_test
])
with open(f'{save_dir}/testdata.npy', 'wb') as f:
    np.save(f, loss_combination_test)
    
plt.figure()
plt.plot(epoch_list, pde_losses_test, label='pde_loss', color='red')
plt.plot(epoch_list, out_losses_test, label='out_loss', color='green')
plt.plot(epoch_list, bc_losses_test, label='bc_loss', color='purple')
plt.plot(epoch_list, init_losses_test, label='init_loss', color='orange')
plt.plot(epoch_list, distillation_losses_test, label='distillation_loss', color='brown')
plt.legend()
plt.yscale('log')
plt.title('Losses of the student model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(f'{save_dir}/test_losses.png')
plt.close()

from torch.utils.data import TensorDataset
from torch.func import vmap, jacrev, hessian

# Empty cache just to be sure
torch.cuda.empty_cache()
# Create tensors for I/O
pde_x = []
pde_y = []
pde_Dy = []
pde_Hy = []

for train_data in train_dataloader:
    x = train_data[0]
    pde_x.append(x)
    pde_y.append(model.forward(x.to(device).float()).detach().cpu())
    pde_Dy.append(vmap(jacrev(model.forward_single))(x.to(device).float()).detach().cpu())
    pde_Hy.append(vmap(hessian(model.forward_single))(x.to(device).float()).detach().cpu())
    torch.cuda.empty_cache()

pde_x = torch.cat(pde_x)
pde_y = torch.cat(pde_y)
pde_Dy = torch.cat(pde_Dy)
pde_Hy = torch.cat(pde_Hy)

print(f'pde_x.shape: {pde_x.shape}')
print(f'pde_y.shape: {pde_y.shape}')
print(f'pde_Dy.shape: {pde_Dy.shape}')
print(f'pde_Hy.shape: {pde_Hy.shape}')
pde_distillation_dataset = TensorDataset(pde_x, pde_y, pde_Dy, pde_Hy)
torch.save(pde_distillation_dataset, os.path.join(save_dir, f'phase1_distillation_data.pth'))
print('Created PDE distillation dataset')