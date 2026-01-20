import torch
from model_derl import KdVPINN, SinActivation
import argparse
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import os
from torch.func import vmap, jacrev, hessian
from itertools import cycle
from tuner_results import kdv_best_params

seed = 30
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--init_weight', default=1., type=float, help='Weight for the init loss')
parser.add_argument('--pde_weight', default=1., type=float, help='Weight for the F loss')
parser.add_argument('--bc_weight', default=1., type=float, help='Weight for the F loss')
parser.add_argument('--sys_weight', default=1., type=float, help='Weight for the F loss')
parser.add_argument('--lr_init', default=1e-3, type=float, help='Starting learning rate')
parser.add_argument('--device', default='cuda:2', type=str, help='Device to use')
parser.add_argument('--name', default='pinn', type=str, help='Experiment name')
parser.add_argument('--train_steps', default=100000, type=int, help='Number of training steps')
parser.add_argument('--epochs', default=200, type=int, help='Number of epochs')
parser.add_argument('--batch_size', default=64, type=int, help='Number of samples per step')
parser.add_argument('--layers', default=9, type=int, help='Number of layers in the network')
parser.add_argument('--units', default=50, type=int, help='Number of units per layer in the network')
parser.add_argument('--mode', default='Derivative', type=str, help='Learning mode')

args = parser.parse_args()
init_weight = args.init_weight
pde_weight = args.pde_weight
bc_weight = args.bc_weight
device = args.device
name = args.name
train_steps = args.train_steps
epochs = args.epochs
batch_size = args.batch_size
layers = args.layers
units = args.units
lr_init = args.lr_init
sys_weight = args.sys_weight
mode = args.mode

sys_weight = kdv_best_params[f'{mode}']['sys_weight']
bc_weight = kdv_best_params[f'{mode}']['bc_weight']
init_weight = kdv_best_params[f'{mode}']['init_weight']
lr_init = kdv_best_params[f'{mode}']['lr_init']
pde_weight = 0.
    
batch_size = 64
use_hessian = False
title_mode = mode


print(f'Running experiment with name {name}')
print(f'sys_weight {sys_weight}, bc_weight {bc_weight}, init_weight {init_weight}, pde_weight {pde_weight}')

pde_dataset = torch.load(os.path.join('data', f'kdv_sol_dataset.pth'), weights_only=False)
init_dataset = torch.load(os.path.join('data', f'kdv_init_dataset.pth'), weights_only=False)
bc1_dataset = torch.load(os.path.join('data', f'kdv_bc1_dataset.pth'), weights_only=False)
bc2_dataset = torch.load(os.path.join('data', f'kdv_bc2_dataset.pth'), weights_only=False)

# Generate the dataloaders
pde_dataloader = DataLoader(pde_dataset, batch_size, generator=gen, shuffle=True)
init_dataloader = DataLoader(init_dataset, batch_size, generator=gen, shuffle=True)
bc1_dataloader = DataLoader(bc1_dataset, batch_size, generator=gen, shuffle=True)
bc2_dataloader = DataLoader(bc2_dataset, batch_size, generator=gen, shuffle=True)    
test_dataloader = DataLoader(pde_dataset, 256, generator=gen, shuffle=True)
    

model = KdVPINN(hidden_units=[units for _ in range(layers)],
                pde_weight=0.,
                sys_weight=sys_weight,
                bc_weight=bc_weight,
                init_weight=init_weight,
                device=device,
                activation=torch.nn.Tanh(),
                lr=lr_init)

model.train()
# Prepare the lists

step_list= []
out_losses_train = []
bc_losses_train = []
pde_losses_train = []
der_losses_train = []
hess_losses_train = []
init_losses_train = []
tot_losses_train = []
out_true_losses_train = []
bc_true_losses_train = []
init_true_losses_train = []

step_list_test = []
out_losses_test = []
bc_losses_test = []
pde_losses_test = []
der_losses_test = []
hess_losses_test = []
init_losses_test = []
tot_losses_test = []
out_true_losses_test = []
bc_true_losses_test = []
init_true_losses_test = []

import time
times_test = []

def train_loop(epochs:int,
        pde_dataloader:DataLoader,
        init_dataloader:DataLoader,
        bc1_dataloader:DataLoader,
        bc2_dataloader:DataLoader,
        test_dataloader:DataLoader=None,
        print_every:int=100):
    
    # Training mode for the network
    
    for epoch in range(epochs):
        model.train()
        step_prefix = epoch*min(len(pde_dataloader),train_steps)
        print(f'Epoch: {epoch}, step_prefix: {step_prefix}')
        start_time = time.time()
        for step, (pde_data, init_data, bc1_data, bc2_data) in enumerate(zip(pde_dataloader, cycle(init_dataloader), cycle(bc1_dataloader), cycle(bc2_dataloader))):
            if step > train_steps:
                break
            # Load batches from dataloaders
            x_pde = pde_data[0].to(device).float().requires_grad_(True)
            y_pde = pde_data[1].to(device).float()
            Dy_pde = pde_data[2].to(device).float().unsqueeze(1)
            #Hy_pde = pde_data[3].to(device).float()
            #y_pde_true = pde_data[4].to(device).float()
            
            x_init = init_data[0].to(device).float()
            y_init = init_data[1].to(device).float()
            #y_init_true = init_data[4].to(device).float()
            
            x_bc1 = bc1_data[0].to(device).float().requires_grad_(True)
            y_bc1 = bc1_data[1].to(device).float()
            
            x_bc2 = bc2_data[0].to(device).float().requires_grad_(True)
            y_bc2 = bc2_data[1].to(device).float()
            
            # Call zero grad on optimizer
            model.opt.zero_grad()
            
            loss = model.student_loss_fn(
                x_pde=x_pde, x_bc1=x_bc1, x_bc2=x_bc2, x_init=x_init, y_init=y_init, y_pde=y_pde, Dy_pde=Dy_pde, Hy_pde=None, mode=mode, y_bc1=y_bc1, y_bc2=y_bc2, y_init_true=y_init, use_hessian=use_hessian
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
                    step_val, teacher_losses, true_losses = model.student_eval_losses(
                        step=step_prefix+step, x_pde=x_pde, y_pde=y_pde, x_bc1=x_bc1, y_bc1=y_bc1, y_bc2=y_bc2, x_bc2=x_bc2, x_init=x_init, y_init=y_init,  y_init_true=y_init, y_pde_true=y_pde, Dy_pde=Dy_pde, Hy_pde=None, mode=mode, print_to_screen=True , use_hessian=use_hessian
                    )
                    step_list.append(step_val)
                    tot_losses_train.append(teacher_losses[0])
                    out_losses_train.append(teacher_losses[1])
                    der_losses_train.append(teacher_losses[2])
                    hess_losses_train.append(teacher_losses[3])
                    pde_losses_train.append(teacher_losses[4])
                    init_losses_train.append(teacher_losses[5])
                    bc_losses_train.append(teacher_losses[6])
                    
                    out_true_losses_train.append(true_losses[0])
                    init_true_losses_train.append(true_losses[1])
                    bc_true_losses_train.append(true_losses[2])
                    
                    
                    #print(f'Step: {step_prefix+step}, Out loss: {out_loss_train}, PDE loss: {pde_loss_train}, Init loss: {init_loss_train}, BC loss: {bc_loss_train}, Tot loss: {tot_loss_train}')        
        # Calculate and average the loss over the test dataloader
        end_time = time.time()
        print(f'Epoch {epoch} took {end_time-start_time} seconds')
        times_test.append(end_time-start_time)
        model.eval()
        test_loss = 0.0
        out_loss_test = 0.0
        der_loss_test = 0.0
        hess_loss_test = 0.0
        pde_loss_test = 0.0
        init_loss_test = 0.0
        tot_loss_test = 0.0
        bc_loss_test = 0.0
        
        out_true_loss_test = 0.0
        init_true_loss_test = 0.0
        bc_true_loss_test = 0.0
        
        with torch.no_grad():
            for (pde_data, init_data, bc1_data, bc2_data) in zip(test_dataloader, cycle(init_dataloader), cycle(bc1_dataloader), cycle(bc2_dataloader)):
                # Load batches from dataloaders
                x_pde = pde_data[0].to(device).float().requires_grad_(True)
                y_pde = pde_data[1].to(device).float()
                Dy_pde = pde_data[2].to(device).float().unsqueeze(1)
                #Hy_pde = pde_data[3].to(device).float()
                #y_pde_true = pde_data[4].to(device).float()
                
                x_init = init_data[0].to(device).float()
                y_init = init_data[1].to(device).float()
                #y_init_true = init_data[4].to(device).float()
                
                x_bc1 = bc1_data[0].to(device).float().requires_grad_(True)
                y_bc1 = bc1_data[1].to(device).float()
                
                x_bc2 = bc2_data[0].to(device).float().requires_grad_(True)
                y_bc2 = bc2_data[1].to(device).float()
                
                step_test, teacher_losses, true_losses = model.student_eval_losses(step=step_prefix+step,
                    x_pde=x_pde, y_pde=y_pde, x_bc1=x_bc1, x_bc2=x_bc2, x_init=x_init, y_init=y_init, y_bc1=y_bc1, y_bc2=y_bc2, mode=mode, y_init_true=y_init, y_pde_true=y_pde, Dy_pde=Dy_pde, Hy_pde=None, use_hessian=use_hessian
                )
                tot_loss_test += teacher_losses[0]
                out_loss_test += teacher_losses[1]
                der_loss_test += teacher_losses[2]
                hess_loss_test += teacher_losses[3]
                pde_loss_test += teacher_losses[4]
                init_loss_test += teacher_losses[5]
                bc_loss_test += teacher_losses[6]
                
                out_true_loss_test += true_losses[0]
                init_true_loss_test += true_losses[1]
                bc_true_loss_test += true_losses[2]
                
                
                test_loss += teacher_losses[0]
                
            test_loss /= len(test_dataloader)
            out_loss_test /= len(test_dataloader)
            der_loss_test /= len(test_dataloader)
            hess_loss_test /= len(test_dataloader)
            pde_loss_test /= len(test_dataloader)
            init_loss_test /= len(test_dataloader)
            bc_loss_test /= len(test_dataloader)
            
            out_true_loss_test /= len(test_dataloader)
            init_true_loss_test /= len(test_dataloader)
            bc_true_loss_test /= len(test_dataloader)
            
        
        step_list_test.append(step_test)
        out_losses_test.append(out_loss_test)
        der_losses_test.append(der_loss_test)
        hess_losses_test.append(hess_loss_test)
        pde_losses_test.append(pde_loss_test)
        init_losses_test.append(init_loss_test)
        bc_losses_test.append(bc_loss_test)
        tot_losses_test.append(test_loss)
        
        out_true_losses_test.append(out_true_loss_test)
        init_true_losses_test.append(init_true_loss_test)
        bc_true_losses_test.append(bc_true_loss_test)
        
            
        print(f"Average test loss: {test_loss}")
        print(f"Average output loss: {out_loss_test}")
        print(f"Average derivative loss: {der_loss_test}")
        print(f"Average 2nd derivative loss: {hess_loss_test}")
        print(f"Average PDE loss: {pde_loss_test}")
        print(f"Average initialization loss: {init_loss_test}")
        print(f"Average boundary conditions loss: {bc_loss_test}")
        print(f"Average total loss: {tot_loss_test}")
        
        print(f"Average true output loss: {out_true_loss_test}")
        print(f"Average true initialization loss: {init_true_loss_test}")
        print(f"Average true boundary conditions loss: {bc_true_loss_test}")
        
        
train_loop(epochs=epochs, pde_dataloader=pde_dataloader, init_dataloader=init_dataloader, bc1_dataloader=bc1_dataloader, bc2_dataloader=bc2_dataloader, test_dataloader=test_dataloader, print_every=100)



# Save the model
if not os.path.exists(f'./student{title_mode}'):
    os.mkdir(f'./student{title_mode}')
    
if not os.path.exists(f'./saved_models'):
    os.mkdir(f'.saved_models')
torch.save(model.state_dict(), f'./student{title_mode}/kdv_net')

# Load it to be sure it works
model.load_state_dict(torch.load(f'./student{title_mode}/kdv_net'))
    
from matplotlib import pyplot as plt 


step_list = torch.tensor(step_list).cpu().numpy()
pde_losses = torch.tensor(pde_losses_train).cpu().numpy()
init_losses = torch.tensor(init_losses_train).cpu().numpy()
bc_losses = torch.tensor(bc_losses_train).cpu().numpy()
tot_losses = torch.tensor(tot_losses_train).cpu().numpy()
out_losses = torch.tensor(out_losses_train).cpu().numpy()

loss_combination_train = np.stack([step_list, pde_losses, init_losses, bc_losses, tot_losses, out_losses], axis=1)
with open(f'student{title_mode}/traindata.npy', 'wb') as f:
    np.save(f, loss_combination_train)

plt.figure()
plt.plot(step_list, pde_losses, label='pde_loss', color='red')
plt.plot(step_list, out_losses, label='out_loss', color='green')
plt.plot(step_list, init_losses, label='init_loss', color='orange')
plt.plot(step_list, bc_losses, label='bc_loss', color='purple')
#plt.plot(step_list, tot_losses, label='tot_loss', color='black')
plt.legend()
plt.yscale('log')
plt.savefig(f'student{title_mode}/trainlosses.png', dpi=300)



step_list = torch.tensor(step_list_test).cpu().numpy()
pde_losses = torch.tensor(pde_losses_test).cpu().numpy()
init_losses = torch.tensor(init_losses_test).cpu().numpy()
bc_losses = torch.tensor(bc_losses_test).cpu().numpy()
tot_losses = torch.tensor(tot_losses_test).cpu().numpy()
out_losses = torch.tensor(out_losses_test).cpu().numpy()

out_true_losses = torch.tensor(out_true_losses_test).cpu().numpy()
init_true_losses = torch.tensor(init_true_losses_test).cpu().numpy()
bc_true_losses = torch.tensor(bc_true_losses_test).cpu().numpy()

times_test = np.array(times_test)

loss_combination_test = np.stack([step_list, pde_losses, init_losses, bc_losses, tot_losses, out_losses, times_test, out_true_losses, init_true_losses, bc_true_losses], axis=1)
with open(f'student{title_mode}/testdata.npy', 'wb') as f:
    np.save(f, loss_combination_test)

plt.figure()
plt.plot(step_list, pde_losses, label='pde_loss', color='red')
plt.plot(step_list, out_losses, label='out_loss', color='green')
plt.plot(step_list, init_losses, label='init_loss', color='orange')
plt.plot(step_list, bc_losses, label='bc_loss', color='purple')
#plt.plot(step_list, tot_losses, label='tot_loss', color='black')
plt.legend()
plt.yscale('log')
plt.savefig(f'student{title_mode}/testlosses.png', dpi=300)

print('Plotting done!')
print('Start saving I/O dataset')


with open(f'data/kdv_data.npy', 'rb') as f:
    kdv_data = np.load(f)
    print(f'kdv_data.shape: {kdv_data.shape}')




# Plot the solution
dt = 0.01
dx = 0.01
x_plot = np.arange(start=-1, stop=1+dx, step=dx)
t_plot = np.arange(start=0, stop=1+dx, step=dx)

X, Y = np.meshgrid(t_plot, x_plot)
pts = np.vstack([X.reshape(-1),Y.reshape(-1)]).T

u_pred = model.forward(torch.from_numpy(pts).to(device).float().requires_grad_(True)).cpu().detach().numpy().reshape(X.shape)
fig = plt.figure()
cmap = plt.colormaps["jet"]
cmap = cmap.with_extremes(bad=cmap(0))
pcm = plt.pcolormesh(X, Y, u_pred, cmap=cmap,
                        rasterized=True)
fig.colorbar(pcm, label="value", pad=0)
plt.title("Predicted KdV solution in the domain")
plt.xlabel("t")
plt.ylabel("x")

plt.savefig(f'student{title_mode}/kdv_tx_plot.png', dpi=300)
plt.close()



fig, axs = plt.subplots(2, 2)
X_multiplot = torch.column_stack([torch.zeros(x_plot.shape), torch.tensor(x_plot, dtype=torch.float32)]).to(device=device)
u = model.forward(X_multiplot).cpu().detach().numpy()
axs[0, 0].plot(x_plot, u)
axs[0, 0].set_title('t = 0')

X_multiplot = torch.column_stack([0.33*torch.ones(x_plot.shape), torch.tensor(x_plot, dtype=torch.float32)]).to(device=device)
u = model.forward(X_multiplot).cpu().detach().numpy()
axs[0, 1].plot(x_plot, u, 'tab:orange')
axs[0, 1].set_title(' t = 0.33')

X_multiplot = torch.column_stack([0.66*torch.ones(x_plot.shape), torch.tensor(x_plot, dtype=torch.float32)]).to(device=device)
print(X_multiplot.shape)
u = model.forward(X_multiplot).cpu().detach().numpy()
axs[1, 0].plot(x_plot, u, 'tab:green')
axs[1, 0].set_title('t = 0.66')

X_multiplot = torch.column_stack([1.*torch.ones(x_plot.shape), torch.tensor(x_plot, dtype=torch.float32)]).to(device=device)
print(X_multiplot.shape)
u = model.forward(X_multiplot).cpu().detach().numpy()
axs[1, 1].plot(x_plot, u, 'tab:red')
axs[1, 1].set_title('t = 1')

for ax in axs.flat:
    ax.set(xlabel='x', ylabel='u(x,t)')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.savefig(f'student{title_mode}/kdv_multiplot.png', dpi=300)



step_list = torch.tensor(step_list).cpu().numpy()
pde_losses = torch.tensor(pde_losses_train).cpu().numpy()
init_losses = torch.tensor(init_losses_train).cpu().numpy()
bc_losses = torch.tensor(bc_losses_train).cpu().numpy()
tot_losses = torch.tensor(tot_losses_train).cpu().numpy()
out_losses = torch.tensor(out_losses_train).cpu().numpy()

loss_combination_train = np.stack([step_list, pde_losses, init_losses, bc_losses, tot_losses, out_losses], axis=1)
with open(f'student{title_mode}/traindata.npy', 'wb') as f:
    np.save(f, loss_combination_train)

plt.figure()
plt.plot(step_list, pde_losses, label='pde_loss', color='red')
plt.plot(step_list, out_losses, label='out_loss', color='green')
plt.plot(step_list, init_losses, label='init_loss', color='orange')
plt.plot(step_list, bc_losses, label='bc_loss', color='purple')
#plt.plot(step_list, tot_losses, label='tot_loss', color='black')
plt.legend()
plt.yscale('log')
plt.savefig(f'./student{title_mode}/trainlosses.png', dpi=300)



step_list = torch.tensor(step_list_test).cpu().numpy()
pde_losses = torch.tensor(pde_losses_test).cpu().numpy()
init_losses = torch.tensor(init_losses_test).cpu().numpy()
bc_losses = torch.tensor(bc_losses_test).cpu().numpy()
tot_losses = torch.tensor(tot_losses_test).cpu().numpy()
out_losses = torch.tensor(out_losses_test).cpu().numpy()

out_true_losses = torch.tensor(out_true_losses_test).cpu().numpy()
init_true_losses = torch.tensor(init_true_losses_test).cpu().numpy()
bc_true_losses = torch.tensor(bc_true_losses_test).cpu().numpy()

times_test = np.array(times_test)

loss_combination_test = np.stack([step_list, pde_losses, init_losses, bc_losses, tot_losses, out_losses, times_test, out_true_losses, init_true_losses, bc_true_losses], axis=1)
with open(f'./student{title_mode}/testdata.npy', 'wb') as f:
    np.save(f, loss_combination_test)

plt.figure()
plt.plot(step_list, pde_losses, label='pde_loss', color='red')
plt.plot(step_list, out_losses, label='out_loss', color='green')
plt.plot(step_list, init_losses, label='init_loss', color='orange')
plt.plot(step_list, bc_losses, label='bc_loss', color='purple')
#plt.plot(step_list, tot_losses, label='tot_loss', color='black')
plt.legend()
plt.yscale('log')
plt.savefig(f'./student{title_mode}/testlosses.png', dpi=300)

print('Plotting done!')
print('Start saving I/O dataset')