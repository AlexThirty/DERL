import torch
from model import KdVPINN, SinActivation
import argparse
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import os
from torch.func import vmap, jacrev, hessian
from itertools import cycle

from tuner_results import kdv_best_params
import time 

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
parser.add_argument('--lr_init', default=1e-4, type=float, help='Starting learning rate')
parser.add_argument('--device', default='cuda:2', type=str, help='Device to use')
parser.add_argument('--name', default='pinn', type=str, help='Experiment name')
parser.add_argument('--train_steps', default=100000, type=int, help='Number of training steps')
parser.add_argument('--epochs', default=200, type=int, help='Number of epochs')
parser.add_argument('--batch_size', default=64, type=int, help='Number of samples per step')
parser.add_argument('--layers', default=9, type=int, help='Number of layers in the network')
parser.add_argument('--units', default=50, type=int, help='Number of units per layer in the network')

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


bc_weight = kdv_best_params['PINN']['bc_weight']
pde_weight = kdv_best_params['PINN']['pde_weight']
init_weight = kdv_best_params['PINN']['init_weight']
lr_init = kdv_best_params['PINN']['lr_init']
#batch_size = kdv_best_params['PINN']['batch_size']
batch_size = 64
sys_weight = kdv_best_params['PINN']['sys_weight']
print('Using best parameters')
print(f'bc_weight: {bc_weight}')
print(f'pde_weight: {pde_weight}')
print(f'init_weight: {init_weight}')
print(f'lr_init: {lr_init}')
print(f'batch_size: {batch_size}')
print(f'sys_weight: {sys_weight}')

pde_dataset = torch.load(os.path.join('data', f'kdv_pde_dataset.pth'))
init_dataset = torch.load(os.path.join('data', f'kdv_init_dataset.pth'))
bc1_dataset = torch.load(os.path.join('data', f'kdv_bc1_dataset.pth'))
bc2_dataset = torch.load(os.path.join('data', f'kdv_bc2_dataset.pth'))

# Generate the dataloaders
pde_dataloader = DataLoader(pde_dataset, batch_size, generator=gen, shuffle=True, num_workers=12)
init_dataloader = DataLoader(init_dataset, batch_size, generator=gen, shuffle=True, num_workers=12)
bc1_dataloader = DataLoader(bc1_dataset, batch_size, generator=gen, shuffle=True, num_workers=12)
bc2_dataloader = DataLoader(bc2_dataset, batch_size, generator=gen, shuffle=True, num_workers=12)    
test_dataloader = DataLoader(pde_dataset, 256, generator=gen, shuffle=True, num_workers=12)
    

model = KdVPINN(hidden_units=[units for _ in range(layers)],
                pde_weight=pde_weight,
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
init_losses_train = []
tot_losses_train = []

step_list_test = []
out_losses_test = []
bc_losses_test = []
pde_losses_test = []
init_losses_test = []
tot_losses_test = []
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
            
            x_init = init_data[0].to(device).float()
            y_init = init_data[1].to(device).float()
            
            x_bc1 = bc1_data[0].to(device).float().requires_grad_(True)
            y_bc1 = bc1_data[1].to(device).float()
            
            x_bc2 = bc2_data[0].to(device).float().requires_grad_(True)
            y_bc2 = bc2_data[1].to(device).float()
            
            # Call zero grad on optimizer
            model.opt.zero_grad()
            
            loss = model.loss_fn(
                x_pde=x_pde, x_bc1=x_bc1, x_bc2=x_bc2, x_init=x_init, y_init=y_init, y_pde=None
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
                    step_val, out_loss_train, pde_loss_train, init_loss_train, bc_loss_train, tot_loss_train = model.eval_losses(
                        step=step_prefix+step, x_pde=x_pde, y_pde=y_pde, x_bc1=x_bc1, x_bc2=x_bc2, x_init=x_init, y_init=y_init
                    )
                    step_list.append(step_val)
                    tot_losses_train.append(tot_loss_train)
                    out_losses_train.append(out_loss_train)
                    pde_losses_train.append(pde_loss_train)
                    init_losses_train.append(init_loss_train)
                    bc_losses_train.append(bc_loss_train)
                    print(f'Step: {step_prefix+step}, Out loss: {out_loss_train}, PDE loss: {pde_loss_train}, Init loss: {init_loss_train}, BC loss: {bc_loss_train}, Tot loss: {tot_loss_train}')        
        # Calculate and average the loss over the test dataloader
        end_time = time.time()
        epoch_time = end_time - start_time
        times_test.append(epoch_time)
        print(f'Epoch time: {epoch_time}')
        
        model.eval()
        test_loss = 0.0
        out_loss_test = 0.0
        pde_loss_test = 0.0
        init_loss_test = 0.0
        tot_loss_test = 0.0
        bc_loss_test = 0.0
        
        with torch.no_grad():
            for (pde_data, init_data, bc1_data, bc2_data) in zip(test_dataloader, cycle(init_dataloader), cycle(bc1_dataloader), cycle(bc2_dataloader)):
                # Load batches from dataloaders
                x_pde = pde_data[0].to(device).float().requires_grad_(True)
                y_pde = pde_data[1].to(device).float()
                
                x_init = init_data[0].to(device).float()
                y_init = init_data[1].to(device).float()
                
                x_bc1 = bc1_data[0].to(device).float()
                y_bc1 = bc1_data[1].to(device).float()
                
                x_bc2 = bc2_data[0].to(device).float()
                y_bc2 = bc2_data[1].to(device).float()
                
                step_test, out_loss, pde_loss, init_loss, bc_loss, tot_loss = model.eval_losses(step=step_prefix+step,
                    x_pde=x_pde, y_pde=y_pde, x_bc1=x_bc1, x_bc2=x_bc2, x_init=x_init, y_init=y_init, print_to_screen=False
                )
                
                out_loss_test += out_loss.item()
                pde_loss_test += pde_loss.item()
                init_loss_test += init_loss.item()
                tot_loss_test += tot_loss.item()
                bc_loss_test += bc_loss.item()
                
                test_loss += tot_loss.item()
                
            test_loss /= len(test_dataloader)
            out_loss_test /= len(test_dataloader)
            pde_loss_test /= len(test_dataloader)
            init_loss_test /= len(test_dataloader)
            tot_loss_test /= len(test_dataloader)
            bc_loss_test /= len(test_dataloader)
        
        step_list_test.append(step_test)
        out_losses_test.append(out_loss_test)
        pde_losses_test.append(pde_loss_test)
        init_losses_test.append(init_loss_test)
        tot_losses_test.append(tot_loss_test)
        bc_losses_test.append(bc_loss_test)
            
        print(f"Average test loss: {test_loss}")
        print(f"Average output loss: {out_loss_test}")
        print(f"Average PDE loss: {pde_loss_test}")
        print(f"Average initialization loss: {init_loss_test}")
        print(f"Average boundary conditions loss: {bc_loss_test}")
        print(f"Average total loss: {tot_loss_test}")
        
train_loop(epochs=epochs, pde_dataloader=pde_dataloader, init_dataloader=init_dataloader, bc1_dataloader=bc1_dataloader, bc2_dataloader=bc2_dataloader, test_dataloader=test_dataloader, print_every=100)



# Save the model
if not os.path.exists(f'./pinn'):
    os.mkdir(f'./pinn')
    
torch.save(model.state_dict(), f'./pinn/kdv_net')

# Load it to be sure it works
model.load_state_dict(torch.load(f'./pinn/kdv_net'))

from matplotlib import pyplot as plt 

with open(f'./kdv_data.npy', 'rb') as f:
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

plt.savefig(f'./pinn/kdv_tx_plot.png', dpi=300)
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

plt.savefig(f'./pinn/kdv_multiplot.png', dpi=300)



step_list = torch.tensor(step_list).cpu().numpy()
pde_losses = torch.tensor(pde_losses_train).cpu().numpy()
init_losses = torch.tensor(init_losses_train).cpu().numpy()
bc_losses = torch.tensor(bc_losses_train).cpu().numpy()
tot_losses = torch.tensor(tot_losses_train).cpu().numpy()
out_losses = torch.tensor(out_losses_train).cpu().numpy()

loss_combination_train = np.stack([step_list, pde_losses, init_losses, bc_losses, tot_losses, out_losses], axis=1)
with open(f'./pinn/traindata.npy', 'wb') as f:
    np.save(f, loss_combination_train)

plt.figure()
plt.plot(step_list, pde_losses, label='pde_loss', color='red')
plt.plot(step_list, out_losses, label='out_loss', color='green')
plt.plot(step_list, init_losses, label='init_loss', color='orange')
plt.plot(step_list, bc_losses, label='bc_loss', color='purple')
#plt.plot(step_list, tot_losses, label='tot_loss', color='black')
plt.legend()
plt.yscale('log')
plt.savefig(f'./pinn/trainlosses.png', dpi=300)



step_list = torch.tensor(step_list_test).cpu().numpy()
pde_losses = torch.tensor(pde_losses_test).cpu().numpy()
init_losses = torch.tensor(init_losses_test).cpu().numpy()
bc_losses = torch.tensor(bc_losses_test).cpu().numpy()
tot_losses = torch.tensor(tot_losses_test).cpu().numpy()
out_losses = torch.tensor(out_losses_test).cpu().numpy()
times_test = np.array(times_test)

loss_combination_test = np.stack([step_list, pde_losses, init_losses, bc_losses, tot_losses, out_losses, times_test], axis=1)
with open(f'./pinn/testdata.npy', 'wb') as f:
    np.save(f, loss_combination_test)

plt.figure()
plt.plot(step_list, pde_losses, label='pde_loss', color='red')
plt.plot(step_list, out_losses, label='out_loss', color='green')
plt.plot(step_list, init_losses, label='init_loss', color='orange')
plt.plot(step_list, bc_losses, label='bc_loss', color='purple')
#plt.plot(step_list, tot_losses, label='tot_loss', color='black')
plt.legend()
plt.yscale('log')
plt.savefig(f'./pinn/testlosses.png', dpi=300)

print('Plotting done!')
print('Start saving I/O dataset')





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
torch.save(pde_distillation_dataset, os.path.join('data', f'kdv_pdedistillation_dataset.pth'))
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
torch.save(init_distillation_dataset, os.path.join('data', f'kdv_initdistillation_dataset.pth'))
print('Created init distillation dataset')



# Empty cache just to be sure
torch.cuda.empty_cache()
# Create tensors for I/O
bc1_x = []
bc1_y = []
bc1_Dy = []
bc1_Hy = []
bc1_ytrue = []
for (x,y) in bc1_dataloader:
    bc1_x.append(x)
    bc1_ytrue.append(y)
    bc1_y.append(model.forward(x.to(device).float()).detach().cpu())
    bc1_Dy.append(torch.func.vmap(torch.func.jacrev(model.forward_single))(x.to(device).float()).detach().cpu())
    bc1_Hy.append(torch.func.vmap(torch.func.hessian(model.forward_single))(x.to(device).float()).detach().cpu())
    torch.cuda.empty_cache()

bc1_x = torch.cat(bc1_x)
bc1_y = torch.cat(bc1_y)
bc1_Dy = torch.cat(bc1_Dy)
bc1_Hy = torch.cat(bc1_Hy)
bc1_ytrue = torch.cat(bc1_ytrue)

print(f'bc1_x.shape: {bc1_x.shape}')
print(f'bc1_y.shape: {bc1_y.shape}')
print(f'bc1_Dy.shape: {bc1_Dy.shape}')
print(f'bc1_Hy.shape: {bc1_Hy.shape}')
print(f'bc1_ytrue.shape: {bc1_ytrue.shape}')
bc1_distillation_dataset = TensorDataset(bc1_x, bc1_y, bc1_Dy, bc1_Hy, bc1_ytrue)
torch.save(bc1_distillation_dataset, os.path.join('data', f'kdv_bc1distillation_dataset.pth'))
print('Created BC1 distillation dataset')


# Empty cache just to be sure
torch.cuda.empty_cache()
# Create tensors for I/O
bc2_x = []
bc2_y = []
bc2_Dy = []
bc2_Hy = []
bc2_ytrue = []
for (x,y) in bc2_dataloader:
    bc2_x.append(x)
    bc2_ytrue.append(y)
    bc2_y.append(model.forward(x.to(device).float()).detach().cpu())
    bc2_Dy.append(torch.func.vmap(torch.func.jacrev(model.forward_single))(x.to(device).float()).detach().cpu())
    bc2_Hy.append(torch.func.vmap(torch.func.hessian(model.forward_single))(x.to(device).float()).detach().cpu())
    torch.cuda.empty_cache()

bc2_x = torch.cat(bc2_x)
bc2_y = torch.cat(bc2_y)
bc2_Dy = torch.cat(bc2_Dy)
bc2_Hy = torch.cat(bc2_Hy)
bc2_ytrue = torch.cat(bc2_ytrue)

print(f'bc2_x.shape: {bc2_x.shape}')
print(f'bc2_y.shape: {bc2_y.shape}')
print(f'bc2_Dy.shape: {bc2_Dy.shape}')
print(f'bc2_Hy.shape: {bc2_Hy.shape}')
print(f'bc2_ytrue.shape: {bc2_ytrue.shape}')
bc2_distillation_dataset = TensorDataset(bc2_x, bc2_y, bc2_Dy, bc2_Hy, bc2_ytrue)
torch.save(bc2_distillation_dataset, os.path.join('data', f'kdv_bc2distillation_dataset.pth'))
print('Created BC2 distillation dataset')

print('Saved distillation datasets!')
