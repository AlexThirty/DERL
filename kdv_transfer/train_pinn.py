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
parser.add_argument('--init_weight', default=.001, type=float, help='Weight for the init loss')
parser.add_argument('--pde_weight', default=1., type=float, help='Weight for the F loss')
parser.add_argument('--bc_weight', default=1., type=float, help='Weight for the F loss')
parser.add_argument('--sys_weight', default=1., type=float, help='Weight for the F loss')
parser.add_argument('--lr_init', default=1e-3, type=float, help='Starting learning rate')
parser.add_argument('--device', default='cuda', type=str, help='Device to use')
parser.add_argument('--name', default='pinn', type=str, help='Experiment name')
parser.add_argument('--train_steps', default=1000, type=int, help='Number of training steps')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('--batch_size', default=1000, type=int, help='Number of samples per step')
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


# Generate the times
t = torch.distributions.Uniform(0.,1.).sample((batch_size*train_steps,1))
# Generate the point
points = torch.distributions.Uniform(-1.,1.).sample((batch_size*train_steps, 1))
# Stack them
x_pde = torch.column_stack((t, points))

train_dataset = torch.utils.data.TensorDataset(x_pde) 

# Generate the times
#t =  t_max*torch.distributions.Uniform(0.,1.).sample((num_boundary,1))
# Generate the points
t = torch.distributions.Uniform(0.,1.).sample((batch_size*train_steps,1))
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
t = torch.distributions.Uniform(0.,1.).sample((batch_size*train_steps,1))
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

eval_dataset = torch.load('data/kdv_pde_dataset_phase0.pth', weights_only=False)
init_dataset = torch.load('data/kdv_init_dataset.pth', weights_only=False)

#else:
#    bc_dataset = None
# Generate the dataloader
train_dataloader = DataLoader(train_dataset, batch_size, generator=gen, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, 2048, generator=gen, shuffle=True)
bc1_dataloader = DataLoader(bc1_dataset, batch_size, generator=gen, shuffle=True)
bc2_dataloader = DataLoader(bc2_dataset, batch_size, generator=gen, shuffle=True)
init_dataloader = DataLoader(init_dataset, batch_size, generator=gen, shuffle=True)
    
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
        print_every:int=100):
    
    # Training mode for the network
    
    for epoch in range(epochs):
        model.train()
        step_prefix = epoch*min(len(train_dataloader),train_steps)
        print(f'Epoch: {epoch}, step_prefix: {step_prefix}')
        for step, (train_data, eval_data, bc1_data, bc2_data, init_data) in enumerate(zip(train_dataloader, cycle(eval_dataloader), cycle(bc1_dataloader), cycle(bc2_dataloader), cycle(init_dataloader))):
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

            #def closure():
            #    model.opt.zero_grad()
            #    loss = model.loss_fn(
            #        x_pde=x_pde, x_bc1=x_bc1, x_bc2=x_bc2, x_init=x_init, y_init=y_init
            #    )
            #    loss.backward()
            #    return loss
            #
            #optim.step(closure)
            
            # Call zero grad on optimizer
            model.opt.zero_grad()
            
            loss = model.loss_fn(
                x_pde=x_pde, x_bc1=x_bc1, x_bc2=x_bc2, x_init=x_init, y_init=y_init
            )
            # Backward the loss, calculate gradients
            loss.backward()
            # Optimizer step
            model.opt.step()
            # Update the learning rate scheduling
            
            # Printing
            if (step_prefix+step) % print_every == 0:                #print('Train losses')
                with torch.no_grad():
                    step_val, out_loss_train, pde_loss_train, init_loss_train, bc_loss_train, distillation_loss_train, tot_loss_train = model.eval_losses(
                        step=step_prefix+step, x_pde=x_eval, y_pde=y_eval, x_bc1=x_bc1, x_bc2=x_bc2, x_init=x_init, y_init=y_init, print_to_screen=True,
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
        hes_loss_test = 0.0
        init_loss_test = 0.0
        distillation_loss_test = 0.0

        with torch.no_grad():
            for (eval_data, bc1_data, bc2_data, init_data) in zip(eval_dataloader, cycle(bc1_dataloader), cycle(bc2_dataloader), cycle(init_dataloader)):
                x_eval = eval_data[0].to(device).float()
                y_eval = eval_data[1].to(device).float()

                x_bc1 = bc1_data[0].to(device).float()
                y_bc1 = bc1_data[1].to(device).float()
                x_bc2 = bc2_data[0].to(device).float()
                y_bc2 = bc2_data[1].to(device).float()
                x_init = init_data[0].to(device).float()
                y_init = init_data[1].to(device).float()

                step_test, out_loss, pde_loss, init_loss, bc_loss, distillation_loss, tot_loss = model.eval_losses(
                    step=step_prefix+step,
                    x_pde=x_eval, y_pde=y_eval, x_bc1=x_bc1, x_bc2=x_bc2, x_init=x_init, y_init=y_init, print_to_screen=False
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

train_loop(epochs=epochs, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, bc1_dataloader=bc1_dataloader, bc2_dataloader=bc2_dataloader, init_dataloader=init_dataloader, print_every=100)



# Save the model
if not os.path.exists(f'./pinn'):
    os.mkdir(f'./pinn')
    
torch.save(model.state_dict(), f'./saved_models/pinn')

# Load it to be sure it works
model.load_state_dict(torch.load(f'./saved_models/pinn'))

from matplotlib import pyplot as plt 

with open(f'./data/kdv_data.npy', 'rb') as f:
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
