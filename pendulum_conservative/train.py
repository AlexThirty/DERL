import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np
import random
from torch import nn
import os
seed = 30
from torch.func import vmap, jacrev
from itertools import cycle

from model import PendulumNet
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)
from model import u_vec
from tuner_results import pendulum_best_params
import time

b = 0.

parser = argparse.ArgumentParser()
parser.add_argument('--init_weight', default=1., type=float, help='Weight for the init loss')
parser.add_argument('--sys_weight', default=1., type=float, help='Weight for the rho loss')
parser.add_argument('--lr_init', default=5e-4, type=float, help='Starting learning rate')
parser.add_argument('--device', default='cuda:2', type=str, help='Device to use')
parser.add_argument('--name', default='true', type=str, help='Experiment name')
parser.add_argument('--train_steps', default=100000, type=int, help='Number of training steps')
parser.add_argument('--epochs', default=200, type=int, help='Number of epochs')
parser.add_argument('--mode', default='Derivative', type=str, help='Mode: -1 for PINN learning, 0 for derivative learning, 1 for output learning')
parser.add_argument('--batch_size', default=32, type=int, help='Number of samples per step')
parser.add_argument('--layers', default=4, type=int, help='Number of layers in the network')
parser.add_argument('--units', default=50, type=int, help='Number of units per layer in the network')

args = parser.parse_args()
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
dt = 1e-2

# Export path and the type of pendulum
EXP_PATH = '.'
pendulum_type = 'damped'
    
print(f'Working on {EXP_PATH}, b={b}')

if not os.path.exists(f'{EXP_PATH}/{name}'):
    os.mkdir(f'{EXP_PATH}/{name}')

if name == 'true':
    prefix = 'true'
elif name == 'extrapolate':
    prefix = 'true_extrapolate'
elif name == 'interpolate':
    prefix = 'true_interpolate'
elif name == 'adapt':
    prefix = 'true_adapt'
elif name == 'emp':
    prefix = 'emp'
else:
    raise ValueError(f'name value is not in the options')

title_mode = mode


best_params = pendulum_best_params[str(mode)]
init_weight = best_params['init_weight']
batch_size = 64
if mode != 'PINN':
    sys_weight = best_params['sys_weight']
else:
    sys_weight = 0.
lr_init = best_params['lr_init']
if mode == 'PINN' or mode == 'PINN+Output':
    pde_weight = best_params['pde_weight'] 
else:
    pde_weight = 0.  
    
print('Loading the data...')    

# Load the data
train_dataset = torch.load(os.path.join('data', f'{prefix}_dataset_train.pth'))
test_dataset = torch.load(os.path.join('data', f'{prefix}_dataset_test.pth'))

print(train_dataset[:][0].shape)
if name in ['adapt', 'interpolate']:
    bc_dataset = torch.load(os.path.join('data', f'{prefix}_bc_train.pth'))
else:
    bc_dataset = None
# Generate the dataloader
val_dataset = torch.load(os.path.join('data', f'{prefix}_dataset_val.pth'))
#train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

gen = torch.Generator().manual_seed(seed)
train_dataloader = DataLoader(train_dataset, batch_size, generator=torch.Generator().manual_seed(seed), shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size, generator=torch.Generator().manual_seed(seed), shuffle=True)

print('Data loaded!')

activation = torch.nn.Tanh()

model = PendulumNet(
    init_weight=init_weight,
    sys_weight=sys_weight,
    pde_weight=pde_weight,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)

print(batch_size)
step_list= []
out_losses_train = []
der_losses_train = []
pde_losses_train = []
init_losses_train = []
tot_losses_train = []
init_pde_losses_train = []

step_list_test = []
out_losses_test = []
der_losses_test = []
pde_losses_test = []
init_losses_test = []
tot_losses_test = []
init_pde_losses_test = []
times_test = []

def train_loop(epochs:int,
        train_dataloader:DataLoader,
        test_dataloader:DataLoader,
        print_every:int=100):
    
    # Training mode for the network
    
    for epoch in range(epochs):
        model.train()
        step_prefix = epoch*min(len(train_dataloader),train_steps)
        print(f'Epoch: {epoch}, step_prefix: {step_prefix}')
        start_time = time.time()
        for step, train_data in enumerate(train_dataloader):
            if step > train_steps:
                break
            # Load batches from dataloaders
            x_train = train_data[0].to(device).float()
            
            y_train = train_data[1].to(device).float()[:,0].reshape((-1,1))
            Dy_train = train_data[2].to(device).float()[:,0]
            
            if bc_dataset is not None:
                x_bc = bc_dataset[:][0].to(device).float()
                y_bc = bc_dataset[:][1].to(device).float()
            else:
                x_bc = None
                y_bc = None
            
            # Call zero grad on optimizer
            model.opt.zero_grad()
            
            loss = model.loss_fn(mode=mode,
                x=x_train, y=y_train, Dy=Dy_train, x_bc=x_bc, y_bc=y_bc
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
                    step_val, out_loss_train, der_loss_train, pde_loss_train, init_loss_train, tot_loss_train, init_pde_loss_train = model.eval_losses(
                        step=step_prefix+step, mode=mode,
                        x=x_train, y=y_train, Dy=Dy_train, x_bc=x_bc, y_bc=y_bc
                    )
                    step_list.append(step_val)
                    tot_losses_train.append(tot_loss_train)
                    out_losses_train.append(out_loss_train)
                    der_losses_train.append(der_loss_train)
                    init_losses_train.append(init_loss_train)
                    pde_losses_train.append(pde_loss_train)
                    init_pde_losses_train.append(init_pde_loss_train)
        
                    print(f'Step: {step_prefix+step}, Out loss: {out_loss_train}, Der loss: {der_loss_train}, PDE loss: {pde_loss_train}, Init loss: {init_loss_train}, Tot loss: {tot_loss_train}, Init PDE loss: {init_pde_loss_train}')
        stop_time = time.time()
        epoch_time = stop_time - start_time
        print(f'Epoch time: {epoch_time}')
        times_test.append(epoch_time)
        # Calculate and average the loss over the test dataloader
        model.eval()
        test_loss = 0.0
        out_loss_test = 0.0
        der_loss_test = 0.0
        pde_loss_test = 0.0
        init_loss_test = 0.0
        tot_loss_test = 0.0
        init_pde_loss_test = 0.0
        
        with torch.no_grad():
            for test_data in test_dataloader:
                x_test = test_data[0].to(device).float()
                y_test = test_data[1].to(device).float()[:,0].reshape((-1,1))
                Dy_test = test_data[2].to(device).float()[:,0]
                
                step_test, out_loss, der_loss, pde_loss, init_loss, tot_loss, init_pde_loss = model.eval_losses(step=step_prefix+step, mode=mode,
                                                                                        x=x_test, y=y_test, Dy=Dy_test, x_bc=None, y_bc=None)
                
                out_loss_test += out_loss.item()
                der_loss_test += der_loss.item()
                pde_loss_test += pde_loss.item()
                init_loss_test += init_loss.item()
                tot_loss_test += tot_loss.item()
                init_pde_loss_test += init_pde_loss.item()
                
                test_loss += tot_loss.item()
                
            test_loss /= len(test_dataloader)
            out_loss_test /= len(test_dataloader)
            der_loss_test /= len(test_dataloader)
            pde_loss_test /= len(test_dataloader)
            init_loss_test /= len(test_dataloader)
            tot_loss_test /= len(test_dataloader)
            init_pde_loss_test /= len(test_dataloader)
        
        step_list_test.append(step_test)
        out_losses_test.append(out_loss_test)
        der_losses_test.append(der_loss_test)
        pde_losses_test.append(pde_loss_test)
        init_losses_test.append(init_loss_test)
        tot_losses_test.append(tot_loss_test)
        init_pde_losses_test.append(init_pde_loss_test)
            
        print(f"Average test loss: {test_loss}")
        print(f"Average output loss: {out_loss_test}")
        print(f"Average derivative loss: {der_loss_test}")
        print(f"Average PDE loss: {pde_loss_test}")
        print(f"Average initialization loss: {init_loss_test}")
        print(f"Average total loss: {tot_loss_test}")
        print(f"Average initialization PDE loss: {init_pde_loss_test}")
train_loop(epochs=epochs, train_dataloader=train_dataloader, test_dataloader=test_dataloader)


torch.cuda.empty_cache()
model.eval()
# %%
import os
if not os.path.exists(f'{EXP_PATH}/{name}/saved_models'):
    os.mkdir(f'{EXP_PATH}/{name}/saved_models')
torch.save(model.state_dict(), f'{EXP_PATH}/{name}/saved_models/pendulum_net{title_mode}')
# %%
model.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/pendulum_net{title_mode}'))

# %%
from matplotlib import pyplot as plt
import matplotlib.patches as patches

if not os.path.exists(f'{EXP_PATH}/{name}/plots{title_mode}'):
    os.mkdir(f'{EXP_PATH}/{name}/plots{title_mode}')

# Number of points for the field
N = 500
xlim = np.pi/2
ylim = 2.
X,Y = np.meshgrid(np.linspace(-xlim,xlim,N),np.linspace(-ylim,ylim,N))
pts = np.vstack([X.reshape(-1),Y.reshape(-1)]).T

#plots the streamplot for the velocity field
#print(pts)
vel = u_vec(torch.from_numpy(pts))
#print(vel)
U = np.array(vel[:,0].reshape(X.shape))
V = np.array(vel[:,1].reshape(Y.shape))
#mask the outside of the ball

plt.streamplot(X,Y,U,V,density=1,color=U**2 + V**2, linewidth=0.15)

plt.xlim((-xlim,xlim))
plt.ylim((-ylim,ylim))
#add outline for aesthetics
points = 100
t_max = 10
steps = int(t_max/dt)+1

x0 = -np.pi/4. + np.pi/2.*np.random.random((points))
v0 = -1.5 + 3*np.random.random((points))
xv0 = np.column_stack((x0.reshape((-1,1)),v0.reshape((-1,1))))

xv = torch.zeros((points,steps,2))
u_save = torch.zeros((points,steps,2))
xv[:,0,0] = torch.from_numpy(x0)
xv[:,0,1] = torch.from_numpy(v0)

for i in range(1,steps):
   # v[i] = v[i-1] + dt*u(x[i-1])
   xv[:,i,0] = xv[:,i-1,0] + dt*u_vec(xv[:,i-1,:])[:,0]
   xv[:,i,1] = xv[:,i-1,1] + dt*u_vec(xv[:,i-1,:])[:,1]


xv_pred = model.evaluate_trajectory(x0=xv[:,0,:].float(), time_steps=steps).detach().cpu().numpy()
print(xv_pred.shape)
print(xv.shape)
xv = xv.numpy()
for i in range(10):
    plt.plot(xv[i,:,0], xv[i,:,1], color='blue')
    plt.plot(xv_pred[i,:,0], xv_pred[i,:,1], color='red')
    #plt.legend()
blue_patch = patches.Patch(color='blue', label='True trajectories')
red_patch = patches.Patch(color='red', label='Predicted trajectories')
plt.legend(handles=[blue_patch,red_patch])
plt.xlabel(r'Angle: $\theta$')
plt.ylabel(r'Angular speed: $\omega$')

plt.title(f'{title_mode} learning phase trajectories')
plt.savefig(f'{EXP_PATH}/{name}/plots{title_mode}/pendulum_phase_trajectory.png', dpi=300)
    
plt.close()


t_base = np.arange(start=0, stop=t_max+dt, step=dt)
print(t_base.shape)
print(xv[0,:,0].shape)
for i in range(10):
    plt.plot(t_base, xv[i,:,0], color='blue')
    plt.plot(t_base, xv_pred[i,:,0], color='red')
blue_patch = patches.Patch(color='blue', label='True trajectories')
red_patch = patches.Patch(color='red', label='Predicted trajectories')
plt.legend(handles=[blue_patch,red_patch])
plt.xlabel(r'Time: $t$')
plt.ylabel(r'Angle: $\theta$')
plt.title(f'{title_mode} learning time trajectories')
plt.savefig(f'{EXP_PATH}/{name}/plots{title_mode}/pendulum_trajectory.png', dpi=300)
plt.close()

#plots the streamplot for the velocity field
plt.figure(figsize=(5,5))
#print(pts)
#vel = vmap(jacrev(model.forward_single))(torch.column_stack((0*torch.ones((pts.shape[0],1)),torch.from_numpy(pts).float())).to(device)).detach().cpu().numpy()[:,:,0]
vel = model.evaluate_field(torch.column_stack((0*torch.ones((pts.shape[0],1)),torch.from_numpy(pts).float())).to(device)).detach().cpu().numpy()

#print(vel)
U = np.array(vel[:,0].reshape(X.shape))
V = np.array(vel[:,1].reshape(Y.shape))
#mask the outside of the ball



plt.streamplot(X,Y,U,V,density=1,color=U**2 + V**2, linewidth=0.15)
for i in range(10):
    plt.plot(xv_pred[i,:,0], xv_pred[i,:,1], label=f'trajectory{i}', color='red')
plt.xlim((-xlim,xlim))
plt.ylim((-ylim,ylim))

plt.xlabel(r'Angle: $\theta$')
plt.ylabel(r'Angular speed: $\omega$')

plt.title(f'{title_mode} learning predicted field')
plt.savefig(f'{EXP_PATH}/{name}/plots{title_mode}/predicted_field.png')
plt.close()

vel_true = u_vec(torch.from_numpy(pts))
#print(vel)
U_true = np.array(vel_true[:,0].reshape(X.shape))
V_true = np.array(vel_true[:,1].reshape(Y.shape))
plt.contourf(X,Y,np.sqrt((U-U_true)**2+(V-V_true)**2),100,cmap='jet')
plt.title('Error in predicted fields')
plt.colorbar()
plt.xlim((-xlim,xlim))
plt.ylim((-ylim,ylim))
plt.xlabel(r'Angle: $\theta$')
plt.ylabel(r'Angular speed: $\omega$')

plt.title(f'{title_mode} learning field error')
plt.savefig(f'{EXP_PATH}/{name}/plots{title_mode}/error_field.png')
plt.close()

# Convert the losses arrays
epoch_list = torch.tensor(step_list).cpu().numpy()
out_losses_train = torch.tensor(out_losses_train).cpu().numpy()
der_losses_train = torch.tensor(der_losses_train).cpu().numpy()
pde_losses_train = torch.tensor(pde_losses_train).cpu().numpy()
init_losses_train = torch.tensor(init_losses_train).cpu().numpy()
tot_losses_train = torch.tensor(tot_losses_train).cpu().numpy()
init_pde_losses_train = torch.tensor(init_pde_losses_train).cpu().numpy()    


loss_combination_train = np.column_stack([epoch_list, out_losses_train, der_losses_train, pde_losses_train, init_losses_train, tot_losses_train, init_pde_losses_train])
with open(f'{EXP_PATH}/{name}/plots{title_mode}/traindata.npy', 'wb') as f:
    np.save(f, loss_combination_train)

N = 100
l = len(np.convolve(out_losses_train, np.ones(N)/N, mode='valid'))
plt.figure()
plt.plot(epoch_list[:l], np.convolve(pde_losses_train, np.ones(N)/N, mode='valid'), label='pde_loss', color='red')
plt.plot(epoch_list[:l], np.convolve(out_losses_train, np.ones(N)/N, mode='valid'), label='out_loss', color='green')
plt.plot(epoch_list[:l], np.convolve(der_losses_train, np.ones(N)/N, mode='valid'), label='der_loss', color='blue')
plt.plot(epoch_list[:l], np.convolve(init_losses_train, np.ones(N)/N, mode='valid'), label='init_loss', color='orange')
plt.plot(epoch_list[:l], np.convolve(init_pde_losses_train, np.ones(N)/N, mode='valid'), label='init_pde_loss', color='purple')
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
init_pde_losses_test = torch.tensor(init_pde_losses_test).cpu().numpy()
times_test = np.array(times_test)


loss_combination_test = np.column_stack([epoch_list, out_losses_test, der_losses_test, pde_losses_test, init_losses_test, tot_losses_test, init_pde_losses_test, times_test])
with open(f'{EXP_PATH}/{name}/plots{title_mode}/testdata.npy', 'wb') as f:
    np.save(f, loss_combination_test)
    
plt.figure()
plt.plot(epoch_list, pde_losses_test, label='pde_loss', color='red')
plt.plot(epoch_list, out_losses_test, label='out_loss', color='green')
plt.plot(epoch_list, der_losses_test, label='der_loss', color='blue')
plt.plot(epoch_list, init_losses_test, label='init_loss', color='orange')
plt.plot(epoch_list, init_pde_losses_test, label='init_pde_loss', color='purple')
plt.legend()
plt.yscale('log')
plt.title('Losses of the student model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(f'{EXP_PATH}/{name}/plots{title_mode}/test_losses.png')