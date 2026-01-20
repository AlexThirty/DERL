import torch
from model import KdVPINN, SinActivation
import argparse
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import os
from torch.func import vmap, jacrev, hessian
from itertools import cycle

font = {'size'   : 16}
import matplotlib
matplotlib.rc('font', **font)

seed = 30
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--init_weight', default=5., type=float, help='Weight for the init loss')
parser.add_argument('--pde_weight', default=0.1, type=float, help='Weight for the F loss')
parser.add_argument('--sys_weight', default=0.1, type=float, help='Weight for the system loss')
parser.add_argument('--bc_weight', default=1., type=float, help='Weight for the F loss')
parser.add_argument('--lr_init', default=1e-4, type=float, help='Starting learning rate')
parser.add_argument('--device', default='cuda:2', type=str, help='Device to use')
parser.add_argument('--name', default='kdv', type=str, help='Experiment name')
parser.add_argument('--train_steps', default=100000, type=int, help='Number of training steps')
parser.add_argument('--epochs', default=500, type=int, help='Number of epochs')
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
    

model_pinn = KdVPINN(hidden_units=[units for _ in range(layers)],
                pde_weight=pde_weight,
                sys_weight=sys_weight,
                bc_weight=bc_weight,
                init_weight=init_weight,
                device=device,
                activation=torch.nn.Tanh(),
                lr=lr_init)

model_pinn.eval()

model_0 = KdVPINN(hidden_units=[units for _ in range(layers)],
                pde_weight=pde_weight,
                sys_weight=sys_weight,
                bc_weight=bc_weight,
                init_weight=init_weight,
                device=device,
                activation=torch.nn.Tanh(),
                lr=lr_init)

model_0.eval()

model_1 = KdVPINN(hidden_units=[units for _ in range(layers)],
                pde_weight=pde_weight,
                sys_weight=sys_weight,
                bc_weight=bc_weight,
                init_weight=init_weight,
                device=device,
                activation=torch.nn.Tanh(),
                lr=lr_init)

model_1.eval()


model_hes = KdVPINN(hidden_units=[units for _ in range(layers)],
                pde_weight=pde_weight,
                sys_weight=sys_weight,
                bc_weight=bc_weight,
                init_weight=init_weight,
                device=device,
                activation=torch.nn.Tanh(),
                lr=lr_init)

model_hes.eval()

model_sob = KdVPINN(hidden_units=[units for _ in range(layers)],
                pde_weight=pde_weight,
                sys_weight=sys_weight,
                bc_weight=bc_weight,
                init_weight=init_weight,
                device=device,
                activation=torch.nn.Tanh(),
                lr=lr_init)

model_sob.eval()


model_sobhes = KdVPINN(hidden_units=[units for _ in range(layers)],
                pde_weight=pde_weight,
                sys_weight=sys_weight,
                bc_weight=bc_weight,
                init_weight=init_weight,
                device=device,
                activation=torch.nn.Tanh(),
                lr=lr_init)

model_sobhes.eval()

model_0hes = KdVPINN(hidden_units=[units for _ in range(layers)],
                pde_weight=pde_weight,
                sys_weight=sys_weight,
                bc_weight=bc_weight,
                init_weight=init_weight,
                device=device,
                activation=torch.nn.Tanh(),
                lr=lr_init)

model_0hes.eval()

# Load it to be sure it works
model_pinn.load_state_dict(torch.load(f'./pinn/saved_models/kdv_net'))
model_0.load_state_dict(torch.load(f'./studentDerivative/saved_models/kdv_net'))
model_1.load_state_dict(torch.load(f'./studentOutput/saved_models/kdv_net'))
model_hes.load_state_dict(torch.load(f'./studentHessian/saved_models/kdv_net'))
model_sobhes.load_state_dict(torch.load(f'./studentSobolev+Hessian/saved_models/kdv_net'))
model_sob.load_state_dict(torch.load(f'./studentSobolev/saved_models/kdv_net'))
model_0hes.load_state_dict(torch.load(f'./studentDerivative+Hessian/saved_models/kdv_net'))



import numpy as np
from matplotlib import pyplot as plt
#plotting function to generate the figures for the ball problem
from matplotlib.colors import TwoSlopeNorm

if not os.path.exists(f'plotsjoint'):
    os.makedirs(f'plotsjoint')

xlim = 1.
ylim = 1.

# Import the true function
with open(f'data/kdv_data.npy', 'rb') as f:
    kdv_data = np.load(f)
    print(f'kdv_data.shape: {kdv_data.shape}')



from plotting_utils import plot_errors

plot_errors([model_0, model_1, model_sob, model_0hes, model_hes, model_sobhes], ['DERL', 'OUTL', 'SOB', 'DER+HESL', 'HESL', 'SOB+HESL'], f'plotsjoint', kdv_data, model_pinn,)
#plot_errors([model_0, model_1, model_0hes, model_hes], ['Derivative', 'Vanilla', 'Derivative+Hessian', 'Hessian'], f'./plotsjoint', kdv_data, model_pinn, '_no_sob')

def plot_model(model:KdVPINN, plot_title):
    if not os.path.exists(f'{plot_title}/plots'):
        os.mkdir(f'{plot_title}/plots')
    # Plot the solution
    dt = 0.005
    dx = 0.005

    points_x = np.arange(start=-1, stop=1+dx, step=dx)
    points_t = np.arange(start=0, stop=1+dt, step=dt)

    X,Y = np.meshgrid(points_t, points_x)
    pts = np.vstack([X.reshape(-1),Y.reshape(-1)]).T
    x_plot = points_x.copy()

    # Get the predictions
    u_pred = model.forward(torch.from_numpy(pts).to(device).float().requires_grad_(True)).cpu().detach().numpy().reshape(X.shape)
    
    # Plotting the solution in time space
    fig = plt.figure()
    cmap = plt.colormaps["jet"]
    cmap = cmap.with_extremes(bad=cmap(0))
    pcm = plt.pcolormesh(X, Y, u_pred, cmap=cmap,
                            rasterized=True)
    fig.colorbar(pcm, label="value", pad=0)
    plt.title("Predicted KdV solution in the domain")
    plt.xlabel("t")
    plt.ylabel("x")

    plt.savefig(f'./{plot_title}/kdv_tx_plot.pdf', format='pdf')
    plt.close()
    
    # Plotting the true solution
    # Plotting the solution in time space
    fig = plt.figure()
    cmap = plt.colormaps["jet"]
    cmap = cmap.with_extremes(bad=cmap(0))
    pcm = plt.pcolormesh(X, Y, kdv_data.T, cmap=cmap,
                            rasterized=True)
    fig.colorbar(pcm, label="value", pad=0)
    plt.title("Predicted KdV solution in the domain")
    plt.xlabel("t")
    plt.ylabel("x")

    plt.savefig(f'./{plot_title}/kdv_true_plot.pdf', format='pdf')
    plt.close()
    
    # Plotting the error wrt the ground truth
    fig = plt.figure()
    cmap = plt.colormaps["jet"]
    cmap = cmap.with_extremes(bad=cmap(0))
    pcm = plt.pcolormesh(X, Y, np.abs(u_pred-kdv_data.T), cmap=cmap, rasterized=True)
    fig.colorbar(pcm, label="value", pad=0)
    plt.title("Absolute error of the KdV solution in the domain")
    plt.xlabel("t")
    plt.ylabel("x")
    
    plt.savefig(f'./{plot_title}/kdv_tx_error.pdf', format='pdf')
    plt.close()
    
    
    # Now lets plot the PDE consistency
    # Calculate the PDE residual
    pde_pred = model.evaluate_consistency(torch.from_numpy(pts).to(device).float().requires_grad_(True)).cpu().detach().numpy().reshape(X.shape)
    fig = plt.figure()
    cmap = plt.colormaps["jet"]
    cmap = cmap.with_extremes(bad=cmap(0))
    pcm = plt.pcolormesh(X, Y, np.abs(pde_pred), cmap=cmap, rasterized=True)
    fig.colorbar(pcm, label="value", pad=0)
    plt.title("PDE residual in the domain")
    plt.xlabel("t")
    plt.ylabel("x")
    
    plt.savefig(f'./{plot_title}/kdv_pde_consistency.pdf', format='pdf')
    plt.close()
    
    # Plotting the solution in x space
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f'KdV solution at for {plot_title}')
    X_multiplot = torch.column_stack([torch.zeros(x_plot.shape), torch.tensor(x_plot, dtype=torch.float32)]).to(device=device)
    u = model.forward(X_multiplot).cpu().detach().numpy()
    axs[0, 0].plot(x_plot, u, label='Predicted', color='blue')
    axs[0, 0].plot(x_plot, kdv_data[0, :], label='True', color='red')
    axs[0, 0].set_title('t = 0')

    X_multiplot = torch.column_stack([0.33*torch.ones(x_plot.shape), torch.tensor(x_plot, dtype=torch.float32)]).to(device=device)
    u = model.forward(X_multiplot).cpu().detach().numpy()
    axs[0, 1].plot(x_plot, u, label='Predicted', color='blue')
    axs[0, 1].plot(x_plot, kdv_data[int(0.33/dt), :], label='True', color='red')
    axs[0, 1].set_title(' t = 0.33')

    X_multiplot = torch.column_stack([0.66*torch.ones(x_plot.shape), torch.tensor(x_plot, dtype=torch.float32)]).to(device=device)
    u = model.forward(X_multiplot).cpu().detach().numpy()
    axs[1, 0].plot(x_plot, u, label='Predicted', color='blue')
    axs[1, 0].plot(x_plot, kdv_data[int(0.66/dt), :], label='True', color='red')
    axs[1, 0].set_title('t = 0.66')

    X_multiplot = torch.column_stack([1.*torch.ones(x_plot.shape), torch.tensor(x_plot, dtype=torch.float32)]).to(device=device)
    u = model.forward(X_multiplot).cpu().detach().numpy()
    axs[1, 1].plot(x_plot, u, label='Predicted', color='blue')
    axs[1, 1].plot(x_plot, kdv_data[int(0.99/dt), :], label='True', color='red')
    axs[1, 1].set_title('t = 1')


    handles, labels = axs[0,0].get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper right')
    for ax in axs.flat:
        ax.set(xlabel='x', ylabel='u(x,t)')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(f'./{plot_title}/kdv_multiplot.pdf', format='pdf')

    with open(f'./{plot_title}/traindata.npy', 'rb') as f:
        loss_combination_train = np.load(f)
        
    with open(f'./{plot_title}/testdata.npy', 'rb') as f:
        loss_combination_test = np.load(f)
        
    N = 20
    step_list = loss_combination_train[:, 0]
    pde_losses = loss_combination_train[:, 1]
    init_losses = loss_combination_train[:, 2]
    bc_losses = loss_combination_train[:, 3]
    tot_losses = loss_combination_train[:, 4]
    out_losses = loss_combination_train[:, 5]
    

    l = len(np.convolve(pde_losses, np.ones(N)/N, mode='valid'))
    plt.figure()
    conv_pde_losses = np.convolve(pde_losses, np.ones(N)/N, mode='valid')
    conv_out_losses = np.convolve(out_losses, np.ones(N)/N, mode='valid')
    conv_init_losses = np.convolve(init_losses, np.ones(N)/N, mode='valid')
    conv_bc_losses = np.convolve(bc_losses, np.ones(N)/N, mode='valid')

    plt.plot(step_list[:l], conv_pde_losses, label='pde_loss', color='red')
    plt.plot(step_list[:l], conv_out_losses, label='out_loss', color='green')
    plt.plot(step_list[:l], conv_init_losses, label='init_loss', color='orange')
    plt.plot(step_list[:l], conv_bc_losses, label='bc_loss', color='purple')
    #plt.plot(step_list, tot_losses, label='tot_loss', color='black')
    plt.legend()
    plt.yscale('log')
    plt.savefig(f'./{plot_title}/trainlosses.pdf', format='pdf')

    N = 20

    
    step_list = loss_combination_test[:, 0]
    pde_losses = loss_combination_test[:, 1]
    init_losses = loss_combination_test[:, 2]
    bc_losses = loss_combination_test[:, 3]
    tot_losses = loss_combination_test[:, 4]
    out_losses = loss_combination_test[:, 5]
    if plot_title != 'pinn':
        test_times = loss_combination_test[:, 6]
    else:
        test_times = np.zeros_like(pde_losses)
    
    with open(f'./plotsjoint/losses.txt', 'a') as f:
        print(f'{plot_title}', file=f)
        print(f'pde_loss: {np.mean(np.sqrt(pde_losses[-10:]))}', file=f)
        print(f'init_loss: {np.mean(np.sqrt(init_losses[-10:]))}', file=f)
        print(f'bc_loss: {np.mean(np.sqrt(bc_losses[-10:]))}', file=f)
        print(f'tot_loss: {np.mean(np.sqrt(tot_losses[-10:]))}', file=f)
        print(f'out_loss: {np.mean(np.sqrt(out_losses[-10:]))}', file=f)
        print(f'test_times: {np.mean(test_times[-10:])}', file=f)
    
    l = len(np.convolve(pde_losses, np.ones(N)/N, mode='valid'))
    plt.figure()
    conv_pde_losses = np.convolve(pde_losses, np.ones(N)/N, mode='valid')
    conv_out_losses = np.convolve(out_losses, np.ones(N)/N, mode='valid')
    conv_init_losses = np.convolve(init_losses, np.ones(N)/N, mode='valid')
    conv_bc_losses = np.convolve(bc_losses, np.ones(N)/N, mode='valid')

    plt.plot(step_list[:l], conv_pde_losses, label='pde_loss', color='red')
    plt.plot(step_list[:l], conv_out_losses, label='out_loss', color='green')
    plt.plot(step_list[:l], conv_init_losses, label='init_loss', color='orange')
    plt.plot(step_list[:l], conv_bc_losses, label='bc_loss', color='purple')
    #plt.plot(step_list[:l], pde_losses, label='pde_loss', color='red')
    #plt.plot(step_list[:l], out_losses, label='out_loss', color='green')
    #plt.plot(step_list[:l], init_losses, label='init_loss', color='orange')
    #plt.plot(step_list[:l], bc_losses, label='bc_loss', color='purple')
    #plt.plot(step_list, tot_losses, label='tot_loss', color='black')
    plt.legend()
    plt.yscale('log')
    plt.savefig(f'./{plot_title}/testlosses.pdf', format='pdf')

    print('Plotting done!')

plot_model(model_0, 'studentDerivative')
plot_model(model_1, 'studentOutput')
plot_model(model_hes, 'studentHessian')
plot_model(model_sobhes, 'studentSobolev+Hessian')
plot_model(model_sob, 'studentSobolev')
plot_model(model_0hes, 'studentDerivative+Hessian')
plot_model(model_pinn, 'pinn')

N = 21

from scipy import ndimage, signal
import scipy.fftpack
def plot_loss_curves(to_plot, step_list, names, path, title, colors):
    plt.figure()
    for i in range(len(to_plot)):
        y2 = ndimage.median_filter(to_plot[i], size=51)
        plt.plot(step_list, y2, label=names[i], color = colors[i])
    plt.legend()
    plt.yscale('log')
    plt.title(title)
    plt.savefig(path, dpi=300)
    plt.close()
    
with open(f'./studentDerivative/traindata.npy', 'rb') as f:
    derivative_losses = np.load(f)

with open(f'./studentOutput/traindata.npy', 'rb') as f:
    output_losses = np.load(f)

with open(f'./studentSobolev/traindata.npy', 'rb') as f:
    sobolev_losses = np.load(f)
    
with open(f'./pinn/traindata.npy', 'rb') as f:
    negdata_losses = np.load(f)
    
with open(f'./studentHessian/traindata.npy', 'rb') as f:
    hessian_losses = np.load(f)

with open(f'./studentSobolev+Hessian/traindata.npy', 'rb') as f:
    sobolev_hessian_losses = np.load(f)
    
with open(f'./studentDerivative+Hessian/traindata.npy', 'rb') as f:
    derivative_hessian_losses = np.load(f)
    
step_list = derivative_losses[:,0]


colors = ['blue', 'red', 'purple', 'green', 'orange', 'darkblue', 'darkred']


print(derivative_hessian_losses[-20:, 1])

plot_loss_curves([derivative_losses[:,1], output_losses[:,1], sobolev_losses[:,1], negdata_losses[:,1], hessian_losses[:,1], sobolev_hessian_losses[:,1], derivative_hessian_losses[:,1]], step_list, ['DERL', 'OUTL', 'SOB', 'PINN', 'HESL', 'SOB+HESL', 'DER+HESL'], f'./{name}/plotsjoint/pde_loss.pdf', 'PDE Loss', colors)
plot_loss_curves([derivative_losses[:,2], output_losses[:,2], sobolev_losses[:,2], negdata_losses[:,2], hessian_losses[:,2], sobolev_hessian_losses[:,2], derivative_hessian_losses[:,2]], step_list, ['DERL', 'OUTL', 'SOB', 'PINN', 'HESL', 'SOB+HESL', 'DER+HESL'], f'./{name}/plotsjoint/init_loss.pdf', 'Init Loss', colors)
plot_loss_curves([derivative_losses[:,3], output_losses[:,3], sobolev_losses[:,3], negdata_losses[:,3], hessian_losses[:,3], sobolev_hessian_losses[:,3], derivative_hessian_losses[:,3]], step_list, ['DERL', 'OUTL', 'SOB', 'PINN', 'HESL', 'SOB+HESL', 'DER+HESL'], f'./{name}/plotsjoint/bc_loss.pdf', 'BC Loss', colors)
plot_loss_curves([derivative_losses[:,5], output_losses[:,5], sobolev_losses[:,5], negdata_losses[:,5], hessian_losses[:,5], sobolev_hessian_losses[:,5], derivative_hessian_losses[:,5]], step_list, ['DERL', 'OUTL', 'SOB', 'PINN', 'HESL', 'SOB+HESL', 'DER+HESL'], f'./{name}/plotsjoint/out_loss.pdf', 'Output Loss', colors)
