import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from torch import nn
from collections import OrderedDict
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

font = {'size'   : 16}
import matplotlib
matplotlib.rc('font', **font)
parser = argparse.ArgumentParser()
parser.add_argument('--init_weight', default=1., type=float, help='Weight for the init loss')
parser.add_argument('--sys_weight', default=1., type=float, help='Weight for the rho loss')
parser.add_argument('--lr_init', default=5e-4, type=float, help='Starting learning rate')
parser.add_argument('--device', default='cuda:2', type=str, help='Device to use')
parser.add_argument('--name', default='true', type=str, help='Experiment name')
parser.add_argument('--train_steps', default=10000, type=int, help='Number of training steps')
parser.add_argument('--epochs', default=500, type=int, help='Number of epochs')
parser.add_argument('--mode', default=0, type=int, help='Mode: -1 for LNN learning, 0 for derivative learning, 1 for output learning')
parser.add_argument('--batch_size', default=256, type=int, help='Number of samples per step')
parser.add_argument('--layers', default=4, type=int, help='Number of layers in the network')
parser.add_argument('--units', default=50, type=int, help='Number of units per layer in the network')
b = 0
EXP_PATH = '.'

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
dt = 1e-3

if not os.path.exists(f'{EXP_PATH}/{name}'):
    os.mkdir(f'{EXP_PATH}/{name}')

new_data = None
if name == 'true':
    prefix = 'true'
    new_data = torch.load(os.path.join('data', f'true_dataset_test.pth'))
elif name == 'extrapolate':
    prefix = 'true_extrapolate'
    new_data = torch.load(os.path.join('data', f'true_extrapolate_dataset_new.pth'))
elif name == 'interpolate':
    prefix = 'true_interpolate'
    new_data = torch.load(os.path.join('data', f'true_interpolate_dataset_new.pth'))
elif name == 'adapt':
    prefix = 'true_adapt'
    new_data = torch.load(os.path.join('data', f'true_adapt_dataset_new.pth'))
elif name == 'emp':
    prefix = 'emp'
    new_data = torch.load(os.path.join('data', f'emp_dataset_test.pth'))
else:
    raise ValueError(f'name value is not in the options')



if mode == 0:
    title_mode = 'Derivative'
elif mode == 1:
    title_mode = 'HNN'
elif mode == -1:
    title_mode = 'LNN'
else:
    raise ValueError('Mode is not valid')
    
print('Loading the data...')    

# Load the data
train_dataset = torch.load(os.path.join('data', f'{prefix}_dataset_train.pth'))
test_dataset = torch.load(os.path.join('data', f'{prefix}_dataset_test.pth'))

if name in ['adapt', 'interpolate']:
    bc_dataset = torch.load(os.path.join('data', f'{prefix}_bc_train.pth'))
else:
    bc_dataset = None
# Generate the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size, generator=gen, shuffle=True, num_workers=12)
test_dataloader = DataLoader(test_dataset, batch_size, generator=gen, shuffle=True, num_workers=12)

print('Data loaded!')

activation = torch.nn.Tanh()

model_0 = PendulumNet(
    init_weight=init_weight,
    sys_weight=sys_weight,
    pde_weight=0.,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)




torch.cuda.empty_cache()
# %%
import os
# %%
model_0.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/pendulum_netDerivative'))

from external_models.hnn import HNN
from external_models.hnn_nn_models import MLP
output_dim = 2
nn_model = MLP(2, 20,  output_dim, 'tanh')
model_1 = HNN(2, differentiable_model=nn_model,
              field_type='solenoidal', baseline=False, device=device)

model_1.load_state_dict(torch.load(f'{EXP_PATH}/{name}/plotshnn/pend-hnn.tar'))

from external_models.lnn_hps import learned_dynamics, extended_mlp
from external_models.lnn import raw_lagrangian_eom, raw_lagrangian_eom_damped

### Load the LNN
class ObjectView(object):
    def __init__(self, d): self.__dict__ = d
args = ObjectView({'dataset_size': 200,
 'fps': 10,
 'samples': 100,
 'num_epochs': 80000,
 'seed': 30,
 'loss': 'l1',
 'act': 'softplus',
 'hidden_dim': 600,
 'output_dim': 1,
 'layers': 3,
 'n_updates': 1,
 'lr': 0.001,
 'lr2': 2e-05,
 'dt': 0.1,
 'model': 'gln',
 'batch_size': 512,
 'l2reg': 5.7e-07,
})
import jax
rng = jax.random.PRNGKey(args.seed)

init_random_params, nn_forward_fn = extended_mlp(args)
from external_models import lnn_hps
lnn_hps.nn_forward_fn = nn_forward_fn
_, init_params = init_random_params(rng+1, (-1, 2))
model = (nn_forward_fn, init_params)

from flax.training import checkpoints, train_state

(nn_forward_fn, init_params) = model

checkpoint_dir = os.path.abspath(f'{EXP_PATH}/true/saved_models')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, 'pendulum_lnn')
# Load the model parameters
params = checkpoints.restore_checkpoint(checkpoint_path, init_params)


# %%
import numpy as np
from matplotlib import pyplot as plt
#plotting function to generate the figures for the ball problem
from matplotlib.colors import TwoSlopeNorm

if not os.path.exists(f'{EXP_PATH}/{name}/plotsextras'):
    os.mkdir(f'{EXP_PATH}/{name}/plotsextras')




# Number of points for the field
N = 250
xlim = np.pi/2
ylim = 1.5
X,Y = np.meshgrid(np.linspace(-xlim,xlim,N),np.linspace(-ylim,ylim,N))
pts = np.vstack([X.reshape(-1),Y.reshape(-1)]).T

#plots the streamplot for the velocity field
plt.figure(figsize=(5,5))
#print(pts)
vel = u_vec(torch.from_numpy(pts), b=b)
#print(vel)
U = np.array(vel[:,0].reshape(X.shape))
V = np.array(vel[:,1].reshape(Y.shape))
#mask the outside of the ball

vel_true = u_vec(torch.from_numpy(pts), b=b)
#print(vel)
U_true = np.array(vel_true[:,0].reshape(X.shape))
V_true = np.array(vel_true[:,1].reshape(Y.shape))

vel_0 = model_0.evaluate_field(torch.column_stack((0*torch.ones((pts.shape[0],1)),torch.from_numpy(pts).float())).to(device)).detach().cpu().numpy()
#print(vel)
U_0 = np.array(vel_0[:,0].reshape(X.shape))
V_0 = np.array(vel_0[:,1].reshape(Y.shape))
error_0 = np.sqrt((U_0-U_true)**2+(V_0-V_true)**2)

vel_1 = model_1.time_derivative(torch.from_numpy(pts).to(device).float().requires_grad_(True)).detach().cpu().numpy()
vel_1[:,1] = vel_1[:,1] - b*pts[:,1]
#print(vel)
U_1 = np.array(vel_1[:,0].reshape(X.shape))
V_1 = np.array(vel_1[:,1].reshape(Y.shape))
error_1 = np.sqrt((U_1-U_true)**2+(V_1-V_true)**2)
from functools import partial
import jax
vel_neg = jax.vmap(partial(raw_lagrangian_eom_damped, learned_dynamics(params)))(pts)

#print(vel)
U_neg = np.array(vel_neg[:,0].reshape(X.shape))
V_neg = np.array(vel_neg[:,1].reshape(Y.shape))
error_neg = np.sqrt((U_neg-U_true)**2+(V_neg-V_true)**2)

grid_size = (2*xlim/N)*(2*ylim/N)

    
from plotting_utils import error_plotting, comparison_plotting

error_plotting([error_0, error_1, error_neg], X, Y, [U_0, U_1, U_neg], [V_0, V_1, V_neg], U_true, V_true, ['DERL', 'HNN', 'LNN'], path=f'{EXP_PATH}/{name}/plotsextras/error_field_wrttrue.pdf')
comparison_plotting([error_1-error_0, error_neg-error_0], X, Y, [U_0, U_1, U_neg], [V_0,  V_1, V_neg], U_true, V_true, ['HNN', 'LNN'], path=f'{EXP_PATH}/{name}/plotsextras/error_field_wrt0.pdf')


with open(f'{EXP_PATH}/{name}/plotsextras/losses.txt', 'w') as f:
    print('Field error averaged over the domain', file=f)
    print(f'Derivative learning: mean {np.mean(error_0)}, std {np.std(error_0)}', file=f)
    print(f'Derivative L2 norm: {np.sqrt(grid_size*np.sum(error_0**2))}', file=f)
    print(f'Derivative L2 norm normalized: {np.sqrt(np.sum(error_0**2)/np.sum(U_true**2))}', file=f)
    
    print(f'HNN : mean {np.mean(error_1)}, std {np.std(error_1)}', file=f)
    print(f'HNN L2 norm: {np.sqrt(grid_size*np.sum(error_1**2))}', file=f)
    print(f'HNN L2 norm normalized: {np.sqrt(np.sum(error_1**2)/np.sum(U_true**2))}', file=f)
    
    print(f'LNN: mean {np.mean(error_neg)}, std {np.std(error_neg)}', file=f)
    print(f'LNN L2 norm: {np.sqrt(grid_size*np.sum(error_neg**2))}', file=f) 
    print(f'LNN L2 norm normalized: {np.sqrt(np.sum(error_neg**2)/np.sum(U_true**2))}', file=f)   
print('Field error averaged over the domain')
print(f'Derivative learning: mean {np.mean(error_0)}, std {np.std(error_0)}')
print(f'HNN : mean {np.mean(error_1)}, std {np.std(error_1)}')
print(f'LNN: mean {np.mean(error_neg)}, std {np.std(error_neg)}')
import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from torch import nn
from collections import OrderedDict
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

font = {'size'   : 16}
import matplotlib
matplotlib.rc('font', **font)
parser = argparse.ArgumentParser()
parser.add_argument('--init_weight', default=1., type=float, help='Weight for the init loss')
parser.add_argument('--sys_weight', default=1., type=float, help='Weight for the rho loss')
parser.add_argument('--lr_init', default=5e-4, type=float, help='Starting learning rate')
parser.add_argument('--device', default='cuda:2', type=str, help='Device to use')
parser.add_argument('--name', default='true', type=str, help='Experiment name')
parser.add_argument('--train_steps', default=10000, type=int, help='Number of training steps')
parser.add_argument('--epochs', default=500, type=int, help='Number of epochs')
parser.add_argument('--mode', default=0, type=int, help='Mode: -1 for LNN learning, 0 for derivative learning, 1 for output learning')
parser.add_argument('--batch_size', default=256, type=int, help='Number of samples per step')
parser.add_argument('--layers', default=4, type=int, help='Number of layers in the network')
parser.add_argument('--units', default=50, type=int, help='Number of units per layer in the network')
b = 0.
EXP_PATH = '.'

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
dt = 1e-3

if not os.path.exists(f'{EXP_PATH}/{name}'):
    os.mkdir(f'{EXP_PATH}/{name}')

new_data = None
if name == 'true':
    prefix = 'true'
    new_data = torch.load(os.path.join('data', f'true_dataset_test.pth'))
elif name == 'extrapolate':
    prefix = 'true_extrapolate'
    new_data = torch.load(os.path.join('data', f'true_extrapolate_dataset_new.pth'))
elif name == 'interpolate':
    prefix = 'true_interpolate'
    new_data = torch.load(os.path.join('data', f'true_interpolate_dataset_new.pth'))
elif name == 'adapt':
    prefix = 'true_adapt'
    new_data = torch.load(os.path.join('data', f'true_adapt_dataset_new.pth'))
elif name == 'emp':
    prefix = 'emp'
    new_data = torch.load(os.path.join('data', f'emp_dataset_test.pth'))
else:
    raise ValueError(f'name value is not in the options')



if mode == 0:
    title_mode = 'Derivative'
elif mode == 1:
    title_mode = 'HNN'
elif mode == -1:
    title_mode = 'LNN'
else:
    raise ValueError('Mode is not valid')
    
print('Loading the data...')    

# Load the data
train_dataset = torch.load(os.path.join('data', f'{prefix}_dataset_train.pth'))
test_dataset = torch.load(os.path.join('data', f'{prefix}_dataset_test.pth'))

if name in ['adapt', 'interpolate']:
    bc_dataset = torch.load(os.path.join('data', f'{prefix}_bc_train.pth'))
else:
    bc_dataset = None
# Generate the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size, generator=gen, shuffle=True, num_workers=12)
test_dataloader = DataLoader(test_dataset, batch_size, generator=gen, shuffle=True, num_workers=12)

print('Data loaded!')

activation = torch.nn.Tanh()

model_0 = PendulumNet(
    init_weight=init_weight,
    sys_weight=sys_weight,
    pde_weight=0.,
    hidden_units=[units for _ in range(layers)],
    lr_init=lr_init,
    device=device,
    activation=activation,    
    last_activation=False,
).to(device)




torch.cuda.empty_cache()
# %%
import os
# %%
model_0.load_state_dict(torch.load(f'{EXP_PATH}/{name}/saved_models/pendulum_netDerivative'))

from external_models.hnn import HNN
from external_models.hnn_nn_models import MLP
output_dim = 2
nn_model = MLP(2, 20,  output_dim, 'tanh')
model_1 = HNN(2, differentiable_model=nn_model,
              field_type='solenoidal', baseline=False, device=device)

model_1.load_state_dict(torch.load(f'{EXP_PATH}/{name}/plotshnn/pend-hnn.tar'))

from external_models.lnn_hps import learned_dynamics, extended_mlp
from external_models.lnn import raw_lagrangian_eom, raw_lagrangian_eom_damped

### Load the LNN
class ObjectView(object):
    def __init__(self, d): self.__dict__ = d
args = ObjectView({'dataset_size': 200,
 'fps': 10,
 'samples': 100,
 'num_epochs': 80000,
 'seed': 30,
 'loss': 'l1',
 'act': 'softplus',
 'hidden_dim': 600,
 'output_dim': 1,
 'layers': 3,
 'n_updates': 1,
 'lr': 0.001,
 'lr2': 2e-05,
 'dt': 0.1,
 'model': 'gln',
 'batch_size': 512,
 'l2reg': 5.7e-07,
})
import jax
rng = jax.random.PRNGKey(args.seed)

init_random_params, nn_forward_fn = extended_mlp(args)
from external_models import lnn_hps
lnn_hps.nn_forward_fn = nn_forward_fn
_, init_params = init_random_params(rng+1, (-1, 2))
model = (nn_forward_fn, init_params)

from flax.training import checkpoints, train_state

(nn_forward_fn, init_params) = model

checkpoint_dir = os.path.abspath(f'{EXP_PATH}/true/saved_models')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, 'pendulum_lnn')
# Load the model parameters
params = checkpoints.restore_checkpoint(checkpoint_path, init_params)


# %%
import numpy as np
from matplotlib import pyplot as plt
#plotting function to generate the figures for the ball problem
from matplotlib.colors import TwoSlopeNorm

if not os.path.exists(f'{EXP_PATH}/{name}/plotsextras'):
    os.mkdir(f'{EXP_PATH}/{name}/plotsextras')




# Number of points for the field
N = 250
xlim = np.pi/2
ylim = 1.5
X,Y = np.meshgrid(np.linspace(-xlim,xlim,N),np.linspace(-ylim,ylim,N))
pts = np.vstack([X.reshape(-1),Y.reshape(-1)]).T

#plots the streamplot for the velocity field
plt.figure(figsize=(5,5))
#print(pts)
vel = u_vec(torch.from_numpy(pts), b=b)
#print(vel)
U = np.array(vel[:,0].reshape(X.shape))
V = np.array(vel[:,1].reshape(Y.shape))
#mask the outside of the ball

vel_true = u_vec(torch.from_numpy(pts), b=b)
#print(vel)
U_true = np.array(vel_true[:,0].reshape(X.shape))
V_true = np.array(vel_true[:,1].reshape(Y.shape))

vel_0 = model_0.evaluate_field(torch.column_stack((0*torch.ones((pts.shape[0],1)),torch.from_numpy(pts).float())).to(device)).detach().cpu().numpy()
#print(vel)
U_0 = np.array(vel_0[:,0].reshape(X.shape))
V_0 = np.array(vel_0[:,1].reshape(Y.shape))
error_0 = np.sqrt((U_0-U_true)**2+(V_0-V_true)**2)

vel_1 = model_1.time_derivative(torch.from_numpy(pts).to(device).float().requires_grad_(True)).detach().cpu().numpy()
vel_1[:,1] = vel_1[:,1] - b*pts[:,1]
#print(vel)
U_1 = np.array(vel_1[:,0].reshape(X.shape))
V_1 = np.array(vel_1[:,1].reshape(Y.shape))
error_1 = np.sqrt((U_1-U_true)**2+(V_1-V_true)**2)
from functools import partial
import jax
vel_neg = jax.vmap(partial(raw_lagrangian_eom_damped, learned_dynamics(params)))(pts)

#print(vel)
U_neg = np.array(vel_neg[:,0].reshape(X.shape))
V_neg = np.array(vel_neg[:,1].reshape(Y.shape))
error_neg = np.sqrt((U_neg-U_true)**2+(V_neg-V_true)**2)

grid_size = (2*xlim/N)*(2*ylim/N)

    
from plotting_utils import error_plotting, comparison_plotting

error_plotting([error_0, error_1, error_neg], X, Y, [U_0, U_1, U_neg], [V_0, V_1, V_neg], U_true, V_true, ['DERL', 'HNN', 'LNN'], path=f'{EXP_PATH}/{name}/plotsextras/error_field_wrttrue.pdf')
comparison_plotting([error_1-error_0, error_neg-error_0], X, Y, [U_0, U_1, U_neg], [V_0,  V_1, V_neg], U_true, V_true, ['HNN', 'LNN'], path=f'{EXP_PATH}/{name}/plotsextras/error_field_wrt0.pdf')


with open(f'{EXP_PATH}/{name}/plotsextras/losses.txt', 'w') as f:
    print('Field error averaged over the domain', file=f)
    print(f'Derivative learning: mean {np.mean(error_0)}, std {np.std(error_0)}', file=f)
    print(f'Derivative L2 norm: {np.sqrt(grid_size*np.sum(error_0**2))}', file=f)
    print(f'Derivative L2 norm normalized: {np.sqrt(np.sum(error_0**2)/np.sum(U_true**2))}', file=f)
    
    print(f'HNN : mean {np.mean(error_1)}, std {np.std(error_1)}', file=f)
    print(f'HNN L2 norm: {np.sqrt(grid_size*np.sum(error_1**2))}', file=f)
    print(f'HNN L2 norm normalized: {np.sqrt(np.sum(error_1**2)/np.sum(U_true**2))}', file=f)
    
    print(f'LNN: mean {np.mean(error_neg)}, std {np.std(error_neg)}', file=f)
    print(f'LNN L2 norm: {np.sqrt(grid_size*np.sum(error_neg**2))}', file=f) 
    print(f'LNN L2 norm normalized: {np.sqrt(np.sum(error_neg**2)/np.sum(U_true**2))}', file=f)   
print('Field error averaged over the domain')
print(f'Derivative learning: mean {np.mean(error_0)}, std {np.std(error_0)}')
print(f'HNN : mean {np.mean(error_1)}, std {np.std(error_1)}')
print(f'LNN: mean {np.mean(error_neg)}, std {np.std(error_neg)}')