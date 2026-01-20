import torch
from torch.utils.data import TensorDataset
from model import ball_boundary_uniform, ball_uniform, v_init_ball
import random
import numpy as np
import os

seed = 30
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)

EXP_PATH = '.'
if not os.path.exists(EXP_PATH):
    os.makedirs(EXP_PATH)

# Parameters
t_max = 0.55
steps = 10000
batch_size = 1000
radius = 1.0
dim = 3
t_init_max = 0.01

tot_domain = steps*batch_size
tot_boundary = steps*batch_size

# Generate the times
t =  t_max*torch.distributions.Uniform(0.,1.).sample((tot_domain,1))
# Generate the points
points = ball_uniform(tot_domain, radius, dim)
# Stack them
x_pde = torch.column_stack((t, points))
print(f'x_pde.shape: {x_pde.shape}')
# Fake target
y_pde = torch.zeros(tot_domain).reshape((-1,1))
print(f'y_pde.shape: {y_pde.shape}')
# Generate the dataset
pde_dataset = TensorDataset(x_pde, y_pde)

# Generate the init times
t =  t_init_max*torch.distributions.Uniform(0.,1.).sample((tot_domain,1))
# Generate the points
points_init = ball_uniform(tot_domain, radius, dim)
# Stack them
x_init = torch.column_stack((t, points_init))
print(f'x_init.shape: {x_init.shape}')
# Get the initial values
y_init = v_init_ball(x_init)
print(f'y_init.shape: {y_init.shape}')
# Generate the dataset
init_dataset = TensorDataset(x_init, y_init)

# Generate the times
t =  t_max*torch.distributions.Uniform(0.,1.).sample((tot_boundary,1))
# Generate the points
points_bc = ball_boundary_uniform(tot_boundary, radius, dim)
# Stack them
x_bc = torch.column_stack((t, points_bc))
print(f'x_bc.shape: {x_bc.shape}')
# Fake target
y_bc = torch.zeros(tot_boundary).reshape((-1,1))
print(f'y_bc.shape: {y_bc.shape}')
# Generate the dataset
bc_dataset = TensorDataset(x_bc, y_bc)

torch.save(pde_dataset, f'{EXP_PATH}/data/pde_dataset.pth')
torch.save(init_dataset, f'{EXP_PATH}/data/init_dataset.pth')
torch.save(bc_dataset, f'{EXP_PATH}/data/bc_dataset.pth')
