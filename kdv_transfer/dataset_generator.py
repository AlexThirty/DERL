import torch
from torch.utils.data import TensorDataset
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator

dt = 0.005
dx = 0.005
xs = np.arange(start=-1, stop=1+dx, step=dx)
ts = np.arange(0, 1+dt, step=dt)

def kdv_init(x):
    return torch.cos(torch.pi*x[:,1]).reshape((-1,1))

if not os.path.exists('data'):
    os.makedirs('data')

with open('data/kdv_data.npy', 'rb') as f:
    kdv_data = np.load(f)
    print(kdv_data.shape)
    
with open('data/kdv_ders.npy', 'rb') as f:
    dudt, ux = np.load(f, allow_pickle=True)
    print(dudt.shape, ux.shape)
    

kdv_ders = np.column_stack((dudt.reshape((-1,1)), ux.reshape((-1,1))))
    
xmin = -1.
xmax = 1.
tmin = 0.
tmax = 1.   
dt = 0.005
dx = 0.005
grid_size = dx*dt

points_x = np.arange(start=xmin, stop=xmax+dx, step=dx)
points_t = np.arange(start=tmin, stop=tmax+dt, step=dt)

X,Y = np.meshgrid(points_t, points_x)
pts = np.vstack([X.reshape(-1),Y.reshape(-1)]).T

kdv_sol_dataset = TensorDataset(torch.from_numpy(pts), torch.from_numpy(kdv_data.reshape((-1,1))), torch.from_numpy(kdv_ders))
torch.save(kdv_sol_dataset, os.path.join('data', f'kdv_sol_dataset.pth'))

kdv_interp = RegularGridInterpolator((ts, xs), kdv_data, method='cubic')

num_domain = 100000
num_boundary = 1000
t_max = 1.0


# Generate the times
t =  t_max*torch.distributions.Uniform(0.,1.).sample((num_domain,1))
# Generate the points
points = torch.distributions.Uniform(-1.,1.).sample((num_domain, 1))
# Stack them
x_pde = torch.column_stack((t, points))
print(f'x_pde.shape: {x_pde.shape}')
# True target
y_pde = kdv_interp(x_pde).reshape((-1,1))
print(f'y_pde.shape: {y_pde.shape}')
# Generate the dataset
pde_dataset = TensorDataset(x_pde, torch.from_numpy(y_pde))
torch.save(pde_dataset, os.path.join('data', f'kdv_pde_dataset.pth'))


t, x = np.meshgrid(ts, xs, indexing='ij')
x_pde = np.column_stack((t.reshape((-1,1)), x.reshape((-1,1))))
y_pde = kdv_data.reshape((-1,1))
pde_dataset = TensorDataset(torch.from_numpy(x_pde), torch.from_numpy(y_pde))
print(f'x_pde.shape: {x_pde.shape}')
print(f'y_pde.shape: {y_pde.shape}')
torch.save(pde_dataset, os.path.join('data', f'kdv_pde_dataset.pth'))

# Filter points with t < 0.35
mask = x_pde[:,0] <= 0.35
x_pde_filt = x_pde[mask]
y_pde_filt = y_pde[mask]
pde_dataset = TensorDataset(torch.from_numpy(x_pde_filt), torch.from_numpy(y_pde_filt))
print(f'Filtered x_pde.shape: {x_pde_filt.shape}')
print(f'Filtered y_pde.shape: {y_pde_filt.shape}')
torch.save(pde_dataset, os.path.join('data', f'kdv_pde_dataset_phase0.pth'))

# Filter points with t<0.7
mask = x_pde[:,0] <= 0.7
x_pde_filt = x_pde[mask]
y_pde_filt = y_pde[mask]
pde_dataset = TensorDataset(torch.from_numpy(x_pde_filt), torch.from_numpy(y_pde_filt))
print(f'Filtered x_pde.shape: {x_pde_filt.shape}')
print(f'Filtered y_pde.shape: {y_pde_filt.shape}')
torch.save(pde_dataset, os.path.join('data', f'kdv_pde_dataset_phase1.pth'))

# Generate the init times
# Generate the times
# Generate the points
#points = torch.distributions.Uniform(-1.,1.).sample((num_boundary, 1))
points = torch.from_numpy(xs.copy())
t = torch.zeros_like(points)
# Stack them
x_init = torch.column_stack((t, points))
print(f'x_init.shape: {x_init.shape}')
# Get the initial values
y_init = kdv_init(x_init).reshape((-1,1))

points = torch.distributions.Uniform(-1.,1.).sample((num_boundary, 1))
points = torch.from_numpy(xs.copy())
t = torch.ones_like(points)
x_init = torch.cat((x_init, torch.column_stack((t, points))), dim=0)
print(f'x_init.shape: {x_init.shape}')
y_init = torch.cat((y_init.reshape((-1)), torch.from_numpy(kdv_data[-1,:].reshape((-1)))), dim=0).reshape((-1,1))
print(f'y_init.shape: {y_init.shape}')
# Generate the dataset
init_dataset = TensorDataset(x_init, y_init)
torch.save(init_dataset, os.path.join('data', f'kdv_init_dataset.pth'))

# Generate the times
#t =  t_max*torch.distributions.Uniform(0.,1.).sample((num_boundary,1))
# Generate the points
t = torch.from_numpy( ts.copy() )
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
t = torch.from_numpy( ts.copy() )
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
torch.save(bc1_dataset, os.path.join('data', f'kdv_bc1_dataset.pth'))
torch.save(bc2_dataset, os.path.join('data', f'kdv_bc2_dataset.pth'))