import numpy as np
import torch
from torch.utils.data import TensorDataset
seed = 30
import random
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)
font = {'size'   : 16}
import matplotlib
matplotlib.rc('font', **font)
lam = 0.01

import os
if not os.path.exists('data'):
    os.makedirs('data')

def allen_cahn_true(x: np.array):
    return np.sin(np.pi*x[:,0])*np.sin(np.pi*x[:,1])

def allen_cahn_forcing(x: np.array):
    return -2*lam*np.pi**2*allen_cahn_true(x) + allen_cahn_true(x)**3 - allen_cahn_true(x)

def allen_cahn_pdv(x: np.array):
    ux = np.pi*np.cos(np.pi*x[:,0])*np.sin(np.pi*x[:,1])
    uy = np.pi*np.sin(np.pi*x[:,0])*np.cos(np.pi*x[:,1])
    return np.column_stack((ux, uy))

def allen_cahn_hes(x: np.array):
    uxx = -np.pi**2*np.sin(np.pi*x[:,0])*np.sin(np.pi*x[:,1])
    uxy = np.pi**2*np.cos(np.pi*x[:,0])*np.cos(np.pi*x[:,1])
    uyy = -np.pi**2*np.sin(np.pi*x[:,0])*np.sin(np.pi*x[:,1])
    return np.column_stack((uxx, uxy, uxy, uyy)).reshape((-1,2,2))

xmin = -1.
xmax = 1.
dx = 0.02

n_rand = 1000

x = np.arange(xmin+dx, xmax, dx)
y = np.arange(xmin+dx, xmax, dx)
x_pts, y_pts = np.meshgrid(x, y)
x_pts = x_pts.reshape((-1,1))
y_pts = y_pts.reshape((-1,1))
pts = np.column_stack((x_pts, y_pts))
u_grid = allen_cahn_true(pts).reshape((-1,1))
pdv_grid = allen_cahn_pdv(pts)
hes_grid = allen_cahn_hes(pts)

print(f'u_grid.shape: {u_grid.shape}')
print(f'pts.shape: {pts.shape}')
print(f'pdv_grid.shape: {pdv_grid.shape}')
print(f'hes_grid.shape: {hes_grid.shape}')

grid_dataset = TensorDataset(torch.tensor(pts, dtype=torch.float32), torch.tensor(u_grid, dtype=torch.float32), torch.tensor(pdv_grid, dtype=torch.float32), torch.tensor(hes_grid, dtype=torch.float32))
torch.save(grid_dataset, f'data/grid_dataset.pth')


# Random points
random_pts = np.random.rand(n_rand, 2)*2 - 1
u_rand = allen_cahn_true(random_pts).reshape((-1,1))
pts_rand = random_pts
pdv_radn = allen_cahn_pdv(random_pts)
hes_rand = allen_cahn_hes(random_pts)

print(f'u_rand.shape: {u_rand.shape}')
print(f'pts_rand.shape: {pts_rand.shape}')
print(f'pdv_radn.shape: {pdv_radn.shape}')
print(f'hes_rand.shape: {hes_rand.shape}')

rand_dataset = TensorDataset(torch.tensor(pts_rand, dtype=torch.float32), torch.tensor(u_rand, dtype=torch.float32), torch.tensor(pdv_radn, dtype=torch.float32), torch.tensor(hes_rand, dtype=torch.float32))
torch.save(rand_dataset, f'data/rand_dataset.pth')

# Boundary conditions
x = np.arange(xmin, xmax+dx, dx)
y = np.array([xmin]*len(x))
x_pts = x.reshape((-1,1))
y_pts = y.reshape((-1,1))
pts = np.column_stack((x_pts, y_pts))
# Get first boudary condition
u_bc = allen_cahn_true(pts).reshape((-1,1))
pts_bc = pts

x = np.arange(xmin, xmax+dx, dx)
y = np.array([xmax]*len(x))
x_pts = x.reshape((-1,1))
y_pts = y.reshape((-1,1))
# Append second boundary condition
pts = np.column_stack((x_pts, y_pts))
u_bc = np.row_stack((u_bc, allen_cahn_true(pts).reshape((-1,1))))
pts_bc = np.row_stack((pts_bc, pts))


x = np.array([xmin]*len(y))
y = np.arange(xmin, xmax+dx, dx)
x_pts = x.reshape((-1,1))
y_pts = y.reshape((-1,1))
# Append third boundary condition
pts = np.column_stack((x_pts, y_pts))
u_bc = np.row_stack((u_bc, allen_cahn_true(pts).reshape((-1,1))))
pts_bc = np.row_stack((pts_bc, pts))

x = np.array([xmax]*len(y))
y = np.arange(xmin, xmax+dx, dx)
x_pts = x.reshape((-1,1))
y_pts = y.reshape((-1,1))
# Append fourth boundary condition
pts = np.column_stack((x_pts, y_pts))
u_bc = np.row_stack((u_bc, allen_cahn_true(pts).reshape((-1,1))))
pts_bc = np.row_stack((pts_bc, pts))

print(f'u_bc.shape: {u_bc.shape}')
print(f'pts_bc.shape: {pts_bc.shape}')

bc_dataset = TensorDataset(torch.tensor(pts_bc, dtype=torch.float32), torch.tensor(u_bc, dtype=torch.float32))
torch.save(bc_dataset, f'data/bc_dataset.pth')


# Plot the solution
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Plot the solution on a grid of points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.arange(xmin, xmax+dx, dx)
y = np.arange(xmin, xmax+dx, dx)
x_pts, y_pts = np.meshgrid(x, y)
z = allen_cahn_true(np.column_stack((x_pts.reshape((-1,1)), y_pts.reshape((-1,1))))).reshape(x_pts.shape)
ax.plot_surface(x_pts, y_pts, z, cmap=cm.jet)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
plt.savefig(f'data/true.pdf')
plt.close()

# Now plot with a heatmap
fig, ax = plt.subplots()
im = ax.imshow(z, cmap=cm.jet)
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.colorbar(im, ax=ax, orientation='vertical')
plt.savefig(f'data/true_heatmap.pdf')
plt.close()

# Plot the forcing on a grid of points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.arange(xmin, xmax+dx, dx)
y = np.arange(xmin, xmax+dx, dx)
x_pts, y_pts = np.meshgrid(x, y)
z = allen_cahn_forcing(np.column_stack((x_pts.reshape((-1,1)), y_pts.reshape((-1,1))))).reshape((x_pts.shape))
ax.plot_surface(x_pts, y_pts, z, cmap=cm.jet)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f')
plt.savefig(f'data/forcing.pdf')
plt.close()

# Not plot with a heatmap
fig, ax = plt.subplots()
im = ax.imshow(z, cmap=cm.jet)
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.colorbar(im, ax=ax, orientation='vertical')
plt.savefig(f'data/forcing_heatmap.pdf')
plt.close()

