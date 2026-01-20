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
dx = 0.01

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
torch.save(grid_dataset, f'data/dataset_full.pth')

# Save in a dataset just the values for y < 0
mask = pts[:, 1] < 0
pts_down = pts[mask]
u_down = u_grid[mask]
pdv_down = pdv_grid[mask]
hes_down = hes_grid[mask]

print(f'u_down.shape: {u_down.shape}')
print(f'pts_down.shape: {pts_down.shape}')
print(f'pdv_down.shape: {pdv_down.shape}')
print(f'hes_down.shape: {hes_down.shape}')

down_dataset = TensorDataset(torch.tensor(pts_down, dtype=torch.float32), torch.tensor(u_down, dtype=torch.float32), torch.tensor(pdv_down, dtype=torch.float32), torch.tensor(hes_down, dtype=torch.float32))
torch.save(down_dataset, f'data/dataset_down.pth')

# Save in a dataset just for the values of y < 0 and x < 0
mask = pts[:, 1] < 0
mask = mask & (pts[:, 0] < 0)
pts_down_left = pts[mask]
u_down_left = u_grid[mask]
pdv_down_left = pdv_grid[mask]
hes_down_left = hes_grid[mask]

print(f'u_down_left.shape: {u_down_left.shape}')
print(f'pts_down_left.shape: {pts_down_left.shape}')
print(f'pdv_down_left.shape: {pdv_down_left.shape}')
print(f'hes_down_left.shape: {hes_down_left.shape}')

down_left_dataset = TensorDataset(torch.tensor(pts_down_left, dtype=torch.float32), torch.tensor(u_down_left, dtype=torch.float32), torch.tensor(pdv_down_left, dtype=torch.float32), torch.tensor(hes_down_left, dtype=torch.float32))
torch.save(down_left_dataset, f'data/dataset_down_left.pth')

# Save in a dataset just for the values of y < 0 and x > 0
mask = pts[:, 1] < 0
mask = mask & (pts[:, 0] > 0)
pts_down_right = pts[mask]
u_down_right = u_grid[mask]
pdv_down_right = pdv_grid[mask]
hes_down_right = hes_grid[mask]

print(f'u_down_right.shape: {u_down_right.shape}')
print(f'pts_down_right.shape: {pts_down_right.shape}')
print(f'pdv_down_right.shape: {pdv_down_right.shape}')
print(f'hes_down_right.shape: {hes_down_right.shape}')

down_right_dataset = TensorDataset(torch.tensor(pts_down_right, dtype=torch.float32), torch.tensor(u_down_right, dtype=torch.float32), torch.tensor(pdv_down_right, dtype=torch.float32), torch.tensor(hes_down_right, dtype=torch.float32))
torch.save(down_right_dataset, f'data/dataset_down_right.pth')

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
torch.save(rand_dataset, f'data/dataset_rand_full.pth')

# Save in a dataset just for the values of y < 0
mask_down = random_pts[:, 1] < 0
random_pts_down = random_pts[mask_down]

# Generate random points on the lower half of the domain
#random_pts_down = np.random.rand(n_rand, 2)*2 - 1
#random_pts_down[:, 1] = -np.abs(random_pts_down[:, 1])

u_rand_down = allen_cahn_true(random_pts_down).reshape((-1, 1))
pdv_rand_down = allen_cahn_pdv(random_pts_down)
hes_rand_down = allen_cahn_hes(random_pts_down)

print(f'u_rand_down.shape: {u_rand_down.shape}')
print(f'pts_rand_down.shape: {random_pts_down.shape}')
print(f'pdv_rand_down.shape: {pdv_rand_down.shape}')
print(f'hes_rand_down.shape: {hes_rand_down.shape}')

rand_dataset_down = TensorDataset(torch.tensor(random_pts_down, dtype=torch.float32), torch.tensor(u_rand_down, dtype=torch.float32), torch.tensor(pdv_rand_down, dtype=torch.float32), torch.tensor(hes_rand_down, dtype=torch.float32))
torch.save(rand_dataset_down, f'data/dataset_rand_down.pth')

# Random points on the up half of the domain
mask_up = random_pts[:, 1] > 0
random_pts_up = random_pts[mask_up]

# Generate random points on the upper half of the domain
#random_pts_up = np.random.rand(n_rand, 2)*2 - 1
#random_pts_up[:, 1] = np.abs(random_pts_up[:, 1])

u_rand_up = allen_cahn_true(random_pts_up).reshape((-1, 1))
pdv_rand_up = allen_cahn_pdv(random_pts_up)
hes_rand_up = allen_cahn_hes(random_pts_up)

print(f'u_rand_up.shape: {u_rand_up.shape}')
print(f'pts_rand_up.shape: {random_pts_up.shape}')
print(f'pdv_rand_up.shape: {pdv_rand_up.shape}')
print(f'hes_rand_up.shape: {hes_rand_up.shape}')

rand_dataset_up = TensorDataset(torch.tensor(random_pts_up, dtype=torch.float32), torch.tensor(u_rand_up, dtype=torch.float32), torch.tensor(pdv_rand_up, dtype=torch.float32), torch.tensor(hes_rand_up, dtype=torch.float32))
torch.save(rand_dataset_up, f'data/dataset_rand_up.pth')

# Save in a dataset just for the values of y < 0 and x < 0
mask_down_left = random_pts[:, 1] < 0
mask_down_left = mask_down_left & (random_pts[:, 0] < 0)
random_pts_down_left = random_pts[mask_down_left]

# Generate random points on the lower left quarter of the domain
#random_pts_down_left = np.random.rand(n_rand, 2)*2 - 1
#random_pts_down_left[:, 1] = -np.abs(random_pts_down_left[:, 1])
#random_pts_down_left[:, 0] = -np.abs(random_pts_down_left[:, 0])

u_rand_down_left = allen_cahn_true(random_pts_down_left).reshape((-1, 1))
pdv_rand_down_left = allen_cahn_pdv(random_pts_down_left)
hes_rand_down_left = allen_cahn_hes(random_pts_down_left)

print(f'u_rand_down_left.shape: {u_rand_down_left.shape}')
print(f'pts_rand_down_left.shape: {random_pts_down_left.shape}')
print(f'pdv_rand_down_left.shape: {pdv_rand_down_left.shape}')
print(f'hes_rand_down_left.shape: {hes_rand_down_left.shape}')

rand_dataset_down_left = TensorDataset(torch.tensor(random_pts_down_left, dtype=torch.float32), torch.tensor(u_rand_down_left, dtype=torch.float32), torch.tensor(pdv_rand_down_left, dtype=torch.float32), torch.tensor(hes_rand_down_left, dtype=torch.float32))
torch.save(rand_dataset_down_left, f'data/dataset_rand_down_left.pth')


# Save in a dataset just for the values of y < 0 and x > 0
mask_down_right = random_pts[:, 1] < 0
mask_down_right = mask_down_right & (random_pts[:, 0] > 0)
random_pts_down_right = random_pts[mask_down_right]

# Generate random points on the lower right quarter of the domain
#random_pts_down_right = np.random.rand(n_rand, 2)*2 - 1
#random_pts_down_right[:, 1] = -np.abs(random_pts_down_right[:, 1])
#random_pts_down_right[:, 0] = np.abs(random_pts_down_right[:, 0])


u_rand_down_right = allen_cahn_true(random_pts_down_right).reshape((-1, 1))
pdv_rand_down_right = allen_cahn_pdv(random_pts_down_right)
hes_rand_down_right = allen_cahn_hes(random_pts_down_right)

print(f'u_rand_down_right.shape: {u_rand_down_right.shape}')
print(f'pts_rand_down_right.shape: {random_pts_down_right.shape}')
print(f'pdv_rand_down_right.shape: {pdv_rand_down_right.shape}')
print(f'hes_rand_down_right.shape: {hes_rand_down_right.shape}')

rand_dataset_down_right = TensorDataset(torch.tensor(random_pts_down_right, dtype=torch.float32), torch.tensor(u_rand_down_right, dtype=torch.float32), torch.tensor(pdv_rand_down_right, dtype=torch.float32), torch.tensor(hes_rand_down_right, dtype=torch.float32))
torch.save(rand_dataset_down_right, f'data/dataset_rand_down_right.pth')

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
torch.save(bc_dataset, f'data/boundary_full.pth')


# Boundary conditions for the lower half of the domain
x = np.arange(xmin, xmax+dx, dx)
y = np.array([xmin]*len(x))
x_pts = x.reshape((-1,1))
y_pts = y.reshape((-1,1))
pts = np.column_stack((x_pts, y_pts))
# Get first boundary condition for lower half
u_bc_down = allen_cahn_true(pts).reshape((-1,1))
pts_bc_down = pts

x = np.arange(xmin, xmax+dx, dx)
y = np.array([0]*len(x))
x_pts = x.reshape((-1,1))
y_pts = y.reshape((-1,1))
# Append second boundary condition for lower half
pts = np.column_stack((x_pts, y_pts))
u_bc_down = np.row_stack((u_bc_down, allen_cahn_true(pts).reshape((-1,1))))
pts_bc_down = np.row_stack((pts_bc_down, pts))

y = np.arange(xmin, 0+dx, dx)
x = np.array([xmin]*len(y))
x_pts = x.reshape((-1,1))
y_pts = y.reshape((-1,1))
# Append third boundary condition for lower half
pts = np.column_stack((x_pts, y_pts))
u_bc_down = np.row_stack((u_bc_down, allen_cahn_true(pts).reshape((-1,1))))
pts_bc_down = np.row_stack((pts_bc_down, pts))

y = np.arange(xmin, 0+dx, dx)
x = np.array([xmax]*len(y))
x_pts = x.reshape((-1,1))
y_pts = y.reshape((-1,1))
# Append fourth boundary condition for lower half
pts = np.column_stack((x_pts, y_pts))
u_bc_down = np.row_stack((u_bc_down, allen_cahn_true(pts).reshape((-1,1))))
pts_bc_down = np.row_stack((pts_bc_down, pts))

print(f'u_bc_down.shape: {u_bc_down.shape}')
print(f'pts_bc_down.shape: {pts_bc_down.shape}')

bc_dataset_down = TensorDataset(torch.tensor(pts_bc_down, dtype=torch.float32), torch.tensor(u_bc_down, dtype=torch.float32))
torch.save(bc_dataset_down, f'data/boundary_down.pth')


# Boundary conditions for the lower left quarter of the domain
x = np.arange(xmin, 0+dx, dx)
y = np.array([xmin]*len(x))
x_pts = x.reshape((-1,1))
y_pts = y.reshape((-1,1))
pts = np.column_stack((x_pts, y_pts))
# Get first boundary condition for lower left quarter
u_bc_down_left = allen_cahn_true(pts).reshape((-1,1))
pts_bc_down_left = pts

x = np.arange(xmin, 0+dx, dx)
y = np.array([0]*len(x))
x_pts = x.reshape((-1,1))
y_pts = y.reshape((-1,1))
# Append second boundary condition for lower left quarter
pts = np.column_stack((x_pts, y_pts))
u_bc_down_left = np.row_stack((u_bc_down_left, allen_cahn_true(pts).reshape((-1,1))))
pts_bc_down_left = np.row_stack((pts_bc_down_left, pts))

y = np.arange(xmin, 0+dx, dx)
x = np.array([xmin]*len(y))
x_pts = x.reshape((-1,1))
y_pts = y.reshape((-1,1))
# Append third boundary condition for lower left quarter
pts = np.column_stack((x_pts, y_pts))
u_bc_down_left = np.row_stack((u_bc_down_left, allen_cahn_true(pts).reshape((-1,1))))
pts_bc_down_left = np.row_stack((pts_bc_down_left, pts))

y = np.arange(xmin, 0+dx, dx)
x = np.array([0]*len(y))
x_pts = x.reshape((-1,1))
y_pts = y.reshape((-1,1))
# Append fourth boundary condition for lower left quarter
pts = np.column_stack((x_pts, y_pts))
u_bc_down_left = np.row_stack((u_bc_down_left, allen_cahn_true(pts).reshape((-1,1))))
pts_bc_down_left = np.row_stack((pts_bc_down_left, pts))

print(f'u_bc_down_left.shape: {u_bc_down_left.shape}')
print(f'pts_bc_down_left.shape: {pts_bc_down_left.shape}')


bc_dataset_down_left = TensorDataset(torch.tensor(pts_bc_down_left, dtype=torch.float32), torch.tensor(u_bc_down_left, dtype=torch.float32))
torch.save(bc_dataset_down_left, f'data/boundary_down_left.pth')


# Boundary conditions for the lower right quarter of the domain
x = np.arange(0, xmax+dx, dx)
y = np.array([xmin]*len(x))
x_pts = x.reshape((-1,1))
y_pts = y.reshape((-1,1))
pts = np.column_stack((x_pts, y_pts))
# Get first boundary condition for lower right quarter
u_bc_down_right = allen_cahn_true(pts).reshape((-1,1))
pts_bc_down_right = pts


x = np.arange(0, xmax+dx, dx)
y = np.array([0]*len(x))
x_pts = x.reshape((-1,1))
y_pts = y.reshape((-1,1))
# Append second boundary condition for lower right quarter
pts = np.column_stack((x_pts, y_pts))
u_bc_down_right = np.row_stack((u_bc_down_right, allen_cahn_true(pts).reshape((-1,1))))
pts_bc_down_right = np.row_stack((pts_bc_down_right, pts))

y = np.arange(xmin, 0+dx, dx)
x = np.array([xmax]*len(y))
x_pts = x.reshape((-1,1))
y_pts = y.reshape((-1,1))
# Append third boundary condition for lower right quarter
pts = np.column_stack((x_pts, y_pts))
u_bc_down_right = np.row_stack((u_bc_down_right, allen_cahn_true(pts).reshape((-1,1))))
pts_bc_down_right = np.row_stack((pts_bc_down_right, pts))

y = np.arange(xmin, 0+dx, dx)
x = np.array([0]*len(y))
x_pts = x.reshape((-1,1))
y_pts = y.reshape((-1,1))   
# Append fourth boundary condition for lower right quarter
pts = np.column_stack((x_pts, y_pts))
u_bc_down_right = np.row_stack((u_bc_down_right, allen_cahn_true(pts).reshape((-1,1))))
pts_bc_down_right = np.row_stack((pts_bc_down_right, pts))

print(f'u_bc_down_right.shape: {u_bc_down_right.shape}')
print(f'pts_bc_down_right.shape: {pts_bc_down_right.shape}')

bc_dataset_down_right = TensorDataset(torch.tensor(pts_bc_down_right, dtype=torch.float32), torch.tensor(u_bc_down_right, dtype=torch.float32))
torch.save(bc_dataset_down_right, f'data/boundary_down_right.pth')

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
x = np.arange(xmin, xmax+dx, dx)
y = np.arange(xmin, xmax+dx, dx)
x_pts, y_pts = np.meshgrid(x, y)
z = allen_cahn_true(np.column_stack((x_pts.reshape((-1,1)), y_pts.reshape((-1,1))))).reshape(x_pts.shape)
contour = ax.contourf(x_pts, y_pts, z, cmap=cm.jet, levels=50)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
fig.colorbar(contour, ax=ax, orientation='vertical')
plt.savefig(f'data/true_contourf.png')
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