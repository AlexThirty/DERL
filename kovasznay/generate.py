import numpy as np
import matplotlib.pyplot as plt
import torch
from models.params import L, Re, lam, nu, x_min, x_max, y_min, y_max, nx, ny, N
# Load the data
    
def u_true(x:torch.Tensor):
    return torch.column_stack((1 - torch.exp(lam * x[:, 0]/L) * torch.cos(2 * np.pi * x[:, 1]/L),
                      lam/(2*np.pi) * torch.exp(lam*x[:, 0]/L)* torch.sin(2 * np.pi * x[:, 1]/L)))

def Du_true(x:torch.Tensor):
    return torch.column_stack((-lam/L * torch.exp(lam * x[:, 0]/L) * torch.cos(2 * np.pi * x[:, 1]/L),
                                -torch.exp(lam * x[:, 0]/L) * (-2*np.pi/L)*torch.sin(2 * np.pi * x[:, 1]/L),
                                lam**2/(2*np.pi*L)*torch.exp(lam*x[:, 0]/L)*torch.sin(2*np.pi*x[:, 1]/L),  
                                lam/L*torch.exp(lam*x[:, 0]/L)*torch.cos(2*np.pi*x[:, 1]/L)))

def vorticity_true(x: torch.Tensor):
    return Re*lam*torch.exp(lam*x[:, 0]/L)*torch.sin(2*np.pi*x[:, 1]/L)/(2*np.pi)

def stream_function(x: torch.Tensor):
    return x[:, 1]/L - 1/(2*np.pi)*torch.exp(lam*x[:, 0]/L)*torch.sin(2*np.pi*x[:, 1]/L)


def p_e_expr(x):
    """Expression for the exact pressure solution to Kovasznay flow"""
    return (1 / 2) * (1 - np.exp(2 * lam * x[0]/L))

def p_true(x:torch.Tensor):
    return 0.5 * (1 - torch.exp(2 * lam * x[:, 0]/L))

def Dp_true(x:torch.Tensor):
    return torch.column_stack((-lam/L * torch.exp(2 * lam * x[:, 0]/L), torch.zeros(x.shape[0])))


# Create a grid in the unit square
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x, y)
points = np.vstack([X.ravel(), Y.ravel()]).T


# Evaluate u and p on the grid
u_exact = u_true(torch.tensor(points)).detach().numpy()
p_exact = p_true(torch.tensor(points)).detach().numpy()

vorticity_exact = vorticity_true(torch.tensor(points)).detach().numpy()
stream_exact = stream_function(torch.tensor(points)).detach().numpy()

out_exact = np.column_stack([u_exact, p_exact])
Du_exact = Du_true(torch.tensor(points)).detach().numpy().reshape(-1, 2, 2)
Dp_exact = Dp_true(torch.tensor(points)).detach().numpy().reshape(-1, 1, 2)
Dout_exact = np.concatenate([Du_exact, Dp_exact], axis=1)

import os
if not os.path.exists('true_figures'):
    os.makedirs('true_figures')

# Plot the exact velocity field using streamplot
# Plot the exact velocity using quiver and contourf for magnitude
velocity_magnitude = np.sqrt(u_exact[:,0]**2 + u_exact[:,1]**2)
plt.figure(figsize=(10, 5))
plt.streamplot(X, Y, u_exact[:, 0].reshape(X.shape), u_exact[:, 1].reshape(X.shape), color='white', linewidth=1)
plt.contourf(X, Y, velocity_magnitude.reshape(X.shape), levels=50, cmap='viridis')
plt.colorbar(label='Velocity magnitude')
plt.title('Exact Velocity Field')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('true_figures/velocity_field.png')

# Plot the exact pressure field
plt.figure(figsize=(10, 5))
plt.contourf(X, Y, p_exact.reshape(X.shape), levels=50, cmap='viridis')
plt.colorbar(label='Pressure')
plt.title('Exact Pressure Field')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('true_figures/pressure_field.png')

# Plot the vorticity field
plt.figure(figsize=(10, 5))
plt.contourf(X, Y, vorticity_exact.reshape(X.shape), levels=50, cmap='viridis')
plt.colorbar(label='Vorticity')
plt.title('Vorticity Field')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('true_figures/vorticity_field.png')

# Plot the streamline field
plt.figure(figsize=(10, 5))
plt.contour(X, Y, stream_exact.reshape(X.shape), levels=100, cmap='viridis')
plt.colorbar(label='Streamline')
plt.title('Streamline Field')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('true_figures/streamline_field.png')


# Save in a dataset
from torch.utils.data import TensorDataset
import torch

x_sol = torch.tensor(points).float()
u_sol = u_true(x_sol)
p_sol = p_true(x_sol).reshape(-1, 1)
out_sol = torch.cat([u_sol, p_sol], dim=1)

Du_sol = Du_true(x_sol).reshape(-1, 2, 2)
Dp_sol = Dp_true(x_sol).reshape(-1, 1, 2)
Dout_sol = torch.cat([Du_sol, Dp_sol], dim=1)

vorticity_sol = vorticity_true(x_sol)
stream_sol = stream_function(x_sol)

import os
if not os.path.exists('data'):
    os.makedirs('data')
    
dataset = TensorDataset(x_sol, out_sol, Dout_sol, vorticity_sol, stream_sol)
torch.save(dataset, 'data/grid_data.pt')
print('Dataset saved successfully!')

# Save numpy arrays
np.save('data/x_sol.npy', x_sol.numpy())
np.save('data/u_sol.npy', u_sol.detach().numpy())
np.save('data/p_sol.npy', p_sol.detach().numpy())
np.save('data/vorticity_sol.npy', vorticity_sol.detach().numpy())
np.save('data/stream_sol.npy', stream_sol.detach().numpy())

print('Numpy arrays saved successfully!')

# Create a dataset with N random points
random_points = torch.rand((N, 2)) * torch.tensor([x_max-x_min, y_max-y_min]) + torch.tensor([x_min, y_min])
# Evaluate u, p, vorticity, and stream function on the random points
u_random = u_true(random_points)
p_random = p_true(random_points).reshape(-1, 1)
out_random = torch.cat([u_random, p_random], dim=1)

Du_random = Du_true(random_points).reshape(-1, 2, 2)
Dp_random = Dp_true(random_points).reshape(-1, 1, 2)
Dout_random = torch.cat([Du_random, Dp_random], dim=1)

vorticity_random = vorticity_true(random_points)
stream_random = stream_function(random_points)

# Save the random dataset
random_dataset = TensorDataset(random_points, out_random, Dout_random, vorticity_random, stream_random)
torch.save(random_dataset, 'data/random_data.pt')
print('Random dataset saved successfully!')

# Save numpy arrays for random points
np.save('data/random_points.npy', random_points.numpy())
np.save('data/out_random.npy', out_random.detach().numpy())
np.save('data/Dout_random.npy', Dout_random.detach().numpy())
np.save('data/vorticity_random.npy', vorticity_random.detach().numpy())
np.save('data/stream_random.npy', stream_random.detach().numpy())

print('Random numpy arrays saved successfully!')

# Define boundary points
boundary_points = np.vstack([
    np.column_stack([np.linspace(x_min, x_max, nx), np.full(nx, y_min)]),  # Bottom boundary
    np.column_stack([np.linspace(x_min, x_max, nx), np.full(nx, y_max)]),   # Top boundary
    np.column_stack([np.full(ny, x_min), np.linspace(y_min, y_max, ny)]),  # Left boundary
    np.column_stack([np.full(ny, x_max), np.linspace(y_min, y_max, ny)])   # Right boundary
])

# Evaluate u, p, vorticity, and stream function on the boundary points
u_boundary = u_true(torch.tensor(boundary_points)).detach().numpy()
p_boundary = p_true(torch.tensor(boundary_points)).detach().numpy().reshape(-1, 1)
out_boundary = np.column_stack([u_boundary, p_boundary])

Du_boundary = Du_true(torch.tensor(boundary_points)).detach().numpy().reshape(-1, 2, 2)
Dp_boundary = Dp_true(torch.tensor(boundary_points)).detach().numpy().reshape(-1, 1, 2)
Dout_boundary = np.column_stack([Du_boundary, Dp_boundary])

vorticity_boundary = vorticity_true(torch.tensor(boundary_points)).detach().numpy()
stream_boundary = stream_function(torch.tensor(boundary_points)).detach().numpy()
boundary_dataset = TensorDataset(torch.tensor(boundary_points).float(), 
                                 torch.tensor(out_boundary).float(), 
                                 torch.tensor(Dout_boundary).float(), 
                                 torch.tensor(vorticity_boundary).float(), 
                                 torch.tensor(stream_boundary).float())
torch.save(boundary_dataset, 'data/boundary_data.pt')
print('Boundary dataset saved successfully!')

# Save the boundary conditions
np.save('data/boundary_points.npy', boundary_points)
np.save('data/out_boundary.npy', out_boundary)
np.save('data/Dout_boundary.npy', Dout_boundary)
np.save('data/vorticity_boundary.npy', vorticity_boundary)
np.save('data/stream_boundary.npy', stream_boundary)

print('Boundary conditions saved successfully!')