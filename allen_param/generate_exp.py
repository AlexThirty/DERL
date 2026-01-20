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
font = {'size'   : 12}
import matplotlib
import os
matplotlib.rc('font', **font)

# The case 1 is the Allen-Cahn equation with an exponential term
def allen_cahn_true_exp(x: np.array, lam: float):
    return np.sin(np.pi*x[:,0])*np.sin(np.pi*x[:,1])*np.exp(-lam*(x[:,0]+0.7))

# Forcing term of the Allen-Cahn equation with an exponential term
def allen_cahn_forcing_exp(x: np.array, lam: float):
    hes = allen_cahn_hes_exp(x, lam)
    return lam*(hes[:,0,0] + hes[:,1,1]) + allen_cahn_true_exp(x, lam)**3 - allen_cahn_true_exp(x, lam)

# Partial derivative of the Allen-Cahn equation with an exponential term
def allen_cahn_pdv_exp(x: np.array, lam: float):
    ux = np.pi*np.cos(np.pi*x[:,0])*np.sin(np.pi*x[:,1])*np.exp(-lam*(x[:,0]+0.7)) \
        - lam*np.exp(-lam*(x[:,0]+0.7))*np.sin(np.pi*x[:,0])*np.sin(np.pi*x[:,1])
    uy = np.pi*np.sin(np.pi*x[:,0])*np.cos(np.pi*x[:,1])*np.exp(-lam*(x[:,0]+0.7))
    return np.column_stack((ux, uy))

# Hessian of the Allen-Cahn equation with an exponential term
def allen_cahn_hes_exp(x: np.array, lam: float):
    uxx = -(np.pi**2 - lam**2)*allen_cahn_true_exp(x, lam) \
        - 2*np.pi*lam*np.cos(np.pi*x[:,0])*np.sin(np.pi*x[:,1])*np.exp(-lam*(x[:,0]+0.7))
        
    uxy = np.pi**2*np.cos(np.pi*x[:,0])*np.cos(np.pi*x[:,1])*np.exp(-lam*(x[:,0]+0.7)) \
        - lam*np.pi*np.sin(np.pi*x[:,0])*np.cos(np.pi*x[:,1])*np.exp(-lam*(x[:,0]+0.7))
    
    uyy = -np.pi**2*np.sin(np.pi*x[:,0])*np.sin(np.pi*x[:,1])*np.exp(-lam*(x[:,0]+0.7))
    return np.column_stack((uxx, uxy, uxy, uyy)).reshape((-1,2,2))


# Number of generated lambda values
n_lambda = 10
# Generate the lambda values
lam_list = np.random.rand(n_lambda)*np.pi
# Split the lambda values into train and val sets
train_size = int(0.8 * n_lambda)
val_size = n_lambda - train_size

# Shuffle the lambda values
np.random.shuffle(lam_list)

# Divide into train and val sets
lam_train = lam_list[:train_size]
lam_val = lam_list[train_size:]

print(f'lam_train: {lam_train}')
print(f'lam_val: {lam_val}')

# Save the lambda values to a file
np.save('allen_lam_train_exp.npy', lam_train)
np.save('allen_lam_val_exp.npy', lam_val)

# Generate the dataset for the Allen-Cahn equation with an exponential term
# Boundaries of the domain
xmin = -1.
xmax = 1.
# Step size
dx = 0.01

# Generate the grid points
x = np.arange(xmin+dx, xmax+dx, dx)
y = np.arange(xmin+dx, xmax+dx, dx)
# Create the meshgrid
x_pts, y_pts = np.meshgrid(x, y)
x_pts = x_pts.reshape((-1,1))
y_pts = y_pts.reshape((-1,1))
pts = np.column_stack((x_pts, y_pts))
# Save the points to a file
np.save('allen_pts.npy', pts)
np.save('allen_x.npy', x)
np.save('allen_y.npy', y)

# Create directory if it does not exist
if not os.path.exists('train_data'):
    os.makedirs('train_data')

# Training part
# Generate the true solution, forcing term, partial derivative, and Hessian
# Generate datasets for training lambdas
train_data = []
for lam in lam_train:
    u_grid = allen_cahn_true_exp(pts, lam).reshape((-1,1))
    pdv_grid = allen_cahn_pdv_exp(pts, lam)
    hes_grid = allen_cahn_hes_exp(pts, lam)
    forcing_grid = allen_cahn_forcing_exp(pts, lam).reshape((-1,1))
    lam_col = np.full((pts.shape[0], 1), lam)
    pts_lam = np.hstack((pts, lam_col))
    train_data.append((pts_lam, u_grid, pdv_grid, hes_grid, forcing_grid))


# Combine all training datasets into a single dataset
all_train_pts = []
all_train_u = []
all_train_pdv = []
all_train_hes = []
all_train_forcing = []

for data in train_data:
    all_train_pts.append(data[0])
    all_train_u.append(data[1])
    all_train_pdv.append(data[2])
    all_train_hes.append(data[3])
    all_train_forcing.append(data[4])

all_train_pts = np.vstack(all_train_pts)
all_train_u = np.vstack(all_train_u)
all_train_pdv = np.vstack(all_train_pdv)
all_train_hes = np.vstack(all_train_hes)
all_train_forcing = np.vstack(all_train_forcing)

# Convert to TensorDataset
all_train_pts_tensor = torch.tensor(all_train_pts, dtype=torch.float32)
all_train_u_tensor = torch.tensor(all_train_u, dtype=torch.float32)
all_train_pdv_tensor = torch.tensor(all_train_pdv, dtype=torch.float32)
all_train_hes_tensor = torch.tensor(all_train_hes, dtype=torch.float32)
all_train_forcing_tensor = torch.tensor(all_train_forcing, dtype=torch.float32)

train_dataset = TensorDataset(all_train_pts_tensor, all_train_u_tensor, all_train_pdv_tensor, all_train_hes_tensor, all_train_forcing_tensor)

# Save the combined dataset
torch.save(train_dataset, 'train_data/train_data.pt')

# Generate 10000 random points in the domain
n_rand = 1000
pts_rand = np.random.uniform(xmin, xmax, (n_rand, 2))

# Generate datasets for training lambdas with random points
train_data_rand = []
for lam in lam_train:
    u_rand = allen_cahn_true_exp(pts_rand, lam).reshape((-1, 1))
    pdv_rand = allen_cahn_pdv_exp(pts_rand, lam)
    hes_rand = allen_cahn_hes_exp(pts_rand, lam)
    forcing_rand = allen_cahn_forcing_exp(pts_rand, lam).reshape((-1, 1))
    lam_col = np.full((n_rand, 1), lam)
    pts_lam_rand = np.hstack((pts_rand, lam_col))
    train_data_rand.append((pts_lam_rand, u_rand, pdv_rand, hes_rand, forcing_rand))

# Convert to TensorDataset

# Combine all training datasets with random points into a single dataset
all_train_rand_pts = []
all_train_rand_u = []
all_train_rand_pdv = []
all_train_rand_hes = []
all_train_rand_forcing = []

for data in train_data_rand:
    all_train_rand_pts.append(data[0])
    all_train_rand_u.append(data[1])
    all_train_rand_pdv.append(data[2])
    all_train_rand_hes.append(data[3])
    all_train_rand_forcing.append(data[4])

all_train_rand_pts = np.vstack(all_train_rand_pts)
all_train_rand_u = np.vstack(all_train_rand_u)
all_train_rand_pdv = np.vstack(all_train_rand_pdv)
all_train_rand_hes = np.vstack(all_train_rand_hes)
all_train_rand_forcing = np.vstack(all_train_rand_forcing)

# Convert to TensorDataset
all_train_rand_pts_tensor = torch.tensor(all_train_rand_pts, dtype=torch.float32)
all_train_rand_u_tensor = torch.tensor(all_train_rand_u, dtype=torch.float32)
all_train_rand_pdv_tensor = torch.tensor(all_train_rand_pdv, dtype=torch.float32)
all_train_rand_hes_tensor = torch.tensor(all_train_rand_hes, dtype=torch.float32)
all_train_rand_forcing_tensor = torch.tensor(all_train_rand_forcing, dtype=torch.float32)

train_dataset_rand = TensorDataset(all_train_rand_pts_tensor, all_train_rand_u_tensor, all_train_rand_pdv_tensor, all_train_rand_hes_tensor, all_train_rand_forcing_tensor)

# Save the combined dataset
torch.save(train_dataset_rand, 'train_data/train_data_rand.pt')
# valing part
# Generate datasets for valing lambdas
val_data = []
for lam in lam_val:
    u_grid = allen_cahn_true_exp(pts, lam).reshape((-1,1))
    pdv_grid = allen_cahn_pdv_exp(pts, lam)
    hes_grid = allen_cahn_hes_exp(pts, lam)
    forcing_grid = allen_cahn_forcing_exp(pts, lam).reshape((-1,1))
    lam_col = np.full((pts.shape[0], 1), lam)
    pts_lam = np.hstack((pts, lam_col))
    val_data.append((pts_lam, u_grid, pdv_grid, hes_grid, forcing_grid))



if not os.path.exists('val_data'):
    os.makedirs('val_data')
# Combine all val datasets into a single dataset
all_val_pts = []
all_val_u = []
all_val_pdv = []
all_val_hes = []
all_val_forcing = []

for data in val_data:
    all_val_pts.append(data[0])
    all_val_u.append(data[1])
    all_val_pdv.append(data[2])
    all_val_hes.append(data[3])
    all_val_forcing.append(data[4])

all_val_pts = np.vstack(all_val_pts)
all_val_u = np.vstack(all_val_u)
all_val_pdv = np.vstack(all_val_pdv)
all_val_hes = np.vstack(all_val_hes)
all_val_forcing = np.vstack(all_val_forcing)

# Convert to TensorDataset
all_val_pts_tensor = torch.tensor(all_val_pts, dtype=torch.float32)
all_val_u_tensor = torch.tensor(all_val_u, dtype=torch.float32)
all_val_pdv_tensor = torch.tensor(all_val_pdv, dtype=torch.float32)
all_val_hes_tensor = torch.tensor(all_val_hes, dtype=torch.float32)
all_val_forcing_tensor = torch.tensor(all_val_forcing, dtype=torch.float32)

val_dataset = TensorDataset(all_val_pts_tensor, all_val_u_tensor, all_val_pdv_tensor, all_val_hes_tensor, all_val_forcing_tensor)

# Save the combined dataset
torch.save(val_dataset, 'val_data/val_data.pt')
    
from matplotlib import pyplot as plt
# Plot the data for each lambda in the training set

# Generate 10000 random points in the domain
pts_rand = np.random.uniform(xmin, xmax, (n_rand, 2))

# Generate datasets for valing lambdas with random points
val_data_rand = []
for lam in lam_val:
    u_rand = allen_cahn_true_exp(pts_rand, lam).reshape((-1, 1))
    pdv_rand = allen_cahn_pdv_exp(pts_rand, lam)
    hes_rand = allen_cahn_hes_exp(pts_rand, lam)
    forcing_rand = allen_cahn_forcing_exp(pts_rand, lam).reshape((-1, 1))
    lam_col = np.full((n_rand, 1), lam)
    pts_lam_rand = np.hstack((pts_rand, lam_col))
    val_data_rand.append((pts_lam_rand, u_rand, pdv_rand, hes_rand, forcing_rand))

# Combine all val datasets with random points into a single dataset
all_val_rand_pts = []
all_val_rand_u = []
all_val_rand_pdv = []
all_val_rand_hes = []
all_val_rand_forcing = []

for data in val_data_rand:
    all_val_rand_pts.append(data[0])
    all_val_rand_u.append(data[1])
    all_val_rand_pdv.append(data[2])
    all_val_rand_hes.append(data[3])
    all_val_rand_forcing.append(data[4])

all_val_rand_pts = np.vstack(all_val_rand_pts)
all_val_rand_u = np.vstack(all_val_rand_u)
all_val_rand_pdv = np.vstack(all_val_rand_pdv)
all_val_rand_hes = np.vstack(all_val_rand_hes)
all_val_rand_forcing = np.vstack(all_val_rand_forcing)

# Convert to TensorDataset
all_val_rand_pts_tensor = torch.tensor(all_val_rand_pts, dtype=torch.float32)
all_val_rand_u_tensor = torch.tensor(all_val_rand_u, dtype=torch.float32)
all_val_rand_pdv_tensor = torch.tensor(all_val_rand_pdv, dtype=torch.float32)
all_val_rand_hes_tensor = torch.tensor(all_val_rand_hes, dtype=torch.float32)
all_val_rand_forcing_tensor = torch.tensor(all_val_rand_forcing, dtype=torch.float32)

val_dataset_rand = TensorDataset(all_val_rand_pts_tensor, all_val_rand_u_tensor, all_val_rand_pdv_tensor, all_val_rand_hes_tensor, all_val_rand_forcing_tensor)

# Save the combined dataset
torch.save(val_dataset_rand, 'val_data/val_data_rand.pt')
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.contourf(x, y, u_grid.reshape(len(x), len(y)), levels=50, cmap='viridis')
plt.colorbar(label='u(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('True Solution of Allen-Cahn Equation')
plt.savefig('allen_true_exp.png')
# Plot the forcing term with a heatmap
plt.close()

forcing_grid = allen_cahn_forcing_exp(pts, lam).reshape((-1,1))

plt.figure(figsize=(8, 6))
plt.contourf(x, y, forcing_grid.reshape(len(x), len(y)), levels=50, cmap='viridis')
plt.colorbar(label='f(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Forcing Term of Allen-Cahn Equation')
plt.savefig('allen_forcing_exp.png')
plt.close()
# Generate the boundary conditions, which are 0 on each side of the [-1,1] square
boundary_pts = []
boundary_values = []

# Left and right boundaries (x = -1 and x = 1)
for y in np.arange(xmin, xmax + dx, dx):
    boundary_pts.append([-1, y])
    boundary_pts.append([1, y])
    boundary_values.append([0])
    boundary_values.append([0])

# Bottom and top boundaries (y = -1 and y = 1)
for x in np.arange(xmin, xmax + dx, dx):
    boundary_pts.append([x, -1])
    boundary_pts.append([x, 1])
    boundary_values.append([0])
    boundary_values.append([0])


boundary_pts = np.array(boundary_pts)
boundary_values = np.array(boundary_values)

# Repeat the boundary points and values for each lambda in the training set
boundary_pts_repeated = []
boundary_values_repeated = []

for lam in lam_train:
    lam_col = np.full((boundary_pts.shape[0], 1), lam)
    boundary_pts_lam = np.hstack((boundary_pts, lam_col))
    boundary_pts_repeated.append(boundary_pts_lam)
    boundary_values_repeated.append(boundary_values)

boundary_pts_repeated = np.vstack(boundary_pts_repeated)
boundary_values_repeated = np.vstack(boundary_values_repeated)

# Save the boundary conditions to a file
np.save('allen_boundary_pts.npy', boundary_pts)
np.save('allen_boundary_values.npy', boundary_values)

# Save it in a TensorDataset
boundary_pts_tensor = torch.tensor(boundary_pts_repeated, dtype=torch.float32)
boundary_values_tensor = torch.tensor(boundary_values_repeated, dtype=torch.float32)

boundary_dataset = TensorDataset(boundary_pts_tensor, boundary_values_tensor)

# Save the boundary dataset
torch.save(boundary_dataset, 'boundary_dataset.pt')