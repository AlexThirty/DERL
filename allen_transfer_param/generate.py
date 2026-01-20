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
# Part 2 is for the second way to parametrize the Allen-Cahn equation
def allen_cahn_true_multi(x: np.array, lam_coef: list):
    u = 0.
    for i in range(len(lam_coef)):
        u += lam_coef[i]*np.sin((i+1)*np.pi*x[:,0])*np.sin((i+1)*np.pi*x[:,1])/((i+1)**2)
    return u/len(lam_coef)

lam_fix = 0.1

def allen_cahn_forcing_multi(x: np.array, lam_coef: list):
    hes = allen_cahn_hes_multi(x, lam_coef)
    return lam_fix*(hes[:,0,0] + hes[:,1,1]) + allen_cahn_true_multi(x, lam_coef)**3 - allen_cahn_true_multi(x, lam_coef)

def allen_cahn_pdv_multi(x: np.array, lam_coef: list):
    pdv = np.zeros((len(x), 2))
    for i in range(len(lam_coef)):
        ux = lam_coef[i]*(i+1)*np.pi*np.cos((i+1)*np.pi*x[:,0])*np.sin((i+1)*np.pi*x[:,1])/((i+1)**2)
        uy = lam_coef[i]*(i+1)*np.pi*np.sin((i+1)*np.pi*x[:,0])*np.cos((i+1)*np.pi*x[:,1])/((i+1)**2)
        pdv += np.column_stack((ux, uy))
    return pdv/len(lam_coef)

def allen_cahn_hes_multi(x: np.array, lam_coef: list):
    hes = np.zeros((len(x), 2, 2))
    for i in range(len(lam_coef)):
        uxx = -lam_coef[i]*(i+1)**2*np.pi**2*np.sin((i+1)*np.pi*x[:,0])*np.sin((i+1)*np.pi*x[:,1])/((i+1)**2)
        uxy = lam_coef[i]*(i+1)**2*np.pi**2*np.cos((i+1)*np.pi*x[:,0])*np.cos((i+1)*np.pi*x[:,1])/((i+1)**2)
        uyy = -lam_coef[i]*(i+1)**2*np.pi**2*np.sin((i+1)*np.pi*x[:,0])*np.sin((i+1)*np.pi*x[:,1])/((i+1)**2)
        hes += np.column_stack((uxx, uxy, uxy, uyy)).reshape((-1,2,2))
    return hes/len(lam_coef)


num_lam = 2
# Number of generated lambda values
n_lambda = 20
# Generate the lambda values
lam_list = np.random.random((n_lambda, num_lam))
# Split the lambda values into train and val sets
train_size = int(0.5 * n_lambda)
val_size = n_lambda - train_size

# Shuffle the lambda values
np.random.shuffle(lam_list)

# Divide into train and val sets
lam_train = lam_list[:train_size]
lam_val = lam_list[train_size:]

lam_test = np.random.random((5, num_lam))

print(f'lam_train: {lam_train}')
print(f'lam_val: {lam_val}')

# Save the lambda values to a file
np.save('allen_lam_train.npy', lam_train)
np.save('allen_lam_val.npy', lam_val)
np.save('allen_lam_test.npy', lam_test)

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
    u_grid = allen_cahn_true_multi(pts, lam).reshape((-1,1))
    pdv_grid = allen_cahn_pdv_multi(pts, lam)
    hes_grid = allen_cahn_hes_multi(pts, lam)
    forcing_grid = allen_cahn_forcing_multi(pts, lam).reshape((-1,1))
    lam_col = np.full((pts.shape[0], num_lam), lam)
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
torch.save(train_dataset, 'train_data/train_data_multi.pt')

# Generate 10000 random points in the domain
n_rand = 1000
pts_rand = np.random.uniform(xmin, xmax, (n_rand, 2))

# Generate datasets for training lambdas with random points
train_data_rand = []
for lam in lam_train:
    u_rand = allen_cahn_true_multi(pts_rand, lam).reshape((-1, 1))
    pdv_rand = allen_cahn_pdv_multi(pts_rand, lam)
    hes_rand = allen_cahn_hes_multi(pts_rand, lam)
    forcing_rand = allen_cahn_forcing_multi(pts_rand, lam).reshape((-1, 1))
    lam_col = np.full((n_rand, num_lam), lam)
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
print(all_train_rand_pts_tensor.shape)

train_dataset_rand = TensorDataset(all_train_rand_pts_tensor, all_train_rand_u_tensor, all_train_rand_pdv_tensor, all_train_rand_hes_tensor, all_train_rand_forcing_tensor)

# Save the combined dataset
torch.save(train_dataset_rand, 'train_data/train_data_multi_rand.pt')
# valing part
# Generate datasets for valing lambdas
val_data = []
for lam in lam_val:
    u_grid = allen_cahn_true_multi(pts, lam).reshape((-1,1))
    pdv_grid = allen_cahn_pdv_multi(pts, lam)
    hes_grid = allen_cahn_hes_multi(pts, lam)
    forcing_grid = allen_cahn_forcing_multi(pts, lam).reshape((-1,1))
    lam_col = np.full((pts.shape[0], num_lam), lam)
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
torch.save(val_dataset, 'val_data/val_data_multi.pt')


# Save also the joint train and val dataset
joint_dataset = TensorDataset(torch.tensor(np.vstack((all_train_pts, all_val_pts)), dtype=torch.float32), torch.tensor(np.vstack((all_train_u, all_val_u)), dtype=torch.float32), torch.tensor(np.vstack((all_train_pdv, all_val_pdv)), dtype=torch.float32), torch.tensor(np.vstack((all_train_hes, all_val_hes)), dtype=torch.float32), torch.tensor(np.vstack((all_train_forcing, all_val_forcing)), dtype=torch.float32))
torch.save(joint_dataset, 'val_data/joint_data_multi.pt')

from matplotlib import pyplot as plt
# Generate 10000 random points in the domain
pts_rand = np.random.uniform(xmin, xmax, (n_rand, 2))

# Generate datasets for valing lambdas with random points
val_data_rand = []
for lam in lam_val:
    u_rand = allen_cahn_true_multi(pts_rand, lam).reshape((-1, 1))
    pdv_rand = allen_cahn_pdv_multi(pts_rand, lam)
    hes_rand = allen_cahn_hes_multi(pts_rand, lam)
    forcing_rand = allen_cahn_forcing_multi(pts_rand, lam).reshape((-1, 1))
    lam_col = np.full((n_rand, num_lam), lam)
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

print(all_val_rand_pts_tensor.shape)

val_dataset_rand = TensorDataset(all_val_rand_pts_tensor, all_val_rand_u_tensor, all_val_rand_pdv_tensor, all_val_rand_hes_tensor, all_val_rand_forcing_tensor)

# Save the combined dataset
torch.save(val_dataset_rand, 'val_data/val_data_multi_rand.pt')
import matplotlib.pyplot as plt
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
    lam_col = np.full((boundary_pts.shape[0], num_lam), lam)
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
torch.save(boundary_dataset, 'train_data/boundary_train_data.pt')

# Generate the boundary conditions, which are 0 on each side of the [-1,1] square
boundary_pts_repeated = []
boundary_values_repeated = []

for lam in lam_val:
    lam_col = np.full((boundary_pts.shape[0], num_lam), lam)
    boundary_pts_lam = np.hstack((boundary_pts, lam_col))
    boundary_pts_repeated.append(boundary_pts_lam)
    boundary_values_repeated.append(boundary_values)

boundary_pts_repeated = np.vstack(boundary_pts_repeated)
boundary_values_repeated = np.vstack(boundary_values_repeated)

# Save it in a TensorDataset
boundary_pts_tensor = torch.tensor(boundary_pts_repeated, dtype=torch.float32)
boundary_values_tensor = torch.tensor(boundary_values_repeated, dtype=torch.float32)

boundary_dataset = TensorDataset(boundary_pts_tensor, boundary_values_tensor)

# Save the boundary dataset
torch.save(boundary_dataset, f'val_data/boundary_val_data.pt')


# Save also the joint train and val dataset with random points
joint_rand_dataset = TensorDataset(
    torch.tensor(np.vstack((all_train_rand_pts, all_val_rand_pts)), dtype=torch.float32),
    torch.tensor(np.vstack((all_train_rand_u, all_val_rand_u)), dtype=torch.float32),
    torch.tensor(np.vstack((all_train_rand_pdv, all_val_rand_pdv)), dtype=torch.float32),
    torch.tensor(np.vstack((all_train_rand_hes, all_val_rand_hes)), dtype=torch.float32),
    torch.tensor(np.vstack((all_train_rand_forcing, all_val_rand_forcing)), dtype=torch.float32)
)
torch.save(joint_rand_dataset, 'val_data/joint_data_multi_rand.pt')