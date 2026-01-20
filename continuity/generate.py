### IMPORTS
import numpy as np
import torch
import random
from torch.utils.data import TensorDataset
import os
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--recreate', default=False, type=bool, help='Whether to regenerate the data', action=argparse.BooleanOptionalAction)

recreate = parser.parse_args().recreate

### PATH
EXP_PATH = 'data'
if not os.path.exists(EXP_PATH):
    os.mkdir(EXP_PATH)
    
    

### INTERPOLATION PARAMETERS
interp_method = 'cubic'
der_dx = 1e-3
der_method = 'forward'
test_prop = 0.3
points = 100000

### REPRODUCIBILITY
seed = 30
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)

### PARAMETERS
# Parameters for the initial density
sigma0 = 0.1
sigma = 0.05 
divide = 1.

# Centers for the initial density, 200 random points in [-0.7,0.7]x[-0.7,0.7]
#c = -0.7 + 1.4*np.random.random((20,2))[-5:]
c = np.array([[-0.3,-0.6],
              [-0.1,0.2],
              [0.5,-0.5],
              [+0.5,0.4]
                ])

# Initial density
def rho0(x, c:np.array):
    # Basically a sum of gaussians with different centers
    rho0 = np.exp(-((x[:,0]-c[0,0])**2 + (x[:,1]-c[0,1])**2)/sigma)
    for i in range(1,c.shape[0]):
        rho0 += np.exp(-((x[:,0]-c[i,0])**2 + (x[:,1]-c[i,1])**2)/sigma)
    return rho0*(rho0>0) / divide

# The equation is dr/dt + div(ur) = 0 which is the continuity equation
# dr/dt + div(u)r + u*grad(r) = 0 
# Parameters for the dynamical system

# Parameters for the simulation
# Time variables
t_max = 10.
### dt must be small enough to avoid numerical instability, in particular the CFL condition dt < dx/|u_max| where u_max is the maximum velocity
dt = 0.001
# Space discretization
dx = 0.01
x_max = 1.5
x_min = -1.5

# Number of steps
t_steps = int(t_max/dt)+1
x_steps = int((x_max-x_min)/dx)+1
# Calculate the vector of coordinates
x_vec = x_min + np.arange(x_steps)*dx
y_vec = x_min + np.arange(x_steps)*dx

print('x_vec:', x_vec)
print('x_steps:', x_steps)
print('x_vec.shape:', x_vec.shape)

# Function for the velocity field, it is basically a rotation
def u(x) -> np.array:
    #return np.array([-x[1]/np.sqrt(x[1]**2+x[0]**2), x[0]/np.sqrt(x[1]**2+x[0]**2)])
    return np.array([-x[1], x[0]])

# Divergence of the velocity field, it is 0.
def div_u(x) -> float:
    return 0.

# Vectorial versions of the same functions  
def u_vec(x):   
    #return np.column_stack([-x[:,1]/np.sqrt(x[:,1]**2+x[:,0]**2), x[:,0]/np.sqrt(x[:,1]**2+x[:,0]**2)])
    return np.column_stack([-x[:,1], x[:,0]])
    
def div_u_vec(x) -> float:
    return np.zeros_like(x[:,0])

# Initialize the density and the masses (the volumes for the finite volumes method)
rho = np.zeros((t_steps,x_steps,x_steps))
Q = np.zeros((t_steps,x_steps,x_steps))

if not os.path.isfile(f'{EXP_PATH}/rho.npy') or recreate:

    print('Generating initial conditions...')

    tot_mass = 0.
    total_masses = []
    for i in range(x_steps):
        x = x_min + i*dx
        for j in range(x_steps):
            y = x_min + j*dx
            #rho[0,i,j] += np.exp(-((x)**2 + (y)**2)/sigma0)
            for k in range(c.shape[0]):
                rho[0,i,j] += np.exp(-((x-c[k,0])**2 + (y-c[k,1])**2)/sigma) / divide
            Q[0,i,j] = rho[0,i,j]*dx**2
            tot_mass += Q[0,i,j]

    total_masses.append(tot_mass)
    print('Finished!')
    print(f'The total mass is: {tot_mass}')
    print(f'The maximum and minimum values of the initial density are: {np.max(rho[0,:,:])} and {np.min(rho[0,:,:])}') 

    # Calculate the initial density on a grid for plotting
    X,Y = np.meshgrid(np.linspace(x_min,x_max+dx,x_steps),np.linspace(x_min,x_max+dx,x_steps))
    pts = np.vstack([X.reshape(-1),Y.reshape(-1)]).T
    density = np.array(rho0(pts, c=c)).reshape(X.shape)

    #plots the streamplot for the velocity field
    plt.figure(figsize=(6,5))
    # Velocity field
    vel = u_vec(pts)
    U = np.array(vel[:,0].reshape(X.shape))
    V = np.array(vel[:,1].reshape(Y.shape))
    # Streamplot of the velocity field
    plt.streamplot(X,Y,U,V,density=0.4,linewidth=0.05,color='grey')
    # Limits of the plot
    plt.xlim((x_min - 0.01,x_max + 0.01))
    plt.ylim((x_min - 0.01,x_max + 0.01))
    # Density plot
    plt.contourf(X,Y,density,50,cmap='Greys')
    plt.colorbar()
    plt.savefig(f'{EXP_PATH}/dynsys_init.pdf', format='pdf')

    # Function to calculate the minmod function for the finite volumes method
    def minmod(a,b):
        # If opposite sign or one of them is 0 return 0
        if a*b <= 0:
            return 0
        # Otherwise return the minimum of the two in modulus
        elif abs(a)<abs(b):
            return a
        else:
            return b

    ### DYNAMICAL SYSTEM SIMULATION WITH FINITE VOLUMES METHOD
    print('Simulating dynamical system...')
    # Cycle through time
    for t in range(1,t_steps):
        tot_mass = 0.
        # Cycle through x
        for i in range(1,x_steps-1):
            # Get the value of x at that point
            x = x_min + i*dx
            # Cycle the value of y
            for j in range(1,x_steps-1):
                # Get the value of y at that point
                y = x_min + j*dx
                # Velocities on boundaries of the volumes
                ul = u([x-dx/2,y])[0]
                ur = u([x+dx/2,y])[0]
                uu = u([x,y+dx/2])[1]
                ud = u([x,y-dx/2])[1]
                
                # Fluxes, with upwind method
                # X flux from left
                if i==1:
                    Fxl = 0.
                else:
                    Fxl = ((ul+np.abs(ul))/2)*rho[t-1,i-1,j] + ((ul-np.abs(ul))/2)*rho[t-1,i,j]
                # X flux from the right
                if i==x_steps-2:
                    Fxr = 0.
                else:
                    Fxr = ((ur+np.abs(ur))/2)*rho[t-1,i,j] + ((ur-np.abs(ur))/2)*rho[t-1,i+1,j]
                
                # Y flux from down
                if j==1:
                    Fxd = 0.
                else:
                    Fxd = ((ud+np.abs(ud))/2)*rho[t-1,i,j-1] + ((ud-np.abs(ud))/2)*rho[t-1,i,j]
                # Y flux from the up
                if j==x_steps-2:
                    Fxu = 0.
                else:
                    Fxu = ((uu+np.abs(uu))/2)*rho[t-1,i,j] + ((uu-np.abs(uu))/2)*rho[t-1,i,j+1]
        
                # Now we are ready to update the density at the next time step following
                # dr/dt + div(u)r + u*grad(r) = 0
                # We are working with volumes so first we update the total mass in the volume       
                # Then we update the density
                rho[t,i,j] = rho[t-1,i,j] - dt/dx*(Fxr-Fxl) - dt/dx*(Fxu-Fxd)
                # We update the total mass
                Q[t,i,j] = rho[t,i,j]*dx**2
                tot_mass += Q[t,i,j]
        total_masses.append(tot_mass)       
        if t%10 == 0:
            print(f'Time: {t*dt}')
            print(f'Tot_mass: {tot_mass}')
            print(f'Max and min values of the density: {np.max(rho[t,:,:])} and {np.min(rho[t,:,:])}')
                
    print('Finished!')

    

    print('Saving complete trajectory...')
    with open(f'{EXP_PATH}/rho.npy', 'wb') as f:
        np.save(f, rho)
    with open(f'{EXP_PATH}/Q.npy', 'wb') as f:
        np.save(f, Q)
    print('Finished!')
    with open(f'{EXP_PATH}/total_masses.npy', 'wb') as f:
        np.save(f, total_masses)

else:
    with open(f'{EXP_PATH}/rho.npy', 'rb') as f:
        rho = np.load(f)
    with open(f'{EXP_PATH}/Q.npy', 'rb') as f:
        Q = np.load(f)
    with open(f'{EXP_PATH}/total_masses.npy', 'rb') as f:
        total_masses = np.load(f)

print('Plotting and saving the dynamical system...')
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
print(f'Maximum value of the masses: {np.max(total_masses)}')
# Max and min values for the density
vmax = np.max(rho)
vmin = 0
levels = np.linspace(vmin,vmax,50)
print(f'Maximum value of the density: {vmax}')

rho = rho[::10]
dt = 0.01

for t in range(10):
    # Calculate the time
    t_plot = t/10.*t_max
    print(f'Plotting time: {t_plot}')
    # Index of the time step
    t_ind = int(t_plot/dt)
    rho_interp = RegularGridInterpolator((x_vec,y_vec), rho[t_ind,:,:], method=interp_method)    

    # Calculate the grid for the density
    N = 250
    X,Y = np.meshgrid(np.linspace(x_min,x_max,N),np.linspace(x_min,x_max,N))
    pts = np.vstack([X.reshape(-1),Y.reshape(-1)]).T

    # Calculate the density on the grid
    rho_plot = rho_interp(pts).reshape(X.shape)
    #print(rho_plot)
    #plots the streamplot for the velocity field
    plt.figure(figsize=(6,5))
    # Velocity field
    vel = u_vec(pts)
    U = np.array(vel[:,0].reshape(X.shape))
    V = np.array(vel[:,1].reshape(Y.shape))
    # Streamplot of the velocity field
    plt.streamplot(X,Y,U,V,density=0.4,linewidth=0.05,color='grey')
    # Limits of the plot
    plt.xlim((x_min - 0.01,x_max + 0.01))
    plt.ylim((x_min - 0.01,x_max + 0.01))
    plt.title(f'Time: {t_plot}')
    # Density plot
    plt.contourf(X,Y,rho_plot,50,cmap='Greys', levels=levels, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig(f'{EXP_PATH}/dynsys_traj{t}.pdf', format='pdf')
    plt.close()

#raise ValueError('Stop here')
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(13,4), layout='compressed')
for i, t in enumerate([0,5,9]):
    # Calculate the grid for the density
    t_plot = t/10.*t_max
    print(f'Plotting time: {t_plot}')
    # Index of the time step
    t_ind = int(t_plot/dt)
    rho_interp = RegularGridInterpolator((x_vec,y_vec), rho[t_ind,:,:], method=interp_method)
    
    N = 250
    X,Y = np.meshgrid(np.linspace(x_min,x_max,N),np.linspace(x_min,x_max,N))
    pts = np.vstack([X.reshape(-1),Y.reshape(-1)]).T

    # Calculate the density on the grid
    rho_plot = rho_interp(pts).reshape(X.shape)
    #print(rho_plot)
    #plots the streamplot for the velocity field
    # Velocity field
    vel = u_vec(pts)
    U = np.array(vel[:,0].reshape(X.shape))
    V = np.array(vel[:,1].reshape(Y.shape))
    # Streamplot of the velocity field
    ax[i].streamplot(X,Y,U,V,density=0.4,linewidth=0.05,color='grey')
    # Limits of the plot
    ax[i].set_xlim((x_min - 0.01,x_max + 0.01))
    ax[i].set_ylim((x_min - 0.01,x_max + 0.01))
    ax[i].set_title(f'Time: {t}')
    ax[i].set_aspect('equal')
    # Density plot
    contour = ax[i].contourf(X,Y,rho_plot,50,cmap='Greys', levels=levels, vmin=vmin, vmax=vmax)
fig.colorbar(contour, ax=ax, orientation='vertical', fraction=0.02, pad=0.05)
plt.savefig(f'{EXP_PATH}/dynsys_traj_multi.pdf', format='pdf')
    

from torch.utils.data import TensorDataset

# Vector of times
t_vec = np.arange(0.,t_max+dt,dt)

### INITIAL DATA GENERATION
print('Generating inital data...')

# Initial data is given via the original rho
# Meshgrid for the initial data
x_init, y_init = np.meshgrid(x_vec, x_vec, indexing='ij')
pts = np.column_stack((x_init.reshape((-1,1)), y_init.reshape((-1,1))))
# Extract the initial values
rho_init = rho[0,:,:].reshape((-1,1))
pts_save = torch.column_stack((torch.zeros_like(torch.tensor(rho_init)), torch.from_numpy(pts)))
u_save = u_vec(pts)

init_data = TensorDataset(pts_save, torch.tensor(rho_init), torch.tensor(u_save))
torch.save(init_data, f'{EXP_PATH}/rho_init_dataset.pth')
print('Finished!')


print('Generating boundary data...')
# Boundary data is given by the boundary values of the density
# Vector of values between 0 and 1 
downsample = 10
# Vectors for the coordinates on the boundary x = -1
t_bc1 = torch.from_numpy(t_vec)
y_bc1 = torch.from_numpy(y_vec)
# Meshgrid for the boundary data
t_bc1, y_bc1 = torch.meshgrid(t_bc1, y_bc1, indexing='ij')
x_bc1 = -1.*torch.ones_like(t_bc1)
pts_bc1 = torch.column_stack((t_bc1.reshape((-1,1)), x_bc1.reshape((-1,1)), y_bc1.reshape((-1,1))))
print('rho_bc1:', np.max(rho[:,0,:]))
out_bc1 = torch.zeros_like(t_bc1).reshape((-1,1))

# Same for x = 1
x_bc2 = 1.*torch.ones_like(t_bc1)
pts_bc2 = torch.column_stack((t_bc1.reshape((-1,1)), x_bc2.reshape((-1,1)), y_bc1.reshape((-1,1))))
print('rho_bc2:', np.max(rho[:,-1,:]))
out_bc2 = torch.zeros_like(t_bc1).reshape((-1,1))

# Now on to y = -1
x_bc3 = torch.from_numpy(x_vec)
t_bc3 = torch.from_numpy(t_vec)
# Meshgrid for the boundary data
t_bc3, x_bc3 = torch.meshgrid(t_bc3, x_bc3, indexing='ij')
y_bc3 = -1.*torch.ones_like(t_bc3)
pts_bc3 = torch.column_stack((t_bc3.reshape((-1,1)), x_bc3.reshape((-1,1)), y_bc3.reshape((-1,1))))
print('rho_bc3:', np.max(rho[:,:,0]))
out_bc3 = torch.zeros_like(t_bc3).reshape((-1,1))

# Now y = 1
y_bc4 = 1.*torch.ones_like(t_bc3)
pts_bc4 = torch.column_stack((t_bc3.reshape((-1,1)), x_bc3.reshape((-1,1)), y_bc4.reshape((-1,1))))
print('rho_bc4:', np.max(rho[:,:,-1]))
out_bc4 = torch.zeros_like(t_bc3).reshape((-1,1))

pts_bc = torch.cat((pts_bc1, pts_bc2, pts_bc3, pts_bc4), dim=0)
out_bc = torch.cat((out_bc1, out_bc2, out_bc3, out_bc4), dim=0)

bc_dataset = TensorDataset(pts_bc, out_bc)
torch.save(bc_dataset, f'{EXP_PATH}/rho_bc_dataset.pth')

t_save, x_save, y_save = np.meshgrid(t_vec[1:-1:downsample], x_vec[1:-1:downsample], y_vec[1:-1:downsample], indexing='ij')
pts_save = torch.from_numpy(np.column_stack((t_save.reshape((-1,1)), x_save.reshape((-1,1)), y_save.reshape((-1,1)))))
rho_save = torch.tensor(rho[1:-1:downsample,1:-1:downsample,1:-1:downsample].reshape((-1,1)))
u_save = torch.tensor(u_vec(pts_save))
if der_method == 'centered':
    pdv_x_grid = (rho[1:-1,2:,1:-1]-rho[1:-1,:-2,1:-1])/(2*dx)
    pdv_y_grid = (rho[1:-1,1:-1,2:]-rho[1:-1,1:-1,:-2])/(2*dx)
    pdv_t_grid = (rho[2:,1:-1,1:-1]-rho[:-2,1:-1,1:-1])/(2*dt)
else:
    pdv_x_grid = (rho[1:-1,2:,1:-1]-rho[1:-1,1:-1,1:-1])/(dx)
    pdv_y_grid = (rho[1:-1,1:-1,2:]-rho[1:-1,1:-1,1:-1])/(dx)
    pdv_t_grid = (rho[2:,1:-1,1:-1]-rho[1:-1,1:-1,1:-1])/(dt)
pdv_save = torch.from_numpy(np.column_stack((pdv_t_grid[::downsample,::downsample,::downsample].reshape((-1,1)), pdv_x_grid[::downsample,::downsample,::downsample].reshape((-1,1)), pdv_y_grid[::downsample,::downsample,::downsample].reshape((-1,1)))))
print(pts_save.shape)
print(pdv_save.shape)
print(rho_save.shape)
print(u_save.shape)
rho_data = TensorDataset(pts_save, rho_save, pdv_save, u_save)
torch.save(rho_data, f'{EXP_PATH}/rho_grid_dataset.pth')


### DENSITY DATA GENERATION
print('Generating global data...')
### Define the function interpolator
rho_interp = RegularGridInterpolator((t_vec,x_vec,y_vec), rho, method=interp_method)
# Define the random points
points_x = torch.distributions.Uniform(-1.5+der_dx,1.5-der_dx).sample((points,1))
points_y = torch.distributions.Uniform(-1.5+der_dx,1.5-der_dx).sample((points,1))
points_t = torch.distributions.Uniform(0.+der_dx,10.-der_dx).sample((points,1))
pts = torch.column_stack((points_t, points_x, points_y))

# Get the dx vectors
dx_vec = np.column_stack((np.zeros(points_x.shape), der_dx*np.ones(points_x.shape), np.zeros(points_x.shape)))
dy_vec = np.column_stack((np.zeros(points_x.shape), np.zeros(points_x.shape), der_dx*np.ones(points_x.shape)))
dt_vec = np.column_stack((der_dx*np.ones(points_x.shape), np.zeros(points_x.shape), np.zeros(points_x.shape)))

# Get the discrete partial derivatives
if der_method == 'centered':
    pdv_x = torch.tensor((rho_interp(pts + dx_vec) - rho_interp(pts-dx_vec))/(2*der_dx))
    pdv_y = torch.tensor((rho_interp(pts + dy_vec) - rho_interp(pts-dy_vec))/(2*der_dx))
    pdv_t = torch.tensor((rho_interp(pts + dt_vec) - rho_interp(pts-dt_vec))/(2*der_dx))
elif der_method == 'forward':
    pdv_x = torch.tensor((rho_interp(pts + dx_vec) - rho_interp(pts))/(der_dx))
    pdv_y = torch.tensor((rho_interp(pts + dy_vec) - rho_interp(pts))/(der_dx))
    pdv_t = torch.tensor((rho_interp(pts + dt_vec) - rho_interp(pts))/(der_dx))
else:
    raise ValueError('Invalid derivative method')



# Get the final pdv tensor
pdv_save = torch.column_stack((pdv_t, pdv_x, pdv_y))
rho_save = torch.tensor(rho_interp(pts).reshape((-1,1)))
u_save = torch.tensor(u_vec(np.column_stack((points_x, points_y))))

rho_data = TensorDataset(pts, rho_save, pdv_save, u_save)
torch.save(rho_data, f'{EXP_PATH}/rho_full_dataset.pth')
print('Finished!')




### ------------------------------------------------------------------ ###
### EXTRAPOLATION DATA GENERATION
print('Generating extrapolation experiment data...')

# Define the random points
points_x = torch.distributions.Uniform(-1.+der_dx,1.-der_dx).sample((points,1))
points_y = torch.distributions.Uniform(-1.+der_dx,1.-der_dx).sample((points,1))
points_t = torch.distributions.Uniform(0.+der_dx,1.-test_prop).sample((points,1))
pts = torch.column_stack((points_t, points_x, points_y)).numpy()

# Get the dx vectors
dx_vec = np.column_stack((np.zeros(points_x.shape), der_dx*np.ones(points_x.shape), np.zeros(points_x.shape)))
dy_vec = np.column_stack((np.zeros(points_x.shape), np.zeros(points_x.shape), der_dx*np.ones(points_x.shape)))
dt_vec = np.column_stack((der_dx*np.ones(points_x.shape), np.zeros(points_x.shape), np.zeros(points_x.shape)))

# Get the discrete partial derivatives
if der_method == 'centered':
    pdv_x = torch.tensor((rho_interp(pts + dx_vec) - rho_interp(pts-dx_vec))/(2*der_dx))
    pdv_y = torch.tensor((rho_interp(pts + dy_vec) - rho_interp(pts-dy_vec))/(2*der_dx))
    pdv_t = torch.tensor((rho_interp(pts + dt_vec) - rho_interp(pts-dt_vec))/(2*der_dx))
elif der_method == 'forward':
    pdv_x = torch.tensor((rho_interp(pts + dx_vec) - rho_interp(pts))/(der_dx))
    pdv_y = torch.tensor((rho_interp(pts + dy_vec) - rho_interp(pts))/(der_dx))
    pdv_t = torch.tensor((rho_interp(pts + dt_vec) - rho_interp(pts))/(der_dx))
else:
    raise ValueError('Invalid derivative method')

# Get the final pdv tensor
pdv_save = torch.column_stack((pdv_t, pdv_x, pdv_y))

rho_save = torch.tensor(rho_interp(pts).reshape((-1,1)))
u_save = torch.tensor(u_vec(np.column_stack((points_x, points_y))))

rho_data = TensorDataset(torch.tensor(pts), rho_save, pdv_save, u_save)
torch.save(rho_data, f'{EXP_PATH}/rho_extrapolate_dataset_train.pth')

# Define the random points
points_x = torch.distributions.Uniform(-1.+der_dx,1.-der_dx).sample((points,1))
points_y = torch.distributions.Uniform(-1.+der_dx,1.-der_dx).sample((points,1))
points_t = torch.distributions.Uniform(1.-test_prop,1.-der_dx).sample((points,1))
pts = torch.column_stack((points_t, points_x, points_y))

# Get the dx vectors
dx_vec = np.column_stack((np.zeros(points_x.shape), der_dx*np.ones(points_x.shape), np.zeros(points_x.shape)))
dy_vec = np.column_stack((np.zeros(points_x.shape), np.zeros(points_x.shape), der_dx*np.ones(points_x.shape)))
dt_vec = np.column_stack((der_dx*np.ones(points_x.shape), np.zeros(points_x.shape), np.zeros(points_x.shape)))

# Get the discrete partial derivatives
if der_method == 'centered':
    pdv_x = torch.tensor((rho_interp(pts + dx_vec) - rho_interp(pts-dx_vec))/(2*der_dx))
    pdv_y = torch.tensor((rho_interp(pts + dy_vec) - rho_interp(pts-dy_vec))/(2*der_dx))
    pdv_t = torch.tensor((rho_interp(pts + dt_vec) - rho_interp(pts-dt_vec))/(2*der_dx))
elif der_method == 'forward':
    pdv_x = torch.tensor((rho_interp(pts + dx_vec) - rho_interp(pts))/(der_dx))
    pdv_y = torch.tensor((rho_interp(pts + dy_vec) - rho_interp(pts))/(der_dx))
    pdv_t = torch.tensor((rho_interp(pts + dt_vec) - rho_interp(pts))/(der_dx))
else:
    raise ValueError('Invalid derivative method')

# Get the final pdv tensor
pdv_save = torch.column_stack((pdv_t, pdv_x, pdv_y))

rho_save = torch.tensor(rho_interp(pts).reshape((-1,1)))
u_save = torch.tensor(u_vec(np.column_stack((points_x, points_y))))

rho_data = TensorDataset(pts, rho_save, pdv_save, u_save)
torch.save(rho_data, f'{EXP_PATH}/rho_extrapolate_dataset_test.pth')
print('Finished!')


### INTERPOLATION DATA GENERATION
print('Generating adaptation experiment data...')

# Define the random points
points_x = torch.distributions.Uniform(-1.+der_dx,1.-der_dx).sample((points,1))
points_y = torch.distributions.Uniform(-1.+der_dx,1.-der_dx).sample((points,1))
points_t = torch.distributions.Uniform(0.+test_prop,1.-der_dx).sample((points,1))
pts = torch.column_stack((points_t, points_x, points_y))

# Get the dx vectors
dx_vec = np.column_stack((np.zeros(points_x.shape), der_dx*np.ones(points_x.shape), np.zeros(points_x.shape)))
dy_vec = np.column_stack((np.zeros(points_x.shape), np.zeros(points_x.shape), der_dx*np.ones(points_x.shape)))
dt_vec = np.column_stack((der_dx*np.ones(points_x.shape), np.zeros(points_x.shape), np.zeros(points_x.shape)))

# Get the discrete partial derivatives
if der_method == 'centered':
    pdv_x = torch.tensor((rho_interp(pts + dx_vec) - rho_interp(pts-dx_vec))/(2*der_dx))
    pdv_y = torch.tensor((rho_interp(pts + dy_vec) - rho_interp(pts-dy_vec))/(2*der_dx))
    pdv_t = torch.tensor((rho_interp(pts + dt_vec) - rho_interp(pts-dt_vec))/(2*der_dx))
elif der_method == 'forward':
    pdv_x = torch.tensor((rho_interp(pts + dx_vec) - rho_interp(pts))/(der_dx))
    pdv_y = torch.tensor((rho_interp(pts + dy_vec) - rho_interp(pts))/(der_dx))
    pdv_t = torch.tensor((rho_interp(pts + dt_vec) - rho_interp(pts))/(der_dx))
else:
    raise ValueError('Invalid derivative method')

# Get the final pdv tensor
pdv_save = torch.column_stack((pdv_t, pdv_x, pdv_y))

rho_save = torch.tensor(rho_interp(pts).reshape((-1,1)))
u_save = torch.tensor(u_vec(np.column_stack((points_x, points_y))))

rho_data = TensorDataset(pts, rho_save, pdv_save, u_save)
torch.save(rho_data, f'{EXP_PATH}/rho_adapt_dataset_train.pth')


# Define the random points
points_x = torch.distributions.Uniform(-1.+der_dx,1.-der_dx).sample((points,1))
points_y = torch.distributions.Uniform(-1.+der_dx,1.-der_dx).sample((points,1))
points_t = torch.distributions.Uniform(0.+der_dx,0.+test_prop).sample((points,1))
pts = torch.column_stack((points_t, points_x, points_y))

# Get the dx vectors
dx_vec = np.column_stack((np.zeros(points_x.shape), der_dx*np.ones(points_x.shape), np.zeros(points_x.shape)))
dy_vec = np.column_stack((np.zeros(points_x.shape), np.zeros(points_x.shape), der_dx*np.ones(points_x.shape)))
dt_vec = np.column_stack((der_dx*np.ones(points_x.shape), np.zeros(points_x.shape), np.zeros(points_x.shape)))

# Get the discrete partial derivatives
if der_method == 'centered':
    pdv_x = torch.tensor((rho_interp(pts + dx_vec) - rho_interp(pts-dx_vec))/(2*der_dx))
    pdv_y = torch.tensor((rho_interp(pts + dy_vec) - rho_interp(pts-dy_vec))/(2*der_dx))
    pdv_t = torch.tensor((rho_interp(pts + dt_vec) - rho_interp(pts-dt_vec))/(2*der_dx))
elif der_method == 'forward':
    pdv_x = torch.tensor((rho_interp(pts + dx_vec) - rho_interp(pts))/(der_dx))
    pdv_y = torch.tensor((rho_interp(pts + dy_vec) - rho_interp(pts))/(der_dx))
    pdv_t = torch.tensor((rho_interp(pts + dt_vec) - rho_interp(pts))/(der_dx))
else:
    raise ValueError('Invalid derivative method')

# Get the final pdv tensor
pdv_save = torch.column_stack((pdv_t, pdv_x, pdv_y))

rho_save = torch.tensor(rho_interp(pts).reshape((-1,1)))
u_save = torch.tensor(u_vec(np.column_stack((points_x, points_y))))

rho_data = TensorDataset(pts, rho_save, pdv_save, u_save)
torch.save(rho_data, f'{EXP_PATH}/rho_adapt_dataset_test.pth')

print('Finished!')


print('Generating interpolation experiment data...')
# Define the random points
points_x = torch.distributions.Uniform(-1.+der_dx,1.-der_dx).sample((points,1))
points_y = torch.distributions.Uniform(-1.+der_dx,1.-der_dx).sample((points,1))
points_t1 = torch.distributions.Uniform(0.+der_dx,0.5-test_prop/2).sample((int(points/2),1))
points_t2 = torch.distributions.Uniform(0.5+test_prop/2,1.-der_dx).sample((int(points/2),1))
points_t = torch.vstack((points_t1, points_t2))
pts = torch.column_stack((points_t, points_x, points_y))

# Get the dx vectors
dx_vec = np.column_stack((np.zeros(points_x.shape), der_dx*np.ones(points_x.shape), np.zeros(points_x.shape)))
dy_vec = np.column_stack((np.zeros(points_x.shape), np.zeros(points_x.shape), der_dx*np.ones(points_x.shape)))
dt_vec = np.column_stack((der_dx*np.ones(points_x.shape), np.zeros(points_x.shape), np.zeros(points_x.shape)))

# Get the discrete partial derivatives
if der_method == 'centered':
    pdv_x = torch.tensor((rho_interp(pts + dx_vec) - rho_interp(pts-dx_vec))/(2*der_dx))
    pdv_y = torch.tensor((rho_interp(pts + dy_vec) - rho_interp(pts-dy_vec))/(2*der_dx))
    pdv_t = torch.tensor((rho_interp(pts + dt_vec) - rho_interp(pts-dt_vec))/(2*der_dx))
elif der_method == 'forward':
    pdv_x = torch.tensor((rho_interp(pts + dx_vec) - rho_interp(pts))/(der_dx))
    pdv_y = torch.tensor((rho_interp(pts + dy_vec) - rho_interp(pts))/(der_dx))
    pdv_t = torch.tensor((rho_interp(pts + dt_vec) - rho_interp(pts))/(der_dx))
else:
    raise ValueError('Invalid derivative method')

# Get the final pdv tensor
pdv_save = torch.column_stack((pdv_t, pdv_x, pdv_y))

rho_save = torch.tensor(rho_interp(pts).reshape((-1,1)))
u_save = torch.tensor(u_vec(np.column_stack((points_x, points_y))))

rho_data = TensorDataset(pts, rho_save, pdv_save, u_save)
torch.save(rho_data, f'{EXP_PATH}/rho_interpolate_dataset_train.pth')


# Define the random points
points_x = torch.distributions.Uniform(-1.+der_dx,1.-der_dx).sample((points,1))
points_y = torch.distributions.Uniform(-1.+der_dx,1.-der_dx).sample((points,1))
points_t = torch.distributions.Uniform(0.5-test_prop/2,0.5+test_prop/2).sample((points,1))
pts = torch.column_stack((points_t, points_x, points_y))

# Get the dx vectors
dx_vec = np.column_stack((np.zeros(points_x.shape), der_dx*np.ones(points_x.shape), np.zeros(points_x.shape)))
dy_vec = np.column_stack((np.zeros(points_x.shape), np.zeros(points_x.shape), der_dx*np.ones(points_x.shape)))
dt_vec = np.column_stack((der_dx*np.ones(points_x.shape), np.zeros(points_x.shape), np.zeros(points_x.shape)))

# Get the discrete partial derivatives
if der_method == 'centered':
    pdv_x = torch.tensor((rho_interp(pts + dx_vec) - rho_interp(pts-dx_vec))/(2*der_dx))
    pdv_y = torch.tensor((rho_interp(pts + dy_vec) - rho_interp(pts-dy_vec))/(2*der_dx))
    pdv_t = torch.tensor((rho_interp(pts + dt_vec) - rho_interp(pts-dt_vec))/(2*der_dx))
elif der_method == 'forward':
    pdv_x = torch.tensor((rho_interp(pts + dx_vec) - rho_interp(pts))/(der_dx))
    pdv_y = torch.tensor((rho_interp(pts + dy_vec) - rho_interp(pts))/(der_dx))
    pdv_t = torch.tensor((rho_interp(pts + dt_vec) - rho_interp(pts))/(der_dx))
else:
    raise ValueError('Invalid derivative method')

# Get the final pdv tensor
pdv_save = torch.column_stack((pdv_t, pdv_x, pdv_y))

rho_save = torch.tensor(rho_interp(pts).reshape((-1,1)))
u_save = torch.tensor(u_vec(np.column_stack((points_x, points_y))))

rho_data = TensorDataset(pts, rho_save, pdv_save, u_save)
torch.save(rho_data, f'{EXP_PATH}/rho_interpolate_dataset_test.pth')
print('Finished!')
