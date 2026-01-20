import numpy as np
import torch
import random
from torch.utils.data import TensorDataset
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import os
# Parameters of the pendulum
m = 1. # old was 0.1
l = 10. # old was 1.
g = 9.81
b = 0.5

EXP_PATH = 'data'
if not os.path.exists(EXP_PATH):
    os.mkdir(EXP_PATH)

# Set seeds
seed = 30
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)

# Fraction of trajectories for train
train_frac = 0.75
# Sequence length for the RNN dataset
seq_len = 1

# Number of trajectories
points = 50
train_points = 30
test_points = 10
# Starting points
x0 = -np.pi/4. + np.pi/2.*np.random.random((points))
v0 = -1.5 + 3*np.random.random((points))
xv0 = np.column_stack((x0.reshape((-1,1)),v0.reshape((-1,1))))
# Time variables
t_max = 10
dt = 0.01

# Function for the velocity field
def u(xv):
    return np.array([xv[1],-(g/l)*np.sin(xv[0])-(b/m)*xv[1]])
    #return np.array([xv[1],-m*g*np.sin(xv[0])-b*l*xv[1]])
    
# Function for the velocity field, but vectorial
def u_vec(xv):
    return np.column_stack((xv[:,1],-(g/l)*np.sin(xv[:,0])-(b/m)*xv[:,1]))
    #return np.column_stack((xv[:,1],-m*g*np.sin(xv[:,0])-b*l*xv[:,1]))
    
# Function for the velocity field, but for tensors
def u_tens(xv):
    return np.column_stack((xv[:,:,1],-(g/l)*np.sin(xv[:,:,0])-(b/m)*xv[:,:,1]))
    #return np.column_stack((xv[:,1],-m*g*np.sin(xv[:,0])-b*l*xv[:,1]))

# Number of steps
steps = int(t_max/dt)+1
print(f'Max simulation time: {t_max}, dt: {dt}, total time steps: {steps}')

# Generate the vectors
xv = np.zeros((points,steps,2))
u_save = np.zeros((points,steps,2))
# Set initial
xv[:,0,0] = x0
xv[:,0,1] = v0

print('Simulation...')
# Iteratively generate new positions
#for i in range(1,steps):
#    if i % 1000 == 0:
#        print(f'Time step: {i}')
#    xv[:,i,0] = xv[:,i-1,0] + dt*u_vec(xv[:,i-1,:])[:,0]
#    xv[:,i,1] = xv[:,i-1,1] + dt*u_vec(xv[:,i-1,:])[:,1]
    
for i in range(points):
    xv[i,:,:] = solve_ivp(lambda t, y: u(y), [0,t_max], xv0[i,:], t_eval=np.linspace(0,t_max,steps)).y.T
print('Simulation done!')

xv_test = np.zeros((points,steps,2))



    


############# PLOTTING THE TRAJECTORIES #############
print('Plotting...')
N = 500
xlim = np.pi/2
ylim = 1.5
X,Y = np.meshgrid(np.linspace(-xlim,xlim,N),np.linspace(-ylim,ylim,N))
pts = np.vstack([X.reshape(-1),Y.reshape(-1)]).T

#plots the streamplot for the velocity field
plt.figure(figsize=(5,5))
#print(pts)
vel = u_vec(pts)
#print(vel)
U = np.array(vel[:,0].reshape(X.shape))
V = np.array(vel[:,1].reshape(Y.shape))
#mask the outside of the ball

plt.streamplot(X,Y,U,V,density=1,color=U**2 + V**2, linewidth=0.15)

plt.xlim((-xlim,xlim))
plt.ylim((-ylim,ylim))
for i in range(min(50, train_points)):
    plt.plot(xv[i,:,0], xv[i,:,1], label=f'trajectory{i}')
plt.xlabel(r'Angle: $u$')
plt.ylabel(r'Angular speed: $\dot{u}$')
plt.title(f'Phase space trajectories')
plt.savefig(f'{EXP_PATH}/pendulum_train_phase_space.png', dpi=300)
    
plt.close()



#plots the streamplot for the velocity field
plt.figure(figsize=(5,5))
#print(pts)
vel = u_vec(pts)
#print(vel)
U = np.array(vel[:,0].reshape(X.shape))
V = np.array(vel[:,1].reshape(Y.shape))
#mask the outside of the ball

plt.streamplot(X,Y,U,V,density=1,color=U**2 + V**2, linewidth=0.15)

plt.xlim((-xlim,xlim))
plt.ylim((-ylim,ylim))
for i in range(points - test_points, points):
    plt.plot(xv[i,:,0], xv[i,:,1], label=f'trajectory{i}')
plt.xlabel(r'Angle: $u$')
plt.ylabel(r'Angular speed: $\dot{u}$')
plt.title(f'Phase space trajectories')
plt.savefig(f'{EXP_PATH}/pendulum_test_phase_space.png', dpi=300)
    
plt.close()

t_base = np.arange(start=0, stop=t_max+dt, step=dt)
for i in range(min(50, points)):
    plt.scatter(t_base, xv[i,:,0], marker='.')
plt.ylabel(r'Angle: $u$')
plt.xlabel(r'Time: $t$')


plt.title(f'Pendulum trajectories')
plt.savefig(f'{EXP_PATH}/pendulum_trajectory.png', dpi=300)
print('Plotting done!')




############# GENERATING THE DATASETS #############
print('Generating the datasets...')
train_dim = train_points
test_dim = test_points
val_dim = points - train_dim - test_dim
xv = torch.tensor(xv)
print(f'Number of training trajectories: {train_dim}')
print(f'Number of validation trajectories: {val_dim}')
print(f'Number of testing trajectories: {test_dim}')


# Vector of the time values
t_base = np.arange(start=0, stop=t_max+dt, step=dt)

# Empirical derivative dt
dt = 1e-3
# Random trajectory points
n_t = 1000
points_t = []
points_x = []
points_u = []
for i in range(points):
    f = interp1d(x=t_base, y=xv[i,:,:].T, kind='cubic')
    ts = (t_max-dt)*np.random.random((n_t))
    points_t.append(torch.tensor(ts))
    xs = f(ts).T
    points_x.append(torch.tensor(xs))
    us = (f(ts+dt).T - f(ts).T)/dt
    points_u.append(torch.tensor(us))

points_t = torch.stack(points_t, dim=0) 
points_x = torch.stack(points_x, dim=0)
points_u = torch.stack(points_u, dim=0)

points_t_train = torch.zeros((train_dim, n_t, 3))
points_t_train[:,:,0] = points_t[:train_dim]
points_t_train[:,:,1:] = torch.tensor(xv0)[:train_dim].unsqueeze(1).repeat((1,n_t,1))

points_x_train = points_x[:train_dim]
points_u_train = points_u[:train_dim]
points_q_train = points_x[:train_dim,:,0]
points_p_train = points_x[:train_dim,:,1]*m
points_dq_train = points_u[:train_dim,:,0]
points_dp_train = points_u[:train_dim,:,1]*m

points_t_test = torch.zeros((test_dim, n_t, 3))
points_t_test[:,:,0] = points_t[-test_dim:]
points_t_test[:,:,1:] = torch.tensor(xv0)[-test_dim:].unsqueeze(1).repeat((1,n_t,1))

points_x_test = points_x[-test_dim:]
points_u_test = points_u[-test_dim:]
points_q_test = points_x[-test_dim:,:,0]
points_p_test = points_x[-test_dim:,:,1]*m
points_dq_test = points_u[-test_dim:,:,0]
points_dp_test = points_u[-test_dim:,:,1]*m

points_t_val = torch.zeros((val_dim, n_t, 3))
points_t_val[:,:,0] = points_t[train_dim:train_dim+val_dim]
points_t_val[:,:,1:] = torch.tensor(xv0)[train_dim:train_dim+val_dim].unsqueeze(1).repeat((1,n_t,1))

points_x_val = points_x[train_dim:train_dim+val_dim]
points_u_val = points_u[train_dim:train_dim+val_dim]
points_q_val = points_x[train_dim:train_dim+val_dim,:,0]
points_p_val = points_x[train_dim:train_dim+val_dim,:,1]*m
points_dq_val = points_u[train_dim:train_dim+val_dim,:,0]
points_dp_val = points_u[train_dim:train_dim+val_dim,:,1]*m

print(f'points_t_train.shape: {points_t_train.shape}')
print(f'points_x_train.shape: {points_x_train.shape}')
print(f'points_u_train.shape: {points_u_train.shape}')

print('Saving random dataset...')
emp_train_dataset = TensorDataset(points_t_train.reshape((-1,3)), points_x_train.reshape((-1,2)), points_u_train.reshape((-1,2)))
emp_test_dataset = TensorDataset(points_t_test.reshape((-1,3)), points_x_test.reshape((-1,2)), points_u_test.reshape((-1,2)))
emp_val_dataset = TensorDataset(points_t_val.reshape((-1,3)), points_x_val.reshape((-1,2)), points_u_val.reshape((-1,2)))
emp_hnn_train_dataset = TensorDataset(points_t_train.reshape((-1,3)), points_q_train.reshape((-1,1)), points_p_train.reshape((-1,1)), points_dq_train.reshape((-1,1)), points_dp_train.reshape((-1,1)))
emp_hnn_val_dataset = TensorDataset(points_t_val.reshape((-1,3)), points_q_val.reshape((-1,1)), points_p_val.reshape((-1,1)), points_dq_val.reshape((-1,1)), points_dp_val.reshape((-1,1)))
emp_hnn_test_dataset = TensorDataset(points_t_test.reshape((-1,3)), points_q_test.reshape((-1,1)), points_p_test.reshape((-1,1)), points_dq_test.reshape((-1,1)), points_dp_test.reshape((-1,1)))

torch.save(emp_train_dataset, f'{EXP_PATH}/emp_dataset_train.pth')
torch.save(emp_test_dataset, f'{EXP_PATH}/emp_dataset_test.pth')
torch.save(emp_val_dataset, f'{EXP_PATH}/emp_dataset_val.pth')
torch.save(emp_hnn_train_dataset, f'{EXP_PATH}/emp_hnn_dataset_train.pth')
torch.save(emp_hnn_val_dataset, f'{EXP_PATH}/emp_hnn_dataset_val.pth')
torch.save(emp_hnn_test_dataset, f'{EXP_PATH}/emp_hnn_dataset_test.pth')

print('Random dataset saved!')


print('Saving regular dataset...')
# Derivative vector
u_save = torch.zeros((points, steps, 2))
for i in range(steps):
    u_save[:,i,:] = torch.tensor(u_vec(xv[:,i,:]))

# Train/test split
xv_train = xv[:train_dim]
xv_val = xv[train_dim:train_dim+val_dim]
xv_test = xv[-test_dim:]

u_train = u_save[:train_dim]
u_val = u_save[train_dim:train_dim+val_dim]
u_test = u_save[-test_dim:]


q_train = xv[:train_dim,:,0]
p_train = xv[:train_dim,:,1]*m
dq_train = u_save[:train_dim,:,0]
dp_train = u_save[:train_dim,:,1]*m

q_val = xv[train_dim:train_dim+val_dim,:,0]
p_val = xv[train_dim:train_dim+val_dim,:,1]*m
dq_val = u_save[train_dim:train_dim+val_dim,:,0]
dp_val = u_save[train_dim:train_dim+val_dim,:,1]*m

q_test = xv[-test_dim:,:,0]
p_test = xv[-test_dim:,:,1]*m
dq_test = u_save[-test_dim:,:,0]
dp_test = u_save[-test_dim:,:,1]*m


t_train = torch.zeros((train_dim, steps, 3))
t_train[:,:,0] = torch.tile(torch.tensor(t_base), (train_dim,1))
t_train[:,:,1:] = torch.tensor(xv0)[:train_dim].unsqueeze(1).repeat((1,steps,1))

point_dataset = TensorDataset(t_train.reshape((-1,3)), xv_train.reshape((-1,2)), u_train.reshape((-1,2)))
hnn_dataset = TensorDataset(t_train.reshape((-1,3)), q_train.reshape((-1,1)), p_train.reshape((-1,1)), dq_train.reshape((-1,1)), dp_train.reshape((-1,1)))
print(f't_train.shape: {t_train.shape}')
print(f'xv_train.shape: {xv_train.shape}')
print(f'u_train.shape: {u_train.shape}')
print(f'q_train.shape: {q_train.shape}')
print(f'p_train.shape: {p_train.shape}')
print(f'dq_train.shape: {dq_train.shape}')
print(f'dp_train.shape: {dp_train.shape}')

torch.save(point_dataset, f'{EXP_PATH}/true_dataset_train.pth')
torch.save(hnn_dataset, f'{EXP_PATH}/true_hnn_dataset_train.pth')

t_test = torch.zeros((test_dim, steps, 3))
t_test[:,:,0] = torch.tile(torch.tensor(t_base), (test_dim,1))
t_test[:,:,1:] = torch.tensor(xv0)[-test_dim:].unsqueeze(1).repeat((1,steps,1))


point_dataset = TensorDataset(t_test.reshape((-1,3)), xv_test.reshape((-1,2)), u_test.reshape((-1,2)))
hnn_dataset_test = TensorDataset(t_test.reshape((-1,3)), q_test.reshape((-1,1)), p_test.reshape((-1,1)), dq_test.reshape((-1,1)), dp_test.reshape((-1,1)))
torch.save(point_dataset, f'{EXP_PATH}/true_dataset_test.pth')
torch.save(hnn_dataset_test, f'{EXP_PATH}/true_hnn_dataset_test.pth')


t_val = torch.zeros((val_dim, steps, 3))
t_val[:,:,0] = torch.tile(torch.tensor(t_base), (val_dim,1))
t_val[:,:,1:] = torch.tensor(xv0)[train_dim:train_dim+val_dim].unsqueeze(1).repeat((1,steps,1))

point_dataset = TensorDataset(t_val.reshape((-1,3)), xv_val.reshape((-1,2)), u_val.reshape((-1,2)))
hnn_dataset_val = TensorDataset(t_val.reshape((-1,3)), q_val.reshape((-1,1)), p_val.reshape((-1,1)), dq_val.reshape((-1,1)), dp_val.reshape((-1,1)))

torch.save(point_dataset, f'{EXP_PATH}/true_dataset_val.pth')
torch.save(hnn_dataset_val, f'{EXP_PATH}/true_hnn_dataset_val.pth')

print('Done!')

print('Generating time series data...')
def gen_ts_data(x: torch.Tensor, seq_len: int):
    data_x = []
    data_y = []
    data_10y = []
    for i in range(x.shape[0]):
        for j in range(x.shape[1]-seq_len-10):
            data_x.append(x[i,j].reshape((1,2)))
            data_y.append(x[i,j+1].reshape((1,2)))
            data_10y.append(x[i,j+10].reshape((1,2)))

    return torch.concatenate(data_x), torch.concatenate(data_y), torch.concatenate(data_10y)

print(xv_train.shape)
ts_train_x, ts_train_y, ts_train_10y = gen_ts_data(xv_train, seq_len=seq_len)
print(ts_train_x.shape)
u_train = torch.from_numpy(u_vec(ts_train_x.reshape((-1,2))).reshape(ts_train_x.shape))
ts_dataset = TensorDataset(ts_train_x, ts_train_y, ts_train_10y)

torch.save(ts_dataset, f'{EXP_PATH}/ts_dataset_train.pth')

ts_test_x, ts_test_y, ts_test_10y = gen_ts_data(xv_test, seq_len=seq_len)
u_test = torch.from_numpy(u_vec(ts_test_x.reshape((-1,2))).reshape(ts_test_x.shape))
ts_dataset = TensorDataset(ts_test_x, ts_test_y, ts_test_10y)

torch.save(ts_dataset, f'{EXP_PATH}/ts_dataset_test.pth')
print('Done!')


print('Generating interpolation data')
train_ts = int(xv.shape[1]*train_frac)
test_ts = steps - train_ts
interp_start = int(train_ts/2)
xv_train = np.concatenate((xv[:train_dim,:interp_start,:], xv[:train_dim,interp_start+test_ts:,:]), axis=1)
xv_val = np.concatenate((xv[train_dim:train_dim+val_dim,:interp_start,:], xv[train_dim:train_dim+val_dim,interp_start+test_ts:,:]), axis=1)
xv_test = xv[:train_dim+val_dim,interp_start:interp_start+test_ts,:]

u_train = u_vec(xv_train.reshape((-1,2)))
u_val = u_vec(xv_val.reshape((-1,2)))
u_test = u_vec(xv_test.reshape((-1,2)))

t_train = torch.zeros((train_dim, train_ts, 3))
t_train[:,:,0] = torch.tile(torch.concatenate((torch.tensor(t_base[:interp_start]), torch.tensor(t_base[interp_start+test_ts:]))), (train_dim,1))
t_train[:,:,1:] = torch.tensor(xv0)[:train_dim].unsqueeze(1).repeat((1,train_ts,1))

point_dataset = TensorDataset(t_train.reshape((-1,3)), torch.from_numpy(xv_train.reshape((-1,2))), torch.from_numpy(u_train))
torch.save(point_dataset, f'{EXP_PATH}/true_interpolate_dataset_train.pth')

t_train_bc = torch.zeros((train_dim, 3))
t_train_bc[:,0] = t_base[interp_start+test_ts]*torch.ones_like(t_train_bc[:,0])
t_train_bc[:,1:] = torch.tensor(xv0)[:train_dim]

xv_train_bc = xv[:train_dim,interp_start+test_ts,:]
bc_dataset = TensorDataset(t_train_bc, xv_train_bc)
torch.save(bc_dataset, f'{EXP_PATH}/true_interpolate_bc_train.pth')

t_val = torch.zeros((val_dim, train_ts, 3))
t_val[:,:,0] = torch.tile(torch.concatenate((torch.tensor(t_base[:interp_start]), torch.tensor(t_base[interp_start+test_ts:]))), (val_dim,1))
t_val[:,:,1:] = torch.tensor(xv0)[train_dim:train_dim+val_dim].unsqueeze(1).repeat((1,train_ts,1))

point_dataset = TensorDataset(t_val.reshape((-1,3)), torch.from_numpy(xv_val.reshape((-1,2))), torch.from_numpy(u_val))
torch.save(point_dataset, f'{EXP_PATH}/true_interpolate_dataset_val.pth')

t_val_bc = torch.zeros((val_dim, 3))
t_val_bc[:,0] = t_base[interp_start+test_ts]*torch.ones_like(t_val_bc[:,0])
t_val_bc[:,1:] = torch.tensor(xv0)[train_dim:train_dim+val_dim]

xv_val_bc = xv[train_dim:train_dim+val_dim,interp_start+test_ts,:]
bc_dataset = TensorDataset(t_val_bc, xv_val_bc)

torch.save(bc_dataset, f'{EXP_PATH}/true_interpolate_bc_val.pth')

t_test = torch.zeros((train_dim+val_dim, test_ts, 3))
t_test[:,:,0] = torch.tile(torch.tensor(t_base[interp_start:interp_start+test_ts]), (train_dim+val_dim,1))
t_test[:,:,1:] = torch.tensor(xv0)[:train_dim+val_dim].unsqueeze(1).repeat((1,test_ts,1))
point_dataset = TensorDataset(t_test.reshape((-1,3)), xv_test.reshape((-1,2)), torch.from_numpy(u_test))
torch.save(point_dataset, f'{EXP_PATH}/true_interpolate_dataset_test.pth')

xv_new = xv[-test_dim:,:,:]
u_new = u_vec(xv_new.reshape((-1,2)))
t_new = torch.zeros((test_dim, steps, 3))
t_new[:,:,0] = torch.tile(torch.tensor(t_base), (test_dim,1))
t_new[:,:,1:] = torch.tensor(xv0)[-test_dim:].unsqueeze(1).repeat((1,steps,1))
point_dataset = TensorDataset(t_new.reshape((-1,3)), xv_new.reshape((-1,2)), torch.from_numpy(u_new).reshape((-1,2)))
torch.save(point_dataset, f'{EXP_PATH}/true_interpolate_dataset_new.pth')
print('Done!')