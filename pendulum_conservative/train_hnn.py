# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch, argparse
import numpy as np

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from external_models.hnn_nn_models import MLP
from external_models.hnn import HNN
from external_models.hnn_utils import L2_loss, rk4, integrate_model
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
import random
seed=30
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)
batch_size = 512
from model import u_vec
def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=2, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=20, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=200, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=1, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='pend', type=str, help='only one option right now')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--use_rk4', dest='use_rk4', action='store_true', help='integrate derivative with RK4')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device to use')
    parser.set_defaults(feature=True)
    return parser.parse_args()


b = 0.
EXP_PATH = '.'
save_dir = f'{EXP_PATH}/true/plotshnn'
name = 'true'
title = 'hnn'
t_max = 10.
dt = 0.01

def train(args):
  # set random seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # init model and optimizer
  if args.verbose:
    print("Training baseline model:" if args.baseline else "Training HNN model:")

  output_dim = args.input_dim if args.baseline else 2
  nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
  model = HNN(args.input_dim, differentiable_model=nn_model,
              field_type=args.field_type, baseline=args.baseline, device=args.device)
  optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)

  train_dataset = torch.load(os.path.join('data', f'true_hnn_dataset_train.pth'))
  val_dataset = torch.load(os.path.join('data', f'true_hnn_dataset_val.pth'))
  test_dataset = torch.load(os.path.join('data', f'true_hnn_dataset_test.pth'))
  
  train_dataloader = DataLoader(train_dataset, batch_size, generator=gen, shuffle=True, num_workers=2)
  test_dataloader = DataLoader(test_dataset, batch_size, generator=gen, shuffle=True, num_workers=2)
  # arrange data
  

  # vanilla train loop
  stats = {'train_loss': [], 'test_loss': []}
  for epoch in range(args.total_steps+1):
    train_losses = []
    for step, train_data in enumerate(train_dataloader):
      x = torch.column_stack((train_data[1], train_data[2])).float().requires_grad_(True).to(args.device)
      dxdt = torch.column_stack((train_data[3], train_data[4])).float().to(args.device)
      # train step
      dxdt_hat = model.rk4_time_derivative(x, dt=dt) if args.use_rk4 else model.time_derivative(x)
      dxdt_hat[:,1] = dxdt_hat[:,1] - b*x[:,1]
      loss = L2_loss(dxdt, dxdt_hat)
      loss.backward() ; optim.step() ; optim.zero_grad()
      #print(loss)
      train_losses.append(loss.item())
      
    test_losses = []
    for test_data in test_dataloader:
      x = torch.column_stack((test_data[1], test_data[2])).float().requires_grad_(True).to(args.device)
      dxdt = torch.column_stack((test_data[3], test_data[4])).float().to(args.device)
      # run test data
      test_dxdt_hat = model.rk4_time_derivative(x, dt=dt) if args.use_rk4 else model.time_derivative(x)
      #if b != 0:
      #  # mx'' + bx' + sin(x) = 0
      test_dxdt_hat[:,1] = test_dxdt_hat[:,1] - b*x[:,1]
      test_loss = L2_loss(dxdt, test_dxdt_hat)
      test_losses.append(test_loss.item())

      # logging
    stats['train_loss'].append(np.mean(train_losses))
    stats['test_loss'].append(np.mean(test_losses))
    if args.verbose and epoch % args.print_every == 0:
      print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(epoch, loss.item(), test_loss.item()))



  x = torch.column_stack((train_dataset[:][1], train_dataset[:][2])).float().requires_grad_(True).to(args.device)
  test_x = torch.column_stack((test_dataset[:][1], test_dataset[:][2])).float().requires_grad_(True).to(args.device)
  dxdt = torch.column_stack((train_dataset[:][3], train_dataset[:][4])).float().to(args.device)
  test_dxdt = torch.column_stack((test_dataset[:][3], test_dataset[:][4])).float().to(args.device)
  
  train_dxdt_hat = model.time_derivative(x)
  if b != 0:
    train_dxdt_hat[:,1] = train_dxdt_hat[:,1] - b*x[:,1]
  train_dist = (dxdt - train_dxdt_hat)**2
  test_dxdt_hat = model.time_derivative(test_x)
  if b != 0:
    test_dxdt_hat[:,1] = test_dxdt_hat[:,1] - b*test_x[:,1]
  test_dist = (test_dxdt - test_dxdt_hat)**2
  print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
    .format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]),
            test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))

  return model, stats

if __name__ == "__main__":
    args = get_args()
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    model, stats = train(args)
    device = args.device
    
    
    # --------------- Plotting ----------------
    new_data = torch.load(os.path.join('data', f'true_dataset_test.pth'))
    new_data_xv0 = new_data[:][0].reshape((10,-1,3))[:,0,1:]
    xv0 = new_data_xv0.numpy()
    x0 = xv0[:,0]
    v0 = xv0[:,1]


    new_data_xv0 = new_data[:][0].reshape((10,-1,3))[:,0,1:]
    points = 10
    t_max = 10
    steps = int(t_max/dt)+1
    xv = torch.zeros((points,steps,2))
    u_save = torch.zeros((points,steps,2))
    xv[:,0,0] = torch.from_numpy(x0)
    xv[:,0,1] = torch.from_numpy(v0)
    for i in range(1,steps):
    # v[i] = v[i-1] + dt*u(x[i-1])
        xv[:,i,0] = xv[:,i-1,0] + dt*u_vec(xv[:,i-1,:])[:,0]
        xv[:,i,1] = xv[:,i-1,1] + dt*u_vec(xv[:,i-1,:])[:,1]
    t_base = np.arange(start=0, stop=t_max+dt, step=dt)
    
    # Number of points for the field
    N = 500
    xlim = np.pi/2
    ylim = 2.
    X,Y = np.meshgrid(np.linspace(-xlim,xlim,N),np.linspace(-ylim,ylim,N))
    pts = np.vstack([X.reshape(-1),Y.reshape(-1)]).T

    #plots the streamplot for the velocity field
    plt.figure(figsize=(5,5))
    #print(pts)
    vel = u_vec(torch.from_numpy(pts))
    #print(vel)
    U = np.array(vel[:,0].reshape(X.shape))
    V = np.array(vel[:,1].reshape(Y.shape))
    #mask the outside of the ball

    plt.streamplot(X,Y,U,V,density=1,color=U**2 + V**2, linewidth=0.15)

    plt.xlim((-xlim,xlim))
    plt.ylim((-ylim,ylim))
    #add outline for aesthetics
    t_base = np.arange(start=0, stop=t_max+dt, step=dt)

    #xv_pred = model.evaluate_trajectory(x0=xv[:,0,:].float(), time_steps=steps).detach().cpu().numpy()
    xv_pred = torch.zeros((xv.shape))
    xv_iter = torch.zeros((xv.shape))
    for i in range(xv.shape[0]):
        in_pts = torch.column_stack((torch.from_numpy(t_base).reshape((-1,1)), torch.tile(torch.from_numpy(xv0[i]), (steps,1)))).float()
        xv_pred[i,:,:] = torch.from_numpy(integrate_model(model, (0,t_max), xv0[i], b=b, fun=None, t_eval=t_base).y.T)
    #xv_iter = model.evaluate_trajectory(x0=xv[:,0,:], time_steps=steps).detach().cpu()

    xv = xv.numpy()
    for i in range(10):
        plt.plot(xv[i,:,0], xv[i,:,1], color='blue')
        plt.plot(xv_pred[i,:,0], xv_pred[i,:,1], color='red')
        #plt.plot(xv_iter[i,:,0], xv_iter[i,:,1], color='green')
        #plt.legend()
    blue_patch = patches.Patch(color='blue', label='True trajectories')
    red_patch = patches.Patch(color='red', label='Predicted trajectories')
    #green_patch = patches.Patch(color='green', label='Iterative trajectories')
    plt.legend(handles=[blue_patch,red_patch])
    plt.xlabel(r'Angle: $\theta$')
    plt.ylabel(r'Angular speed: $\omega$')

    plt.title(f'HNN learning phase trajectories')
    plt.savefig(f'{EXP_PATH}/{name}/plots{title}/pendulum_phase_trajectory.png', dpi=300)
        
    plt.close()

    plt.figure(figsize=(8,5))
    t_base = np.arange(start=0, stop=t_max+dt, step=dt)
    for i in range(10):
        plt.plot(t_base, xv[i,:,0], color='blue')
        plt.plot(t_base, xv_pred[i,:,0], color='red')
    blue_patch = patches.Patch(color='blue', label='True trajectories')
    red_patch = patches.Patch(color='red', label='Predicted trajectories')
    plt.legend(handles=[blue_patch,red_patch])
    plt.xlabel(r'Time: $t$')
    plt.ylabel(r'Angle: $\theta$')
    plt.title(f'HNN learning time trajectories')
    plt.savefig(f'{EXP_PATH}/{name}/plots{title}/pendulum_trajectory.png', dpi=300)
    plt.close()

    #plots the streamplot for the velocity field
    plt.figure(figsize=(5,5))
    #print(pts)
    x = torch.from_numpy(pts).to(device).float().requires_grad_(True)  
    vel = model.rk4_time_derivative(x, dt=dt).detach().cpu().numpy() if args.use_rk4 else model.time_derivative(x).detach().cpu().numpy()
    vel[:,1] = vel[:,1] - b*x[:,1].detach().cpu().numpy()
    #print(vel)
    U = np.array(vel[:,0].reshape(X.shape))
    V = np.array(vel[:,1].reshape(Y.shape))
    #mask the outside of the ball



    plt.streamplot(X,Y,U,V,density=1,color=U**2 + V**2, linewidth=0.15)
    for i in range(10):
        plt.plot(xv_pred[i,:,0], xv_pred[i,:,1], label=f'trajectory{i}', color='red')
    plt.xlim((-xlim,xlim))
    plt.ylim((-ylim,ylim))

    plt.xlabel(r'Angle: $\theta$')
    plt.ylabel(r'Angular speed: $\omega$')

    plt.title(f'HNN learning predicted field')
    plt.savefig(f'{EXP_PATH}/{name}/plots{title}/predicted_field.png')
    plt.close()
    
    plt.figure(figsize=(6,5))
    vel_true = u_vec(torch.from_numpy(pts))
    #print(vel)
    U_true = np.array(vel_true[:,0].reshape(X.shape))
    V_true = np.array(vel_true[:,1].reshape(Y.shape))
    plt.contourf(X,Y,np.sqrt((U-U_true)**2+(V-V_true)**2),100,cmap='jet')
    plt.title('Error in predicted fields')
    plt.colorbar()
    plt.xlim((-xlim,xlim))
    plt.ylim((-ylim,ylim))
    plt.xlabel(r'Angle: $\theta$')
    plt.ylabel(r'Angular speed: $\omega$')

    plt.title(f'HNN learning field error')
    plt.savefig(f'{EXP_PATH}/{name}/plots{title}/error_field.png')
    plt.close()
    # save
    
    # --------------- Save ----------------
    
    
    label = '-baseline' if args.baseline else '-hnn'
    label = '-rk4' + label if args.use_rk4 else label
    path = '{}/{}{}.tar'.format(save_dir, args.name, label)
    torch.save(model.state_dict(), path)