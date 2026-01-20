import torch.utils
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from torch.func import vmap, jacrev, jacfwd, hessian
import numpy as np
from .params import nu, lam, x_min, x_max, y_min, y_max, nx, ny, N

def generate_random_points(N:int, x_min:float, x_max:float, y_min:float, y_max:float):
    random_points = torch.rand((N, 2)) * torch.tensor([x_max-x_min, y_max-y_min]) + torch.tensor([x_min, y_min])
    return random_points

def generate_boundary_points(N:int, x_min:float, x_max:float, y_min:float, y_max:float):
    boundary_points = np.vstack([
        np.column_stack([np.linspace(x_min, x_max, N), np.full(N, y_min)]),  # Bottom boundary
        np.column_stack([np.linspace(x_min, x_max, N), np.full(N, y_max)]),   # Top boundary
        np.column_stack([np.full(N, x_min), np.linspace(y_min, y_max, N)]),  # Left boundary
        np.column_stack([np.full(N, x_max), np.linspace(y_min, y_max, N)])   # Right boundary
    ])
    return boundary_points
    


class PINN(torch.nn.Module):
    def __init__(self,
                 hidden_units:list,
                 mom_weight:float=1.,
                 div_weight:float=1.,
                 sys_weight:float=1.,
                 bc_weight:float=1.,
                 lr:float=1e-3,
                 activation:nn.Module=nn.Tanh(),
                 device: str='cuda:0',
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        
        self.in_dim = 2
        self.bc_weight = bc_weight
        self.sys_weight = sys_weight
        self.div_weight = div_weight
        self.mom_weight = mom_weight
        self.hidden_units = hidden_units
        
        self.loss_container = torch.nn.MSELoss(reduction='mean') 
        self.discrepancy_loss_container = torch.nn.MSELoss(reduction='mean') 
        
        # Divergence free network
        out_dim = 3
        self.out_dim = out_dim
        self.hidden_units = [self.in_dim] + hidden_units
    
        net = nn.Sequential()
        for i in range(len(self.hidden_units)-1):
            net.add_module(f'div_lin{i}', nn.Linear(self.hidden_units[i], self.hidden_units[i+1]))
            net.add_module(f'div_act{i}', activation)
        net.add_module(f'div_lin{len(self.hidden_units)-1}', nn.Linear(self.hidden_units[-1], self.out_dim))
        
        self.net = net.to(self.device)

        # Save the optimizer
        self.lr = lr
        
        self.device = device
        
        # Calculate the total number of parameters
        self.total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"Total number of parameters: {self.total_params}")
    
    def forward_single(self, tx: torch.Tensor, return_final:bool=False):
        return self.net(tx.reshape(-1, self.in_dim)).reshape(-1)
    
    def forward(self, tx, return_final:bool=False):
        return self.net(tx)
    
    def forward_single_final(self, tx: torch.Tensor, return_final:bool=False):
        return self.forward_single(tx, return_final=True)
    
    def loss_fn(self, 
                x_pde:torch.Tensor, y_pde:torch.Tensor, Dy_pde:torch.Tensor,
                x_bc:torch.Tensor, y_bc: torch.Tensor,
                mode:str='Derivative'
        ) -> torch.Tensor:
        
        # Get the prediction
        y_pred = self.forward(x_pde)
        
        # Get the derivatives
        Dy_pred = vmap(jacrev(self.forward_single))(x_pde)
        
        # Get the hessians
        Hy_pred = vmap(hessian(self.forward_single))(x_pde)
        
        # Now we impose the pinn loss on the momentum branch
        # Calculate the pde_residual
        if mode == 'Derivative':
            spec_loss = self.sys_weight*self.loss_container(Dy_pred, Dy_pde)
        elif mode == 'Output':
            spec_loss = self.sys_weight*self.loss_container(y_pred, y_pde)
        elif mode == 'Sobolev':
            spec_loss = self.sys_weight*self.loss_container(y_pred, y_pde) + self.sys_weight*self.loss_container(Dy_pred, Dy_pde)
        elif mode == 'PINN+Output':
            spec_loss = self.sys_weight*self.loss_container(y_pred, y_pde)
            lapl_u = torch.diagonal(Hy_pred[:,:2,:,:], dim1=2, dim2=3).sum(dim=2)
            mom_pde = - nu*lapl_u + torch.einsum('bij,bj->bi', Dy_pred[:,:2,:], y_pred[:,:2]) + Dy_pred[:,-1,:]
            
            div_pde = torch.einsum('bii->b', Dy_pred[:,:2,:])
            
            mom_loss = self.loss_container(mom_pde, torch.zeros_like(mom_pde))
            div_loss = self.loss_container(div_pde, torch.zeros_like(div_pde))
            
            spec_loss += self.mom_weight*mom_loss + self.div_weight*div_loss
            
        y_bc_pred = self.forward(x_bc)
        bc_loss = self.loss_container(y_bc_pred, y_bc)
    
        return bc_loss*self.bc_weight + spec_loss
    
    def eval_losses(self, 
                    x_pde:torch.Tensor, y_pde:torch.Tensor, Dy_pde:torch.Tensor,
                    x_bc:torch.Tensor, y_bc:torch.Tensor,
                    step:int) -> torch.Tensor:
        
            # Get the prediction
        y_pred = self.forward(x_pde)
        
        # Get the derivatives
        Dy_pred = vmap(jacrev(self.forward_single))(x_pde)
        
        # Get the hessians
        Hy_pred = vmap(hessian(self.forward_single))(x_pde)
        
        # Now we impose the pinn loss on the momentum branch
        # Calculate the pde_residual
        lapl_u = torch.diagonal(Hy_pred[:,:2,:,:], dim1=2, dim2=3).sum(dim=2)
        mom_pde = - nu*lapl_u + torch.einsum('bij,bj->bi', Dy_pred[:,:2,:], y_pred[:,:2]) + Dy_pred[:,-1,:]
        
        div_pde = torch.einsum('bii->b', Dy_pred[:,:2,:])
        
        mom_loss = self.loss_container(mom_pde, torch.zeros_like(mom_pde))
        div_loss = self.loss_container(div_pde, torch.zeros_like(div_pde))
        
        Dy_loss = self.loss_container(Dy_pred, Dy_pde)
             
        y_bc_pred = self.forward(x_bc)
        bc_loss = self.loss_container(y_bc_pred, y_bc)
    
        y_loss = self.loss_container(y_pred, y_pde)
        
        tot_loss_val = bc_loss + mom_loss + div_loss
        #print(f'Step: {step}, F_loss: {F_loss_val}, div_loss: {div_loss_val}, bc_loss: {bc_loss_val}, init_loss: {init_loss_val}')
        return step, mom_loss, div_loss, y_loss, bc_loss, Dy_loss, tot_loss_val
    
    def evaluate_consistency(self, x_pde:torch.Tensor):
        Dy_pred = vmap(jacrev(self.forward_single_final))(x_pde)
        y_pred = self.forward(x_pde, return_final=True)
        # Get the hessians
        Hy_pred = vmap(hessian(self.forward_single_final))(x_pde)
        
        # Now we impose the pinn loss on the momentum branch
        # Calculate the pde_residual
        lapl_u = torch.diagonal(Hy_pred[:,:2,:,:], dim1=2, dim2=3).sum(dim=2)
        mom_pde = - nu*lapl_u + torch.einsum('bij,bj->bi', Dy_pred[:,:2,:], y_pred[:,:2]) + Dy_pred[:,-1,:]
        
        div_pde = torch.einsum('bii->b', Dy_pred[:,:2,:])
        
        
        
        return torch.abs(mom_pde), torch.abs(div_pde)