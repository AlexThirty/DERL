import torch
from torch import nn
import torch
from torch import nn
from collections import OrderedDict
from torch.func import vmap, jacrev, hessian
import numpy as np

def ball_boundary_uniform(n: int, radius: float, dim:int):
    angle = torch.distributions.Normal(0., 1.).sample((n, dim))
    #angle = torch.distributions.Uniform(0., 1.).sample((n, dim))*2 - 1
    norms = torch.norm(angle, p=2., dim=1).reshape((-1,1))
    angle = angle/norms    
    pts = radius*angle
    return pts

def allen_cahn_true_exp(x):
    return torch.sin(torch.pi*x[:,0])*torch.sin(torch.pi*x[:,1])*torch.exp(-x[:,2]*(x[:,0]+0.7))

# Forcing term of the Allen-Cahn equation with an exponential term
def allen_cahn_forcing_exp(x: torch.tensor):
    hes = allen_cahn_hes_exp(x)
    return x[:,2]*(hes[:,0,0] + hes[:,1,1]) + allen_cahn_true_exp(x)**3 - allen_cahn_true_exp(x)

# Partial derivative of the Allen-Cahn equation with an exponential term
def allen_cahn_pdv_exp(x: torch.tensor):
    ux = torch.pi*torch.cos(torch.pi*x[:,0])*torch.sin(torch.pi*x[:,1])*torch.exp(-x[:,2]*(x[:,0]+0.7)) \
        - x[:,2]*torch.exp(-x[:,2]*(x[:,0]+0.7))*torch.sin(torch.pi*x[:,0])*torch.sin(torch.pi*x[:,1])
    uy = torch.pi*torch.sin(torch.pi*x[:,0])*torch.cos(torch.pi*x[:,1])*torch.exp(-x[:,2]*(x[:,0]+0.7))
    return torch.column_stack((ux, uy))

# Hessian of the Allen-Cahn equation with an exponential term
def allen_cahn_hes_exp(x: torch.tensor):
    uxx = -(torch.pi**2 - x[:,2]**2)*allen_cahn_true_exp(x) \
        - 2*torch.pi*x[:,2]*torch.cos(torch.pi*x[:,0])*torch.sin(torch.pi*x[:,1])*torch.exp(-x[:,2]*(x[:,0]+0.7))
        
    uxy = torch.pi**2*torch.cos(torch.pi*x[:,0])*torch.cos(torch.pi*x[:,1])*torch.exp(-x[:,2]*(x[:,0]+0.7)) \
        - x[:,2]*torch.pi*torch.sin(torch.pi*x[:,0])*torch.cos(torch.pi*x[:,1])*torch.exp(-x[:,2]*(x[:,0]+0.7))
    
    uyy = -torch.pi**2*torch.sin(torch.pi*x[:,0])*torch.sin(torch.pi*x[:,1])*torch.exp(-x[:,2]*(x[:,0]+0.7))
    return torch.column_stack((uxx, uxy, uxy, uyy)).reshape((-1,2,2))

# Density network
class AllenNet(torch.nn.Module):
    def __init__(self,
                 bc_weight: float,
                 sys_weight: float,
                 pde_weight: float,
                 hidden_units: list,
                 lr_init: float,
                 device: str,
                 activation: nn.Module=nn.Tanh,
                 last_activation: bool=True,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # Save the parameters
        self.bc_weight = bc_weight
        self.pde_weight = pde_weight
        self.sys_weight = sys_weight
        self.hidden_units = hidden_units
        self.lr_init = lr_init
        self.device = device

        # Define the net, first layer
        net_dict = OrderedDict(
            {'lin0': nn.Linear(3, hidden_units[0]),
            'act0': activation}
        )

        # Define the net, hidden layers
        for i in range(1, len(hidden_units)):
            net_dict.update({f'lin{i}': nn.Linear(in_features=hidden_units[i-1], out_features=hidden_units[i])})
            net_dict.update({f'act{i}': activation})
        # Define the net, last layer
        net_dict.update({f'lin{len(hidden_units)}': nn.Linear(in_features=hidden_units[-1], out_features=1)})
        
        if last_activation:
            net_dict.update({f'act{len(hidden_units)}': activation})
        # Save the network
        self.net = nn.Sequential(net_dict).to(self.device)
        # Define the optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr_init)
        self.loss_container = nn.MSELoss(reduction='mean')
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # Forward function
        return self.net(x)
    
    def forward_single(self, x:torch.Tensor) -> torch.Tensor:
        # Forward function for individual samples
        return self.net(x.reshape((1,-1))).reshape((-1))
    
    def loss_fn(self,
        x:torch.Tensor,
        y:torch.Tensor,
        Dy:torch.Tensor,
        mode:str,
        x_bc:torch.Tensor=None,
        y_bc:torch.Tensor=None,
        Hy: torch.Tensor=None,
    ) -> torch.Tensor:
        # Check that the mode parameter is correct
        modes = ['PINN', 'PINN+Output', 'Derivative', 'Output', 'Sobolev', 'Hessian', 'Derivative+Hessian', 'Sobolev+Hessian']
        if mode not in modes:
            raise ValueError(f'mode should be in {modes}, but found {mode}')
        lam = x[:,2]
        x_in = x[:,:2]
                
        if mode == 'PINN':
            # Get the prediction
            y_pred = self.forward(x)
            # Get the partial derivatives from the network
            Dy_pred = vmap(jacrev(self.forward_single))(x)[:,0,:2]
            Hy_pred = vmap(hessian(self.forward_single))(x)[:,0,:2,:2]
            
            # lambda*(uxx + uyy) - u + u^3 = 0
            # Calculate the pde_residual
            pde_pred = lam*(Hy_pred[:,0,0] + Hy_pred[:,1,1]) - y_pred.reshape((-1)) + y_pred.reshape((-1))**3
            # Calculate the loss
            pde_loss = self.loss_container(pde_pred, allen_cahn_forcing_exp(x))
            spec_loss = self.pde_weight*pde_loss
            
        if mode == 'PINN+Output':
            # Get the prediction
            y_pred = self.forward(x)
            # Get the partial derivatives from the network
            Dy_pred = vmap(jacrev(self.forward_single))(x)[:,0,:2]
            Hy_pred = vmap(hessian(self.forward_single))(x)[:,0,:2,:2]
            
            # lambda*(uxx + uyy) - u + u^3 = 0
            # Calculate the pde_residual
            pde_pred = lam*(Hy_pred[:,0,0] + Hy_pred[:,1,1]) - y_pred.reshape((-1)) + y_pred.reshape((-1))**3
            # Calculate the loss
            pde_loss = self.loss_container(pde_pred, allen_cahn_forcing_exp(x))
            
            out_loss = self.loss_container(y_pred, y)
            spec_loss = self.pde_weight*pde_loss + self.sys_weight*out_loss
        
        elif mode == 'Derivative':
            # Get the partial derivatives from the network
            Dy_pred = vmap(jacrev(self.forward_single))(x)[:,0,:2]
            # In this case, we learn by supervision the partial derivatives
            der_loss = self.loss_container(Dy_pred, Dy)
            spec_loss = self.sys_weight*der_loss
        
        elif mode == 'Output':
            # Get the prediction
            y_pred = self.forward(x)
            # Out loss
            out_loss = self.loss_container(y_pred, y)
            spec_loss = self.sys_weight*out_loss
        
        elif mode == 'Sobolev':
            # Get the prediction
            y_pred = self.forward(x)
            # Get the partial derivatives from the network
            Dy_pred = vmap(jacrev(self.forward_single))(x)[:,0,:2]            
            # In this case, we learn by supervision the partial derivatives
            der_loss = self.loss_container(Dy_pred, Dy)
            # Out loss
            out_loss = self.loss_container(y_pred, y)
            # Sobolev loss
            sob_loss = der_loss+out_loss
            
            spec_loss = self.sys_weight*sob_loss
        
    
        elif mode == 'Hessian':
            # Get the prediction
            # Get the partial derivatives from the network
            Hy_pred = vmap(hessian(self.forward_single))(x)[:,0,:2,:2]
            # Hessian loss
            hes_loss = self.loss_container(Hy_pred, Hy)
            spec_loss = self.sys_weight*hes_loss
        
        elif mode == 'Derivative+Hessian':
            # Get the partial derivatives from the network
            Dy_pred = vmap(jacrev(self.forward_single))(x)[:,0,:2]
            Hy_pred = vmap(hessian(self.forward_single))(x)[:,0,:2,:2]
            # In this case, we learn by supervision the partial derivatives
            der_loss = self.loss_container(Dy_pred, Dy)
            # Hessian loss
            hes_loss = self.loss_container(Hy_pred, Hy)
            spec_loss = self.sys_weight*der_loss + self.sys_weight*hes_loss
        
        elif mode == 'Sobolev+Hessian':
            # Get the prediction
            y_pred = self.forward(x)
            # Get the partial derivatives from the network
            Dy_pred = vmap(jacrev(self.forward_single))(x)[:,0,:]
            Hy_pred = vmap(hessian(self.forward_single))(x)[:,0,:,:]
            
            out_loss = self.loss_container(y_pred, y)
            der_loss = self.loss_container(Dy_pred, Dy)
            sob_loss = der_loss+out_loss    
            spec_loss = self.sys_weight*sob_loss + self.sys_weight*self.hessian_sobolev_error(x, Hy)
        
        if x_bc is not None:
            y_bc_pred = self.forward(x_bc)
            bc_loss = self.loss_container(y_bc_pred.reshape((-1)), y_bc.reshape((-1)))
        else:
            bc_loss = torch.tensor([0.])
    
        # Total loss
        tot_loss = spec_loss + self.bc_weight*bc_loss
        return tot_loss
    
    def hessian_sobolev_error(self, x_pde:torch.Tensor, H_pde:torch.Tensor):
        rand_vec = ball_boundary_uniform(n=100, radius=1., dim=2).to(self.device)
        sigma = 0.01
        noise = rand_vec*sigma
        def rand_proj(x):
            der_pred = jacrev(self.forward_single)(x)
            der_eps_pred = vmap(jacrev(self.forward_single))(x+noise)
            hess_pred = (der_eps_pred[:,0,:]-der_pred)/sigma
            return hess_pred
        hess_proj_pred = vmap(rand_proj)(x_pde) 
        hess_proj_true = torch.einsum('bij,pj->bpi', H_pde, rand_vec)
        error = torch.norm(hess_proj_pred - hess_proj_true, p=2, dim=2).mean()
        return error
    
    def eval_losses(self, step:int,
        x:torch.Tensor,
        y:torch.Tensor,
        Dy:torch.Tensor,
        mode:str,
        x_bc:torch.Tensor=None,
        y_bc:torch.Tensor=None,
        Hy:torch.Tensor=None,
        print_to_screen:bool=False,    
    ):
        # Check that the mode parameter is correct
        modes = ['PINN', 'PINN+Output', 'Derivative', 'Output', 'Sobolev', 'Hessian', 'Derivative+Hessian', 'Sobolev+Hessian']
        if mode not in modes:
            raise ValueError(f'mode should be in {modes}, but found {mode}')
        lam = x[:,2]
        x_in = x[:,:2]
        # Get the prediction
        y_pred = self.forward(x)
        # Get the partial derivatives from the network
        Dy_pred = vmap(jacrev(self.forward_single))(x)[:,0,:2]
        Hy_pred = vmap(hessian(self.forward_single))(x)[:,0,:2,:2]
            
        # lambda*(uxx + uyy) - u + u^3 = 0
        # Calculate the pde_residual
        pde_pred = lam*(Hy_pred[:,0,0] + Hy_pred[:,1,1]) - y_pred.reshape((-1)) + y_pred.reshape((-1))**3
        # Calculate the loss
        pde_loss = self.loss_container(pde_pred, allen_cahn_forcing_exp(x))
        
        # In this case, we learn by supervision the partial derivatives
        der_loss = self.loss_container(Dy_pred, Dy)
        
        out_loss = self.loss_container(y_pred, y)
        
        hes_loss = self.loss_container(Hy_pred, Hy)
        
        
        # Sobolev loss
        sob_loss = der_loss+out_loss
        
        if mode == 'PINN':
            spec_loss = self.pde_weight*pde_loss
        elif mode == 'PINN+Output':
            spec_loss = self.pde_weight*pde_loss + self.sys_weight*out_loss
        elif mode == 'Derivative':
            spec_loss = self.sys_weight*der_loss
        elif mode == 'Output':
            spec_loss = self.sys_weight*out_loss
        elif mode == 'Sobolev':
            spec_loss = self.sys_weight*sob_loss
        elif mode == 'Hessian':
            spec_loss = self.sys_weight*hes_loss
        elif mode == 'Derivative+Hessian':
            spec_loss = self.sys_weight*der_loss + self.sys_weight*hes_loss
        elif mode == 'Sobolev+Hessian':
            spec_loss = self.sys_weight*sob_loss + self.sys_weight*self.hessian_sobolev_error(x, Hy)
        else:
            spec_loss = self.sys_weight*sob_loss
        if x_bc is not None:
            y_bc_pred = self.forward(x_bc)
            bc_loss = self.loss_container(y_bc_pred.reshape((-1)), y_bc.reshape((-1)))
            #x_init = torch.cat((x_init, x_bc), dim=0)
            #y_init = torch.cat((y_init, y_bc), dim=0)
        else:
            bc_loss = torch.tensor([0.])
        
        # Total loss
        tot_loss = spec_loss + self.bc_weight*bc_loss
        
        
        if print_to_screen:
            print(f'Step: {step}, total loss: {tot_loss}')
            print(f'pde loss: {pde_loss}, rho loss {out_loss}, Drho loss: {der_loss}, bc loss: {bc_loss}, hes loss: {hes_loss}')
        
        
        return step, out_loss, der_loss, hes_loss, pde_loss, bc_loss, tot_loss
    
    def evaluate_forcing(self, x):
        lam = x[:,2]
        x_in = x[:,:2]
        # Get the prediction
        y_pred = self.forward(x)
        # Get the partial derivatives from the network
        Dy_pred = vmap(jacrev(self.forward_single))(x)[:,0,:2]
        Hy_pred = vmap(hessian(self.forward_single))(x)[:,0,:2,:2]
            
        # lambda*(uxx + uyy) - u + u^3 = 0
        # Calculate the pde_residual
        pde_pred = lam*(Hy_pred[:,0,0] + Hy_pred[:,1,1]) - y_pred.reshape((-1)) + y_pred.reshape((-1))**3
                
        return pde_pred
    
    def evaluate_consistency(self, x):
        lam = x[:,2]
        x_in = x[:,:2]
        # Get the prediction
        y_pred = self.forward(x)
        # Get the partial derivatives from the network
        Dy_pred = vmap(jacrev(self.forward_single))(x)[:,0,:2]
        Hy_pred = vmap(hessian(self.forward_single))(x)[:,0,:2,:2]
            
        # lambda*(uxx + uyy) - u + u^3 = 0
        # Calculate the pde_residual
        pde_pred = lam*(Hy_pred[:,0,0] + Hy_pred[:,1,1]) - y_pred.reshape((-1)) + y_pred.reshape((-1))**3
                
        return torch.abs(pde_pred - allen_cahn_forcing_exp(x))
            
    
    
    
