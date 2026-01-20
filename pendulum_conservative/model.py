from torch import nn
import torch


import torch
from torch import nn
from collections import OrderedDict
from torch.func import vmap, jacrev, hessian

m = 1. # old was 0.1
l = 10. # old was 1.
g = 9.81
b = 0

def u_vec(xv: torch.Tensor, b=b):
    return torch.column_stack((xv[:,1],-(g/l)*torch.sin(xv[:,0])-(b/m)*xv[:,1]))


def ball_boundary_uniform(n: int, radius: float, dim:int):
    angle = torch.distributions.Normal(0., 1.).sample((n, dim))
    #angle = torch.distributions.Uniform(0., 1.).sample((n, dim))*2 - 1
    norms = torch.norm(angle, p=2., dim=1).reshape((-1,1))
    angle = angle/norms    
    pts = radius*angle
    return pts

class PendulumNet(torch.nn.Module):
    def __init__(self,
                 init_weight: float,
                 sys_weight: float,
                 pde_weight: float,
                 hidden_units: list,
                 lr_init: float,
                 device: str,
                 activation: nn.Module=nn.Tanh,
                 last_activation: bool=False,
                 b = 0,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # Save state variables
        self.sys_weight = sys_weight
        self.init_weight = init_weight
        self.hidden_units = hidden_units
        self.pde_weight = pde_weight
        self.lr_init = lr_init
        self.device = device
        self.b = b
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
        #self.net.double()
        #self.double()
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
    ) -> torch.Tensor:
        # Check that the mode parameter is correct
        modes = ['PINN', 'PINN+Output', 'Derivative', 'Output', 'Sobolev']
        if mode not in modes:
            raise ValueError(f'mode should be in {modes}, but found {mode}')
        
        
        if mode == 'PINN':
                # Get the prediction
            y_pred = self.forward(x)
            # Get the partial derivatives from the network
            Dy_pred = vmap(jacrev(self.forward_single))(x)
            D2y_pred = vmap(hessian(self.forward_single))(x)
            
            
            # In this case, we learn through the pde in PINN style
            # dx/dt = v
            # dv/dt = -sin 
            # d2x/dt2 = -sin
            # Calculate the pde_residual
            pde_pred = D2y_pred[:,0,0,0] + g/l*torch.sin(y_pred.reshape((-1))) + b/m*Dy_pred[:,0,0]
            # Calculate the loss
            pde_loss = self.loss_container(pde_pred, torch.zeros_like(pde_pred))
            
            spec_loss = self.pde_weight*pde_loss
        elif mode == 'PINN+Output':
            # Get the prediction
            y_pred = self.forward(x)
            D2y_pred = vmap(hessian(self.forward_single))(x)
            Dy_pred = vmap(jacrev(self.forward_single))(x)
            # In this case, we learn through the pde in PINN style
            # dx/dt = v
            # dv/dt = -sin resto 
            # Calculate the pde_residual
            pde_pred = D2y_pred[:,0,0,0] + g/l*torch.sin(y_pred.reshape((-1))) + b/m*Dy_pred[:,0,0]
            # Calculate the loss
            pde_loss = self.loss_container(pde_pred, torch.zeros_like(pde_pred))
            
            # In this case we learn the output
            out_loss = self.loss_container(y_pred, y)
            
            spec_loss = self.pde_weight*pde_loss + self.sys_weight*out_loss
        
        elif mode == 'Derivative':
            # Get the partial derivatives from the network
            Dy_pred = vmap(jacrev(self.forward_single))(x)
            # In this case, we learn by supervision the partial derivatives
            der_loss = self.loss_container(Dy_pred[:,0,0], Dy)
            
            
            spec_loss = self.sys_weight*der_loss
        
        elif mode == 'Output':            
            # Get the prediction
            y_pred = self.forward(x)
            # In this case we learn the output
            out_loss = self.loss_container(y_pred, y)            
            spec_loss = self.sys_weight*out_loss
        else:
            # Sobolev mode
            # Get the prediction
            y_pred = self.forward(x)
            # Get the partial derivatives from the network
            Dy_pred = vmap(jacrev(self.forward_single))(x)
            der_loss = self.loss_container(Dy_pred[:,0,0], Dy)
            
            # In this case we learn the output
            out_loss = self.loss_container(y_pred, y)
            
            # Sobolev loss
            sob_loss = der_loss + out_loss
            
            spec_loss = self.sys_weight*sob_loss
            
        
        #init = torch.unique(x[:,1:], dim=0)
        init = x[:,1:]
        x_init = torch.column_stack((torch.zeros((init.shape[0],1)).to(self.device), init))
        y_init = x_init[:,1:]
        if x_bc is not None:
            x_init = torch.cat((x_init, x_bc))
            y_init = torch.cat((y_init, y_bc))
        # Calculate the init prediction
        y_init_pred = self.forward(x_init)
        Dy_init_pred = vmap(jacrev(self.forward_single))(x_init)[:,:,0]
        # Initial loss
        init_pred = torch.column_stack((y_init_pred, Dy_init_pred))
        init_loss = self.loss_container(init_pred, y_init)
        # Total loss
        tot_loss = spec_loss + self.init_weight*init_loss
        return tot_loss
    
    def eval_losses(self, step:int,
        x:torch.Tensor,
        y:torch.Tensor,
        Dy:torch.Tensor,
        mode:str,    
        x_bc:torch.Tensor=None,
        y_bc:torch.Tensor=None,
        print_to_screen:bool=False,
    ):
        # Check that the mode parameter is correct
        modes = ['PINN', 'PINN+Output', 'Derivative', 'Output', 'Sobolev']
        if mode not in modes:
            raise ValueError(f'mode should be in {modes}, but found {mode}')
        
        # Get the prediction
        y_pred = self.forward(x)
        # Get the partial derivatives from the network
        Dy_pred = vmap(jacrev(self.forward_single))(x)
        D2y_pred = vmap(hessian(self.forward_single))(x)
        D3y_pred = vmap(jacrev(jacrev(jacrev(self.forward_single))))(x)
        
        pde_pred = D2y_pred[:,0,0,0] + g/l*torch.sin(y_pred.reshape((-1))) + b/m*Dy_pred[:,0,0]
        
        init_pde_pred = D3y_pred[:,0,0,0,1]  + g/l*torch.cos(y_pred.reshape((-1)))*Dy_pred[:,0,1] + b/m*D2y_pred[:,0,0,1]
        init_pde_pred2 = D3y_pred[:,0,0,0,2]  + g/l*torch.cos(y_pred.reshape((-1)))*Dy_pred[:,0,2] + b/m*D2y_pred[:,0,0,2]
        
        init_pde_pred = torch.column_stack((init_pde_pred, init_pde_pred2))
        init_pde_loss = self.loss_container(init_pde_pred, torch.zeros_like(init_pde_pred))

        pde_loss = self.loss_container(pde_pred, torch.zeros_like(pde_pred))
            
        # In this case, we learn by supervision the partial derivatives
        der_loss = self.loss_container(Dy_pred[:,0,0], Dy)
        
        # In this case we learn the output
        out_loss = self.loss_container(y_pred, y)
        #init = torch.unique(x[:,1:], dim=0)
        #
        #init = torch.unique(x[:,1:], dim=0)
        init = x[:,1:]
        x_init = torch.column_stack((torch.zeros((init.shape[0],1)).to(self.device), init))
        y_init = x_init[:,1:]
        if x_bc is not None:
            x_init = torch.cat((x_init, x_bc))
            y_init = torch.cat((y_init, y_bc))
        # Calculate the init prediction
        y_init_pred = self.forward(x_init)
        Dy_init_pred = vmap(jacrev(self.forward_single))(x_init)[:,:,0]
        # Initial loss
        init_pred = torch.column_stack((y_init_pred, Dy_init_pred))
        init_loss = self.loss_container(init_pred, y_init)
        # Total loss
        
        # Sobolev loss
        #rand_vecs = ball_boundary_uniform(1000, 1., 1).to(self.device)
        #scalar_prods_pred = torch.einsum('bij,pj->bip', u_pred[:,:,:1], rand_vecs)
        #print(scalar_prods_pred.shape)
        #scalar_prods_true = torch.einsum('bij,pj->bip', u.reshape((-1,2,1)), rand_vecs)
        #print(scalar_prods_true.shape)
        #der_loss_est = torch.sqrt(((scalar_prods_true - scalar_prods_pred)**2).sum(dim=2)).mean()
        sob_loss = der_loss + out_loss
        
        if mode == 'PINN':
            spec_loss = self.pde_weight*pde_loss
        elif mode == 'PINN+Output':
            spec_loss = self.pde_weight*pde_loss + self.sys_weight*out_loss
        elif mode == 'Derivative':
            spec_loss = self.sys_weight*der_loss
        elif mode == 'Output':
            spec_loss = self.sys_weight*out_loss
        else:
            spec_loss = self.sys_weight*sob_loss
        
        tot_loss = spec_loss + self.init_weight*init_loss
        if print_to_screen:
            print(f'Step: {step}, total loss: {tot_loss}, init loss: {init_loss}')
            print(f'pde loss: {pde_loss}, out loss {out_loss}, der loss: {der_loss}, init pde loss: {init_pde_loss}. Sobolev loss: {sob_loss}')
            
        return step, out_loss, der_loss, pde_loss, init_loss, tot_loss, init_pde_loss
    
    def evaluate_trajectory(self, x0:torch.Tensor, time_steps, dt=0.01):
        out = torch.zeros((x0.shape[0], time_steps, 2)).to(self.device)
        for i in range(time_steps):
            x_in = torch.column_stack((dt*i*torch.ones(x0.shape[0],1).to(self.device), x0.to(self.device)))
            out_i = self.forward(x_in)
            out[:,i,:] = out_i
            
        return out
    
    def evaluate_consistency(self, x:torch.Tensor):
        #  Get the prediction
        y_pred = self.forward(x)
        # Get the partial derivatives from the network
        Dy_pred = vmap(jacrev(self.forward_single))(x)
        D2y_pred = vmap(hessian(self.forward_single))(x)
        
        pde_pred = D2y_pred[:,0,0,0] + g/l*torch.sin(y_pred.reshape((-1))) + b/m*Dy_pred[:,0,0]
        return torch.abs(pde_pred)
    
    
    def evaluate_init_consistency(self, x:torch.Tensor):
        # Get the prediction
        y_pred = self.forward(x)
        # Get the partial derivatives from the network
        Dy_pred = vmap(jacrev(self.forward_single))(x)
        D2y_pred = vmap(hessian(self.forward_single))(x)
        D3y_pred = vmap(jacrev(jacrev(jacrev(self.forward_single))))(x)
                
        init_pde_pred = D3y_pred[:,0,0,0,1]  + g/l*torch.cos(y_pred.reshape((-1)))*Dy_pred[:,0,1] + b/m*D2y_pred[:,0,0,1]
        init_pde_pred2 = D3y_pred[:,0,0,0,2]  + g/l*torch.cos(y_pred.reshape((-1)))*Dy_pred[:,0,2] + b/m*D2y_pred[:,0,0,2]
        
        init_pde_pred = torch.column_stack((init_pde_pred, init_pde_pred2))
                
        return torch.norm(init_pde_pred, p=2, dim=1)
    
    def evaluate_field(self, x:torch.Tensor):
        # Get the prediction
        y_pred = self.forward(x)
        # Get the partial derivatives from the network
        Dy_pred = vmap(jacrev(self.forward_single))(x)
        D2y_pred = vmap(hessian(self.forward_single))(x)
        
        field = torch.column_stack((Dy_pred[:,:,0], D2y_pred[:,:,0,0]))
        return field