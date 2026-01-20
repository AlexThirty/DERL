import torch
from torch import nn
from collections import OrderedDict
from torch.func import vmap, jacrev, hessian

# Functions for the dynamics, velocity field and divergence
def u(x) -> torch.Tensor:
    #return torch.tensor([-x[1]/torch.sqrt(x[1]**2+x[0]**2), x[0]/torch.sqrt(x[1]**2+x[0]**2)])
    return torch.tensor([-x[1], x[0]])
    
def div_u(x: torch.Tensor) -> float:
    return 0.

def div_u_vec(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x[:,0])
    
def u_vec(x: torch.Tensor) -> torch.Tensor:
    #return torch.column_stack([-x[:,1]/torch.sqrt(x[:,1]**2+x[:,0]**2), x[:,0]/torch.sqrt(x[:,1]**2+x[:,0]**2)])
    return torch.column_stack([-x[:,1], x[:,0]])

# Density network
class DensityNet(torch.nn.Module):
    def __init__(self,
                 bc_weight: float,
                 init_weight: float,
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
        self.init_weight = init_weight
        self.hidden_units = hidden_units
        self.lr_init = lr_init
        self.device = device
        # Define the net, first layer
        net_dict = OrderedDict(
            {'lin0': nn.Linear(3, hidden_units[0]),
            'act0': activation}
        )

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
        x_init:torch.Tensor,
        y_init:torch.Tensor,
        mode:str,
        x_bc:torch.Tensor=None,
        y_bc:torch.Tensor=None,
    ) -> torch.Tensor:
        # Check that the mode parameter is correct
        modes = ['PINN', 'Derivative', 'Output', 'Sobolev', 'PINN+Output']
        if mode not in modes:
            raise ValueError(f'mode should be in {modes}, but found {mode}')
        
        
        if mode == 'PINN':
            # Get the prediction
            y_pred = self.forward(x)
            # Get the partial derivatives from the network
            Dy_pred = vmap(jacrev(self.forward_single))(x)[:,0,:]
            
            # dr/dt + div(u)r + u*grad(r) = 0 
            # Calculate the pde_residual
            pde_pred = Dy_pred[:,0] + div_u_vec(x[:,1:])*y_pred.reshape((-1)) + torch.einsum('bi,bi->b', u_vec(x[:,1:]), Dy_pred[:,1:])
            # Calculate the loss
            pde_loss = self.loss_container(pde_pred, torch.zeros_like(pde_pred))
            
            spec_loss = self.pde_weight*pde_loss
        
        elif mode == 'Derivative':
            # Get the partial derivatives from the network
            Dy_pred = vmap(jacrev(self.forward_single))(x)[:,0,:]
            # In this case, we learn by supervision the partial derivatives
            der_loss = self.loss_container(Dy_pred, Dy)

            spec_loss = self.sys_weight*der_loss
        elif mode == 'Output':
            # Get the prediction
            y_pred = self.forward(x)

            # Out loss
            out_loss = self.loss_container(y_pred, y)
            
            spec_loss = self.sys_weight*out_loss
        elif mode == 'PINN+Output':
            # Get the prediction
            y_pred = self.forward(x)
            # Get the partial derivatives from the network
            Dy_pred = vmap(jacrev(self.forward_single))(x)[:,0,:]
            
            # dr/dt + div(u)r + u*grad(r) = 0 
            # Calculate the pde_residual
            pde_pred = Dy_pred[:,0] + div_u_vec(x[:,1:])*y_pred.reshape((-1)) + torch.einsum('bi,bi->b', u_vec(x[:,1:]), Dy_pred[:,1:])
            # Calculate the loss
            pde_loss = self.loss_container(pde_pred, torch.zeros_like(pde_pred))
            
            out_loss = self.loss_container(y_pred, y)
            
            spec_loss = self.pde_weight*pde_loss + self.sys_weight*out_loss    

        else:
            # Get the prediction
            y_pred = self.forward(x)
            # Get the partial derivatives from the network
            Dy_pred = vmap(jacrev(self.forward_single))(x)[:,0,:]            
            # In this case, we learn by supervision the partial derivatives
            der_loss = self.loss_container(Dy_pred, Dy)
            # Out loss
            out_loss = self.loss_container(y_pred, y)
            # Sobolev loss
            sob_loss = der_loss+out_loss
            
            spec_loss = self.sys_weight*sob_loss
        
        if x_bc is not None:
            y_bc_pred = self.forward(x_bc)
            bc_loss = self.loss_container(y_bc_pred.reshape((-1)), y_bc.reshape((-1)))
            #x_init = torch.cat((x_init, x_bc), dim=0)
            #y_init = torch.cat((y_init, y_bc), dim=0)
        else:
            bc_loss = torch.tensor([0.])
        
        # Calculate the init prediction
        y_init_pred = self.forward(x_init)
        # Initial loss
        init_loss = self.loss_container(y_init_pred.reshape((-1)), y_init.reshape((-1)))
        # Total loss
        tot_loss = spec_loss + self.init_weight*init_loss + self.bc_weight*bc_loss
        return tot_loss
    
    def eval_losses(self, step:int,
        x:torch.Tensor,
        y:torch.Tensor,
        Dy:torch.Tensor,
        x_init:torch.Tensor,
        y_init:torch.Tensor,
        mode:str,
        x_bc:torch.Tensor=None,
        y_bc:torch.Tensor=None,
        print_to_screen:bool=False,    
    ):
        # Check that the mode parameter is correct
        modes = ['PINN', 'Derivative', 'Output', 'Sobolev', 'PINN+Output']
        if mode not in modes:
            raise ValueError(f'mode should be in {modes}, but found {mode}')
        
        # Get the prediction
        y_pred = self.forward(x)
        # Get the partial derivatives from the network
        Dy_pred = vmap(jacrev(self.forward_single))(x)[:,0,:]
        
        # dr/dt + div(u)r + u*grad(r) = 0 
        # Calculate the pde_residual
        pde_pred = Dy_pred[:,0] + div_u_vec(x[:,1:])*y_pred.reshape((-1)) + torch.einsum('bi,bi->b', u_vec(x[:,1:]), Dy_pred[:,1:])
        # Calculate the loss
        pde_loss = self.loss_container(pde_pred, torch.zeros_like(pde_pred))
        
        # In this case, we learn by supervision the partial derivatives
        der_loss = self.loss_container(Dy_pred, Dy)
        
        out_loss = self.loss_container(y_pred, y)
        
        
        # Sobolev loss
        sob_loss = der_loss+out_loss
        
        if mode == 'PINN':
            spec_loss = self.pde_weight*pde_loss
        elif mode == 'Derivative':
            spec_loss = self.sys_weight*der_loss
        elif mode == 'Output':
            spec_loss = self.sys_weight*out_loss
        elif mode == 'PINN+Output':
            spec_loss = self.pde_weight*pde_loss + self.sys_weight*out_loss
        else:
            spec_loss = self.sys_weight*sob_loss
        
        if x_bc is not None:
            y_bc_pred = self.forward(x_bc)
            bc_loss = self.loss_container(y_bc_pred.reshape((-1)), y_bc.reshape((-1)))
            #x_init = torch.cat((x_init, x_bc), dim=0)
            #y_init = torch.cat((y_init, y_bc), dim=0)
        else:
            bc_loss = torch.tensor([0.])
        
        # Calculate the init prediction
        y_init_pred = self.forward(x_init)
        # Initial loss
        init_loss = self.loss_container(y_init_pred.reshape((-1)), y_init.reshape((-1)))
        # Total loss
        tot_loss = spec_loss + self.init_weight*init_loss + self.bc_weight*bc_loss
        
        ### Field loss
        #ux_pred = torch.ones_like(y_pred).reshape((-1))
        #uy_pred = -(Dy_pred[:,0] + ux_pred*Dy_pred[:,1])/(Dy_pred[:,2])
        ux_pred = Dy_pred[:,0]/Dy_pred[:,1]
        uy_pred = Dy_pred[:,0]/Dy_pred[:,2]
        u_pred = torch.column_stack((ux_pred.reshape((-1,1)), uy_pred.reshape((-1,1))))
        u_pred = u_pred/(torch.norm(u_pred, dim=1).reshape((-1,1)))
        field_loss = self.loss_container(torch.abs(u_pred), torch.abs(u_vec(x[:,1:])))
        #field_loss = torch.tensor([0.])
        
        if print_to_screen:
            print(f'Step: {step}, total loss: {tot_loss}, init loss: {init_loss}')
            print(f'pde loss: {pde_loss}, rho loss {out_loss}, Drho loss: {der_loss}, bc loss: {bc_loss} field loss: {field_loss}')
        
        
        return step, out_loss, der_loss, pde_loss, init_loss, tot_loss, bc_loss
    
    def evaluate_consistency(self, x):
        # Get the prediction
        y_pred = self.forward(x)
        # Get the partial derivatives from the network
        Dy_pred = vmap(jacrev(self.forward_single))(x)[:,0,:]
        
        # dr/dt + div(u)r + u*grad(r) = 0 
        # Calculate the pde_residual
        pde_pred = Dy_pred[:,0] + div_u_vec(x[:,1:])*y_pred.reshape((-1)) + torch.einsum('bi,bi->b', u_vec(x[:,1:]), Dy_pred[:,1:])
        # Calculate the loss
        #pde_loss = self.loss_container(pde_pred, torch.zeros_like(pde_pred))
        
        return torch.abs(pde_pred)
        
    
    
    
