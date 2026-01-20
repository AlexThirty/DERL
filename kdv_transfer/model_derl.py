import torch
from torch import nn
from torch.func import vmap, jacrev, hessian, jacfwd
from torch.utils.data import DataLoader
from collections import OrderedDict


def ball_boundary_uniform(n: int, radius: float, dim:int):
    angle = torch.distributions.Normal(0., 1.).sample((n, dim))
    #angle = torch.distributions.Uniform(0., 1.).sample((n, dim))*2 - 1
    norms = torch.norm(angle, p=2., dim=1).reshape((-1,1))
    angle = angle/norms    
    pts = radius*angle
    return pts

nu = 0.0025

class SinActivation(torch.nn.Module):
    def __init__(self):
        super(SinActivation, self).__init__()
        return
    def forward(self, x):
        return torch.sin(x)

class KdVPINN(torch.nn.Module):
    def __init__(self,
                 hidden_units,
                 lr: float,
                 pde_weight:float,
                 init_weight:float,
                 bc_weight:float,
                 sys_weight:float,
                 device:str='cuda:0',
                 activation:torch.nn.Module=SinActivation(),
                 last_activation:bool=False,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.pde_weight = pde_weight
        self.init_weight = init_weight
        self.bc_weight = bc_weight
        self.sys_weight = sys_weight
        
        self.device = device
        
        # Define the net, first layer
        net_dict = OrderedDict(
            {'lin0': nn.Linear(2, hidden_units[0]),
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
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        
        self.loss_container = nn.MSELoss(reduction='mean')
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def forward_single(self, x:torch.Tensor) -> torch.Tensor:
        # Forward function for individual samples
        return self.net(x.reshape((1,-1))).reshape((-1))
        
    def eval_losses(self, step,
                    x_pde:torch.Tensor,
                    y_pde:torch.Tensor,
                    x_init:torch.Tensor,
                    y_init:torch.Tensor,
                    x_bc1:torch.Tensor,
                    x_bc2:torch.Tensor,
                    print_to_screen=False,
        ) -> torch.Tensor:
        # Prediction for the PDE
        u_pde = self.forward(x_pde)
        
        out_loss = self.loss_container(u_pde.reshape((-1)), y_pde.reshape((-1)))
        # Derivative for the PDE
        uX = vmap(jacrev(self.forward_single))(x_pde)[:,0,:]
        # Third order derivative for the PDE
        uXXX = vmap(jacrev(jacrev(jacrev(self.forward_single))))(x_pde)[:,0,:,:,:]
        # Get the individual derivatives
        ut = uX[:,0]
        ux = uX[:,1]
        uxxx = uXXX[:,1,1,1]
        # Calculate the PDE residual
        pde_pred = ut + u_pde.reshape((-1))*ux + nu*uxxx
        pde_loss = self.loss_container(pde_pred, torch.zeros_like(pde_pred))
        
        # Prediction for the initial condition        
        u_init = self.forward(x_init)
        init_loss = self.loss_container(u_init.reshape((-1)), y_init.reshape((-1)))
        
        # Prediction for the boundary conditions, periodic both in the output and in the derivative
        u_bc1 = self.forward(x_bc1).reshape((-1))
        u_bc2 = self.forward(x_bc2).reshape((-1))
        ux_bc1 = vmap(jacrev(self.forward_single))(x_bc1)[:,0,1]
        ux_bc2 = vmap(jacrev(self.forward_single))(x_bc2)[:,0,1]
        bc_loss = self.loss_container(u_bc1-u_bc2, torch.zeros_like(u_bc1))
        bc_loss += self.loss_container(ux_bc1-ux_bc2, torch.zeros_like(ux_bc1))
        
        
        
        loss = self.pde_weight*pde_loss + self.init_weight*init_loss + self.bc_weight*bc_loss + self.sys_weight*out_loss
        if print_to_screen:
            print(f'Step: {step}, tot loss: {loss}, out loss: {out_loss}, PDE loss: {pde_loss}, boundary loss: {bc_loss}, initial loss: {init_loss}')
        return step, out_loss, pde_loss, init_loss, bc_loss, loss

            
    def loss_fn(self,
                x_pde:torch.Tensor,
                x_init:torch.Tensor,
                y_init:torch.Tensor,
                x_bc1:torch.Tensor,
                x_bc2:torch.Tensor,
                y_pde:torch.Tensor=None,
        ) -> torch.Tensor:
        # Prediction for the PDE
        u_pde = self.forward(x_pde)
        if y_pde is not None:
            out_loss = self.loss_container(u_pde.reshape((-1)), y_pde.reshape((-1)))
        else:
            out_loss = 0.0
        # Derivative for the PDE
        uX = vmap(jacrev(self.forward_single))(x_pde)[:,0,:]
        # Third order derivative for the PDE
        uXXX = vmap(jacrev(jacrev(jacrev(self.forward_single))))(x_pde)[:,0,:,:,:]
        # Get the individual derivatives
        ut = uX[:,0]
        ux = uX[:,1]
        uxxx = uXXX[:,1,1,1]
        # Calculate the PDE residual
        pde_pred = ut + u_pde.reshape((-1))*ux + nu*uxxx
        pde_loss = self.loss_container(pde_pred, torch.zeros_like(pde_pred))
        
        # Prediction for the initial condition        
        u_init = self.forward(x_init)
        init_loss = self.loss_container(u_init.reshape((-1)), y_init.reshape((-1)))
        
        # Prediction for the boundary conditions, periodic both in the output and in the derivative
        u_bc1 = self.forward(x_bc1)
        u_bc2 = self.forward(x_bc2)
        ux_bc1 = vmap(jacrev(self.forward_single))(x_bc1)[:,0,1]
        ux_bc2 = vmap(jacrev(self.forward_single))(x_bc2)[:,0,1]
        bc_loss = self.loss_container(u_bc1.reshape((-1))-u_bc2.reshape((-1)), torch.zeros_like(u_bc1).reshape((-1)))
        bc_loss += self.loss_container(ux_bc1-ux_bc2, torch.zeros_like(ux_bc1))
        
        loss = self.pde_weight*pde_loss + self.init_weight*init_loss + self.bc_weight*bc_loss + self.sys_weight*out_loss
        return loss  
    
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
    
    def student_loss_fn(self,
                        x_pde:torch.Tensor,
                        y_pde:torch.Tensor,
                        Dy_pde:torch.Tensor,
                        Hy_pde:torch.Tensor,
                        #y_pde_true:torch.Tensor,
                        x_init:torch.Tensor,
                        y_init:torch.Tensor,
                        y_init_true:torch.Tensor,
                        x_bc1:torch.Tensor,
                        y_bc1:torch.Tensor,
                        #y_bc1_true:torch.Tensor,
                        #Dy_bc1_teacher:torch.Tensor,
                        x_bc2:torch.Tensor,
                        y_bc2:torch.Tensor,
                        #y_bc2_true:torch.Tensor,
                        #Dy_bc2_teacher:torch.Tensor,
                        mode:str,
                        use_hessian:bool=False,
        ) -> torch.Tensor:
        
        # Prediction for the PDE
        u_pde = self.forward(x_pde)
        # Derivative for the PDE
        
        # Hessian for the PDE
        
        # Third order derivative for the PDE
        #uXXX = vmap(jacrev(jacrev(jacrev(self.forward_single))))(x_pde)[:,0,:,:,:]
        # Get the individual derivatives
        #ut = uX[:,0]
        #ux = uX[:,1]
        #uxxx = uXXX[:,1,1,1]
        # Calculate the PDE residual
        #pde_pred = ut + u_pde*ux + nu*uxxx
        #pde_loss = self.loss_container(pde_pred, torch.zeros_like(pde_pred))
        
        if mode == 'Derivative':
            uX = vmap(jacrev(self.forward_single))(x_pde)[:,0,:]
            sys_loss = self.loss_container(uX, Dy_pde[:,0,:])
        elif mode == 'Derivative+Hessian':
            uXX = vmap(hessian(self.forward_single))(x_pde)[:,0,:,:]
            uX = vmap(jacrev(self.forward_single))(x_pde)[:,0,:]
            sys_loss = self.loss_container(uX, Dy_pde[:,0,:]) + self.loss_container(uXX, Hy_pde[:,0,:,:])
        elif mode == 'Hessian':
            uXX = vmap(hessian(self.forward_single))(x_pde)[:,0,:,:]
            sys_loss = self.loss_container(uXX, Hy_pde[:,0,:,:])
        elif mode == 'Output':
            sys_loss = self.loss_container(u_pde, y_pde)
        elif mode == 'Sobolev':
            uX = vmap(jacrev(self.forward_single))(x_pde)[:,0,:]
            sys_loss = self.loss_container(u_pde, y_pde) + self.loss_container(uX, Dy_pde[:,0,:])
        elif mode == 'Sobolev+Hessian':
            uX = vmap(jacrev(self.forward_single))(x_pde)[:,0,:]
            hess_error = self.hessian_sobolev_error(x_pde, Hy_pde[:,0,:,:])
            sys_loss = self.loss_container(u_pde, y_pde) + self.loss_container(uX, Dy_pde[:,0,:]) + hess_error
        else:
            raise ValueError(f'Unknown mode {mode}')
        
        # Prediction for the initial condition
        u_init = self.forward(x_init)
        init_loss = self.loss_container(u_init.reshape((-1)), y_init_true.reshape((-1)))
        
        # Prediction for the boundary conditions, periodic both in the output and in the derivative
        u_bc1 = self.forward(x_bc1)
        u_bc2 = self.forward(x_bc2)
        ux_bc1 = vmap(jacrev(self.forward_single))(x_bc1)[:,0,1]
        ux_bc2 = vmap(jacrev(self.forward_single))(x_bc2)[:,0,1]
        bc_loss = self.loss_container(u_bc1.reshape((-1))-u_bc2.reshape((-1)), torch.zeros_like(u_bc1).reshape((-1)))
        bc_loss += self.loss_container(ux_bc1-ux_bc2, torch.zeros_like(ux_bc1))
        #bc_loss = self.loss_container(u_bc1.reshape((-1)), y_bc1.reshape((-1)))
        #bc_loss += self.loss_container(u_bc2.reshape((-1)), y_bc2.reshape((-1)))
        
        loss = self.sys_weight*sys_loss + self.init_weight*init_loss + self.bc_weight*bc_loss
        
        
        return loss
    
    
    
    def student_eval_losses(self, step:int,
                        x_pde:torch.Tensor,
                        y_pde:torch.Tensor,
                        Dy_pde:torch.Tensor,
                        Hy_pde:torch.Tensor,
                        y_pde_true:torch.Tensor,
                        x_init:torch.Tensor,
                        y_init:torch.Tensor,
                        y_init_true:torch.Tensor,
                        x_bc1:torch.Tensor,
                        y_bc1:torch.Tensor,
                        #Dy_bc1_teacher:torch.Tensor,
                        x_bc2:torch.Tensor,
                        y_bc2:torch.Tensor,
                        #Dy_bc2_teacher:torch.Tensor,
                        mode:int,
                        use_hessian:bool=False,
                        print_to_screen:bool=False,
        ) -> torch.Tensor:
        
        # Prediction for the PDE
        u_pde = self.forward(x_pde)
        # Derivative for the PDE
        uX = vmap(jacrev(self.forward_single))(x_pde)[:,0,:]
        # Hessian for the PDE
        uXX = vmap(hessian(self.forward_single))(x_pde)[:,0,:,:]
        # Third order derivative for the PDE
        uXXX = vmap(jacrev(jacrev(jacrev(self.forward_single))))(x_pde)[:,0,:,:,:]
        # Get the individual derivatives
        ut = uX[:,0]
        ux = uX[:,1]
        uxxx = uXXX[:,1,1,1]
        # Calculate the PDE residual
        pde_pred = ut + u_pde.reshape((-1))*ux + nu*uxxx
        pde_loss = self.loss_container(pde_pred, torch.zeros_like(pde_pred))
        
        # Derivative losses
        der_loss = self.loss_container(uX, Dy_pde[:,0,:])
        #hess_loss = self.loss_container(uXX, Hy_pde[:,0,:,:])
        hess_loss = torch.zeros_like(der_loss)
        
        # Output loss
        out_loss = self.loss_container(u_pde, y_pde)
        out_loss_true = self.loss_container(u_pde, y_pde_true)
        
        if mode == 'Derivative':
            sys_loss = self.loss_container(uX, Dy_pde[:,0,:])
        elif mode == 'Derivative+Hessian':
            sys_loss = self.loss_container(uX, Dy_pde[:,0,:]) + self.loss_container(uXX, Hy_pde[:,0,:,:])
        elif mode == 'Hessian':
            sys_loss = self.loss_container(uXX, Hy_pde[:,0,:,:])
        elif mode == 'Output':
            sys_loss = self.loss_container(u_pde, y_pde)
        elif mode == 'Sobolev':
            sys_loss = self.loss_container(u_pde, y_pde) + self.loss_container(uX, Dy_pde[:,0,:])
        elif mode == 'Sobolev+Hessian':
            hess_error = self.hessian_sobolev_error(x_pde, Hy_pde[:,0,:,:])
            sys_loss = self.loss_container(u_pde, y_pde) + self.loss_container(uX, Dy_pde[:,0,:]) + hess_error
        else:
            raise ValueError(f'Unknown mode {mode}')
        
        # Prediction for the initial condition
        u_init = self.forward(x_init)
        init_loss = self.loss_container(u_init.reshape((-1)), y_init.reshape((-1)))
        
        init_loss_true = self.loss_container(u_init.reshape((-1)), y_init_true.reshape((-1)))
        
        # Prediction for the boundary conditions, periodic both in the output and in the derivative
        u_bc1 = self.forward(x_bc1)
        u_bc2 = self.forward(x_bc2)
        ux_bc1 = vmap(jacrev(self.forward_single))(x_bc1)[:,0,1]
        ux_bc2 = vmap(jacrev(self.forward_single))(x_bc2)[:,0,1]
        bc_loss = self.loss_container(u_bc1.reshape((-1)), y_bc1.reshape((-1)))
        bc_loss += self.loss_container(u_bc2.reshape((-1)), y_bc2.reshape((-1)))
        
        bc_loss_true = self.loss_container(u_bc1.reshape((-1))-u_bc2.reshape((-1)), torch.zeros_like(u_bc1).reshape((-1)))
        bc_loss_true += self.loss_container(ux_bc1-ux_bc2, torch.zeros_like(ux_bc1))
        
        loss = self.sys_weight*sys_loss + self.init_weight*init_loss + self.bc_weight*bc_loss
        if print_to_screen:
            print(f'Step: {step}, tot loss: {loss}, der loss: {der_loss}, hess loss: {hess_loss}, out loss: {out_loss}, init loss: {init_loss}, bc loss: {bc_loss}')
            print(f'PDE loss: {pde_loss}, out loss true: {out_loss_true}, init loss true: {init_loss_true}, bc loss true: {bc_loss_true}')
        return step, (loss, out_loss, der_loss, hess_loss, pde_loss, init_loss, bc_loss), (out_loss_true, init_loss_true, bc_loss_true)
    
    def evaluate_consistency(self, x):
        u = self.forward(x).reshape((-1))
        uX = vmap(jacrev(self.forward_single))(x)[:,0,:]
        uXXX = vmap(jacrev(jacrev(jacrev((self.forward_single)))))(x)[:,0,:,:,:]
        
        pde_pred = uX[:,0] + u*uX[:,1] + nu*uXXX[:,1,1,1]
        
        return torch.abs(pde_pred)
        
        
        