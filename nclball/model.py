import torch.utils
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from torch.func import vmap, jacrev, jacfwd


def v_init_ball(x: torch.Tensor):
    init_rho = (3./2.)*torch.ones_like(x[:,0]) - (x[:,1]**2 + x[:,2]**2 + x[:,3]**2)
    init_vx = -2.*torch.ones_like(x[:,0])*init_rho
    init_vy = (x[:,1]-1.)*init_rho
    init_vz = 0.5*torch.ones_like(x[:,0])*init_rho
    return torch.column_stack([init_rho, init_vx, init_vy, init_vz])

def ball_uniform(n: int, radius: float, dim:int):
    angle = torch.distributions.Normal(0., 1.).sample((n, dim))
    norms = torch.norm(angle, p=2., dim=1).reshape((-1,1))
    angle = angle/norms
    rad = torch.distributions.Uniform(0., 1.).sample((n,1))**(1./dim)
    
    points = radius*rad*angle
    return points

def ball_boundary_uniform(n: int, radius: float, dim:int):
    angle = torch.distributions.Normal(0., 1.).sample((n, dim))
    #angle = torch.distributions.Uniform(0., 1.).sample((n, dim))*2 - 1
    norms = torch.norm(angle, p=2., dim=1).reshape((-1,1))
    angle = angle/norms    
    pts = radius*angle
    return pts
    
def ball_generate_dataset(steps: int, num_domain:int, num_boundary: int, radius: float, dim: int, t_max: float, t_init_max: float, init_fun):
    tot_domain = steps*num_domain
    tot_boundary = steps*num_boundary
    
    # Generate the times
    t =  t_max*torch.distributions.Uniform(0.,1.).sample((tot_domain,1))
    # Generate the points
    points = ball_uniform(tot_domain, radius, dim)
    # Stack them
    x_pde = torch.column_stack((t, points))
    print(f'x_pde.shape: {x_pde.shape}')
    # Fake target
    y_pde = torch.zeros(tot_domain).reshape((-1,1))
    print(f'y_pde.shape: {y_pde.shape}')
    # Generate the dataset
    pde_dataset = TensorDataset(x_pde, y_pde)
    
    # Generate the init times
    t =  t_init_max*torch.distributions.Uniform(0.,1.).sample((tot_domain,1))
    # Generate the points
    points_init = ball_uniform(tot_domain, radius, dim)
    # Stack them
    x_init = torch.column_stack((t, points_init))
    print(f'x_init.shape: {x_init.shape}')
    # Get the initial values
    y_init = init_fun(x_init)
    print(f'y_init.shape: {y_init.shape}')
    # Generate the dataset
    init_dataset = TensorDataset(x_init, y_init)
    
    # Generate the times
    t =  t_max*torch.distributions.Uniform(0.,1.).sample((tot_boundary,1))
    # Generate the points
    points_bc = ball_boundary_uniform(tot_boundary, radius, dim)
    # Stack them
    x_bc = torch.column_stack((t, points_bc))
    print(f'x_bc.shape: {x_bc.shape}')
    # Fake target
    y_bc = torch.zeros(tot_boundary).reshape((-1,1))
    print(f'y_bc.shape: {y_bc.shape}')
    # Generate the dataset
    bc_dataset = TensorDataset(x_bc, y_bc)
    
    return pde_dataset, init_dataset, bc_dataset
    
    

class BallNCL(torch.nn.Module):
    def __init__(self,
                 hidden_units:list,
                 sys_weight:float=1.,
                 div_weight:float=1.,
                 F_weight:float=1.,
                 init_weight:float=1.,
                 bc_weight:float=1.,
                 radius: float=1.,
                 lr:float=1e-3,
                 activation:nn.Module=nn.Softplus(beta=25.),
                 device: str='cuda:0',
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        
        self.radius = radius
        self.F_weight = F_weight
        self.init_weight = init_weight
        self.bc_weight = bc_weight
        self.sys_weight = sys_weight
        self.div_weight = div_weight
        self.hidden_units = hidden_units
        
        self.loss_container = torch.nn.MSELoss(reduction='mean')  
        
        # Define the net, first layer
        net_dict = OrderedDict(
            {'lin0': nn.Linear(4, hidden_units[0]),
            'act0': activation}
        )

        # Define the net, hidden layers
        for i in range(1, len(hidden_units)):
            net_dict.update({f'lin{i}': nn.Linear(in_features=hidden_units[i-1], out_features=hidden_units[i])})
            net_dict.update({f'act{i}': activation})
        # Define the net, last layer
        net_dict.update({f'lin{len(hidden_units)}': nn.Linear(in_features=hidden_units[-1], out_features=5)})
        # Save the network
        self.net = nn.Sequential(net_dict).to(self.device)
        # Save the optimizer
        self.lr = lr
        
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.opt, milestones=[300, 50000], gamma=5e-3)
        # Device
        self.device = device
      
    def forward_single(self, x:torch.Tensor) -> torch.Tensor:
        # Get the output
        u = self.net(x.reshape((1,-1))).reshape((-1))
        def A_matrix(x:torch.Tensor) -> torch.Tensor:
            #print(x.shape)
            jacb = jacrev(self.net_out_div)(x)
            #print(jacb.shape)
            A = jacb - torch.transpose(jacb, dim0=0, dim1=1)
            #print(A.shape)
            return A
        
        # Now get the vector
        vec = jacrev(A_matrix)
        return torch.concat([torch.einsum('bii->b', vec(x)), u[4:]])
    
    def net_out_div(self, x:torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape((1,-1))).reshape((-1))[:4]
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # Get the output
        u = self.net(x)
        
        def A_matrix(x:torch.Tensor) -> torch.Tensor:
            #print(x.shape)
            jacb = jacrev(self.net_out_div)(x)
            #print(jacb.shape)
            A = jacb - torch.transpose(jacb, dim0=0, dim1=1)
            #print(A.shape)
            return A
        
        # Now get the vector
        vec = vmap(jacrev(A_matrix))
        return torch.column_stack([torch.einsum('...ii', vec(x)), u[:,4].reshape((-1,1))])
    
    '''
    def forward_single(self, x:torch.Tensor):
        # Get the output
        u = self.net(x.reshape((1,-1))).reshape(-1)
        # Function that obtains the matrix
        def A_matrix(x:torch.Tensor):
            #print(x.shape)
            # Pass through the network
            out = self.net(x.reshape((1,-1))).reshape(-1)[:6]
            #rint(out.shape)
            # Reshape into a matrix form
            mat = torch.zeros((4,4), device=self.device)
            triu_indexes = torch.triu_indices(4,4, offset=1)
            mat = mat.index_put(tuple(triu_indexes), out)
            #print(out.shape)
            # Make the matrix antisymmetric
            A = mat - torch.transpose(mat, dim0=0, dim1=1)
            #print(A.shape)
            return A
        # Now get the vector
        vec = jacrev(A_matrix)
        return torch.concat([torch.einsum('...ii', vec(x)), u[6:]])
    
    
    def forward(self, x:torch.Tensor):
        # Get the output
        u = self.net(x)
        # Function that obtains the matrix
        def A_matrix(x:torch.Tensor):
            #print(x.shape)
            # Pass through the network
            out = self.net(x)[:6]
            #rint(out.shape)
            # Reshape into a matrix form
            mat = torch.zeros((4,4), device=self.device)
            triu_indexes = torch.triu_indices(4,4, offset=1)
            mat = mat.index_put(tuple(triu_indexes), out)
            #print(out.shape)
            # Make the matrix antisymmetric
            A = mat - torch.transpose(mat, dim0=0, dim1=1)
            #print(A.shape)
            return A
        # Now get the vector
        vec = vmap(jacrev(A_matrix))
        return torch.column_stack([torch.einsum('...ii', vec(x)), u[:,6:]])
    '''
    
    def loss_fn(self, x_pde:torch.Tensor, x_bc:torch.Tensor, x_init:torch.Tensor, y_init:torch.Tensor) -> torch.Tensor:
        # Get the output of the net and divide them
        pde_out = self.forward(x_pde)
        rho = pde_out[:,0].reshape((-1,1))
        rhou = pde_out[:,1:4]
        # Get the jacobian of the network
        Dout = vmap(jacrev(self.forward_single))(x_pde)
        # Extract the individual derivatives
        rhou_t = Dout[:,1:4,0]
        # Time derivative of density wrt t
        rho_t = Dout[:,0,0].reshape((-1,1))
        # Calculate the rhou derivative of rhou
        rhou_rhou = torch.einsum('bij,bj->bi', Dout[:,1:4,1:4], rhou)
        # Calculate the derivative of rhou wrt the hard_term
        rho_Xt = Dout[:,0,:]
        rho_X = Dout[:,0,1:4]   
        hard_der = torch.einsum('bi,bj->bji', rho_X, rhou)
        rhou_hard_der = torch.einsum('bij,bj->bi', hard_der, rhou)
        # Calculate the gradient of the pressure
        pressure_X = Dout[:,-1,1:4]
        
        # Now join everything together
        F_pde = rho**2*rhou_t - rho*rho_t*rhou + rho*rhou_rhou - rhou_hard_der + rho**2*pressure_X
        F_loss_val = self.loss_container(F_pde, torch.zeros_like(F_pde))
        
        # Now it is time for the divergence term, easy one
        div_pde = torch.einsum('bi,bi->b', rho_Xt, pde_out[:,:4])
        div_loss_val = self.loss_container(div_pde, torch.zeros_like(div_pde))
        
        # Now the boundary condition loss
        bc_out = self.forward(x_bc)
        bc_rhou = bc_out[:,1:4]
        normal_vec = x_bc[:,1:4]#/torch.norm(x_bc[:,1:4], dim=1).reshape((-1,1))
        bc = torch.einsum('bi,bi->b', bc_rhou, normal_vec)
        bc_loss_val = self.loss_container(bc, torch.zeros_like(bc))
                
        ### Now the initial loss
        init_out = self.forward(x_init)[:,:4]
        init_loss_val = self.loss_container(init_out, y_init)
    
        return F_loss_val*self.F_weight + div_loss_val*self.div_weight + bc_loss_val*self.bc_weight + init_loss_val*self.init_weight
    
    def eval_losses(self, x_pde:torch.Tensor, x_bc:torch.Tensor, x_init:torch.Tensor, y_init:torch.Tensor, step:int):
        # Get the output of the net and divide them
        pde_out = self.forward(x_pde)
        rho = pde_out[:,0].reshape((-1,1))
        rhou = pde_out[:,1:4]
        # Get the jacobian of the network
        Dout = vmap(jacrev(self.forward_single))(x_pde)
        # Extract the individual derivatives
        rhou_t = Dout[:,1:4,0]
        # Time derivative of density wrt t
        rho_t = Dout[:,0,0].reshape((-1,1))
        # Calculate the rhou derivative of rhou
        rhou_rhou = torch.einsum('bij,bj->bi', Dout[:,1:4,1:4], rhou)
        # Calculate the derivative of rhou wrt the hard_term
        rho_Xt = Dout[:,0,:]
        rho_X = Dout[:,0,1:4]   
        hard_der = torch.einsum('bi,bj->bji', rho_X, rhou)
        rhou_hard_der = torch.einsum('bij,bj->bi', hard_der, rhou)
        # Calculate the gradient of the pressure
        pressure_X = Dout[:,-1,1:4]
        
        # Now join everything together
        F_pde = rho**2*rhou_t - rho*rho_t*rhou + rho*rhou_rhou - rhou_hard_der + rho**2*pressure_X
        F_loss_val = self.loss_container(F_pde, torch.zeros_like(F_pde))
        
        # Now it is time for the divergence term, easy one
        div_pde = torch.einsum('bi,bi->b', rho_Xt, pde_out[:,:4])
        div_loss_val = self.loss_container(div_pde, torch.zeros_like(div_pde))
        
        # Now the boundary condition loss
        bc_out = self.forward(x_bc)
        bc_rhou = bc_out[:,1:4]
        normal_vec = x_bc[:,1:4]#/torch.norm(x_bc[:,1:4], dim=1).reshape((-1,1))
        bc = torch.einsum('bi,bi->b', bc_rhou, normal_vec)
        bc_loss_val = self.loss_container(bc, torch.zeros_like(bc))
                
        ### Now the initial loss
        init_out = self.forward(x_init)[:,:4]
        init_loss_val = self.loss_container(init_out, y_init)
        
        tot_loss_val = F_loss_val*self.F_weight + div_loss_val*self.div_weight + bc_loss_val*self.bc_weight + init_loss_val*self.init_weight
        #print(f'Step: {step}, F_loss: {F_loss_val}, div_loss: {div_loss_val}, bc_loss: {bc_loss_val}, init_loss: {init_loss_val}')
        return step, F_loss_val, div_loss_val, bc_loss_val, init_loss_val, tot_loss_val
    
    def multioutput_sobolev_error(self, x_pde:torch.Tensor, D_pde:torch.Tensor):
        rand_vec = ball_boundary_uniform(n=100, radius=1., dim=5).to(self.device)
        
        def rand_proj(x):
            out_pred =self.forward_single(x)
            proj_pred= torch.einsum('pj,j->p', rand_vec, out_pred)
            return proj_pred
        rand_proj_der_pred = vmap(jacfwd(rand_proj))(x_pde)
        rand_proj_der_true = torch.einsum('bij,pi->bpj', D_pde, rand_vec)
        error = torch.norm(rand_proj_der_pred - rand_proj_der_true, p=2, dim=2).mean()
        return error
    
    def eval_student_loss_fn(self, step:int, mode:int,
                             x_pde:torch.Tensor, y_pde:torch.Tensor, Dy_pde:torch.Tensor,
                             x_bc:torch.Tensor, 
                             x_init:torch.Tensor, y_init:torch.Tensor, D2y_pde:torch.Tensor) -> torch.Tensor:
        # Get the output of the net and divide them
        pde_out = self.forward(x_pde)
        rho = pde_out[:,0].reshape((-1,1))
        
        out_loss = self.loss_container(pde_out[:,:4], y_pde[:,:4])
        
        rhou = pde_out[:,1:4]
        # Get the jacobian of the network
        Dout = vmap(jacrev(self.forward_single))(x_pde)
        D2out = vmap(jacrev(jacrev(self.forward_single)))(x_pde)
        
        der_loss = self.loss_container(Dout, Dy_pde)
        # Extract the individual derivatives
        rhou_t = Dout[:,1:4,0]
        # Time derivative of density wrt t
        rho_t = Dout[:,0,0].reshape((-1,1))
        # Calculate the rhou derivative of rhou
        rhou_rhou = torch.einsum('bij,bj->bi', Dout[:,1:4,1:4], rhou)
        # Calculate the derivative of rhou wrt the hard_term
        rho_Xt = Dout[:,0,:]
        rho_X = Dout[:,0,1:4]   
        hard_der = torch.einsum('bi,bj->bji', rho_X, rhou)
        rhou_hard_der = torch.einsum('bij,bj->bi', hard_der, rhou)
        # Calculate the gradient of the pressure
        pressure_X = Dout[:,-1,1:4]
        
        # Now join everything together
        F_pde = rho**2*rhou_t - rho*rho_t*rhou + rho*rhou_rhou - rhou_hard_der + rho**2*pressure_X
        F_loss_val = self.loss_container(F_pde, torch.zeros_like(F_pde))
        
        # Now it is time for the divergence term, easy one
        div_pde = torch.einsum('bi,bi->b', rho_Xt, pde_out[:,:4])
        div_loss_val = self.loss_container(div_pde, torch.zeros_like(div_pde))
        
        # Now the boundary condition loss
        bc_out = self.forward(x_bc)
        bc_rhou = bc_out[:,1:4]
        normal_vec = x_bc[:,1:4]
        bc = torch.einsum('bi,bi->b', bc_rhou, normal_vec)
        
        bc_loss_val = self.loss_container(bc, torch.zeros_like(bc))
        
        ### Now the initial loss
        init_out = self.forward(x_init)[:,:4]
        init_loss_val = self.loss_container(init_out, y_init[:,:4])
        
        hes_loss = self.loss_container(D2out, D2y_pde)
        
        if mode == 'Derivative':
            sys_loss = self.loss_container(Dout, Dy_pde)
        elif mode == 'Output':
            sys_loss = self.loss_container(pde_out, y_pde)
        elif mode == 'Sobolev':
            sys_loss = self.loss_container(Dout, Dy_pde) + self.loss_container(pde_out, y_pde)#self.loss_container(pde_out, y_pde)
        elif mode == 'Hessian':
            sys_loss = self.loss_container(D2out, D2y_pde)
        elif mode == 'Derivative+Hessian':
            sys_loss = self.loss_container(Dout, Dy_pde) + self.loss_container(D2out, D2y_pde)
        elif mode == 'Sobolev+Hessian':
            sys_loss = self.loss_container(pde_out, y_pde) + self.loss_container(Dout, Dy_pde) + self.hessian_sobolev_error(x_pde, D2y_pde[:,0,:,:])
        else:
            raise ValueError('Mode not recognized')
        
        tot_loss_val = self.init_weight*init_loss_val + self.bc_weight*bc_loss_val + self.sys_weight*sys_loss
        
        return step, out_loss, der_loss, hes_loss, F_loss_val, div_loss_val, bc_loss_val, init_loss_val, tot_loss_val
    
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
    
    def student_loss_fn(self, mode:int,
                        x_pde:torch.Tensor, y_pde:torch.Tensor, Dy_pde:torch.Tensor,
                        x_bc:torch.Tensor,
                        x_init:torch.Tensor, y_init:torch.Tensor, D2y_pde:torch.Tensor=None) -> torch.Tensor:
                
        if mode == 'Derivative':
            # Get the jacobian of the network
            Dout = vmap(jacrev(self.forward_single))(x_pde)
            sys_loss = self.loss_container(Dout, Dy_pde)
        elif mode == 'Output':
            # Get the output of the net and divide them
            pde_out = self.forward(x_pde)
            sys_loss = self.loss_container(pde_out, y_pde)
        elif mode == 'Sobolev':
            # Get the output of the net and divide them
            pde_out = self.forward(x_pde)
            # Get the jacobian of the network
            Dout = vmap(jacrev(self.forward_single))(x_pde)
            sys_loss = self.loss_container(Dout, Dy_pde) + self.loss_container(Dout, Dy_pde)#self.loss_container(pde_out, y_pde)
        elif mode == 'Hessian':
            D2out = vmap(jacrev(jacrev(self.forward_single)))(x_pde)
            sys_loss = self.loss_container(D2out, D2y_pde)
        elif mode == 'Derivative+Hessian':
            Dout = vmap(jacrev(self.forward_single))(x_pde)
            D2out = vmap(jacrev(jacrev(self.forward_single)))(x_pde)
            sys_loss = self.loss_container(Dout, Dy_pde) + self.loss_container(D2out, D2y_pde)
        elif mode == 'Sobolev+Hessian':
            pde_out = self.forward(x_pde)
            Dout = vmap(jacrev(self.forward_single))(x_pde)
            hess_error = self.hessian_sobolev_error(x_pde, D2y_pde[:,:,:,:])
            sys_loss = self.loss_container(pde_out, y_pde) + self.loss_container(Dout, Dy_pde[:,0,:]) + hess_error
        else:
            raise ValueError('Mode not recognized')        
        
        # Now the boundary condition loss
        bc_out = self.forward(x_bc)
        bc_rhou = bc_out[:,1:4]
        normal_vec = x_bc[:,1:4]
        bc = torch.einsum('bi,bi->b', bc_rhou, normal_vec)
        
        bc_loss_val = self.loss_container(bc, torch.zeros_like(bc))
        
        ### Now the initial loss
        init_out = self.forward(x_init)[:,:4]
        init_loss_val = self.loss_container(init_out, y_init[:,:4])
             
        loss = self.init_weight*init_loss_val + self.bc_weight*bc_loss_val + self.sys_weight*sys_loss
        
        return loss
    
    def evaluate_consistency(self, x_pde:torch.Tensor):
        # Get the output of the net and divide them
        pde_out = self.forward(x_pde)
        rho = pde_out[:,0].reshape((-1,1))
        rhou = pde_out[:,1:4]
        # Get the jacobian of the network
        Dout = vmap(jacrev(self.forward_single))(x_pde)
        # Extract the individual derivatives
        rhou_t = Dout[:,1:4,0]
        # Time derivative of density wrt t
        rho_t = Dout[:,0,0].reshape((-1,1))
        # Calculate the rhou derivative of rhou
        rhou_rhou = torch.einsum('bij,bj->bi', Dout[:,1:4,1:4], rhou)
        # Calculate the derivative of rhou wrt the hard_term
        rho_Xt = Dout[:,0,:]
        rho_X = Dout[:,0,1:4]   
        hard_der = torch.einsum('bi,bj->bji', rho_X, rhou)
        rhou_hard_der = torch.einsum('bij,bj->bi', hard_der, rhou)
        # Calculate the gradient of the pressure
        pressure_X = Dout[:,-1,1:4]
        
        # Now join everything together
        F_pde = rho**2*rhou_t - rho*rho_t*rhou + rho*rhou_rhou - rhou_hard_der + rho**2*pressure_X
        F_loss_val = self.loss_container(F_pde, torch.zeros_like(F_pde))
        
        # Now it is time for the divergence term, easy one
        div_pde = torch.einsum('bi,bi->b', rho_Xt, pde_out[:,:4])
        div_loss_val = self.loss_container(div_pde, torch.zeros_like(div_pde))
        
        
        return torch.norm(F_pde, p=2, dim=1), torch.abs(div_pde)
    