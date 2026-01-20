from torch import nn
import torch


import torch
from torch import nn
from collections import OrderedDict
from torch.func import vmap, jacrev, hessian

m = 1. # old was 0.1
l = 10. # old was 1.
g = 9.81
b = 0.5

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
                 pred_weight: float,
                 ae_weight: float,
                 lin_weight: float,
                 hidden_units: list,
                 lr_init: float,
                 device: str,
                 activation: nn.Module=nn.Tanh,
                 last_activation: bool=False,
                 b = 0.5,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # Save state variables
        self.pred_weight = pred_weight
        self.ae_weight = ae_weight
        self.lin_weight = lin_weight
        self.hidden_units = hidden_units
        self.lr_init = lr_init
        self.device = device
        self.b = b
        # Define the net, first layer
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_units[0]),
            activation(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            activation(),
            nn.Linear(hidden_units[1], 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, hidden_units[1]),
            activation(),
            nn.Linear(hidden_units[1], hidden_units[0]),
            activation(),
            nn.Linear(hidden_units[0], 2)
        )
        self.loss_container = nn.MSELoss(reduction='mean')

        self.eigen_net = nn.Sequential(
            nn.Linear(2, hidden_units[0]),
            activation(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            activation(),
            nn.Linear(hidden_units[1], 2)
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # Forward function
        return self.decoder(self.update(self.encoder(x)))
    
    
    def forward_single(self, x:torch.Tensor) -> torch.Tensor:
        # Forward function for individual samples
        return self.decoder(self.update(self.encoder(x.reshape((1,-1))))).reshape((-1))
        
    def update(self, x:torch.Tensor) -> torch.Tensor:
        muwu = self.eigen_net(x)
        # Create a rotation matrix using the second component of muwu
        theta = muwu[:, 1]*0.01
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        # Rotation matrices for each sample in the batch
        K_mat = torch.stack([
            torch.stack([cos_theta, -sin_theta], dim=1),
            torch.stack([sin_theta,  cos_theta], dim=1)
        ], dim=2)  # Shape: (batch, 2, 2)
        K_mat = torch.exp(-muwu[:, 0].reshape((-1,1,1))) * K_mat
        return torch.einsum('bij, bj -> bi', K_mat, x)

    def loss_fn(self,
        x:torch.Tensor,
        y:torch.Tensor,
        y_10:torch.Tensor=None,
    ) -> torch.Tensor:
        
        # Part for the prediction loss
        y_pred = self.forward(x)
        pred_loss = self.loss_container(y_pred, y)
        
        # Part for the autoencoder loss
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        ae_loss = self.loss_container(x_decoded, x)
        
        # Part for the update loss
        with torch.no_grad():
            y_encoded = self.encoder(y)
            x_encoded = self.encoder(x)
        y_encoded_pred = self.update(x_encoded)
        update_loss = self.loss_container(y_encoded_pred, y_encoded)
        
        if y_10 is not None:
            with torch.no_grad():
                y_encoded_10 = self.encoder(y_10)
            # Repeat the update 10 times starting from x_encoded
            y_10_encoded_pred = x_encoded
            for _ in range(10):
                y_10_encoded_pred = self.update(y_10_encoded_pred)

            #update_loss += self.loss_container(y_10_encoded_pred, y_encoded_10)
            
        tot_loss = self.ae_weight * ae_loss +\
                   self.pred_weight * pred_loss +\
                   self.lin_weight * update_loss

        return tot_loss
    
    def eval_losses(self,
        x:torch.Tensor,
        y:torch.Tensor,
        print_to_screen:bool=False,
        y_10:torch.Tensor=None,
    ):
        # Part for the prediction loss
        y_pred = self.forward(x)
        pred_loss = self.loss_container(y_pred, y)
        
        # Part for the autoencoder loss
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        ae_loss = self.loss_container(x_decoded, x)
        
        # Part for the update loss
        with torch.no_grad():
            y_encoded = self.encoder(y)
            x_encoded = self.encoder(x)
        y_encoded_pred = self.update(x_encoded)
        update_loss = self.loss_container(y_encoded_pred, y_encoded)
        
        if y_10 is not None:
            with torch.no_grad():
                y_encoded_10 = self.encoder(y_10)
            # Repeat the update 10 times starting from x_encoded
            y_10_encoded_pred = x_encoded
            for _ in range(10):
                y_10_encoded_pred = self.update(y_10_encoded_pred)

            #update_loss += self.loss_container(y_10_encoded_pred, y_encoded_10)
        tot_loss = self.ae_weight * ae_loss +\
                   self.pred_weight * pred_loss +\
                   self.lin_weight * update_loss
                                      
        return pred_loss, ae_loss, update_loss, tot_loss
    
    def evaluate_trajectory(self, x0:torch.Tensor, time_steps, dt=0.01):
        out = torch.zeros((time_steps, 2)).to(self.device)
        out[0,:] = x0
        for i in range(time_steps-1):
            out_i = self.forward(x0.reshape((1,-1)).to(self.device)).reshape((-1))
            out[i+1,:] = out_i
            x0 = out_i
        return out
