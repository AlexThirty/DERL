from torch import nn
from torch.utils.data import DataLoader
from itertools import cycle
import torch
import time
from ray import train



def load_data(data_dir='./data'):
    boundary_dataset = torch.load(f'{data_dir}/boundary_data.pt')
    internal_dataset = torch.load(f'{data_dir}/grid_data.pt')
    
    return internal_dataset, boundary_dataset

def train_loop(epochs:int, model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        internal_dataloader:DataLoader,
        boundary_dataloader:DataLoader,
        print_every:int=100):
    
    # Training mode for the network
    model.train()

    
    for epoch in range(epochs):
        
        
        start_time = time.time()
        step_prefix = epoch*len(internal_dataloader)
        model.train()
        for step, (pde_data, bc_data) in enumerate(zip(internal_dataloader, cycle(boundary_dataloader))):
            # Load batches from dataloaders
            x_pde = pde_data[0].to(model.device).float().requires_grad_(True)
            u_pde = pde_data[1].to(model.device).float()
            p_pde = pde_data[2].to(model.device).float()
            vorticity_pde = pde_data[3].to(model.device).float()
            stream_pde = pde_data[4].to(model.device).float()
            
            y_pde = torch.column_stack([u_pde, p_pde])
            
            x_bc = bc_data[0].to(model.device).float().requires_grad_(True)
            u_bc = bc_data[1].to(model.device).float()
            p_bc = bc_data[2].to(model.device).float()
            vorticity_bc = bc_data[3].to(model.device).float()
            stream_bc = bc_data[4].to(model.device).float()
            
            y_bc = torch.column_stack([u_bc, p_bc])
            
            # Call zero grad on optimizer
            optimizer.zero_grad()
        
            loss = model.loss_fn(
                x_pde=x_pde,
                x_bc=x_bc, y_bc=y_bc,
            )
            # Backward the loss, calculate gradients
            loss.backward()
            # Optimizer step
            optimizer.step()
            
            # Printing
            if (step_prefix+step) % print_every == 0 and step>0:
                with torch.no_grad():
                    _, mom_loss, div_loss, y_loss, bc_loss_val, discrepancy_loss_val, tot_loss_val = model.eval_losses(
                        step=step_prefix+step,
                        x_pde=x_pde, y_pde=y_pde,
                        x_bc=x_bc, y_bc=y_bc,
                    )
    
        end_time = time.time()
        
        epoch_time = end_time - start_time
        print(f'Epoch: {epoch}, time: {epoch_time}')
        
        # Testing the model
        model.eval()
        mom_loss_test = 0.
        div_loss_test = 0.
        y_loss_test = 0.
        bc_loss_test = 0.
        tot_loss_test = 0.
        discrepancy_loss_test = 0.
        
        for _, (pde_data, bc_data) in enumerate(zip(internal_dataloader, cycle(boundary_dataloader))):
            # Load batches from dataloaders
            x_pde = pde_data[0].to(model.device).float().requires_grad_(True)
            u_pde = pde_data[1].to(model.device).float()
            p_pde = pde_data[2].to(model.device).float()
            vorticity_pde = pde_data[3].to(model.device).float()
            stream_pde = pde_data[4].to(model.device).float()
            
            y_pde = torch.column_stack([u_pde, p_pde])
            
            x_bc = bc_data[0].to(model.device).float().requires_grad_(True)
            u_bc = bc_data[1].to(model.device).float()
            p_bc = bc_data[2].to(model.device).float()
            vorticity_bc = bc_data[3].to(model.device).float()
            stream_bc = bc_data[4].to(model.device).float()
            
            y_bc = torch.column_stack([u_bc, p_bc])

            
            _, mom_loss, div_loss, y_loss, bc_loss_val, discrepancy_loss_val, tot_loss_val = model.eval_losses(
                step=step_prefix+step,
                x_pde=x_pde, y_pde=y_pde,
                x_bc=x_bc, y_bc=y_bc,
            )
            
            div_loss_test += div_loss.item()
            mom_loss_test += mom_loss.item()
            y_loss_test += y_loss.item()
            bc_loss_test += bc_loss_val.item()
            tot_loss_test += tot_loss_val.item()
            discrepancy_loss_test += discrepancy_loss_val.item()
            
        div_loss_test /= len(internal_dataloader)
        mom_loss_test /= len(internal_dataloader)
        y_loss_test /= len(internal_dataloader)
        bc_loss_test /= len(internal_dataloader)
        tot_loss_test /= len(internal_dataloader)
        discrepancy_loss_test /= len(internal_dataloader)
        
        
            
        print(f'Test Mom loss: {mom_loss_test}, Test Div loss: {div_loss_test}, Test y loss: {y_loss_test}, Test discrepancy loss: {discrepancy_loss_test}')
        print(f'Test BC loss: {bc_loss_test}, Test Total loss: {tot_loss_test}')
        print('---------------------------------')   
            
        scheduler.step(tot_loss_test)     
        
        train.report({
            'epoch': epoch,
            'loss': tot_loss_test,
            'out_loss': y_loss_test,
            'mom_loss': mom_loss_test,
            'div_loss': div_loss_test,
            'bc_loss': bc_loss_test,
            'discrepancy_loss': discrepancy_loss_test,
        })