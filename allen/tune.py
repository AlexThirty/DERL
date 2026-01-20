from ray import tune, train
import ray
import os
import torch
from torch.utils.data import DataLoader
import random
import numpy as np

seed = 30
from torch.func import vmap, jacrev
from itertools import cycle

from model import AllenNet
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='Derivative', type=str, help='Mode: -1 for PINN learning, 0 for derivative learning, 1 for output learning')
parser.add_argument('--device', default='cuda', type=str, help='Device to run the code')
parser.add_argument('--name', default='rand', type=str, help='Experiment name')
args = parser.parse_args()
mode = args.mode
device_in = args.device
name = args.name

prefix = name
if name not in ['rand', 'grid']:
    raise ValueError('Name must be either rand or grid')
title_mode = mode

abs_path = os.path.abspath('.')

def train_model(config):
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    epochs = 200
    
    if mode != 'PINN':
        sys_weight = config['sys_weight']
    else:
        sys_weight = 0.
    if mode == 'PINN' or mode == 'PINN+Output':
        pde_weight = config['pde_weight']
    else:
        pde_weight = 0.
    lr_init = config['lr_init']
    bc_weight = config['bc_weight']
   
    batch_size = 50
    
    train_dataset = torch.load(os.path.join(abs_path, 'data', f'{prefix}_dataset.pth'))
    bc_dataset = torch.load(os.path.join(abs_path, 'data', f'bc_dataset.pth'))
    #else:
    #    bc_dataset = None
    
    # Generate the dataloader
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, generator=torch.Generator().manual_seed(seed))
    val_dataloader = DataLoader(train_dataset, 256, shuffle=True, num_workers=4, generator=torch.Generator().manual_seed(seed))
    bc_dataloader = DataLoader(bc_dataset, batch_size, shuffle=True, num_workers=4, generator=torch.Generator().manual_seed(seed))
    print('Data loaded!')

    device = "cpu"
    if torch.cuda.is_available():
        device = 'cuda'
    
    activation = torch.nn.Tanh()
    model = AllenNet(
        sys_weight=sys_weight,
        pde_weight=pde_weight,  
        bc_weight=bc_weight,
        hidden_units=[50 for _ in range(4)],
        lr_init=lr_init,
        device=device,
        activation=activation,    
        last_activation=False,
    ).to(device)
    
    
    
    # Training mode for the network
    model.train()
    
    for epoch in range(epochs):
        model.train()
        step_prefix = epoch*len(train_dataloader)
        #print(f'Epoch: {epoch}, step_prefix: {step_prefix}')
        for step, (train_data, bc_data) in enumerate(zip(train_dataloader, cycle(bc_dataloader))):
            # Load batches from dataloaders
            x_train = train_data[0].to(device).float().requires_grad_(True)
            
            y_train = train_data[1].to(device).float()
            Dy_train = train_data[2].to(device).float()
            Hy_train = train_data[3].to(device).float()
            
            x_bc = bc_data[0].to(device).float()
            y_bc = bc_data[1].to(device).float()
            
            # Call zero grad on optimizer
            model.opt.zero_grad()
            
            loss = model.loss_fn(mode=mode,
                x=x_train, y=y_train, Dy=Dy_train, x_bc=x_bc, y_bc=y_bc, Hy=Hy_train
            )
            # Backward the loss, calculate gradients
            loss.backward()
            # Optimizer step
            model.opt.step()
            # Update the learning rate scheduling
        
        out_losses = []
        der_losses = []
        hes_losses = []
        pde_losses = []
        tot_losses = []
        bc_losses = []
        
        
        for step, (val_data, bc_data) in enumerate(zip(val_dataloader, cycle(bc_dataloader))):
            x_val = val_data[0].to(device).float()
            y_val = val_data[1].to(device).float()
            Dy_val = val_data[2].to(device).float()
            Hy_val = val_data[3].to(device).float()
            
            x_bc = bc_data[0].to(device).float()
            y_bc = bc_data[1].to(device).float()
        
            # Printing
            with torch.no_grad():
                step_val, out_loss_val, der_loss_val, hes_loss_val, pde_loss_val, bc_loss_val, tot_loss_val = model.eval_losses(
                    step=step_prefix+step, mode=mode,
                    x=x_val, y=y_val, Dy=Dy_val, x_bc=x_bc, y_bc=y_bc, Hy=Hy_val
                )
            out_losses.append(out_loss_val.cpu().item())
            der_losses.append(der_loss_val.cpu().item())
            hes_losses.append(hes_loss_val.cpu().item())
            pde_losses.append(pde_loss_val.cpu().item()) 
            tot_losses.append(tot_loss_val.cpu().item())
            bc_losses.append(bc_loss_val.cpu().item())
            
        out_loss_val = np.mean(out_losses)
        der_loss_val = np.mean(der_losses)
        hes_loss_val = np.mean(hes_losses)
        pde_loss_val = np.mean(pde_losses)
        tot_loss_val = np.mean(tot_losses)
        bc_loss_val = np.mean(bc_losses)
            
        train.report({
            'step': step_prefix+step,
            'loss': tot_loss_val.item(),
            'out_loss': out_loss_val.item(),
            'der_loss': der_loss_val.item(),
            'hes_loss': hes_loss_val.item(),
            'pde_loss': pde_loss_val.item(),
            'bc_loss': bc_loss_val.item(),
        })
                
                
                
                             
param_space = {
    "bc_weight": tune.loguniform(1e-3, 1e2),
    "lr_init": tune.choice([1e-3, 5e-4, 1e-4]),
}       

if mode != 'PINN':
    param_space['sys_weight'] = tune.loguniform(1e-3, 1e2)
if mode == 'PINN' or mode == 'PINN+Output':
    param_space['pde_weight'] = tune.loguniform(1e-3, 1e2)

from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

algo = HyperOptSearch()     
scheduler = ASHAScheduler(max_t=200, grace_period=100)
           
trainable_with_resources = tune.with_resources(train_model, {'cpu':4, 'gpu':0.25})
 
if os.path.exists(f'{abs_path}/tuner_results/allen_{prefix}{mode}'):
    print('Directory exists')
    tuner = tune.Tuner.restore(f'{abs_path}/tuner_results/allen_{prefix}{mode}', trainable_with_resources)
else:          
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(metric="out_loss", mode="min", search_alg=algo, scheduler=scheduler, num_samples=100, max_concurrent_trials=4,),
        run_config=train.RunConfig(storage_path=os.path.join(abs_path,'tuner_results'), name=f'allen_{prefix}{mode}', log_to_file=True),
        param_space=param_space,
    )
results = tuner.fit()

best_result = results.get_best_result()  # Get best result object
best_config = best_result.config  # Get best trial's hyperparameters
best_logdir = best_result.path  # Get best trial's result directory
best_checkpoint = best_result.checkpoint  # Get best trial's best checkpoint
best_metrics = best_result.metrics  # Get best trial's last results
best_result_df = best_result.metrics_dataframe  # Get best result as pandas dataframe

print(best_result)
print(best_config)
print(best_logdir)
print(best_checkpoint)
print(best_metrics)
print(best_result_df)

results_df = results.get_dataframe()
results_df = results_df.sort_values(by='out_loss')
results_df.to_csv(f'{abs_path}/tuner_results/allen_{prefix}{mode}.csv')