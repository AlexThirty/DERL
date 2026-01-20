from ray import tune, train
import ray
import os
import torch
from torch.utils.data import DataLoader
import random
from torch.optim import LBFGS
import numpy as np

seed = 30
from torch.func import vmap, jacrev
from itertools import cycle

from model_multi import AllenNet

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='Derivative', type=str, help='Mode: -1 for PINN learning, 0 for derivative learning, 1 for output learning')
parser.add_argument('--device', default='cuda', type=str, help='Device to run the code')
args = parser.parse_args()
mode = args.mode
device_in = args.device
name = 'multi'

prefix = name
if name not in ['exp', 'multi']:
    raise ValueError(f'Name must be in [exp, multi], got {name}')
title_mode = mode

abs_path = os.path.abspath('.')

def train_model(config):
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
   
    batch_size = 100
    
    train_dataset = torch.load(f'{abs_path}/train_data/train_data_multi_rand.pt')
    true_dataset = torch.load(f'{abs_path}/val_data/val_data_multi_rand.pt')
    boundary_dataset = torch.load(f'{abs_path}/boundary_dataset_multi.pt')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    bc_loader = DataLoader(boundary_dataset, batch_size=batch_size, shuffle=True)
    true_loader = DataLoader(true_dataset, batch_size=5000, shuffle=True)
    #else:
    #    bc_dataset = None
    
    print('Data loaded!')

    device = "cpu"
    if torch.cuda.is_available():
        device = device_in
    
    activation = torch.nn.Tanh()
    model = AllenNet(
        in_dim=4,
        sys_weight=sys_weight,
        pde_weight=pde_weight,  
        bc_weight=bc_weight,
        hidden_units=[50 for _ in range(4)],
        lr_init=lr_init,
        device=device,
        activation=activation,    
        last_activation=False,
    ).to(device)
    
    optim = LBFGS(model.parameters(), lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)
    epoch_lim = 99
    
    # Training mode for the network
    model.train()
    
    for epoch in range(epochs):
        model.train()
        step_prefix = epoch*len(train_loader)
        print(f'Epoch: {epoch}, step_prefix: {step_prefix}')
        
        if epoch > epoch_lim:
            step = 0
            train_data = train_dataset[:]
            bc_data = boundary_dataset[:]
            
            # Load batches from dataloaders
            x_train = train_data[0].to(device).float().requires_grad_(True)
            
            y_train = train_data[1].to(device).float()
            Dy_train = train_data[2].to(device).float()
            D2y_train = train_data[3].to(device).float()
            
            x_bc = bc_data[0].to(device).float()
            y_bc = bc_data[1].to(device).float()
            
            def closure():
                model.opt.zero_grad()
                loss = model.loss_fn(mode=mode,
                    x=x_train, y=y_train, Dy=Dy_train, x_bc=x_bc, y_bc=y_bc, Hy=D2y_train
                )
                loss.backward()
                return loss
            
            optim.step(closure)
    
        else:
        
            for step, (train_data, bc_data) in enumerate(zip(train_loader, cycle(bc_loader))):
                # Load batches from dataloaders
                x_train = train_data[0].to(device).float().requires_grad_(True)
                
                y_train = train_data[1].to(device).float()
                Dy_train = train_data[2].to(device).float()
                D2y_train = train_data[3].to(device).float()
                
                x_bc = bc_data[0].to(device).float()
                y_bc = bc_data[1].to(device).float()
                # Call zero grad on optimizer
                model.opt.zero_grad()
                
                loss = model.loss_fn(mode=mode,
                    x=x_train, y=y_train, Dy=Dy_train, x_bc=x_bc, y_bc=y_bc, Hy=D2y_train
                )
                # Backward the loss, calculate gradients
                loss.backward()
                # Optimizer step
                model.opt.step()
                # Update the learning rate scheduling
            
        # Calculate and average the loss over the val dataloader
        model.eval()
        val_loss = 0.0
        out_loss_val = 0.0
        der_loss_val = 0.0
        pde_loss_val = 0.0
        tot_loss_val = 0.0
        bc_loss_val = 0.0
        hes_loss_val = 0.0
        
        with torch.no_grad():
            for (val_data, bc_data) in zip(true_loader, cycle(bc_loader)):
                x_val = val_data[0].to(device).float().requires_grad_(True)
                y_val = val_data[1].to(device).float()
                Dy_val = val_data[2].to(device).float()
                D2y_val = val_data[3].to(device).float()

                x_bc = bc_data[0].to(device).float()
                y_bc = bc_data[1].to(device).float()
                
                step_val, out_loss, der_loss, hes_loss, pde_loss, bc_loss, tot_loss = model.eval_losses(step=step_prefix+step, mode=mode,
                                                                                        x=x_val, y=y_val, Dy=Dy_val, x_bc=x_bc, y_bc=y_bc, Hy=D2y_val)
                
                out_loss_val += out_loss.item()
                der_loss_val += der_loss.item()
                pde_loss_val += pde_loss.item()
                tot_loss_val += tot_loss.item()
                bc_loss_val += bc_loss.item()
                hes_loss_val += hes_loss.item()
                
                val_loss += tot_loss.item()
                
            val_loss /= len(true_loader)
            out_loss_val /= len(true_loader)
            der_loss_val /= len(true_loader)
            pde_loss_val /= len(true_loader)
            tot_loss_val /= len(true_loader)
            bc_loss_val /= len(true_loader)
            hes_loss_val /= len(true_loader)
    
            
        train.report({
            'step': step_prefix+step,
            'loss': val_loss,
            'out_loss': out_loss_val,
            'der_loss': der_loss_val,
            'hes_loss': hes_loss_val,
            'pde_loss': pde_loss_val,
            'bc_loss': bc_loss_val,
        })
                
                
                
                             
param_space = {
    "bc_weight": tune.loguniform(1e-3, 1e2),
    "lr_init": tune.choice([1e-3, 5e-4, 1e-4, 5e-5]),
}       

if mode != 'PINN':
    param_space['sys_weight'] = tune.loguniform(1e-3, 1e2)
if mode == 'PINN' or mode == 'PINN+Output':
    param_space['pde_weight'] = tune.loguniform(1e-3, 1e2)

from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

algo = HyperOptSearch()     
scheduler = ASHAScheduler(max_t=200, grace_period=50)
           
trainable_with_resources = tune.with_resources(train_model, {'cpu':8, 'gpu':1})
           
if os.path.exists(f'{abs_path}/tuner_results/allen_{prefix}{mode}'):
    print('Directory exists!')
    tuner = tune.Tuner.restore(f'{abs_path}/tuner_results/allen_{prefix}{mode}', trainable_with_resources, restart_errored=True)
else:
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(metric="out_loss", mode="min", search_alg=algo, scheduler=scheduler, num_samples=100, max_concurrent_trials=1),
        run_config=train.RunConfig(storage_path=os.path.join(os.path.abspath('.'), 'tuner_results'), name=f'allen_{prefix}{mode}', log_to_file=True),
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
