from ray import tune, train
import ray
import os
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from torch import nn

seed = 30
from itertools import cycle

from model import KdVPINN, SinActivation
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--device', default='cuda:0', type=str, help='Device to run the code')
parser.add_argument('--name', default='pinn', type=str, help='Experiment name')
args = parser.parse_args()
device_in = args.device
name = args.name

if name == 'pinn':
    prefix = 'pinn'
elif name == 'student':
    prefix = 'student'
else:
    raise ValueError(f'name value is not in the options')

abs_path = os.path.abspath('.')

def train_model(config):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    epochs = 200
    
    
    
    batch_size = 64
    init_weight = config['init_weight']
    bc_weight = config['bc_weight']
    pde_weight = config['pde_weight']
    lr_init = config['lr_init']
    sys_weight = config['sys_weight']
    
    # Load the data
    if name == 'pinn':
        train_dataset = torch.load(os.path.join(abs_path, 'data', f'kdv_pde_dataset.pth'))
        init_dataset = torch.load(os.path.join(abs_path, 'data', f'kdv_init_dataset.pth'))
        bc1_dataset = torch.load(os.path.join(abs_path, 'data', f'kdv_bc1_dataset.pth'))
        bc2_dataset = torch.load(os.path.join(abs_path, 'data', f'kdv_bc2_dataset.pth'))

    # Generate the dataloaders
    pde_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=12)    
    init_dataloader = DataLoader(init_dataset, batch_size, shuffle=True, num_workers=12)
    bc1_dataloader = DataLoader(bc1_dataset, batch_size, shuffle=True, num_workers=12)
    bc2_dataloader = DataLoader(bc2_dataset, batch_size, shuffle=True, num_workers=12)
    test_dataloader = DataLoader(train_dataset, 512, shuffle=True, num_workers=12)

    device = "cpu"
    if torch.cuda.is_available():
        device = device_in
    
    activation = nn.Tanh()
    model = KdVPINN(
        init_weight=init_weight,
        sys_weight=sys_weight,
        pde_weight=pde_weight,
        bc_weight=bc_weight,
        hidden_units=[50 for _ in range(9)],
        lr=lr_init,
        device=device,
        activation=activation,    
        last_activation=False,
    ).to(device)
    
    
    
    # Training mode for the network
    model.train()
    
    for epoch in range(epochs):
        step_prefix = epoch*len(pde_dataloader)
        model.train()
        for step, (pde_data, init_data, bc1_data, bc2_data) in enumerate(zip(test_dataloader, cycle(init_dataloader), cycle(bc1_dataloader), cycle(bc2_dataloader))):
            # Load batches from dataloaders
            x_pde = pde_data[0].to(device).float().requires_grad_(True)
            y_pde = pde_data[1].to(device).float()
            
            x_init = init_data[0].to(device).float()
            y_init = init_data[1].to(device).float()
            
            x_bc1 = bc1_data[0].to(device).float().requires_grad_(True)
            y_bc1 = bc1_data[1].to(device).float()
            
            x_bc2 = bc2_data[0].to(device).float().requires_grad_(True)
            y_bc2 = bc2_data[1].to(device).float()
            
            # Call zero grad on optimizer
            model.opt.zero_grad()
            
            loss = model.loss_fn(
                x_pde=x_pde, x_bc1=x_bc1, x_bc2=x_bc2, x_init=x_init, y_init=y_init
            )
            # Backward the loss, calculate gradients
            loss.backward()
            # Optimizer step
            model.opt.step()
            # Update the learning rate scheduling
            

        
        model.eval()
        test_loss = 0.0
        out_loss_test = 0.0
        pde_loss_test = 0.0
        init_loss_test = 0.0
        tot_loss_test = 0.0
        bc_loss_test = 0.0
        
        with torch.no_grad():
            for (pde_data, init_data, bc1_data, bc2_data) in zip(pde_dataloader, cycle(init_dataloader), cycle(bc1_dataloader), cycle(bc2_dataloader)):
                # Load batches from dataloaders
                x_pde = pde_data[0].to(device).float().requires_grad_(True)
                y_pde = pde_data[1].to(device).float()
                
                x_init = init_data[0].to(device).float()
                y_init = init_data[1].to(device).float()
                
                x_bc1 = bc1_data[0].to(device).float().requires_grad_(True)
                y_bc1 = bc1_data[1].to(device).float()
                
                x_bc2 = bc2_data[0].to(device).float().requires_grad_(True)
                y_bc2 = bc2_data[1].to(device).float()
                
                step_test, out_loss, pde_loss, init_loss, bc_loss, tot_loss = model.eval_losses(step=step_prefix+step,
                    x_pde=x_pde, y_pde=y_pde, x_bc1=x_bc1, x_bc2=x_bc2, x_init=x_init, y_init=y_init, print_to_screen=False
                )
                
                out_loss_test += out_loss.item()
                pde_loss_test += pde_loss.item()
                init_loss_test += init_loss.item()
                tot_loss_test += tot_loss.item()
                bc_loss_test += bc_loss.item()
                
                test_loss += tot_loss.item()
                
            test_loss /= len(pde_dataloader)
            out_loss_test /= len(pde_dataloader)
            pde_loss_test /= len(pde_dataloader)
            init_loss_test /= len(pde_dataloader)
            tot_loss_test /= len(pde_dataloader)
            bc_loss_test /= len(pde_dataloader)

            
        train.report({
            'step': step_prefix+step,
            'loss': tot_loss_test,
            'out_loss': out_loss_test,
            'pde_loss': pde_loss_test,
            'init_loss': init_loss_test,
            'bc_loss': bc_loss_test,
            'use_loss': out_loss_test+pde_loss_test
        })
                
                
                
                             
param_space = {
    "init_weight": tune.loguniform(1e-3, 1e2),
    "sys_weight": tune.loguniform(1e-3, 1e2),
    "pde_weight": tune.loguniform(1e-3, 1e2),
    "bc_weight": tune.loguniform(1e-3, 1e2),
    "lr_init": tune.choice([5e-4, 1e-4, 5e-5]),   
}       

from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

algo = HyperOptSearch()     
scheduler = ASHAScheduler(max_t=200, grace_period=50)
           
trainable_with_resources = tune.with_resources(train_model, {'cpu':12, 'gpu':0.25})

from os.path import abspath

if os.path.exists(os.path.join(abs_path, 'tuner_results', f'kdv_pinn')):
    print('Directory exists!')
    tuner = tune.Tuner.restore(os.path.join(abs_path, 'tuner_results', f'kdv_pinn'))  
else:
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(metric="out_loss", mode="min", search_alg=algo, scheduler=scheduler, num_samples=100, max_concurrent_trials=4,),
        run_config=train.RunConfig(storage_path=os.path.join(abs_path, 'tuner_results'), name=f'kdv_pinn', log_to_file=True),
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
results_df.to_csv(f'{abs_path}/tuner_results/kdv_pinn.csv')