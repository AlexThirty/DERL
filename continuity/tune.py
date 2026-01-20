from ray import tune, train
import ray
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np

seed = 30
from torch.func import vmap, jacrev
from itertools import cycle

from model import DensityNet
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='Derivative', type=str, help='Mode: -1 for PINN learning, 0 for derivative learning, 1 for output learning')
parser.add_argument('--device', default='cuda:0', type=str, help='Device to run the code')
parser.add_argument('--name', default='grid', type=str, help='Experiment name')
args = parser.parse_args()
mode = args.mode
device_in = args.device
name = args.name


if name == 'full':
    prefix = 'rho_full'
elif name == 'grid':
    prefix = 'rho_grid'
elif name == 'extrapolate':
    prefix = 'rho_extrapolate'
elif name == 'interpolate':
    prefix = 'rho_interpolate'
elif name == 'adapt':
    prefix = 'rho_adapt'
else:
    raise ValueError(f'name value is not in the options')

EXP_PATH = f'{os.path.abspath('.')}'

def train_model(config):
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    epochs = 200
    
    init_weight = config['init_weight']
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
    if name == 'grid':
        batch_size = 128
    elif name == 'full':
        batch_size = 128
    
    # Load the data
    if name in ['adapt', 'interpolate', 'extrapolate']:
        train_dataset = torch.load(os.path.join(EXP_PATH, f'data/{prefix}_dataset_train.pth'))
        test_dataset = torch.load(os.path.join(EXP_PATH, f'data/{prefix}_dataset_test.pth'))
    # If on the full data, there is not test set
    else:
        train_dataset = test_dataset = torch.load(os.path.join(EXP_PATH, f'data/{prefix}_dataset.pth'))

    init_dataset = torch.load(os.path.join(EXP_PATH, f'data/rho_init_dataset.pth'))
    bc_dataset = torch.load(os.path.join(EXP_PATH, f'data/rho_bc_dataset.pth'))

    if name in ['adapt', 'interpolate', 'extrapolate']:
        extra_dataset = torch.load(os.path.join(EXP_PATH, f'data/{prefix}_dataset_extra.pth'))
    
    # Generate the dataloader
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, generator=torch.Generator().manual_seed(seed))
    val_dataloader = DataLoader(train_dataset, 256, shuffle=True, num_workers=4, generator=torch.Generator().manual_seed(seed))
    bc_dataloader = DataLoader(bc_dataset, batch_size, shuffle=True, num_workers=4, generator=torch.Generator().manual_seed(seed))
    init_dataloader = DataLoader(init_dataset, batch_size, shuffle=True, num_workers=4, generator=torch.Generator().manual_seed(seed))
    print('Data loaded!')

    device = "cpu"
    if torch.cuda.is_available():
        device = device_in
    
    activation = torch.nn.Tanh()
    model = DensityNet(
        init_weight=init_weight,
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
        step_prefix = epoch*len(train_dataloader)
        print(f'Epoch: {epoch}, step_prefix: {step_prefix}')
        for step, (train_data, init_data, bc_data) in enumerate(zip(train_dataloader, cycle(init_dataloader), cycle(bc_dataloader))):
            # Load batches from dataloaders
            # Load batches from dataloaders
            x_train = train_data[0].to(device).float()
            
            y_train = train_data[1].to(device).float()
            Dy_train = train_data[2].to(device).float()
            
            x_init = init_data[0].to(device).float()
            y_init = init_data[1].to(device).float()
            
            x_bc = bc_data[0].to(device).float()
            y_bc = bc_data[1].to(device).float()
            
            
            #if bc_dataset is not None:
            #    x_bc = bc_dataset[:][0].to(device).float()
            #    y_bc = bc_dataset[:][1].to(device).float()
            #else:
            #    x_bc = None
            #    y_bc = None
            
            # Call zero grad on optimizer
            model.opt.zero_grad()
            
            loss = model.loss_fn(mode=mode,
                x=x_train, y=y_train, Dy=Dy_train, x_bc=x_bc, y_bc=y_bc, x_init=x_init, y_init=y_init
            )
            # Backward the loss, calculate gradients
            loss.backward()
            # Optimizer step
            model.opt.step()
            # Update the learning rate scheduling
        
        out_losses = []
        der_losses = []
        pde_losses = []
        init_losses = []
        tot_losses = []
        bc_losses = []
        for step, (val_data, init_data, bc_data) in enumerate(zip(val_dataloader, cycle(init_dataloader), cycle(bc_dataloader))):
            x_val = val_data[0].to(device).float()
            y_val = val_data[1].to(device).float()
            Dy_val = val_data[2].to(device).float()
            
            x_init = init_data[0].to(device).float()
            y_init = init_data[1].to(device).float()
            
            x_bc = bc_data[0].to(device).float()
            y_bc = bc_data[1].to(device).float()
        
            # Printing
            with torch.no_grad():
                #step_val, out_loss_train, der_loss_train, pde_loss_train, init_loss_train, tot_loss_train, bc_loss_train = model.eval_losses(
                #    step=step_prefix+step, mode=mode,
                #    x=x_train, y=y_train, u=u_train, x_bc=None, y_bc=None
                #)
                step_val, out_loss_val, der_loss_val, pde_loss_val, init_loss_val, tot_loss_val, bc_loss_val = model.eval_losses(
                    step=step_prefix+step, mode=mode,
                    x=x_val, y=y_val, Dy=Dy_val, x_bc=x_bc, y_bc=y_bc, x_init=x_init, y_init=y_init
                )
            out_losses.append(out_loss_val.item())
            der_losses.append(der_loss_val.item())
            pde_losses.append(pde_loss_val.item()) 
            init_losses.append(init_loss_val.item())
            tot_losses.append(tot_loss_val.item())
            bc_losses.append(bc_loss_val.item())
            
        out_loss_val = np.mean(out_losses)
        der_loss_val = np.mean(der_losses)
        pde_loss_val = np.mean(pde_losses)
        init_loss_val = np.mean(init_losses)
        tot_loss_val = np.mean(tot_losses)
        bc_loss_val = np.mean(bc_losses)
            
        train.report({
            'step': step_prefix+step,
            'loss': tot_loss_val.item(),
            'out_loss': out_loss_val.item(),
            'der_loss': der_loss_val.item(),
            'pde_loss': pde_loss_val.item(),
            'init_loss': init_loss_val.item(),
            'bc_loss': bc_loss_val.item(),
        })
                
                
                
                             
param_space = {
    "init_weight": tune.loguniform(1e-3, 1e2),
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
scheduler = ASHAScheduler(max_t=200, grace_period=50)
           
trainable_with_resources = tune.with_resources(train_model, {'cpu':12, 'gpu':0.25})

if os.path.exists(f'{EXP_PATH}/tuner_results/continuity_{prefix}{mode}'):
    tuner = tune.Tuner.restore(f'{EXP_PATH}/tuner_results/continuity_{prefix}{mode}', trainable_with_resources)

else:
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(metric="out_loss", mode="min", search_alg=algo, scheduler=scheduler, num_samples=100, max_concurrent_trials=4,),
        run_config=train.RunConfig(storage_path=f'{EXP_PATH}/tuner_results', name=f'continuity_{prefix}{mode}', log_to_file=True),
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
results_df.to_csv(f'{EXP_PATH}/tuner_results/continuity_{prefix}{mode}.csv')