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

from models.dualpinn import DualPINN
from models.multiphysics import MultiPhysics
from models.pinn import PINN
from models.ncl import NCL
from tuning_utils.model_configs import *
from tuning_utils.train import train_loop, load_data
#from train_dualpinn import train_loop as train_loop_dualpinn
#from train_ncl import train_loop as train_loop_ncl

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='Derivative', type=str, help='Mode: -1 for PINN learning, 0 for derivative learning, 1 for output learning')
parser.add_argument('--model', default='multiphysics', type=str, help='Model to train')
parser.add_argument('--device', default='cuda:0', type=str, help='Device to run the code')

args = parser.parse_args()
mode = args.mode
device_in = args.device
model = args.model


EXP_PATH = os.path.abspath('.')

def run_experiment(config):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    
    if model == 'multiphysics':
        model_instance = MultiPhysics(
            bc_weight=config['bc_weight'],
            mom_weight=config['mom_weight'],
            discrepancy_weight=config['discrepancy_weight'],
            device='cuda',
            mom_hidden_units=[64 for i in range(4)],
            div_hidden_units=[64 for i in range(4)],
        )
        
    if model == 'pinn':
        model_instance = PINN(
            bc_weight=config['bc_weight'],
            mom_weight=config['mom_weight'],
            div_weight=config['div_weight'],
            device='cuda',
            hidden_units=[64 for i in range(4)],
        )
        
    if model == 'dualpinn':
        model_instance = DualPINN(
            bc_weight=config['bc_weight'],
            mom_weight=config['mom_weight'],
            div_weight=config['div_weight'],
            discrepancy_weight=config['discrepancy_weight'],
            device='cuda',
            div_hidden_units=[64 for i in range(4)],
            mom_hidden_units=[64 for i in range(4)],
        )
        #train_loop = train_loop_dualpinn
    
    if config['model'] == 'ncl':
        model_instance = NCL(
            bc_weight=config['bc_weight'],
            mom_weight=config['mom_weight'],
            device='cuda',
            div_hidden_units=[64 for i in range(4)],
        )
    
    lr_init = config['lr_init']
    config['epochs']
    
    optimizer = torch.optim.Adam(model_instance.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.5)
    
    internal_dataset, boundary_dataset = load_data(config['abs_path'])
    
    internal_dataloader = DataLoader(internal_dataset, batch_size=batch_size, shuffle=True, generator=gen)
    boundary_dataloader = DataLoader(boundary_dataset, batch_size=batch_size, shuffle=True, generator=gen)
    
    train_loop(
        epochs=epochs,
        model=model_instance,
        optimizer=optimizer,
        scheduler=scheduler,
        internal_dataloader=internal_dataloader,
        boundary_dataloader=boundary_dataloader,
    )


batch_size = 1000
epochs = 500
from models.params import x_min, x_max, y_min, y_max
from itertools import product


os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
data_path = os.path.abspath('./data')
                                  
         
if model == 'ncl':
    param_space = ncl_config

if model == 'pinn':
    param_space = pinn_config

if model == 'dualpinn':
    param_space = dualpinn_config

if model == 'multiphysics':
    param_space = multiphysics_config  

def count_combinations(param_space):
    keys, values = zip(*param_space.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    return len(combinations)

total_combinations = count_combinations(param_space)
print(f'Total number of combinations: {total_combinations}')

trainable_with_resources = tune.with_resources(run_experiment, resources={'cpu': 4, 'gpu': 1/3})
param_space.update({'epochs': epochs,
                    'model': model,
                    'abs_path': data_path,
                   })


from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

algo = HyperOptSearch()     
scheduler = ASHAScheduler(max_t=epochs, grace_period=100)

model_path = f'{model}'
if not model in ['ncl', 'pinn']:
    model_path += f'_{mode}'
results_path = f'{EXP_PATH}/tuner_results/{model_path}'
               
if os.path.exists(results_path):
    tuner = tune.Tuner.restore(results_path, trainable_with_resources, restart_errored=True)

else:
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(metric="loss", mode="min", num_samples=200, max_concurrent_trials=6, scheduler=scheduler, search_alg=algo),
        run_config=train.RunConfig(storage_path=f'{EXP_PATH}/tuner_results', name=model_path, log_to_file=True),
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
results_df = results_df.sort_values(by='loss')
results_df.to_csv(f'{EXP_PATH}/tuner_results/{model_path}.csv')