import torch
from torch import nn
from models.ncl import BallNCL
import argparse
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import os
from ray import tune, train

seed = 30
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)

parser = argparse.ArgumentParser()

parser.add_argument('--device', default='cuda:0', type=str, help='Device to run the code')
parser.add_argument('--name', default='student', type=str, help='Experiment name')
parser.add_argument('--mode', default='Derivative', type=str, help='Mode')
args = parser.parse_args()

device_in = args.device
name = args.name
mode = args.mode

abs_path = os.path.abspath('.')

def train_model(config):
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    EXP_PATH = f'{abs_path}/nclball'
    if not os.path.exists(EXP_PATH):
        os.mkdir(EXP_PATH)


    pde_dataset = torch.load(f'{EXP_PATH}/pdedistillation_dataset.pth')
    init_dataset = torch.load(f'{EXP_PATH}/initdistillation_dataset.pth')
    bc_dataset = torch.load(f'{EXP_PATH}/bcdistillation_dataset.pth')
    
    epochs = 10
    
    batch_size = 1000
    init_weight = config['init_weight']
    bc_weight = config['bc_weight']
    sys_weight = config['sys_weight']
    lr_init = config['lr_init']

    # Generate the dataloaders
    pde_dataloader = DataLoader(pde_dataset, batch_size, shuffle=True, num_workers=4, generator=torch.Generator().manual_seed(seed))
    init_dataloader = DataLoader(init_dataset, batch_size, shuffle=True, num_workers=4, generator=torch.Generator().manual_seed(seed))
    bc_dataloader = DataLoader(bc_dataset, batch_size, shuffle=True, num_workers=4, generator=torch.Generator().manual_seed(seed))

    device = "cpu"
    if torch.cuda.is_available():
        device = device_in
    
    model = BallNCL(hidden_units=[128 for _ in range(4)],
                sys_weight=sys_weight,
                div_weight=0,
                F_weight=0,
                init_weight=init_weight,
                bc_weight=bc_weight,
                radius=1.,
                lr=lr_init,
                activation=nn.Softplus(beta=25.),
                device=device).to(device=device)

    
    
    
    # Training mode for the network
    model.train()
    
    for epoch in range(epochs):
        model.train()
        step_prefix = epoch*len(pde_dataloader)
        print(f'Epoch: {epoch}, step_prefix: {step_prefix}')
        # Training mode for the network
        for step, (pde_data, init_data, bc_data) in enumerate(zip(pde_dataloader, init_dataloader, bc_dataloader)):
            # Load batches from dataloaders
            x_pde = pde_data[0].to(device).float().requires_grad_(True)
            y_pde = pde_data[1].to(device).float()
            Dy_pde = pde_data[2].to(device).float()
            D2y_pde = pde_data[3].to(device).float()
                    
            x_init = init_data[0].to(device).float()
            y_init = init_data[1].to(device).float()
            
            x_bc = bc_data[0].to(device).float().requires_grad_(True)
            y_bc = bc_data[1].to(device).float()
            
            model.train()
            # Call zero grad on optimizer
            model.opt.zero_grad()
            
            loss = model.student_loss_fn(mode=mode,
                x_pde=x_pde, y_pde=y_pde, Dy_pde=Dy_pde,
                x_bc=x_bc, 
                x_init=x_init, y_init=y_init, D2y_pde=D2y_pde
            )
            # Backward the loss, calculate gradients
            loss.backward()
            # Optimizer step
            model.opt.step()
            # Update the learning rate scheduling
            #model.lr_scheduler.step()
            torch.cuda.empty_cache()
            
            model.eval()
            # Printing
            with torch.no_grad():
                step_val, out_loss_val, der_loss_val, hes_loss_val, F_loss_val, div_loss_val, bc_loss_val, init_loss_val, tot_loss_val = model.eval_student_loss_fn(
                    step=step+step_prefix, x_pde=x_pde, x_bc=x_bc, x_init=x_init, y_init=y_init, y_pde=y_pde, Dy_pde=Dy_pde, mode=mode, D2y_pde=D2y_pde
                )

                #print(f'Step: {step+step_prefix}, out_loss: {out_loss_val}, der_loss:{der_loss_val}, F_loss: {F_loss_val}, div_loss: {div_loss_val}, bc_loss: {bc_loss_val}, init_loss: {init_loss_val}, tot_loss: {tot_loss_val}')        
    # Calculate and average the loss over the test dataloader
            
            
            
            train.report({
                'step': step_prefix+step,
                'loss': tot_loss_val.item(),
                'out_loss': out_loss_val.item(),
                'der_loss': der_loss_val.item(),
                'hes_loss': hes_loss_val.item(),
                'div_loss': div_loss_val.item(),
                'F_loss': F_loss_val.item(),
                'init_loss': init_loss_val.item(),
                'bc_loss': bc_loss_val.item(),
            })
                    
                
                
                             
param_space = {
    "init_weight": tune.loguniform(1e-3, 1e2),
    "sys_weight": tune.loguniform(1e-3, 1e2),
    "bc_weight": tune.loguniform(1e-3, 1e2),
    "lr_init": tune.choice([1e-3, 5e-4, 1e-4]),
}       

from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

algo = HyperOptSearch()     
scheduler = ASHAScheduler(max_t=100000, grace_period=10000)
           
trainable_with_resources = tune.with_resources(train_model, {'cpu':12, 'gpu':1})
           
tuner = tune.Tuner(
    trainable_with_resources,
    tune_config=tune.TuneConfig(metric="out_loss", mode="min", search_alg=algo, scheduler=scheduler, num_samples=100, max_concurrent_trials=1,),
    run_config=train.RunConfig(storage_path=f'{os.path.abspath('.')}/results_new', name=f'nclball_student{mode}', log_to_file=True),
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
