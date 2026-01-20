from ray import tune, train
import ray
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import random
import tempfile
import numpy as np
seed = 30
from model import PendulumNet
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

b = 0.

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='Derivative', type=str, help='Mode: -1 for PINN learning, 0 for derivative learning, 1 for output learning')
parser.add_argument('--device', default='cuda:0', type=str, help='Device to run the code')
parser.add_argument('--name', default='true', type=str, help='Experiment name')
args = parser.parse_args()
mode = args.mode
device_in = args.device
name = args.name

if name == 'true':
    prefix = 'true'
elif name == 'extrapolate':
    prefix = 'true_extrapolate'
elif name == 'interpolate':
    prefix = 'true_interpolate'
elif name == 'adapt':
    prefix = 'true_adapt'
elif name == 'emp':
    prefix = 'emp'
else:
    raise ValueError(f'name value is not in the options')

EXP_PATH = f'{os.path.abspath('.')}'

def train_model(config):
    #ray.train.torch.enable_reproducibility(seed=seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    print(mode)
    
    epochs = 200
    batch_size = 64
    
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
    
    # Load the data
    train_dataset = torch.load(os.path.join(EXP_PATH, 'data', f'{prefix}_dataset_train.pth'))
    val_dataset = torch.load(os.path.join(EXP_PATH, 'data', f'{prefix}_dataset_val.pth'))
    
    if name in ['adapt', 'interpolate']:
        bc_dataset = torch.load(os.path.join(EXP_PATH, 'data', f'{prefix}_bc_train.pth'))
    else:
        bc_dataset = None

    # Generate the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))

    device = "cpu"
    if torch.cuda.is_available():
        device = device_in
    
    activation = torch.nn.Tanh()
    model = PendulumNet(
        init_weight=init_weight,
        sys_weight=sys_weight,
        pde_weight=pde_weight,
        hidden_units=[50 for _ in range(4)],
        lr_init=lr_init,
        device=device,
        activation=activation,    
        last_activation=False,
    ).to(device)
    
    
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            start = checkpoint_dict["epoch"] + 1
            model.load_state_dict(checkpoint_dict["model_state"])
    
    
    # Training mode for the network
    model.train()
    
    for epoch in range(epochs):
        step_prefix = epoch*len(train_dataloader)
        print(f'Epoch: {epoch}, step_prefix: {step_prefix}')
        for step, train_data in enumerate(train_dataloader):
            # Load batches from dataloaders
            x_train = train_data[0].to(device).float()
            
            y_train = train_data[1].to(device).float()[:,0].reshape((-1,1))
            Dy_train = train_data[2].to(device).float()[:,0]
            
            if bc_dataset is not None:
                x_bc = bc_dataset[:][0].to(device).float()
                y_bc = bc_dataset[:][1].to(device).float()
            else:
                x_bc = None
                y_bc = None
            
            # Call zero grad on optimizer
            model.opt.zero_grad()
            
            loss = model.loss_fn(mode=mode,
                x=x_train, y=y_train, Dy=Dy_train, x_bc=x_bc, y_bc=y_bc
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
        init_pde_losses = []
        for step, val_data in enumerate(val_dataloader):
            x_val = val_data[0].to(device).float()
            y_val = val_data[1].to(device).float()[:,0].reshape((-1,1))
            Dy_val = val_data[2].to(device).float()[:,0]
        
            # Printing
            with torch.no_grad():
                #step_val, out_loss_train, der_loss_train, pde_loss_train, init_loss_train, tot_loss_train, init_pde_loss_train = model.eval_losses(
                #    step=step_prefix+step, mode=mode,
                #    x=x_train, y=y_train, u=u_train, x_bc=None, y_bc=None
                #)
                step_val, out_loss_val, der_loss_val, pde_loss_val, init_loss_val, tot_loss_val, init_pde_loss_val = model.eval_losses(
                    step=step_prefix+step, mode=mode,
                    x=x_val, y=y_val, Dy=Dy_val, x_bc=x_bc, y_bc=y_bc
                )
            out_losses.append(out_loss_val.item())
            der_losses.append(der_loss_val.item())
            pde_losses.append(pde_loss_val.item()) 
            init_losses.append(init_loss_val.item())
            tot_losses.append(tot_loss_val.item())
            init_pde_losses.append(init_pde_loss_val.item())
            
        out_loss_val = np.mean(out_losses)
        der_loss_val = np.mean(der_losses)
        pde_loss_val = np.mean(pde_losses)
        init_loss_val = np.mean(init_losses)
        tot_loss_val = np.mean(tot_losses)
        init_pde_loss_val = np.mean(init_pde_losses)
            
        metrics = {
            'step': step_prefix+step,
            'loss': tot_loss_val.item(),
            'out_loss': out_loss_val.item(),
            'der_loss': der_loss_val.item(),
            'pde_loss': pde_loss_val.item(),
            'init_loss': init_loss_val.item(),
            'init_pde_loss': init_pde_loss_val.item(),
        }
        with tempfile.TemporaryDirectory() as tempdir:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict()},
                os.path.join(tempdir, "checkpoint.pt"),
            )
            train.report(metrics=metrics, checkpoint=train.Checkpoint.from_directory(tempdir))
                
                
                
                             
param_space = {
    "init_weight": tune.loguniform(1e-3, 1e2),
    "lr_init": tune.choice([1e-3, 5e-4, 2e-4]),
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
           
if os.path.exists(f'{EXP_PATH}/tuner_results/pendulum_{prefix}{mode}'):
    print('Directory exists')
    tuner = tune.Tuner.restore(f'{EXP_PATH}/tuner_results/pendulum_{prefix}{mode}', trainable_with_resources)
else:          
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(metric="out_loss", mode="min", num_samples=100, max_concurrent_trials=4, search_alg=algo, scheduler=scheduler),
        run_config=train.RunConfig(storage_path=os.path.join(EXP_PATH,'tuner_results'), name=f'pendulum_{prefix}{mode}', log_to_file=True,
                                   checkpoint_config=train.CheckpointConfig(checkpoint_score_attribute="out_loss", num_to_keep=5, checkpoint_score_order='min')),
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
results_df.to_csv(f'{EXP_PATH}/tuner_results/pendulum_{prefix}{mode}.csv')