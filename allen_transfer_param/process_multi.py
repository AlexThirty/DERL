import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import numpy as np
import sys

import argparse

parser = argparse.ArgumentParser(description='Process results of the Allen-Cahn Multi Parametric equation.')
parser.add_argument('--exp', type=str, default='new', help='Experiment name')
exp = parser.parse_args().exp

train_results_file = os.path.join(exp, 'plotsjoint', 'train_results.txt')
val_results_file = os.path.join(exp, 'plotsjoint', 'val_results.txt')
test_results_file = os.path.join(exp, 'plotsjoint', 'test_results.txt')
with open(train_results_file, 'r') as f:
    train_results = f.readlines()
    
with open(val_results_file, 'r') as f:
    val_results = f.readlines()
    
with open(test_results_file, 'r') as f:
    test_results = f.readlines()
    
    
def get_df(res_lines: list[str]):
    results = [x.strip() for x in res_lines]
    results = [x for x in results if x]
    results_nums = [x for x in results if x[-1].isdigit()]
    results_lams = [x.split('[')[1].strip(']').split() for x in results if '[' in x]
    results_lams = [list(map(float, x)) for x in results_lams]
    final_numbers = [float(line.split()[-1]) for line in results_nums if line]

    avg_errors = final_numbers[:4]
    avg_forcing = final_numbers[4:8]

    rem_numbers = final_numbers[8:]
    derl_errors = rem_numbers[::4]
    outl_errors = rem_numbers[1::4]
    sob_errors = rem_numbers[2::4]
    pinn_errors = rem_numbers[3::4]

    lambdas_u = results_lams[::2]
    lambdas_f = results_lams[1::2]
    
    
    derl_u = derl_errors[::2]
    outl_u = outl_errors[::2]
    sob_u = sob_errors[::2]
    pinn_u = pinn_errors[::2]

    derl_forcing = derl_errors[1::2]
    outl_forcing = outl_errors[1::2]
    sob_forcing = sob_errors[1::2]
    pinn_forcing = pinn_errors[1::2]

    df_u = pd.DataFrame({'lambda_1': [x[0] for x in lambdas_u], 'lambda_2': [x[1] for x in lambdas_u], 'derl': derl_u, 'outl': outl_u, 'sob': sob_u, 'pinn': pinn_u})
    df_f = pd.DataFrame({'lambda_1': [x[0] for x in lambdas_f], 'lambda_2': [x[1] for x in lambdas_f], 'derl': derl_forcing, 'outl': outl_forcing, 'sob': sob_forcing, 'pinn': pinn_forcing})
    df_u.sort_values(by=['lambda_1', 'lambda_2'], inplace=True)
    df_f.sort_values(by=['lambda_1', 'lambda_2'], inplace=True)
    
    return df_u, df_f


train_df_u, train_df_f = get_df(train_results)
val_df_u, val_df_f = get_df(val_results)
test_df_u, test_df_f = get_df(test_results)

train_df_u['type'] = 'train'
val_df_u['type'] = 'val'
test_df_u['type'] = 'test'

train_df_f['type'] = 'train'
val_df_f['type'] = 'val'
test_df_f['type'] = 'test'

df_u = pd.concat([train_df_u, val_df_u, test_df_u])
df_f = pd.concat([train_df_f, val_df_f, test_df_f])
df_u.sort_values(by=['lambda_1', 'lambda_2'], inplace=True)
df_f.sort_values(by=['lambda_1', 'lambda_2'], inplace=True)

print(df_u)

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create grid data for lambda_1 and lambda_2
lambda_1 = np.linspace(df_u['lambda_1'].min(), df_u['lambda_1'].max(), 100)
lambda_2 = np.linspace(df_u['lambda_2'].min(), df_u['lambda_2'].max(), 100)
lambda_1, lambda_2 = np.meshgrid(lambda_1, lambda_2)

# Interpolate data for each method
derl = griddata((df_u['lambda_1'], df_u['lambda_2']), df_u['derl'], (lambda_1, lambda_2), method='cubic')
outl = griddata((df_u['lambda_1'], df_u['lambda_2']), df_u['outl'], (lambda_1, lambda_2), method='cubic')
sob = griddata((df_u['lambda_1'], df_u['lambda_2']), df_u['sob'], (lambda_1, lambda_2), method='cubic')
pinn = griddata((df_u['lambda_1'], df_u['lambda_2']), df_u['pinn'], (lambda_1, lambda_2), method='cubic')

# Plot surfaces with log scale
ax.plot_surface(lambda_1, lambda_2, np.log10(derl), label='derl', alpha=0.5)
ax.plot_surface(lambda_1, lambda_2, np.log10(outl), label='outl', alpha=0.5)
ax.plot_surface(lambda_1, lambda_2, np.log10(sob), label='sob', alpha=0.5)
ax.plot_surface(lambda_1, lambda_2, np.log10(pinn), label='pinn', alpha=0.5)

ax.set_xlabel('lambda_1')
ax.set_ylabel('lambda_2')
ax.set_zlabel('Error on u')
ax.legend()

plt.savefig(f'{exp}/plotsjoint/errors_u.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create grid data for lambda_1 and lambda_2
lambda_1 = np.linspace(df_f['lambda_1'].min(), df_f['lambda_1'].max(), 100)
lambda_2 = np.linspace(df_f['lambda_2'].min(), df_f['lambda_2'].max(), 100)
lambda_1, lambda_2 = np.meshgrid(lambda_1, lambda_2)

# Interpolate data for each method
derl = griddata((df_f['lambda_1'], df_f['lambda_2']), df_f['derl'], (lambda_1, lambda_2), method='cubic')
outl = griddata((df_f['lambda_1'], df_f['lambda_2']), df_f['outl'], (lambda_1, lambda_2), method='cubic')
sob = griddata((df_f['lambda_1'], df_f['lambda_2']), df_f['sob'], (lambda_1, lambda_2), method='cubic') 
pinn = griddata((df_f['lambda_1'], df_f['lambda_2']), df_f['pinn'], (lambda_1, lambda_2), method='cubic')

# Plot surfaces with log scale
ax.plot_surface(lambda_1, lambda_2, np.log10(derl), label='derl', alpha=0.5)
ax.plot_surface(lambda_1, lambda_2, np.log10(outl), label='outl', alpha=0.5)
ax.plot_surface(lambda_1, lambda_2, np.log10(sob), label='sob', alpha=0.5)
ax.plot_surface(lambda_1, lambda_2, np.log10(pinn), label='pinn', alpha=0.5)

ax.set_xlabel('lambda_1')
ax.set_ylabel('lambda_2') 
ax.set_zlabel('Error on f')
ax.legend()

plt.savefig(f'{exp}/plotsjoint/errors_f.png')

# Determine the model with the lowest error for each square in the lambda_1, lambda_2 domain
def get_best_model(df):
    df['best_model'] = df[['derl', 'outl', 'sob', 'pinn']].idxmin(axis=1)
    return df

df_u = get_best_model(df_u)
df_f = get_best_model(df_f)

model_names = ['derl', 'outl', 'sob', 'pinn']

fig, ax = plt.subplots()

# Create grid data for lambda_1 and lambda_2
lambda_1 = np.linspace(df_u['lambda_1'].min(), df_u['lambda_1'].max(), 100)
lambda_2 = np.linspace(df_u['lambda_2'].min(), df_u['lambda_2'].max(), 100)
lambda_1, lambda_2 = np.meshgrid(lambda_1, lambda_2)

# Interpolate best model data
best_model_u = griddata((df_u['lambda_1'], df_u['lambda_2']), df_u['best_model'].astype('category').cat.codes, (lambda_1, lambda_2), method='nearest')

# Plot the best model regions
c = ax.pcolormesh(lambda_1, lambda_2, best_model_u, cmap='viridis', shading='auto', vmin=0, vmax=len(model_names)-1)
fig.colorbar(c, ax=ax, ticks=range(len(model_names)), label='Best Model').ax.set_yticklabels(model_names)

ax.set_xlabel('lambda_1')
ax.set_ylabel('lambda_2')
ax.set_title('Best Model Regions for Error on u')

plt.savefig(f'{exp}/plotsjoint/best_model_u.png')

fig, ax = plt.subplots()

# Interpolate best model data
best_model_f = griddata((df_f['lambda_1'], df_f['lambda_2']), df_f['best_model'].astype('category').cat.codes, (lambda_1, lambda_2), method='nearest')

# Plot the best model regions
c = ax.pcolormesh(lambda_1, lambda_2, best_model_f, cmap='viridis', shading='auto', vmin=0, vmax=len(model_names)-1)
fig.colorbar(c, ax=ax, ticks=range(len(model_names)), label='Best Model').ax.set_yticklabels(model_names)

ax.set_xlabel('lambda_1')
ax.set_ylabel('lambda_2')
ax.set_title('Best Model Regions for Error on f')

plt.savefig(f'{exp}/plotsjoint/best_model_f.png')