import os
import pandas as pd
train_results_file = os.path.join('exp', 'plotsjoint', 'train_results.txt')
val_results_file = os.path.join('exp', 'plotsjoint', 'val_results.txt')
test_results_file = os.path.join('exp', 'plotsjoint', 'test_results.txt')
with open(train_results_file, 'r') as f:
    train_results = f.readlines()
    
with open(val_results_file, 'r') as f:
    val_results = f.readlines()
    
with open(test_results_file, 'r') as f:
    test_results = f.readlines()
      

def get_df(res_lines: list[str]):
    results = [x.strip() for x in res_lines]
    results = [x for x in results if x]
    results = [x for x in results if x[-1].isdigit()]


    final_numbers = [float(line.split()[-1]) for line in results if line]

    avg_errors = final_numbers[:4]
    avg_forcing = final_numbers[4:8]

    rem_numbers = final_numbers[8:]
    lambdas = rem_numbers[::5]
    derl_errors = rem_numbers[1::5]
    outl_errors = rem_numbers[2::5]
    sob_errors = rem_numbers[3::5]
    pinn_errors = rem_numbers[4::5]

    lambdas_u = lambdas[::2]
    lambdas_f = lambdas[1::2]

    derl_u = derl_errors[::2]
    outl_u = outl_errors[::2]
    sob_u = sob_errors[::2]
    pinn_u = pinn_errors[::2]

    derl_forcing = derl_errors[1::2]
    outl_forcing = outl_errors[1::2]
    sob_forcing = sob_errors[1::2]
    pinn_forcing = pinn_errors[1::2]

    df_u = pd.DataFrame({'lambda': lambdas_u, 'derl': derl_u, 'outl': outl_u, 'sob': sob_u, 'pinn': pinn_u})
    df_f = pd.DataFrame({'lambda': lambdas_f, 'derl': derl_forcing, 'outl': outl_forcing, 'sob': sob_forcing, 'pinn': pinn_forcing})
    df_u.sort_values(by='lambda', inplace=True)
    df_f.sort_values(by='lambda', inplace=True)
    
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
df_u.sort_values(by='lambda', inplace=True)
df_f.sort_values(by='lambda', inplace=True)

import matplotlib.pyplot as plt

def plot_errors(df, title, filename):
    markers = {'train': 'o', 'val': 's', 'test': '^'}
    colors = {'derl': 'r', 'outl': 'g', 'sob': 'b', 'pinn': 'k'}
    for error_type in ['derl', 'outl', 'sob', 'pinn']:
        for data_type in df['type'].unique():
            subset = df[df['type'] == data_type]
            plt.scatter(subset['lambda'], subset[error_type], marker=markers[data_type], color=colors[error_type])
    
    # Now add lines independent of the type
    for error_type in ['derl', 'outl', 'sob', 'pinn']:
        plt.plot(df['lambda'], df[error_type], label=error_type, color=colors[error_type])
        
    # Add legend for markers
    for data_type in df['type'].unique():
        plt.scatter([], [], marker=markers[data_type], color='k', label=data_type)

    plt.xlabel('Lambda')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

plot_errors(df_u, 'Errors as a function of Lambda (U)', 'exp/plotsjoint/errors_u.png')
plot_errors(df_f, 'Errors as a function of Lambda (F)', 'exp/plotsjoint/errors_f.png')

