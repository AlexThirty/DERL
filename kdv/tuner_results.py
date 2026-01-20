kdv_best_params = {
    'PINN': {'init_weight': 6.6076, 'sys_weight': 0.18810, 'pde_weight': 0.15237, 'bc_weight': 0.0076557, 'lr_init': 0.0005},
    'Derivative': {'init_weight': 4.6174, 'sys_weight': 3.9150, 'bc_weight': 0.0071801, 'lr_init': 0.0001},
    'Output': {'init_weight': 0.0016136, 'sys_weight': 64.853, 'bc_weight': 0.011702, 'lr_init': 0.0005},
    'Derivative+Hessian': {'init_weight': 9.1558, 'sys_weight': 0.20647, 'bc_weight': 0.055457, 'lr_init': 0.0005},
    'Hessian': {'init_weight': 43.849, 'sys_weight': 0.020549, 'bc_weight': 0.0010580, 'lr_init': 0.0005},
    'Sobolev': {'init_weight': 0.29303, 'sys_weight': 0.81005, 'bc_weight': 0.0067370, 'lr_init': 0.0001},
    'Sobolev+Hessian': {'init_weight': 2.4654, 'sys_weight': 0.020428, 'bc_weight': 0.012218, 'lr_init': 0.0001},
}
