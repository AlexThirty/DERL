continuity_best_params = {
    'grid':{
        'PINN': {'init_weight': 1.2027, 'bc_weight': 0.0028335, 'lr_init': 0.001, 'pde_weight': 3.4905},
        'Derivative': {'init_weight': 8.8612, 'bc_weight': 0.0010685, 'lr_init': 0.0005, 'sys_weight': 9.9791},
        'Sobolev': {'init_weight': 0.0023772, 'bc_weight': 0.30942, 'lr_init': 0.001, 'sys_weight': 16.943},
        'Output': {'init_weight': 8.9267, 'bc_weight': 0.0010879, 'lr_init': 0.001, 'sys_weight': 31.593},
        'PINN+Output': {'init_weight': 0.61519, 'bc_weight': 0.014213, 'lr_init': 0.0005, 'sys_weight': 15.859, 'pde_weight': 0.33899}
    },
    'full':{
        'PINN': {'init_weight': 1.6322, 'bc_weight': 0.0090753, 'lr_init': 0.001, 'pde_weight': 4.3266},
        'Derivative': {'init_weight': 12.484, 'bc_weight': 0.0026753, 'lr_init': 0.001, 'sys_weight': 0.73327},
        'Sobolev': {'init_weight': 2.2939, 'bc_weight': 0.028245, 'lr_init': 0.001, 'sys_weight': 4.7580},
        'Output': {'init_weight': 12.569, 'bc_weight': 0.0073021, 'lr_init': 0.001, 'sys_weight': 30.313},
        'PINN+Output': {'init_weight': 92.265, 'bc_weight': 0.13429, 'lr_init': 0.0005, 'sys_weight': 71.710, 'pde_weight': 0.52159}
    }    
}