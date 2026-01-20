allen_best_params = {
    'grid':{
        'PINN': {'bc_weight': 7.7816, 'lr_init': 0.001, 'pde_weight': 9.2973},
        'Derivative': {'bc_weight': 0.10498, 'lr_init': 0.001, 'sys_weight': 0.22816},
        'Sobolev': {'bc_weight': 0.0012448, 'lr_init': 0.0001, 'sys_weight': 51.338},
        'Output': {'bc_weight': 0.064092, 'lr_init': 0.001, 'sys_weight': 0.74804},
        'Hessian': {'bc_weight': 0.14997, 'lr_init': 0.0005, 'sys_weight': 0.0010124},
        'Derivative+Hessian': {'bc_weight': 1.3394, 'lr_init': 0.001, 'sys_weight': 0.0049381},
        'Sobolev+Hessian': {'bc_weight': 0.13458, 'lr_init': 0.0001, 'sys_weight': 0.037961},
        'PINN+Output': {'bc_weight': 0.85010, 'lr_init': 0.001, 'sys_weight': 12.499, 'pde_weight': 21.711}
    },
    'rand':{
        'PINN': {'bc_weight': 0.7706726054765473, 'lr_init': 0.001, 'pde_weight': 0.9858565485427153},#{'bc_weight': 93.963, 'lr_init': 0.0005, 'pde_weight': 94.385},
        'Derivative': {'bc_weight': 0.10820988332774353, 'lr_init': 0.001, 'sys_weight': 100.},#{'bc_weight': 10.242588675911106, 'lr_init': 0.0005, 'sys_weight': 27.441787089175453},#{'bc_weight': 0.011445, 'lr_init': 0.0005, 'sys_weight': 0.012407},
        'Sobolev': {'bc_weight': 0.5526231085702442, 'lr_init': 0.001, 'sys_weight': 99.35496420589203},#{'bc_weight': 8.8341, 'lr_init': 0.0005, 'sys_weight': 11.288},
        'Output': {'bc_weight': 0.2661364361247872, 'lr_init': 0.001, 'sys_weight': 0.6991614945315253},#{'bc_weight': 0.020645, 'lr_init': 0.0005, 'sys_weight': 0.21809},
        'Hessian': {'bc_weight': 0.32979, 'lr_init': 0.001, 'sys_weight': 0.0026883},
        'PINN+Output': {'bc_weight': 0.023691562319359746, 'lr_init': 0.001, 'sys_weight': 0.9489899687179719, 'pde_weight': 1.8585141698256284},
        'Derivative+Hessian': {'bc_weight': 19.063, 'lr_init': 0.001, 'sys_weight': 0.13547},
        'Sobolev+Hessian': {'bc_weight': 7.6536, 'lr_init': 0.0001, 'sys_weight': 1.0317}
    }
}