dns_config = {
    'two_step': {
        'lambda_fixed': 0.0609,
        'n_factors': 3
    },
    'one_step': {
        'lambda_fixed': None,  
        'n_factors': 3
    }
}

dnss_config = {
    'two_step': {
        'lambda1_fixed': 0.0609,  
        'lambda2_fixed': 0.0292,  
        'n_factors': 4
    },
    'one_step': {
        'lambda1_fixed': None,  
        'lambda2_fixed': None,  
        'n_factors': 4
    }
}

nnss_config = {
    'n_factors': 4,
    'n_hidden': 300,
    'IG_a': 0.1,            
    'IG_b': 0.001,          
    'minnesota_lambda': 0.5,
    'minnesota_gamma': 0.9, 
    'nn_prior_var': 0.05,   
    'learning_rates': {
        'basis_learner': 1e-3,
        'transition_matrix': 1e-3,
        'transition_covariance': 1e-4,
        'observation_covariance': 5e-5
    },
    'n_epochs': 500,
    'batch_size': None,      
    'use_gpu': False
}


dns_grid = {
    'lambda_values': [0.0609, 0.0509, 0.0709, 0.0409, 0.0809]
}

dnss_grid = {
    'lambda1_values': [0.0609, 0.0509, 0.0709],
    'lambda2_values': [0.0292, 0.0192, 0.0392]
}

nnss_grid = {
    'IG_a_values': [0.05, 0.1, 0.2],
    'IG_b_values': [0.0005, 0.001, 0.002],
    'minnesota_lambda_values': [0.3, 0.5, 0.7],
    'minnesota_gamma_values': [0.5, 0.7, 0.9],
    'nn_prior_var_values': [0.01, 0.05, 0.1]
}