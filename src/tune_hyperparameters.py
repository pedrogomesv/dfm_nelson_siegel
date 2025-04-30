import os
import numpy as np
import pandas as pd
import time
import json
import itertools
from pathlib import Path
import matplotlib.pyplot as plt

from src.models.baselines.dns import DynamicNelsonSiegelModel
from src.models.baselines.dnss import DynamicNelsonSiegelSvenssonModel
from src.models.baselines.nnss import NNSSModel
from src.config.hyperparameters import dns_grid, dnss_grid, nnss_grid

def load_yield_data(file_path):
    """
    Carrega os dados da curva de juros de um arquivo.
    
    Args:
        file_path: Caminho para o arquivo de dados
        
    Retorna:
        DataFrame com os dados da curva de juros
    """
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    
    return df

def tune_dns_model(train_data, val_data, horizons, output_dir):
    """
    Otimiza os hiperparâmetros do modelo DNS.
    
    Args:
        train_data: Dados de treinamento
        val_data: Dados de validação
        horizons: Lista de horizontes de previsão
        output_dir: Diretório para salvar os resultados
        
    Retorna:
        Melhores hiperparâmetros
    """
    maturities = [int(col) for col in train_data.columns]
    methods = ['two-step', 'one-step']
    
    results = []
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Tuning DNS model with {method} method")
        print(f"{'='*50}")
        
        for lambda_value in dns_grid['lambda_values']:
            model = DynamicNelsonSiegelModel(
                maturities=maturities,
                method=method,
                lambda_fixed=lambda_value
            )
            
            try:
                start_time = time.time()
                model.fit(train_data)
                training_time = time.time() - start_time
                
                eval_results = model.evaluate(val_data, horizons)
                
                for horizon in horizons:
                    avg_rmse = eval_results[eval_results['horizon'] == horizon][eval_results['maturity'] == 'avg']['rmse'].values[0]
                    
                    results.append({
                        'model': 'DNS',
                        'method': method,
                        'lambda': lambda_value,
                        'horizon': horizon,
                        'rmse': avg_rmse,
                        'training_time': training_time
                    })
                    
                print(f"  λ = {lambda_value:.4f}, Method = {method}, RMSE (60d) = {avg_rmse:.2f} bps")
                
            except Exception as e:
                print(f"Error with λ = {lambda_value}, Method = {method}: {str(e)}")
    
    results_df = pd.DataFrame(results)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(output_path / 'dns_tuning_results.csv', index=False)
    
    best_configs = []
    
    for horizon in horizons:
        horizon_results = results_df[results_df['horizon'] == horizon]
        best_idx = horizon_results['rmse'].idxmin()
        best_config = horizon_results.loc[best_idx]
        
        best_configs.append(best_config)
        
        print(f"\nBest DNS configuration for horizon {horizon}:")
        print(f"  Method: {best_config['method']}")
        print(f"  Lambda: {best_config['lambda']:.4f}")
        print(f"  RMSE: {best_config['rmse']:.2f} bps")
    
    best_configs_df = pd.DataFrame(best_configs)
    best_configs_df.to_csv(output_path / 'dns_best_configs.csv', index=False)
    
    return best_configs_df

def tune_dnss_model(train_data, val_data, horizons, output_dir):
    """
    Otimiza os hiperparâmetros do modelo DNSS.
    
    Args:
        train_data: Dados de treinamento
        val_data: Dados de validação
        horizons: Lista de horizontes de previsão
        output_dir: Diretório para salvar os resultados
        
    Retorna:
        Melhores hiperparâmetros
    """
    maturities = [int(col) for col in train_data.columns]
    methods = ['two-step', 'one-step']
    
    results = []
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Tuning DNSS model with {method} method")
        print(f"{'='*50}")
        
        for lambda1 in dnss_grid['lambda1_values']:
            for lambda2 in dnss_grid['lambda2_values']:
                if abs(lambda1 - lambda2) < 0.01:
                    continue
                
                model = DynamicNelsonSiegelSvenssonModel(
                    maturities=maturities,
                    method=method,
                    lambda1_fixed=lambda1,
                    lambda2_fixed=lambda2
                )
                
                try:
                    start_time = time.time()
                    model.fit(train_data)
                    training_time = time.time() - start_time
                    
                    eval_results = model.evaluate(val_data, horizons)
                    
                    for horizon in horizons:
                        avg_rmse = eval_results[eval_results['horizon'] == horizon][eval_results['maturity'] == 'avg']['rmse'].values[0]
                        
                        results.append({
                            'model': 'DNSS',
                            'method': method,
                            'lambda1': lambda1,
                            'lambda2': lambda2,
                            'horizon': horizon,
                            'rmse': avg_rmse,
                            'training_time': training_time
                        })
                        
                    h60_rmse = next(r['rmse'] for r in results if r['model'] == 'DNSS' and 
                                  r['method'] == method and r['lambda1'] == lambda1 and 
                                  r['lambda2'] == lambda2 and r['horizon'] == 60)
                    
                    print(f"  λ1 = {lambda1:.4f}, λ2 = {lambda2:.4f}, Method = {method}, RMSE (60d) = {h60_rmse:.2f} bps")
                    
                except Exception as e:
                    print(f"Error with λ1 = {lambda1}, λ2 = {lambda2}, Method = {method}: {str(e)}")
    
    results_df = pd.DataFrame(results)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(output_path / 'dnss_tuning_results.csv', index=False)
    
    best_configs = []
    
    for horizon in horizons:
        horizon_results = results_df[results_df['horizon'] == horizon]
        best_idx = horizon_results['rmse'].idxmin()
        best_config = horizon_results.loc[best_idx]
        
        best_configs.append(best_config)
        
        print(f"\nBest DNSS configuration for horizon {horizon}:")
        print(f"  Method: {best_config['method']}")
        print(f"  Lambda1: {best_config['lambda1']:.4f}")
        print(f"  Lambda2: {best_config['lambda2']:.4f}")
        print(f"  RMSE: {best_config['rmse']:.2f} bps")
    
    best_configs_df = pd.DataFrame(best_configs)
    best_configs_df.to_csv(output_path / 'dnss_best_configs.csv', index=False)
    
    return best_configs_df

def tune_nnss_model(train_data, val_data, horizons, output_dir, use_gpu=False):
    """
    Otimiza os hiperparâmetros do modelo NNSS.
    
    Args:
        train_data: Dados de treinamento
        val_data: Dados de validação
        horizons: Lista de horizontes de previsão
        output_dir: Diretório para salvar os resultados
        use_gpu: Se deve usar GPU para o treinamento
        
    Retorna:
        Melhores hiperparâmetros
    """
    maturities = [int(col) for col in train_data.columns]
    device = 'cuda' if use_gpu else 'cpu'
    
    param_combinations = [
        {'IG_a': 0.1, 'IG_b': 0.001, 'minnesota_lambda': 0.5, 'minnesota_gamma': 0.9, 'nn_prior_var': 0.05},
        {'IG_a': 0.05, 'IG_b': 0.001, 'minnesota_lambda': 0.5, 'minnesota_gamma': 0.9, 'nn_prior_var': 0.05},
        {'IG_a': 0.2, 'IG_b': 0.001, 'minnesota_lambda': 0.5, 'minnesota_gamma': 0.9, 'nn_prior_var': 0.05},
        {'IG_a': 0.1, 'IG_b': 0.0005, 'minnesota_lambda': 0.5, 'minnesota_gamma': 0.9, 'nn_prior_var': 0.05},
        {'IG_a': 0.1, 'IG_b': 0.002, 'minnesota_lambda': 0.5, 'minnesota_gamma': 0.9, 'nn_prior_var': 0.05},
        {'IG_a': 0.1, 'IG_b': 0.001, 'minnesota_lambda': 0.3, 'minnesota_gamma': 0.9, 'nn_prior_var': 0.05},
        {'IG_a': 0.1, 'IG_b': 0.001, 'minnesota_lambda': 0.7, 'minnesota_gamma': 0.9, 'nn_prior_var': 0.05},
        {'IG_a': 0.1, 'IG_b': 0.001, 'minnesota_lambda': 0.5, 'minnesota_gamma': 0.5, 'nn_prior_var': 0.05},
        {'IG_a': 0.1, 'IG_b': 0.001, 'minnesota_lambda': 0.5, 'minnesota_gamma': 0.7, 'nn_prior_var': 0.05},
        {'IG_a': 0.1, 'IG_b': 0.001, 'minnesota_lambda': 0.5, 'minnesota_gamma': 0.9, 'nn_prior_var': 0.01},
        {'IG_a': 0.1, 'IG_b': 0.001, 'minnesota_lambda': 0.5, 'minnesota_gamma': 0.9, 'nn_prior_var': 0.1}
    ]
    
    results = []
    
    print(f"\n{'='*50}")
    print(f"Tuning NNSS model")
    print(f"{'='*50}")
    
    for params in param_combinations:
        IG_a = params['IG_a']
        IG_b = params['IG_b']
        minnesota_lambda = params['minnesota_lambda']
        minnesota_gamma = params['minnesota_gamma']
        nn_prior_var = params['nn_prior_var']
        
        print(f"\nTrying parameters: IG_a={IG_a}, IG_b={IG_b}, lambda={minnesota_lambda}, gamma={minnesota_gamma}, prior_var={nn_prior_var}")
        
        model = NNSSModel(
            maturities=maturities,
            n_factors=4,
            n_hidden=300,
            IG_a=IG_a,
            IG_b=IG_b,
            minnesota_lambda=minnesota_lambda,
            minnesota_gamma=minnesota_gamma,
            nn_prior_var=nn_prior_var,
            device=device
        )
        
        try:
            start_time = time.time()
            model.fit(train_data)
            training_time = time.time() - start_time
            
            eval_results = model.evaluate(val_data, horizons)
            
            for horizon in horizons:
                avg_rmse = eval_results[eval_results['horizon'] == horizon][eval_results['maturity'] == 'avg']['rmse'].values[0]
                
                results.append({
                    'model': 'NNSS',
                    'IG_a': IG_a,
                    'IG_b': IG_b,
                    'minnesota_lambda': minnesota_lambda,
                    'minnesota_gamma': minnesota_gamma,
                    'nn_prior_var': nn_prior_var,
                    'horizon': horizon,
                    'rmse': avg_rmse,
                    'training_time': training_time
                })
                
            h60_rmse = next(r['rmse'] for r in results if r['model'] == 'NNSS' and 
                           r['IG_a'] == IG_a and r['IG_b'] == IG_b and 
                           r['minnesota_lambda'] == minnesota_lambda and 
                           r['minnesota_gamma'] == minnesota_gamma and 
                           r['nn_prior_var'] == nn_prior_var and r['horizon'] == 60)
            
            print(f"  RMSE (60d) = {h60_rmse:.2f} bps, Training time: {training_time:.2f}s")
            
        except Exception as e:
            print(f"Error with parameters: {str(e)}")
    
    results_df = pd.DataFrame(results)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(output_path / 'nnss_tuning_results.csv', index=False)
    
    best_configs = []
    
    for horizon in horizons:
        horizon_results = results_df[results_df['horizon'] == horizon]
        best_idx = horizon_results['rmse'].idxmin()
        best_config = horizon_results.loc[best_idx]
        
        best_configs.append(best_config)
        
        print(f"\nBest NNSS configuration for horizon {horizon}:")
        print(f"  IG_a: {best_config['IG_a']}")
        print(f"  IG_b: {best_config['IG_b']}")
        print(f"  minnesota_lambda: {best_config['minnesota_lambda']}")
        print(f"  minnesota_gamma: {best_config['minnesota_gamma']}")
        print(f"  nn_prior_var: {best_config['nn_prior_var']}")
        print(f"  RMSE: {best_config['rmse']:.2f} bps")
    
    best_configs_df = pd.DataFrame(best_configs)
    best_configs_df.to_csv(output_path / 'nnss_best_configs.csv', index=False)
    
    return best_configs_df

def run_hyperparameter_tuning(data_path, output_dir, train_val_test_split=[0.7, 0.15, 0.15], 
                             horizons=[5, 20, 60, 120], tune_models=['dns', 'dnss', 'nnss'],
                             use_gpu=False):
    """
    Executa a otimização de hiperparâmetros para os modelos selecionados.
    
    Args:
        data_path: Caminho para os dados da curva de juros
        output_dir: Diretório para salvar os resultados
        train_val_test_split: Proporções para as divisões treino/validação/teste
        horizons: Lista de horizontes de previsão
        tune_models: Lista de modelos para otimizar ('dns', 'dnss', 'nnss')
        use_gpu: Se deve usar GPU para o modelo NNSS
    
    Retorna:
        Dicionário com as melhores configurações para cada modelo
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    yield_data = load_yield_data(data_path)
    
    train_end = int(len(yield_data) * train_val_test_split[0])
    val_end = train_end + int(len(yield_data) * train_val_test_split[1])
    
    train_data = yield_data.iloc[:train_end]
    val_data = yield_data.iloc[train_end:val_end]
    test_data = yield_data.iloc[val_end:]
    
    print(f"\nData loaded and split:")
    print(f"  Training data: {len(train_data)} observations ({train_data.index[0]} to {train_data.index[-1]})")
    print(f"  Validation data: {len(val_data)} observations ({val_data.index[0]} to {val_data.index[-1]})")
    print(f"  Test data: {len(test_data)} observations ({test_data.index[0]} to {test_data.index[-1]})")
    
    best_configs = {}
    
    if 'dns' in tune_models:
        best_configs['dns'] = tune_dns_model(train_data, val_data, horizons, output_path / 'dns')
    
    if 'dnss' in tune_models:
        best_configs['dnss'] = tune_dnss_model(train_data, val_data, horizons, output_path / 'dnss')
    
    if 'nnss' in tune_models:
        best_configs['nnss'] = tune_nnss_model(train_data, val_data, horizons, output_path / 'nnss', use_gpu)
    
    train_data.to_csv(output_path / 'train_data.csv')
    val_data.to_csv(output_path / 'val_data.csv')
    test_data.to_csv(output_path / 'test_data.csv')
    
    split_info = {
        'train_start': train_data.index[0].strftime('%Y-%m-%d'),
        'train_end': train_data.index[-1].strftime('%Y-%m-%d'),
        'val_start': val_data.index[0].strftime('%Y-%m-%d'),
        'val_end': val_data.index[-1].strftime('%Y-%m-%d'),
        'test_start': test_data.index[0].strftime('%Y-%m-%d'),
        'test_end': test_data.index[-1].strftime('%Y-%m-%d'),
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'train_proportion': train_val_test_split[0],
        'val_proportion': train_val_test_split[1],
        'test_proportion': train_val_test_split[2]
    }
    
    with open(output_path / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=4)
    
    return best_configs

if __name__ == "__main__":
    data_path = Path("data/raw/ettj_padronizado.csv")
    output_dir = Path("results/hyperparameter_tuning")
    
    run_hyperparameter_tuning(
        data_path, 
        output_dir, 
        train_val_test_split=[0.7, 0.15, 0.15],
        horizons=[5, 20, 60, 120], 
        tune_models=['dns', 'dnss', 'nnss'],
        use_gpu=False
    )