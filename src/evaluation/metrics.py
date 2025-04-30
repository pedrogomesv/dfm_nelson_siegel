import numpy as np
import pandas as pd

def calculate_rmse(y_true, y_pred):
    """
    Calcula a Raiz do Erro Quadrático Médio (RMSE) entre os valores verdadeiros e previstos.

    Args:
        y_true: Array de valores verdadeiros
        y_pred: Array de valores previstos

    Returns:
        Valor RMSE (escalar ou array dependendo das dimensões de entrada)
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))

def calculate_mae(y_true, y_pred):
    """
    Calcula o Erro Absoluto Médio (MAE) entre os valores verdadeiros e previstos.

    Args:
        y_true: Array de valores verdadeiros
        y_pred: Array de valores previstos

    Returns:
        Valor MAE (escalar ou array dependendo das dimensões de entrada)
    """
    return np.mean(np.abs(y_true - y_pred), axis=0)

def calculate_directional_accuracy(y_true, y_pred, prev_y=None):
    """
    Calcula a acurácia direcional (percentual de previsões corretas de direção).

    Args:
        y_true: Array de valores verdadeiros
        y_pred: Array de valores previstos
        prev_y: Valores anteriores para calcular a direção (se None, usa o primeiro y_true)

    Returns:
        Percentual de acurácia direcional
    """
    if prev_y is None:
        prev_y = y_true[0]
        y_true = y_true[1:]
        y_pred = y_pred[1:]

    true_direction = np.sign(y_true - prev_y)
    pred_direction = np.sign(y_pred - prev_y)

    correct_direction = (true_direction == pred_direction).astype(int)
    return np.mean(correct_direction, axis=0) * 100

def evaluate_forecasts(true_values, predicted_values, horizons, maturities):
    """
    Avalia as previsões em múltiplos horizontes e maturidades.

    Args:
        true_values: Dicionário com horizontes como chaves e arrays de valores verdadeiros
        predicted_values: Dicionário com horizontes como chaves e arrays de valores previstos
        horizons: Lista de horizontes de previsão
        maturities: Lista de maturidades da curva de juros

    Returns:
        DataFrame com métricas de avaliação
    """
    results = []

    for horizon in horizons:
        y_true = true_values[horizon]
        y_pred = predicted_values[horizon]


        rmse = calculate_rmse(y_true, y_pred)
        mae = calculate_mae(y_true, y_pred)

        for i, maturity in enumerate(maturities):
            results.append({
                'horizon': horizon,
                'maturity': maturity,
                'rmse': rmse[i] * 1000,
                'mae': mae[i] * 1000
            })

        results.append({
            'horizon': horizon,
            'maturity': 'avg',
            'rmse': np.mean(rmse) * 1000,
            'mae': np.mean(mae) * 1000
        })

    return pd.DataFrame(results)

def compare_models(model_results, baseline_results=None):
    """
    Compara os resultados do modelo com os resultados da linha de base (baseline).

    Args:
        model_results: Dicionário com nomes de modelos como chaves e DataFrames de resultados como valores
        baseline_results: Resultados da linha de base opcionais para usar como referência

    Returns:
        DataFrame com métricas comparativas
    """
    if baseline_results is None:
        baseline_name = list(model_results.keys())[0]
        baseline_df = model_results[baseline_name]
    else:
        baseline_name = 'baseline'
        baseline_df = baseline_results

    comparison = []

    for model_name, results_df in model_results.items():
        if model_name == baseline_name and baseline_results is None:
            continue

        merged = pd.merge(
            results_df,
            baseline_df[['horizon', 'maturity', 'rmse']],
            on=['horizon', 'maturity'],
            suffixes=('', '_baseline')
        )

        merged['improvement'] = ((merged['rmse_baseline'] - merged['rmse']) /
                               merged['rmse_baseline']) * 100

        # Add model name
        merged['model'] = model_name

        comparison.append(merged)

    return pd.concat(comparison) if comparison else pd.DataFrame()