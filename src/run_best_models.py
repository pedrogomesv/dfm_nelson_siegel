import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as sklearn_train_test_split

from src.models.baselines.random_walk import RandomWalkModel
from src.models.baselines.dns import DynamicNelsonSiegelModel
from src.models.baselines.dnss import DynamicNelsonSiegelSvenssonModel
from src.models.baselines.nnss import NNSSModel
from src.utils.visualization import plot_model_comparison


def load_yield_data(file_path):
    """
    Carrega os dados da curva de juros de um arquivo.

    Args:
        file_path: Caminho para o arquivo de dados

    Retorna:
        DataFrame com os dados da curva de juros
    """
    df = pd.read_csv(file_path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)

    return df


def load_best_hyperparameters(tuning_dir):
    """
    Carrega os melhores hiperparâmetros dos resultados da otimização.

    Args:
        tuning_dir: Diretório com os resultados da otimização

    Retorna:
        Dicionário com os melhores hiperparâmetros
    """
    best_params = {}

    dns_path = Path(tuning_dir) / "dns" / "dns_best_configs.csv"
    if dns_path.exists():
        dns_best = pd.read_csv(dns_path)
        best_params["dns"] = dns_best

    dnss_path = Path(tuning_dir) / "dnss" / "dnss_best_configs.csv"
    if dnss_path.exists():
        dnss_best = pd.read_csv(dnss_path)
        best_params["dnss"] = dnss_best

    nnss_path = Path(tuning_dir) / "nnss" / "nnss_best_configs.csv"
    if nnss_path.exists():
        nnss_best = pd.read_csv(nnss_path)
        best_params["nnss"] = nnss_best

    return best_params


def run_best_models(
    data_path,
    tuning_dir,
    output_dir,
    train_test_split=0.8,
    horizons=[5, 20, 60, 120],
    use_gpu=False,
    nnss_epochs=500,
    nnss_patience=10,
    validation_split=0.15,
):
    """
    Executa os modelos com os melhores hiperparâmetros, usando parada antecipada para o NNSS.

    Args:
        data_path: Caminho para os dados da curva de juros
        tuning_dir: Diretório com os resultados da otimização de hiperparâmetros
        output_dir: Diretório para salvar os resultados
        train_test_split: Proporção dos dados a serem usados para treinamento
        horizons: Lista de horizontes de previsão
        use_gpu: Se deve usar GPU para o modelo NNSS
        nnss_epochs: Número máximo de épocas para o treinamento do NNSS
        nnss_patience: Paciência para parada antecipada no treinamento do NNSS
        validation_split: Proporção dos dados de treinamento a serem usados para validação no NNSS

    Retorna:
        DataFrame com os resultados da comparação dos modelos
    """
    device = "cuda" if use_gpu else "cpu"

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    yield_data = load_yield_data(data_path)

    maturities = [int(col) for col in yield_data.columns]

    split_idx = int(len(yield_data) * train_test_split)
    initial_train_data = yield_data.iloc[:split_idx]
    test_data = yield_data.iloc[split_idx:]

    train_subset, val_subset = sklearn_train_test_split(
        initial_train_data,
        test_size=validation_split,
        shuffle=False,
    )

    print(f"Data loaded: {len(yield_data)} observations")
    print(f"Initial Train data: {len(initial_train_data)} observations")
    print(f"  -> Train Subset: {len(train_subset)} observations (for NNSS)")
    print(f"  -> Validation Subset: {len(val_subset)} observations (for NNSS)")
    print(f"Test data: {len(test_data)} observations")

    best_params = load_best_hyperparameters(tuning_dir)

    all_results = {}

    rw_model = RandomWalkModel(maturities)
    rw_model.fit(initial_train_data)
    rw_results = rw_model.evaluate(
        test_data, horizons, output_dir=output_path / "RandomWalk"
    )
    all_results["RandomWalk"] = rw_results

    for horizon in horizons:
        print(f"\n{'='*50}")
        print(f"Running models for horizon {horizon}")
        print(f"{'='*50}")

        if "dns" in best_params:
            dns_config = best_params["dns"][
                best_params["dns"]["horizon"] == horizon
            ].iloc[0]
            dns_model = DynamicNelsonSiegelModel(
                maturities=maturities,
                method=dns_config["method"],
                lambda_fixed=dns_config["lambda"],
            )

            print(f"\nTraining DNS model with:")
            print(f"  Method: {dns_config['method']}")
            print(f"  Lambda: {dns_config['lambda']:.4f}")

            dns_model.fit(initial_train_data)
            dns_results = dns_model.evaluate(
                test_data, [horizon], output_dir=output_path / f"DNS_h{horizon}"
            )

            model_name = f"DNS_h{horizon}"
            all_results[model_name] = dns_results

        if "dnss" in best_params:
            dnss_config = best_params["dnss"][
                best_params["dnss"]["horizon"] == horizon
            ].iloc[0]
            dnss_model = DynamicNelsonSiegelSvenssonModel(
                maturities=maturities,
                method=dnss_config["method"],
                lambda1_fixed=dnss_config["lambda1"],
                lambda2_fixed=dnss_config["lambda2"],
            )

            print(f"\nTraining DNSS model with:")
            print(f"  Method: {dnss_config['method']}")
            print(f"  Lambda1: {dnss_config['lambda1']:.4f}")
            print(f"  Lambda2: {dnss_config['lambda2']:.4f}")

            dnss_model.fit(initial_train_data)
            dnss_results = dnss_model.evaluate(
                test_data, [horizon], output_dir=output_path / f"DNSS_h{horizon}"
            )

            model_name = f"DNSS_h{horizon}"
            all_results[model_name] = dnss_results

        if "nnss" in best_params:
            nnss_config = best_params["nnss"][
                best_params["nnss"]["horizon"] == horizon
            ].iloc[0]
            nnss_model = NNSSModel(
                maturities=maturities,
                n_factors=4,
                n_hidden=300,
                IG_a=nnss_config["IG_a"],
                IG_b=nnss_config["IG_b"],
                minnesota_lambda=nnss_config["minnesota_lambda"],
                minnesota_gamma=nnss_config["minnesota_gamma"],
                nn_prior_var=nnss_config["nn_prior_var"],
                device=device,
            )

            print(f"\nTraining NNSS model with best params:")
            print(f"  IG_a: {nnss_config['IG_a']}")
            print(f"  IG_b: {nnss_config['IG_b']}")
            print(f"  minnesota_lambda: {nnss_config['minnesota_lambda']}")
            print(f"  minnesota_gamma: {nnss_config['minnesota_gamma']}")
            print(f"  nn_prior_var: {nnss_config['nn_prior_var']}")
            print(
                f"  Using validation split, epochs={nnss_epochs}, patience={nnss_patience}"
            )

            model_horizon_name = f"NNSS_h{horizon}"
            checkpoint_dir = output_path / model_horizon_name / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            nnss_model.fit(
                train_data=train_subset,
                val_data=val_subset,
                epochs=nnss_epochs,
                patience=nnss_patience,
                checkpoint_dir=checkpoint_dir,
            )
            nnss_results = nnss_model.evaluate(
                test_data, [horizon], output_dir=output_path / model_horizon_name
            )

            model_name = f"NNSS_h{horizon}"
            all_results[model_name] = nnss_results

    comparison = pd.concat([df.assign(model=name) for name, df in all_results.items()])

    comparison.to_csv(output_path / "all_results.csv", index=False)

    plot_model_comparison(comparison, metric="rmse", output_dir=output_dir)

    leaderboard = []

    for horizon in horizons:
        horizon_models = [
            m
            for m in all_results.keys()
            if m == "RandomWalk" or m.endswith(f"_h{horizon}")
        ]

        horizon_results = comparison[
            (comparison["horizon"] == horizon)
            & (comparison["maturity"] == "avg")
            & (comparison["model"].isin(horizon_models))
        ].sort_values("rmse")

        for i, (_, row) in enumerate(horizon_results.iterrows()):
            leaderboard.append(
                {
                    "horizon": horizon,
                    "rank": i + 1,
                    "model": row["model"],
                    "rmse": row["rmse"],
                    "mae": row["mae"],
                }
            )

    leaderboard_df = pd.DataFrame(leaderboard)
    leaderboard_df.to_csv(output_path / "leaderboard.csv", index=False)

    print("\nModel Leaderboard:")
    for horizon in horizons:
        print(f"\nHorizon {horizon}:")
        horizon_board = leaderboard_df[leaderboard_df["horizon"] == horizon]
        for _, row in horizon_board.iterrows():
            print(
                f"  Rank {int(row['rank'])}: {row['model']} (RMSE = {row['rmse']:.2f} bps)"
            )

    return comparison


if __name__ == "__main__":
    data_path = Path("data/raw/ettj_padronizado.csv")
    tuning_dir = Path("results/hyperparameter_tuning")
    output_dir = Path("results/best_models")

    run_best_models(
        data_path,
        tuning_dir,
        output_dir,
        train_test_split=0.8,
        horizons=[5, 20, 60, 120],
        use_gpu=False,
    )
