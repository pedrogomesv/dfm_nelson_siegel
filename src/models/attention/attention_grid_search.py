import sys
from pathlib import Path
import argparse
import itertools
import numpy as np
import pandas as pd
import time
import torch
import json
from tqdm import tqdm

sys.path.append(
    str(Path(__file__).resolve().parents[3])
)

from src.models.attention.graph import NNSSAttentionModel


def run_grid_search(
    data_path="insira_o_path_para_os_dados.csv",
    output_dir="insira_o_path_para_o_diretorio_de_saida",
    train_test_split=0.8,
    horizons=[5, 20, 60, 120],
    use_gpu=False,
    validation_split=0.2,
):
    """
    Executa grid search para os hiperparâmetros do modelo NNSS Attention.
    """
    # Configurar device
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"Usando device: {device}")


    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)


    print(f"Carregando dados de {data_path}")
    if not Path(data_path).exists():
        print(f"ERRO: Arquivo de dados não encontrado em {data_path}")
        print("Por favor, especifique o caminho correto.")
        return None
    try:
        data = pd.read_csv(data_path, parse_dates=["Date"])
        data.set_index("Date", inplace=True)
    except Exception as e:
        print(f"Erro ao carregar dados: {str(e)}")
        return None


    maturities = [int(col) for col in data.columns]


    split_idx = int(len(data) * train_test_split)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]


    val_split_idx = int(len(train_data) * (1 - validation_split))
    grid_train_data = train_data.iloc[:val_split_idx]
    val_data = train_data.iloc[val_split_idx:]

    print(f"Dados carregados: {len(data)} observações")
    print(f"Dados de treino para grid: {len(grid_train_data)} observações")
    print(f"Dados de validação: {len(val_data)} observações")
    print(f"Dados de teste: {len(test_data)} observações")
    print(f"Maturidades: {maturities}")


    param_grid = {
        "n_factors": [4],
        "n_hidden": [
            192,
            256,
            384,
        ],
        "n_blocks": [2, 3, 4],
        "num_heads": [2, 4, 8],
        "IG_a": [0.1, 0.5],
        "IG_b": [0.001, 0.01],
        "minnesota_lambda": [0.3, 0.5, 0.7],
        "minnesota_gamma": [0.7, 0.9],
    }


    keys = list(param_grid.keys())
    raw_combinations = list(itertools.product(*[param_grid[key] for key in keys]))


    combinations = []
    for combo in raw_combinations:
        params = dict(zip(keys, combo))
        if params["n_hidden"] % params["num_heads"] == 0:
            combinations.append(combo)

    print(f"Total de combinações válidas de hiperparâmetros: {len(combinations)}")


    results = []
    best_rmse = float("inf")
    best_params = None
    best_model = None


    eval_horizon = horizons[0]


    for i, combo in enumerate(tqdm(combinations, desc="Grid Search Progress")):
        params = dict(zip(keys, combo))


        print(f"\nCombinação {i+1}/{len(combinations)}:")
        for k, v in params.items():
            print(f"  {k}: {v}")


        try:
            model = NNSSAttentionModel(maturities=maturities, device=device, **params)


            start_time = time.time()
            model.fit(grid_train_data)
            training_time = time.time() - start_time


            val_results = model.evaluate(val_data, [eval_horizon])
            val_avg_rmse = val_results[val_results["horizon"] == eval_horizon][
                val_results["maturity"] == "avg"
            ]["rmse"].values[0]

            print(
                f"RMSE de validação ({eval_horizon} passos à frente): {val_avg_rmse:.2f} bps"
            )
            print(f"Tempo de treinamento: {training_time:.2f} segundos")


            params_with_results = {
                **params,
                "val_rmse": val_avg_rmse,
                "training_time": training_time,
            }
            results.append(params_with_results)


            if val_avg_rmse < best_rmse:
                best_rmse = val_avg_rmse
                best_params = params
                best_model = model
                print(f"Novo melhor modelo encontrado! RMSE: {best_rmse:.2f}")


                model_save_dir = output_path / "best_model"
                model_save_dir.mkdir(exist_ok=True, parents=True)
                model.save(str(model_save_dir))

        except Exception as e:
            print(f"Erro ao treinar modelo com parâmetros {params}: {str(e)}")

            params_with_results = {
                **params,
                "val_rmse": float("nan"),
                "training_time": float("nan"),
                "error": str(e),
            }
            results.append(params_with_results)


        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path / "grid_search_results.csv", index=False)


        with open(output_path / "grid_search_results.json", "w") as f:
            json.dump(results, f, indent=2)


    print("\nGrid search concluído!")
    if best_params:
        print(f"Melhores hiperparâmetros (RMSE de validação: {best_rmse:.2f} bps):")
        for k, v in best_params.items():
            print(f"  {k}: {v}")


        print(
            "\nRetreinando com os melhores hiperparâmetros no conjunto completo de treinamento..."
        )
        final_model = NNSSAttentionModel(
            maturities=maturities, device=device, **best_params
        )
        final_model.fit(train_data)


        test_results = final_model.evaluate(test_data, horizons, output_dir=output_dir)


        print("\nAvaliação final no conjunto de teste:")
        for horizon in horizons:
            avg_rmse = test_results[test_results["horizon"] == horizon][
                test_results["maturity"] == "avg"
            ]["rmse"].values[0]
            print(f"Horizonte {horizon}: RMSE = {avg_rmse:.2f} bps")


        final_model_dir = output_path / "final_model"
        final_model_dir.mkdir(exist_ok=True, parents=True)
        final_model.save(str(final_model_dir))


        test_results.to_csv(output_path / "test_results.csv", index=False)
    else:
        print("Nenhum modelo válido encontrado durante o grid search.")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Executar grid search para o modelo NNSS Attention"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="insira_o_path_para_os_dados.csv",
        help="Caminho para o arquivo de dados",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="insira_o_path_para_o_diretorio_de_saida",
        help="Diretório para salvar resultados",
    )
    parser.add_argument(
        "--train_test_split",
        type=float,
        default=0.8,
        help="Proporção dos dados para treino",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[5, 20, 60, 120],
        help="Horizontes de previsão",
    )
    parser.add_argument("--use_gpu", action="store_true", help="Usar GPU se disponível")

    args = parser.parse_args()


    data_file = Path(args.data_path)
    if not data_file.exists():
        print(f"ERRO: Arquivo de dados não encontrado em {args.data_path}")
        print("Por favor, especifique o caminho correto com --data_path")
        exit(1)


    print("\nIniciando grid search para o modelo NNSS Attention...")
    results = run_grid_search(
        data_path=str(data_file),
        output_dir=args.output_dir,
        train_test_split=args.train_test_split,
        horizons=args.horizons,
        use_gpu=args.use_gpu,
    )
    if results is None:
        print("\nGrid search não pôde ser concluído devido a erros anteriores.")
    else:
        print("\nProcesso concluído!")
