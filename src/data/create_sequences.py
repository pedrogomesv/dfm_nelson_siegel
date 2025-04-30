import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import os


def create_sequential_windows(data, window_size, horizon):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Os dados devem ser um DataFrame pandas")

    values = data.values
    n_total_points = len(data)
    n_maturities = data.shape[1]

    min_points_needed = window_size + horizon
    if n_total_points < min_points_needed:
        print(
            f"    AVISO: Dados insuficientes ({n_total_points}) para criar sequências com window_size={window_size} e horizon={horizon}. Necessário pelo menos {min_points_needed} pontos."
        )
        return None, None

    n_samples = n_total_points - window_size - horizon + 1

    X = np.zeros((n_samples, window_size, n_maturities))
    Y = np.zeros((n_samples, n_maturities))

    for i in range(n_samples):
        X[i] = values[i : i + window_size]
        Y[i] = values[i + window_size + horizon - 1]

    X_tensor = torch.tensor(X, dtype=torch.float64)
    Y_tensor = torch.tensor(Y, dtype=torch.float64)

    return X_tensor, Y_tensor


def create_torch_datasets(
    data_path,
    output_dir,
    window_sizes,
    horizons,
    train_val_test_split=[0.7, 0.15, 0.15],
    normalize_sequences=False,
):
    try:
        data = pd.read_csv(data_path, parse_dates=["Date"])
        data.set_index("Date", inplace=True)
    except Exception as e:
        print(f"Erro ao carregar dados: {str(e)}")
        return

    n_total_data = len(data)
    print(f"Dados carregados: {n_total_data} observações, {data.shape[1]} maturidades")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    max_window = max(window_sizes)
    max_horizon = max(horizons)
    min_required_test_len = max_window + max_horizon

    if n_total_data < min_required_test_len:
        print(
            f"ERRO: Total de dados ({n_total_data}) é insuficiente para a maior combinação de window/horizon ({min_required_test_len}). Não é possível criar conjunto de teste."
        )
        return

    requested_test_len = int(n_total_data * train_val_test_split[2])
    actual_test_len = max(min_required_test_len, requested_test_len)
    actual_test_len = min(
        actual_test_len,
        n_total_data - 1 if n_total_data > min_required_test_len else n_total_data,
    )

    if actual_test_len <= 0:
        print(f"ERRO: Não foi possível alocar pontos para o conjunto de teste.")
        return

    remaining_len = n_total_data - actual_test_len
    actual_train_len = 0
    actual_val_len = 0

    if remaining_len <= 0:
        print(
            "AVISO: Não há dados suficientes para treino/validação após garantir o teste."
        )
    elif remaining_len == 1:
        print(
            "AVISO: Apenas 1 ponto restante para treino/validação. Alocando para treino."
        )
        actual_train_len = 1
        actual_val_len = 0
    else:
        train_prop = train_val_test_split[0]
        val_prop = train_val_test_split[1]
        total_train_val_prop = train_prop + val_prop
        if total_train_val_prop <= 0:
            actual_train_len = remaining_len
            actual_val_len = 0
        else:
            requested_val_proportion_of_remainder = val_prop / total_train_val_prop
            actual_val_len = int(remaining_len * requested_val_proportion_of_remainder)
            if val_prop > 0:
                actual_val_len = max(1, actual_val_len)

            actual_train_len = remaining_len - actual_val_len
            if actual_train_len <= 0:
                actual_train_len = 1
                actual_val_len = remaining_len - 1

    assert (
        actual_train_len + actual_val_len + actual_test_len == n_total_data
    ), "Erro na lógica de cálculo do split"
    assert actual_train_len >= 0 and actual_val_len >= 0 and actual_test_len > 0

    train_data = data.iloc[:actual_train_len]
    val_data = data.iloc[actual_train_len : actual_train_len + actual_val_len]
    test_data = data.iloc[actual_train_len + actual_val_len :]

    print(f"Divisão dos dados (ajustada para garantir teste):")
    print(f"  Treino: {len(train_data)} observações")
    print(f"  Validação: {len(val_data)} observações")
    print(
        f"  Teste: {len(test_data)} observações (mínimo necessário: {min_required_test_len})"
    )

    if len(train_data) > 0:
        train_mean = train_data.mean().values
        train_std = train_data.std().values
        train_std = np.where(train_std < 1e-8, 1.0, train_std)

        print(f"Calculando e salvando média/std do treino...")
        np.save(output_path / "train_mean.npy", train_mean)
        np.save(output_path / "train_std.npy", train_std)
        print(f"  Salvo em: {output_path / 'train_mean.npy'}")
        print(f"  Salvo em: {output_path / 'train_std.npy'}")
    else:
        print(
            "AVISO: Conjunto de treino vazio. Não foi possível calcular/salvar média/std."
        )
        train_mean = np.zeros(data.shape[1])
        train_std = np.ones(data.shape[1])
        np.save(output_path / "train_mean.npy", train_mean)
        np.save(output_path / "train_std.npy", train_std)

    if normalize_sequences:
        print("Normalizando dados ANTES de criar as sequências...")
        if len(train_data) > 0:
            train_data_proc = (train_data - train_mean) / train_std
            if len(val_data) > 0:
                val_data_proc = (val_data - train_mean) / train_std
            else:
                val_data_proc = val_data
            if len(test_data) > 0:
                test_data_proc = (test_data - train_mean) / train_std
            else:
                test_data_proc = test_data
        else:
            print("AVISO: Treino vazio, usando dados originais para val/teste.")
            train_data_proc = train_data
            val_data_proc = val_data
            test_data_proc = test_data
    else:
        print("Usando dados BRUTOS para criar as sequências.")
        train_data_proc = train_data
        val_data_proc = val_data
        test_data_proc = test_data

    for window_size in window_sizes:
        for horizon in horizons:
            print(
                f"Criando sequências para window_size={window_size}, horizon={horizon}"
            )
            base_name = f"seq_w{window_size}_h{horizon}"

            if len(train_data_proc) > 0:
                X_train, y_train = create_sequential_windows(
                    train_data_proc, window_size, horizon
                )
                if X_train is not None:
                    torch.save(X_train, output_path / f"{base_name}_X_train.pt")
                    torch.save(y_train, output_path / f"{base_name}_y_train.pt")
                    print(f"    Train: X={X_train.shape}, y={y_train.shape}")
                else:
                    print(f"    Train: Não foi possível criar.")
            else:
                print("    Train: Conjunto vazio.")

            if len(val_data_proc) > 0:
                X_val, y_val = create_sequential_windows(
                    val_data_proc, window_size, horizon
                )
                if X_val is not None:
                    torch.save(X_val, output_path / f"{base_name}_X_val.pt")
                    torch.save(y_val, output_path / f"{base_name}_y_val.pt")
                    print(f"    Val:   X={X_val.shape}, y={y_val.shape}")
                else:
                    print(f"    Val:   Não foi possível criar.")
            else:
                print("    Val:   Conjunto vazio.")

            if len(test_data_proc) > 0:
                X_test, y_test = create_sequential_windows(
                    test_data_proc, window_size, horizon
                )
                if X_test is not None:
                    torch.save(X_test, output_path / f"{base_name}_X_test.pt")
                    torch.save(y_test, output_path / f"{base_name}_y_test.pt")
                    print(f"    Test:  X={X_test.shape}, y={y_test.shape}")
                else:
                    print(f"    Test:  Não foi possível criar (Inesperado!).")
            else:
                print("    Test:  Conjunto vazio (Inesperado!).")

    maturities = data.columns.tolist()
    with open(output_path / "maturities.txt", "w") as f:
        f.write(",".join(maturities))

    print(f"\nProcessamento concluído. Dados salvos em {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Criar sequências para modelos RNN/LSTM de curvas de juros"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="INSIRA/SEU/PATH/PARA/dados.csv",
        help="Caminho para o arquivo CSV com dados",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="INSIRA/SEU/PATH/PARA/output",
        help="Diretório para salvar as sequências",
    )
    parser.add_argument(
        "--window_sizes",
        type=int,
        nargs="+",
        default=[20, 60, 120],
        help="Tamanhos de janela para criar",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 5, 20, 60, 120],
        help="Horizontes de previsão",
    )
    parser.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Proporções desejadas para treino, validação e teste (soma deve ser 1)",
    )
    parser.add_argument(
        "--normalize_sequences",
        action="store_true",
        help="Normalizar os dados ANTES de salvar as sequências (usando média/std do treino)",
    )

    args = parser.parse_args()

    if not np.isclose(sum(args.split), 1.0):
        raise ValueError(
            f"As proporções do split devem somar 1. Recebido: {args.split} (soma: {sum(args.split)})"
        )
    if any(p < 0 for p in args.split):
        raise ValueError(
            f"As proporções do split não podem ser negativas. Recebido: {args.split}"
        )

    if not Path(args.data_path).exists():
        print(f"ERRO: Arquivo de dados não encontrado em {args.data_path}")
        print("Por favor, forneça um caminho válido usando --data_path.")
        exit(1)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Diretório de saída: {output_path.resolve()}")

    create_torch_datasets(
        data_path=args.data_path,
        output_dir=args.output_dir,
        window_sizes=args.window_sizes,
        horizons=args.horizons,
        train_val_test_split=args.split,
        normalize_sequences=args.normalize_sequences,
    )
