import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
from src.evaluation.metrics import calculate_rmse, calculate_mae


class BaseYieldCurveModel(ABC):
    """
    Classe base abstrata para modelos de curva de juros.
    """

    def __init__(self, name, maturities):
        """
        Inicializa o modelo base.

        Args:
            name: Nome do modelo
            maturities: Lista de maturidades
        """
        self.name = name
        self.maturities = maturities
        self.is_fitted = False
        self.training_time = None
        self.prediction_times = {}
        self.metrics = {}

    @abstractmethod
    def fit(self, train_data):
        """
        Ajusta o modelo aos dados de treino.

        Args:
            train_data: Dados de treino

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, data, horizon):
        """
        Gera previsões para o horizonte especificado.

        Args:
            data: Dados de entrada
            horizon: Horizonte de previsão

        Returns:
            Array de previsões
        """
        pass

    def evaluate(self, test_data, horizons, output_dir=None):
        """
        Avalia o modelo nos dados de teste para múltiplos horizontes.

        Args:
            test_data: Dados de teste
            horizons: Lista de horizontes de previsão
            output_dir: Diretório opcional para salvar os resultados

        Returns:
            DataFrame com métricas de avaliação
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")

        results = []
        forecasts = {}
        actuals = {}

        for horizon in horizons:
            start_time = time.time()

            predictions = self.predict(test_data, horizon)

            self.prediction_times[horizon] = time.time() - start_time

            actual_values = test_data[horizon:].values

            forecasts[horizon] = predictions
            actuals[horizon] = actual_values

            rmse = calculate_rmse(actual_values, predictions)
            mae = calculate_mae(actual_values, predictions)

            for i, maturity in enumerate(self.maturities):
                results.append(
                    {
                        "model": self.name,
                        "horizon": horizon,
                        "maturity": maturity,
                        "rmse": rmse[i] * 100,
                        "mae": mae[i] * 100,
                    }
                )

            results.append(
                {
                    "model": self.name,
                    "horizon": horizon,
                    "maturity": "avg",
                    "rmse": np.mean(rmse) * 100,
                    "mae": np.mean(mae) * 100,
                }
            )

        results_df = pd.DataFrame(results)
        self.metrics = results_df

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)

            results_df.to_csv(output_path / f"{self.name}_metrics.csv", index=False)

            for horizon in horizons:
                np.save(
                    output_path / f"{self.name}_forecasts_h{horizon}.npy",
                    forecasts[horizon],
                )
                np.save(
                    output_path / f"{self.name}_actuals_h{horizon}.npy",
                    actuals[horizon],
                )

            timing_info = {
                "training_time": self.training_time,
                "prediction_times": self.prediction_times,
            }

            with open(output_path / f"{self.name}_timing.json", "w") as f:
                json.dump(timing_info, f, indent=4)

        return results_df

    def plot_fit(self, train_data, test_data=None, output_dir=None):
        """
        Plota o modelo ajustado.

        Args:
            train_data: Dados de treino
            test_data: Dados de teste opcionais
            output_dir: Diretório opcional para salvar os gráficos
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)

        plt.figure(figsize=(12, 8))

        plt.title(f"{self.name} Model Fit")

        if output_dir:
            plt.savefig(
                output_path / f"{self.name}_fit.png", dpi=300, bbox_inches="tight"
            )

        plt.close()

    def save(self, output_dir):
        """
        Salva o modelo.

        Args:
            output_dir: Diretório para salvar o modelo
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        metrics_df = pd.DataFrame([self.metrics])
        metrics_df.to_csv(output_path / f"{self.name}_metrics.csv", index=False)

        timing_info = {
            "training_time": self.training_time,
            "prediction_times": self.prediction_times,
        }

        with open(output_path / f"{self.name}_timing.json", "w") as f:
            json.dump(timing_info, f, indent=4)

        