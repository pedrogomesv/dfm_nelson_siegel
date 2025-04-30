import sys
from pathlib import Path
import argparse
import warnings

warnings.filterwarnings("ignore")

sys.path.append(
    str(Path(__file__).resolve().parents[3])
)  

import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.base_model import BaseYieldCurveModel
from src.utils.kalman import KalmanFilter


class ResidualBlock(nn.Module):
    """
    Bloco residual com normalização e ativação GELU.
    """

    def __init__(self, dim, dropout=0.1, dtype=torch.float64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim, dtype=dtype),
            nn.LayerNorm(dim, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim, dtype=dtype),
            nn.LayerNorm(dim, dtype=dtype),
            nn.Dropout(dropout),
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(x + self.block(x))


class EnhancedBasisLearner(nn.Module):
    """
    Versão melhorada do BasisLearner com arquitetura residual.
    """

    def __init__(
        self,
        n_factors=4,
        n_hidden=300,
        n_blocks=3,
        dropout=0.1,
        intercept_term=True,
        dtype=torch.float64,
    ):
        super().__init__()

        self.n_factors = n_factors
        self.intercept_term = intercept_term
        self.dtype = dtype

        output_size = n_factors - 1 if intercept_term else n_factors

        layers = [
            nn.Linear(1, n_hidden, dtype=dtype),
            nn.LayerNorm(n_hidden, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout),
        ]

        for _ in range(n_blocks):
            layers.append(ResidualBlock(n_hidden, dropout=dropout, dtype=dtype))

        layers.extend(
            [
                nn.Linear(n_hidden, output_size, dtype=dtype),
                nn.Sigmoid(),  
            ]
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass da rede.
        """
        if x.dtype != self.dtype:
            x = x.to(dtype=self.dtype)

        basis_fns = self.network(x)

        if self.intercept_term:
            intercept = torch.ones(x.size(0), 1, device=x.device, dtype=self.dtype)
            return torch.cat([intercept, basis_fns], dim=1)

        return basis_fns


class NNSSResNetModel(BaseYieldCurveModel):
    """
    Modelo NNSS melhorado com arquitetura residual para previsão da curva de rendimentos.
    """

    def __init__(
        self,
        maturities,
        n_factors=4,
        n_hidden=300,
        n_blocks=3,
        IG_a=0.1,
        IG_b=0.001,
        minnesota_lambda=0.5,
        minnesota_gamma=0.9,
        nn_prior_var=0.05,
        device="cpu",
        l1_lambda=1e-5,
    ):
        """
        Inicializar o modelo NNSS ResNet.
        """
        name = f"NNSS_ResNet_{n_factors}"
        super().__init__(name, maturities)

        self.n_factors = n_factors
        self.n_hidden = n_hidden
        self.n_blocks = n_blocks
        self.dtype = torch.float64  

        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.basis_learner = EnhancedBasisLearner(
            n_factors=n_factors, n_hidden=n_hidden, n_blocks=n_blocks, dtype=self.dtype
        ).to(self.device)

        self.IG_a = IG_a
        self.IG_b = IG_b
        self.minnesota_lambda = minnesota_lambda
        self.minnesota_gamma = minnesota_gamma
        self.nn_prior_var = nn_prior_var

        self.transition_matrix = None
        self.transition_covariance = None
        self.observation_covariance = None
        self.initial_state = None
        self.initial_state_covariance = None
        self.data_mean = None

        self.filtered_factors = None
        self.smoothed_factors = None

        self.hyperparams = {
            "n_blocks": n_blocks,
            "IG_a": IG_a,
            "IG_b": IG_b,
            "minnesota_lambda": minnesota_lambda,
            "minnesota_gamma": minnesota_gamma,
            "nn_prior_var": nn_prior_var,
        }

        self.l1_lambda = l1_lambda

    def _initialize_basis_learner(self, train_data):
        """
        Inicializar BasisLearner com pesos que aproximam carregamentos Nelson-Siegel-Svensson.
        """
        print("Inicializando EnhancedBasisLearner com pesos DNS-like...")

        maturities_array = np.array([float(m) for m in self.maturities])
        maturities_years = maturities_array / 12

        maturities_tensor = torch.tensor(
            maturities_years.reshape(-1, 1),
            dtype=torch.float64,
            device=self.device,
        )

        lambda1 = 0.0609
        lambda2 = 0.0292

        loadings = np.zeros((len(self.maturities), 4))

        loadings[:, 0] = 1.0

        loadings[:, 1] = (1 - np.exp(-lambda1 * maturities_years)) / (
            lambda1 * maturities_years
        )

        loadings[:, 2] = (1 - np.exp(-lambda1 * maturities_years)) / (
            lambda1 * maturities_years
        ) - np.exp(-lambda1 * maturities_years)

        loadings[:, 3] = (1 - np.exp(-lambda2 * maturities_years)) / (
            lambda2 * maturities_years
        ) - np.exp(-lambda2 * maturities_years)

        target_loadings = torch.tensor(
            loadings[:, 1:] if self.basis_learner.intercept_term else loadings,
            dtype=torch.float64,
            device=self.device,
        )

        optimizer = optim.Adam(self.basis_learner.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        for epoch in range(10000):
            optimizer.zero_grad()

            output = self.basis_learner(maturities_tensor)

            if self.basis_learner.intercept_term:
                output = output[:, 1:]  

            loss = criterion(output, target_loadings)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 1000 == 0:
                print(f"Pré-treinamento época {epoch+1}/10000, Loss: {loss.item():.6f}")

            if loss.item() < 1e-4:
                print(
                    f"Pré-treinamento convergiu na época {epoch+1} com loss: {loss.item():.6f}"
                )
                break

        return self.basis_learner

    def _ig_minnesota_prior(self, trans_sigma, trans_matrix):
        """
        Calcular log-pdf do prior IG-Minnesota.
        """
        log_pdf = -(self.IG_a + 1) * torch.log(trans_sigma) - self.IG_b / trans_sigma
        log_pdf = log_pdf.sum()

        trans_sigma_col = trans_sigma.unsqueeze(1)
        V = trans_sigma_col / trans_sigma_col.t() * (self.minnesota_lambda**2)

        identity = torch.eye(
            trans_sigma.size(0), device=self.device, dtype=trans_sigma.dtype
        )
        V = V * ((1 - identity) * (self.minnesota_gamma**2) + identity)

        demean_trans_matrix = (trans_matrix - identity).view(-1, 1)

        log_pdf += (
            -0.5 * torch.log(V).sum()
            - 0.5 * (demean_trans_matrix**2 / V.view(-1)).sum()
        )

        return log_pdf

    def _nn_prior(self):
        """
        Calcular log-pdf do prior de pesos da rede neural.
        """
        log_pdf = 0

        for name, param in self.basis_learner.named_parameters():
            if "weight" in name:
                log_pdf -= 0.5 * torch.sum(param**2) / self.nn_prior_var

        return log_pdf

    def _compute_regularization(self):
        """
        Calcular regularização L1 e L2 combinadas
        """
        l1_reg = 0
        l2_reg = 0

        for name, param in self.basis_learner.named_parameters():
            if "weight" in name:
                l1_reg += torch.sum(torch.abs(param))
                l2_reg += torch.sum(param**2) / self.nn_prior_var

        return self.l1_lambda * l1_reg - 0.5 * l2_reg

    def fit(
        self, train_data, val_data=None, epochs=1000, patience=100, checkpoint_dir=None
    ):
        """
        Ajustar o modelo NNSS ResNet aos dados de treinamento com validação.

        Args:
            train_data: Dados de treinamento
            val_data: Dados de validação (opcional)
            epochs: Número de épocas (dobrado para 1000)
            patience: Número de épocas para early stopping (dobrado para 100)
            checkpoint_dir: Diretório para salvar checkpoints
        """
        start_time = time.time()
        print(
            f"Ajustando modelo NNSS ResNet com {self.n_factors} fatores e {self.n_blocks} blocos residuais..."
        )

        train_tensor = torch.tensor(
            train_data.values, dtype=torch.float64, device=self.device
        )

        train_mean = train_tensor.mean(dim=0)
        train_tensor_normalized = train_tensor - train_mean

        maturities_tensor = torch.tensor(
            np.array(self.maturities).reshape(-1, 1) / 12,
            dtype=torch.float64,
            device=self.device,
        )

        self.basis_learner = self._initialize_basis_learner(train_data)

        transition_matrix = torch.eye(
            self.n_factors, dtype=torch.float64, device=self.device
        )
        transition_matrix.requires_grad = True

        transition_covariance_diag = (
            torch.ones(self.n_factors, dtype=torch.float64, device=self.device) * 0.001
        )
        transition_covariance_diag.requires_grad = True

        observation_covariance_diag = (
            torch.ones(len(self.maturities), dtype=torch.float64, device=self.device)
            * 0.001
        )
        observation_covariance_diag.requires_grad = True

        optimizer = optim.AdamW(
            [
                {
                    "params": self.basis_learner.parameters(),
                    "lr": 1e-3,
                    "weight_decay": 0.01,
                },
                {"params": [transition_matrix], "lr": 1e-3, "weight_decay": 0.01},
                {"params": [transition_covariance_diag], "lr": 1e-4},
                {"params": [observation_covariance_diag], "lr": 5e-5},
            ]
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        best_epoch = 0

        history = {"train_loss": [], "val_loss": [] if val_data is not None else None}

        if val_data is not None:
            val_tensor = torch.tensor(
                val_data.values, dtype=torch.float64, device=self.device
            )
            val_tensor_normalized = val_tensor - train_mean

        for epoch in tqdm(range(epochs), desc="Training"):
            optimizer.zero_grad()

            observation_matrix = self.basis_learner(maturities_tensor).t()

            transition_cov_positive = torch.abs(transition_covariance_diag)
            observation_cov_positive = torch.abs(observation_covariance_diag)

            train_log_likelihood = self._compute_log_likelihood(
                train_tensor_normalized,
                observation_matrix,
                transition_matrix,
                transition_cov_positive,
                observation_cov_positive,
            )

            log_prior = self._ig_minnesota_prior(
                transition_cov_positive, transition_matrix
            )
            log_prior += self._nn_prior()
            reg_loss = self._compute_regularization()

            train_loss = -(train_log_likelihood + log_prior) + reg_loss
            history["train_loss"].append(train_loss.item())

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.basis_learner.parameters(), max_norm=1.0
            )
            optimizer.step()

            if val_data is not None:
                self.basis_learner.eval()
                with torch.no_grad():
                    val_loss = self._compute_log_likelihood(
                        val_tensor_normalized,
                        observation_matrix,
                        transition_matrix,
                        transition_cov_positive,
                        observation_cov_positive,
                    )
                    history["val_loss"].append(val_loss.item())

                    if (epoch + 1) % 25 == 0:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_epoch = epoch
                            patience_counter = 0
                            best_model_state = {
                                "basis_learner": self.basis_learner.state_dict(),
                                "transition_matrix": transition_matrix.clone(),
                                "transition_covariance_diag": transition_cov_positive.clone(),
                                "observation_covariance_diag": observation_cov_positive.clone(),
                            }

                            if checkpoint_dir:
                                self._save_checkpoint(
                                    checkpoint_dir, epoch, best_model_state
                                )
                                print(
                                    f"Novo melhor modelo salvo na época {epoch+1} com val_loss: {val_loss.item():.4f}"
                                )
                        else:
                            patience_counter += 1
                            if patience_counter >= patience:
                                print(f"Early stopping triggered at epoch {epoch+1}")
                                print(
                                    f"Usando melhor modelo da época {best_epoch+1} com val_loss: {best_val_loss:.4f}"
                                )
                                break

                self.basis_learner.train()

            if (epoch + 1) % 50 == 0:
                print(
                    f"Época {epoch+1}/{epochs}, Train Loss: {train_loss.item():.4f}"
                    + (
                        f", Val Loss: {val_loss.item():.4f}"
                        if val_data is not None
                        else ""
                    )
                )

            with torch.no_grad():
                transition_covariance_diag.copy_(torch.abs(transition_covariance_diag))
                observation_covariance_diag.copy_(
                    torch.abs(observation_covariance_diag)
                )

        if val_data is not None and best_model_state is not None:
            print(f"\nRestaurando melhor modelo da época {best_epoch+1}")
            print(f"Melhor val_loss: {best_val_loss:.4f}")

            self.basis_learner.load_state_dict(best_model_state["basis_learner"])
            transition_matrix.data.copy_(best_model_state["transition_matrix"])
            transition_covariance_diag.data.copy_(
                best_model_state["transition_covariance_diag"]
            )
            observation_covariance_diag.data.copy_(
                best_model_state["observation_covariance_diag"]
            )

        self.transition_matrix = transition_matrix.detach().cpu().numpy()
        self.transition_covariance = np.diag(
            transition_covariance_diag.detach().cpu().numpy()
        )
        self.observation_covariance = np.diag(
            observation_covariance_diag.detach().cpu().numpy()
        )
        self.initial_state = np.zeros(self.n_factors)
        self.initial_state_covariance = np.eye(self.n_factors)

        with torch.no_grad():
            self.observation_matrix = (
                self.basis_learner(maturities_tensor).t().detach().cpu().numpy()
            )

        kf = KalmanFilter(
            transition_matrix=self.transition_matrix,
            observation_matrix=self.observation_matrix.T,
            transition_covariance=self.transition_covariance,
            observation_covariance=self.observation_covariance,
            initial_state_mean=self.initial_state,
            initial_state_covariance=self.initial_state_covariance,
        )

        self.filtered_factors, _ = kf.filter(train_data.values)
        self.smoothed_factors, _ = kf.smooth(train_data.values)

        self.data_mean = train_mean.detach().cpu().numpy()

        self.is_fitted = True
        self.training_time = time.time() - start_time

        print(f"Modelo NNSS ResNet ajustado em {self.training_time:.2f} segundos")

        self._plot_training_history(history)

        return self

    def _compute_log_likelihood(
        self,
        data_tensor,
        observation_matrix,
        transition_matrix,
        transition_covariance_diag,
        observation_covariance_diag,
    ):
        """
        Compute log-likelihood for a given dataset.
        """
        log_likelihood = 0
        current_state = torch.zeros(
            self.n_factors, dtype=torch.float64, device=self.device
        )
        current_cov = torch.eye(self.n_factors, dtype=torch.float64, device=self.device)

        for t in range(len(data_tensor)):
            pred_state = torch.matmul(transition_matrix, current_state)
            pred_cov = torch.matmul(
                torch.matmul(transition_matrix, current_cov), transition_matrix.t()
            )
            pred_cov = pred_cov + torch.diag(transition_covariance_diag)

            pred_obs = torch.matmul(observation_matrix.t(), pred_state)
            obs_cov = torch.matmul(
                torch.matmul(observation_matrix.t(), pred_cov), observation_matrix
            )
            obs_cov = obs_cov + torch.diag(observation_covariance_diag)

            innovation = data_tensor[t] - pred_obs

            try:
                L = torch.linalg.cholesky(obs_cov)
                log_det = 2 * torch.sum(torch.log(torch.diag(L)))
                solved = torch.linalg.solve(obs_cov, innovation.unsqueeze(1))
                inno_prec_inno = torch.matmul(innovation.unsqueeze(0), solved).squeeze()

                log_likelihood -= 0.5 * (
                    log_det + inno_prec_inno + len(innovation) * np.log(2 * np.pi)
                )

                kalman_gain = torch.matmul(
                    torch.matmul(pred_cov, observation_matrix),
                    torch.linalg.solve(
                        obs_cov,
                        torch.eye(
                            obs_cov.shape[0], device=self.device, dtype=obs_cov.dtype
                        ),
                    ),
                )

                current_state = pred_state + torch.matmul(kalman_gain, innovation)
                current_cov = pred_cov - torch.matmul(
                    torch.matmul(kalman_gain, observation_matrix.t()), pred_cov
                )

            except Exception as e:
                print(
                    f"Aviso: Erro no cálculo de verossimilhança no passo {t}: {str(e)}"
                )
                continue

        return log_likelihood

    def _plot_training_history(self, history):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(history["train_loss"], label="Training Loss")
        if history["val_loss"]:
            plt.plot(history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training History")
        plt.legend()
        plt.grid(True)
        plt.show()

    def _save_checkpoint(self, checkpoint_dir, epoch, state):
        """Save model checkpoint."""
        Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)
        torch.save(state, f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pt")

    def predict(self, data, horizon):
        """
        Gerar previsões para o horizonte dado.
        """
        if not self.is_fitted:
            raise ValueError("Modelo deve ser ajustado antes da previsão")

        kf = KalmanFilter(
            transition_matrix=self.transition_matrix,
            observation_matrix=self.observation_matrix.T,
            transition_covariance=self.transition_covariance,
            observation_covariance=self.observation_covariance,
            initial_state_mean=self.initial_state,
            initial_state_covariance=self.initial_state_covariance,
        )

        forecasts = []

        current_state = self.filtered_factors[-1]
        current_cov = np.eye(self.n_factors) * 0.01

        for i in range(len(data) - horizon):
            current_state, current_cov = kf.update(
                current_state, current_cov, data.iloc[i].values - self.data_mean
            )

            future_states, _ = kf.forecast(current_state, current_cov, horizon)

            yield_forecast = (
                np.dot(self.observation_matrix.T, future_states[-1]) + self.data_mean
            )
            forecasts.append(yield_forecast)

        return np.array(forecasts)

    def save(self, output_dir):
        """
        Salvar os parâmetros do modelo.
        """
        super().save(output_dir)

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        np.savez(
            output_path / f"{self.name}_model_params.npz",
            transition_matrix=self.transition_matrix,
            transition_covariance=self.transition_covariance,
            observation_covariance=self.observation_covariance,
            initial_state=self.initial_state,
            initial_state_covariance=self.initial_state_covariance,
            observation_matrix=self.observation_matrix,
            data_mean=self.data_mean,
        )

        torch.save(
            self.basis_learner.state_dict(),
            output_path / f"{self.name}_basis_learner.pt",
        )

        if self.filtered_factors is not None:
            np.save(
                output_path / f"{self.name}_filtered_factors.npy", self.filtered_factors
            )

        if self.smoothed_factors is not None:
            np.save(
                output_path / f"{self.name}_smoothed_factors.npy", self.smoothed_factors
            )

        self.plot_factor_loadings(output_dir=output_dir)

        if self.smoothed_factors is not None:
            plt.figure(figsize=(12, 8))

            factor_names = ["Level", "Slope", "Curvature 1", "Curvature 2"][
                : self.n_factors
            ]

            for i in range(self.n_factors):
                plt.plot(
                    self.smoothed_factors[:, i], linewidth=2, label=factor_names[i]
                )

            plt.xlabel("Time")
            plt.ylabel("Factor Value")
            plt.title(f"{self.name} Estimated Factors")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.savefig(
                output_path / f"{self.name}_factors.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

    def plot_factor_loadings(self, num_points=1000, output_dir=None):
        """
        Plotar as cargas fatoriais aprendidas.
        """
        if not self.is_fitted:
            raise ValueError("Modelo deve ser ajustado antes de plotar")

        maturities_float = [float(m) for m in self.maturities]

        extended_maturities = np.linspace(
            min(maturities_float) / 12, 10, num_points  
        )

        maturities_tensor = torch.tensor(
            extended_maturities.reshape(-1, 1), dtype=torch.float64, device=self.device
        )

        with torch.no_grad():
            loadings = self.basis_learner(maturities_tensor).detach().cpu().numpy()

        plt.figure(figsize=(12, 8))

        factor_names = ["Level", "Slope", "Curvature 1", "Curvature 2"][
            : self.n_factors
        ]

        for i in range(self.n_factors):
            plt.plot(
                extended_maturities, loadings[:, i], linewidth=2, label=factor_names[i]
            )

        plt.xlabel("Maturity (years)")
        plt.ylabel("Loading")
        plt.title(f"{self.name} Learned Factor Loadings")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(
                output_path / f"{self.name}_factor_loadings.png",
                dpi=300,
                bbox_inches="tight",
            )

        plt.close()


def train_and_evaluate_standalone(
    data_path="insira o path aqui",
    output_dir="insira o path aqui",
    train_test_split=0.7,
    horizons=[5, 20, 60, 120],
    use_gpu=False,
):
    """
    Função para treinar e avaliar o modelo NNSS ResNet de forma independente.

    Args:
        data_path: Caminho para o arquivo de dados
        output_dir: Diretório para salvar resultados
        train_test_split: Proporção dos dados para treino
        horizons: Lista de horizontes de previsão
        use_gpu: Se deve usar GPU
    """
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print(f"Carregando dados de {data_path}")
    try:
        data = pd.read_csv(data_path, parse_dates=["Date"])
        data.set_index("Date", inplace=True)
    except Exception as e:
        print(f"Erro ao carregar dados: {str(e)}")
        return

    maturities = [int(col) for col in data.columns]

    train_size = int(0.7 * len(data))
    val_size = int(0.15 * len(data))

    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size : train_size + val_size]
    test_data = data.iloc[train_size + val_size :]

    print(f"Dados carregados: {len(data)} observações")
    print(f"Dados de treino: {len(train_data)} observações")
    print(f"Dados de validação: {len(val_data)} observações")
    print(f"Dados de teste: {len(test_data)} observações")
    print(f"Maturidades: {maturities}")

    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    model = NNSSResNetModel(
        maturities=maturities,
        n_factors=4,
        n_hidden=300,
        n_blocks=3,
        device=device,
    )

    model.fit(
        train_data=train_data,
        val_data=val_data,
        epochs=1000,
        patience=100,
        checkpoint_dir=checkpoint_dir,
    )

    model.save(output_dir)

    results = model.evaluate(test_data, horizons, output_dir=output_dir)

    print("\nResultados:")
    for horizon in horizons:
        avg_rmse = results[results["horizon"] == horizon][results["maturity"] == "avg"][
            "rmse"
        ].values[0]
        print(f"Horizonte {horizon}: RMSE = {avg_rmse:.2f} bps")

    results.to_csv(output_path / "results.csv", index=False)

    return model, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Treinar e avaliar modelo NNSS ResNet")
    parser.add_argument(
        "--data_path",
        type=str,
        default="insira o path aqui",
        help="Caminho para o arquivo de dados",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="insira o path aqui",
        help="Diretório para salvar resultados",
    )
    parser.add_argument(
        "--train_test_split",
        type=float,
        default=0.7,
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

    if not Path(args.data_path).exists():
        print(f"ERRO: Arquivo de dados não encontrado em {args.data_path}")
        print("Verificando caminhos alternativos...")

        possible_paths = [
            "insira o path aqui",
            "insira o path aqui",
            "insira o path aqui",
            "insira o path aqui",
            "insira o path aqui",
        ]

        for path in possible_paths:
            if Path(path).exists():
                args.data_path = path
                print(f"Arquivo encontrado em {path}")
                break
        else:
            print("Arquivo de dados não encontrado em nenhum local padrão.")
            print("Por favor, especifique o caminho correto com --data_path")
            exit(1)

    print("\nIniciando treinamento e avaliação do modelo NNSS ResNet...")
    model, results = train_and_evaluate_standalone(
        data_path=args.data_path,
        output_dir=args.output_dir,
        train_test_split=args.train_test_split,
        horizons=args.horizons,
        use_gpu=args.use_gpu,
    )
    print("\nProcesso concluído!")
