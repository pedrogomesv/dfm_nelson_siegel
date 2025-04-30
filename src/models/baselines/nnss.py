import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent.parent
sys.path.append(str(root_dir))

from src.models.base_model import BaseYieldCurveModel
from src.utils.kalman import KalmanFilter


class BasisLearner(nn.Module):
    """
    Rede neural para aprender funções base no modelo NNSS.
    """

    def __init__(
        self, n_factors=4, n_hidden=300, intercept_term=True, dtype=torch.float64
    ):
        super().__init__()

        self.n_factors = n_factors
        self.intercept_term = intercept_term
        self.dtype = dtype

        output_size = n_factors - 1 if intercept_term else n_factors

        self.mlp = nn.Sequential(
            nn.Linear(1, n_hidden, dtype=dtype),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden, dtype=dtype),
            nn.Tanh(),
            nn.Linear(n_hidden, output_size, dtype=dtype),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Forward pass da rede.
        """
        if x.dtype != self.dtype:
            x = x.to(dtype=self.dtype)

        basis_fns = self.mlp(x)

        if self.intercept_term:
            intercept = torch.ones(x.size(0), 1, device=x.device, dtype=self.dtype)
            return torch.cat([intercept, basis_fns], dim=1)

        return basis_fns


class NNSSModel(BaseYieldCurveModel):
    """
    Modelo Neural Network Augmented State-Space (NNSS) para previsão da curva de rendimentos.
    """

    def __init__(
        self,
        maturities,
        n_factors=4,
        n_hidden=300,
        IG_a=0.1,
        IG_b=0.001,
        minnesota_lambda=0.5,
        minnesota_gamma=0.9,
        nn_prior_var=0.05,
        device="cpu",
    ):
        """
        Inicializar o modelo NNSS.
        """
        name = f"NNSS_{n_factors}"
        super().__init__(name, maturities)

        self.n_factors = n_factors
        self.n_hidden = n_hidden
        self.dtype = torch.float64

        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.basis_learner = BasisLearner(
            n_factors=n_factors, n_hidden=n_hidden, dtype=self.dtype
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
            "IG_a": IG_a,
            "IG_b": IG_b,
            "minnesota_lambda": minnesota_lambda,
            "minnesota_gamma": minnesota_gamma,
            "nn_prior_var": nn_prior_var,
        }

    def _compute_loss(
        self,
        data_tensor,
        transition_matrix,
        transition_covariance_diag,
        observation_covariance_diag,
    ):
        """
        Compute loss for validation data.

        Args:
            data_tensor: Normalized validation data tensor
            transition_matrix: Current transition matrix tensor
            transition_covariance_diag: Current transition covariance diagonal tensor
            observation_covariance_diag: Current observation covariance diagonal tensor
        """
        maturities_tensor = torch.tensor(
            np.array(self.maturities).reshape(-1, 1) / 12,
            dtype=torch.float64,
            device=self.device,
        )
        observation_matrix = self.basis_learner(maturities_tensor).t()

        current_state = torch.zeros(
            self.n_factors, dtype=torch.float64, device=self.device
        )
        current_cov = torch.eye(self.n_factors, dtype=torch.float64, device=self.device)

        log_likelihood = 0

        for t in range(len(data_tensor)):
            pred_state = torch.matmul(transition_matrix, current_state)
            pred_cov = torch.matmul(
                torch.matmul(transition_matrix, current_cov),
                transition_matrix.t(),
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
                            obs_cov.shape[0],
                            device=self.device,
                            dtype=obs_cov.dtype,
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

        return -log_likelihood  # Retorna negativo da log-verossimilhança como perda

    def _initialize_basis_learner(self, train_data):
        """
        Inicializar BasisLearner com pesos que aproximam carregamentos Nelson-Siegel-Svensson.
        """
        print("Inicializando BasisLearner com pesos DNS-like...")

        maturities_tensor = torch.tensor(
            np.array(self.maturities).reshape(-1, 1) / 12,
            dtype=torch.float64,
            device=self.device,
        )

        lambda1 = 0.0609
        lambda2 = 0.0292

        loadings = np.zeros((len(self.maturities), 4))
        maturities_years = np.array(self.maturities) / 12

        loadings[:, 0] = 1.0

        for i, m in enumerate(maturities_years):
            loadings[i, 1] = (1 - np.exp(-lambda1 * m)) / (lambda1 * m)

        for i, m in enumerate(maturities_years):
            loadings[i, 2] = (1 - np.exp(-lambda1 * m)) / (lambda1 * m) - np.exp(
                -lambda1 * m
            )

        for i, m in enumerate(maturities_years):
            loadings[i, 3] = (1 - np.exp(-lambda2 * m)) / (lambda2 * m) - np.exp(
                -lambda2 * m
            )

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

    def fit(
        self, train_data, val_data=None, epochs=1000, patience=50, checkpoint_dir=None
    ):
        """
        Ajustar o modelo NNSS aos dados de treinamento com validação.

        Args:
            train_data: Dados de treinamento
            val_data: Dados de validação (opcional)
            epochs: Número de épocas (dobrado para 1000)
            patience: Número de épocas para early stopping
            checkpoint_dir: Diretório para salvar checkpoints
        """
        start_time = time.time()
        print(f"Ajustando modelo NNSS com {self.n_factors} fatores...")

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

        optimizer = optim.Adam(
            [
                {"params": self.basis_learner.parameters(), "lr": 1e-3},
                {"params": [transition_matrix], "lr": 1e-3},
                {"params": [transition_covariance_diag], "lr": 1e-4},
                {"params": [observation_covariance_diag], "lr": 5e-5},
            ]
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

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

            log_likelihood = 0
            current_state = torch.zeros(
                self.n_factors, dtype=torch.float64, device=self.device
            )
            current_cov = torch.eye(
                self.n_factors, dtype=torch.float64, device=self.device
            )

            for t in range(len(train_tensor_normalized)):
                pred_state = torch.matmul(transition_matrix, current_state)
                pred_cov = torch.matmul(
                    torch.matmul(transition_matrix, current_cov), transition_matrix.t()
                )
                pred_cov = pred_cov + torch.diag(transition_cov_positive)

                pred_obs = torch.matmul(observation_matrix.t(), pred_state)
                obs_cov = torch.matmul(
                    torch.matmul(observation_matrix.t(), pred_cov), observation_matrix
                )
                obs_cov = obs_cov + torch.diag(observation_cov_positive)

                innovation = train_tensor_normalized[t] - pred_obs

                try:
                    L = torch.linalg.cholesky(obs_cov)
                    log_det = 2 * torch.sum(torch.log(torch.diag(L)))

                    solved = torch.linalg.solve(obs_cov, innovation.unsqueeze(1))
                    inno_prec_inno = torch.matmul(
                        innovation.unsqueeze(0), solved
                    ).squeeze()

                    log_likelihood -= 0.5 * (
                        log_det + inno_prec_inno + len(innovation) * np.log(2 * np.pi)
                    )

                    kalman_gain = torch.matmul(
                        torch.matmul(pred_cov, observation_matrix),
                        torch.linalg.solve(
                            obs_cov,
                            torch.eye(
                                obs_cov.shape[0],
                                device=self.device,
                                dtype=obs_cov.dtype,
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

            log_prior = self._ig_minnesota_prior(
                transition_cov_positive, transition_matrix
            )
            log_prior += self._nn_prior()

            loss = -(log_likelihood + log_prior)

            loss.backward()
            optimizer.step()

            history["train_loss"].append(loss.item())

            if val_data is not None:
                self.basis_learner.eval()
                with torch.no_grad():
                    val_loss = self._compute_loss(
                        val_tensor_normalized,
                        transition_matrix,
                        transition_cov_positive,
                        observation_cov_positive,
                    )
                    history["val_loss"].append(val_loss.item())

                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
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
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Early stopping triggered at epoch {epoch}")
                            break

            if (epoch + 1) % 50 == 0:
                print(
                    f"Época {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}"
                    + (
                        f", Val Loss: {val_loss.item():.4f}"
                        if val_data is not None
                        else ""
                    )
                )

            with torch.no_grad():
                transition_cov_positive.copy_(torch.abs(transition_cov_positive))
                observation_cov_positive.copy_(torch.abs(observation_cov_positive))

        self.transition_matrix = transition_matrix.detach().cpu().numpy()
        self.transition_covariance = np.diag(
            transition_cov_positive.detach().cpu().numpy()
        )
        self.observation_covariance = np.diag(
            observation_cov_positive.detach().cpu().numpy()
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

        print(f"Modelo NNSS ajustado em {self.training_time:.2f} segundos")

        self._plot_training_history(history)

        return self

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

        extended_maturities = np.linspace(
            min(self.maturities) / 12, 10, num_points
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


def main():
    """Main function for local training and testing."""
    try:
        
        data_path = "insira o path aqui" 
        if not os.path.exists(data_path):
            print(f"Arquivo de dados não encontrado em {data_path}")
            print("Por favor, verifique se o arquivo existe no diretório correto.")
            return

        yield_data = pd.read_csv(data_path)
        if "Date" in yield_data.columns:
            yield_data.set_index("Date", inplace=True)

        maturities = [int(col) for col in yield_data.columns]

        train_size = int(0.7 * len(yield_data))
        val_size = int(0.15 * len(yield_data))

        train_data = yield_data.iloc[:train_size]
        val_data = yield_data.iloc[train_size : train_size + val_size]
        test_data = yield_data.iloc[train_size + val_size :]

        print(f"Data loaded: {len(yield_data)} observations")
        print(f"Training data: {len(train_data)} observations")
        print(f"Test data: {len(test_data)} observations")
        print(f"Maturities: {maturities}")

        output_dir = os.path.join(root_dir, "results", "nnss_local")
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        model = NNSSModel(
            maturities=maturities,
            n_factors=4,
            n_hidden=300,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        model.fit(
            train_data=train_data,
            val_data=val_data,
            epochs=1000,
            patience=50,
            checkpoint_dir=checkpoint_dir,
        )

        model.save(output_dir)

        horizon = 20
        predictions = model.predict(test_data, horizon)

        print(f"Test set predictions shape: {predictions.shape}")
        print("Training completed successfully!")
        print(f"Results saved to: {output_dir}")

    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")
        raise


if __name__ == "__main__":
    main()
