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
import seaborn as sns

from src.models.base_model import BaseYieldCurveModel
from src.utils.kalman import KalmanFilter


class MaturityAttention(nn.Module):


    def __init__(self, dim, num_heads=4, dropout=0.3, dtype=torch.float64):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert (
            dim % num_heads == 0
        ), f"Dimensão {dim} deve ser divisível pelo número de cabeças {num_heads}"

        self.q_proj = nn.Linear(dim, dim, dtype=dtype)
        self.k_proj = nn.Linear(dim, dim, dtype=dtype)
        self.v_proj = nn.Linear(dim, dim, dtype=dtype)

        self.out_proj = nn.Linear(dim, dim, dtype=dtype)

        self.norm = nn.LayerNorm(dim, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        self.attention_weights = None

    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        residual = x

        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attention_weights = torch.softmax(scores, dim=-1)


        self.attention_weights = attention_weights.detach().squeeze(
            0
        )

        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, v)

        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        )
        output = self.out_proj(context)

        output = self.norm(output + residual)

        return output


class ResidualBlock(nn.Module):


    def __init__(self, dim, dtype=torch.float64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim, dtype=dtype),
            nn.LayerNorm(dim, dtype=dtype),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(dim, dim, dtype=dtype),
            nn.LayerNorm(dim, dtype=dtype),
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(x + self.block(x))


class AttentionBasisLearner(nn.Module):


    def __init__(
        self,
        n_factors=4,
        n_hidden=64,
        n_blocks=2,
        num_heads=4,
        intercept_term=True,
        dtype=torch.float64,
    ):
        super().__init__()

        self.n_factors = n_factors
        self.intercept_term = intercept_term
        self.dtype = dtype
        self.n_hidden = n_hidden

        self.input_embedding = nn.Sequential(
            nn.Linear(1, n_hidden, dtype=dtype),
            nn.LayerNorm(n_hidden, dtype=dtype),
            nn.GELU(),
        )

        self.attention = MaturityAttention(
            dim=n_hidden, num_heads=num_heads, dtype=dtype
        )

        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(n_hidden, dtype=dtype) for _ in range(n_blocks)]
        )

        output_size = n_factors - 1 if intercept_term else n_factors

        self.output_layer = nn.Sequential(
            nn.Linear(n_hidden, output_size, dtype=dtype),
            nn.Sigmoid(),
        )

    def forward(self, x):

        if x.dtype != self.dtype:
            x = x.to(dtype=self.dtype)

        batch_size = x.size(0)

        embedded = self.input_embedding(x)

        embedded = embedded.unsqueeze(0)

        attended = self.attention(embedded)

        attended = attended.squeeze(0)

        x_res = attended
        for block in self.residual_blocks:
            x_res = block(x_res)

        basis_fns = self.output_layer(x_res)

        if self.intercept_term:
            intercept = torch.ones(batch_size, 1, device=x.device, dtype=self.dtype)
            return torch.cat([intercept, basis_fns], dim=1)

        return basis_fns

    def get_attention_weights(self):

        if hasattr(self.attention, "attention_weights"):
            return self.attention.attention_weights
        return None


class NNSSAttentionModel(BaseYieldCurveModel):


    def __init__(
        self,
        maturities,
        n_factors=4,
        n_hidden=64,
        n_blocks=2,
        num_heads=4,
        IG_a=0.1,
        IG_b=0.001,
        minnesota_lambda=0.5,
        minnesota_gamma=0.9,
        nn_prior_var=0.05,
        device="cpu",
    ):

        name = f"NNSS_Attention_{n_factors}"
        super().__init__(name, maturities)

        self.n_factors = n_factors
        self.n_hidden = n_hidden
        self.n_blocks = n_blocks
        self.num_heads = num_heads
        self.dtype = torch.float64
        self.batch_size = 256

        if n_hidden % num_heads != 0:
            raise ValueError(
                f"n_hidden ({n_hidden}) deve ser divisível por num_heads ({num_heads})"
            )

        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.basis_learner = AttentionBasisLearner(
            n_factors=n_factors,
            n_hidden=n_hidden,
            n_blocks=n_blocks,
            num_heads=num_heads,
            dtype=self.dtype,
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
            "num_heads": num_heads,
            "IG_a": IG_a,
            "IG_b": IG_b,
            "minnesota_lambda": minnesota_lambda,
            "minnesota_gamma": minnesota_gamma,
            "nn_prior_var": nn_prior_var,
        }

    def _initialize_basis_learner(self, train_data):

        print("Inicializando AttentionBasisLearner com pesos DNS-like...")

        maturities_array = np.array([float(m) for m in self.maturities])
        maturities_years = maturities_array / 12

        maturities_tensor = torch.tensor(
            maturities_years.reshape(-1, 1),
            dtype=torch.float64,
            device=self.device,
        )

        lambda1 = 0.0609
        lambda2 = 0.0292

        n_loadings = min(4, self.n_factors)

        loadings = np.zeros((len(self.maturities), n_loadings))

        loadings[:, 0] = 1.0

        if n_loadings > 1:
            loadings[:, 1] = (1 - np.exp(-lambda1 * maturities_years)) / (
                lambda1 * maturities_years
            )

        if n_loadings > 2:
            loadings[:, 2] = (1 - np.exp(-lambda1 * maturities_years)) / (
                lambda1 * maturities_years
            ) - np.exp(-lambda1 * maturities_years)

        if n_loadings > 3:
            loadings[:, 3] = (1 - np.exp(-lambda2 * maturities_years)) / (
                lambda2 * maturities_years
            ) - np.exp(-lambda2 * maturities_years)

        if self.basis_learner.intercept_term:
            target_loadings = torch.tensor(
                (
                    loadings[:, 1:n_loadings]
                    if n_loadings > 1
                    else np.zeros((len(self.maturities), 0))
                ),
                dtype=torch.float64,
                device=self.device,
            )
        else:
            target_loadings = torch.tensor(
                loadings,
                dtype=torch.float64,
                device=self.device,
            )

        optimizer = optim.Adam(self.basis_learner.parameters(), lr=5e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=200, verbose=True
        )
        criterion = nn.MSELoss()

        patience = 300
        best_loss = float("inf")
        counter = 0

        for epoch in range(10000):
            optimizer.zero_grad()

            output = self.basis_learner(maturities_tensor)

            if self.basis_learner.intercept_term:
                output = output[:, 1:]

            if output.shape[1] != target_loadings.shape[1]:
                print(
                    f"WARNING: Output shape {output.shape} doesn't match target shape {target_loadings.shape}"
                )
                print(
                    "This is likely due to a mismatch between n_factors and the number of DNS loadings."
                )
                print(
                    "Will use random initialization instead of DNS-like initialization."
                )
                break

            loss = criterion(output, target_loadings)

            loss.backward()
            optimizer.step()

            scheduler.step(loss.item())

            if loss.item() < best_loss:
                best_loss = loss.item()
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print(
                    f"Pré-treinamento parou com early stopping na época {epoch+1} com loss: {best_loss:.6f}"
                )
                break

            if (epoch + 1) % 1000 == 0:
                print(
                    f"Pré-treinamento época {epoch+1}/10000, Loss: {loss.item():.6f}, Melhor: {best_loss:.6f}"
                )

            if loss.item() < 1e-4:
                print(
                    f"Pré-treinamento convergiu na época {epoch+1} com loss: {loss.item():.6f}"
                )
                break

        return self.basis_learner

    def _ig_minnesota_prior(self, trans_sigma, trans_matrix):

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

        log_pdf = 0

        nn_prior_var_matrix = {
            "q_proj.weight": self.nn_prior_var
            * 0.5,
            "k_proj.weight": self.nn_prior_var * 0.5,
            "v_proj.weight": self.nn_prior_var * 0.5,
            "out_proj.weight": self.nn_prior_var * 0.5,
            "default": self.nn_prior_var,
        }

        for name, param in self.basis_learner.named_parameters():
            if "weight" in name:
                var = nn_prior_var_matrix.get(
                    name.split(".")[-2:][0], nn_prior_var_matrix["default"]
                )
                log_pdf -= 0.5 * torch.sum(param**2) / var

        return log_pdf

    def _compute_log_likelihood(
        self,
        data_tensor,
        transition_matrix,
        transition_cov_positive,
        observation_matrix,
        observation_cov_positive,
    ):

        log_likelihood = 0

        current_state = torch.zeros(
            self.n_factors, dtype=torch.float64, device=self.device
        )
        current_cov = torch.eye(self.n_factors, dtype=torch.float64, device=self.device)

        for t in range(len(data_tensor)):
            jitter_matrix = (
                torch.eye(self.n_factors, dtype=torch.float64, device=self.device)
                * 1e-6
            )

            pred_state = torch.matmul(transition_matrix, current_state)
            pred_cov = torch.matmul(
                torch.matmul(transition_matrix, current_cov), transition_matrix.t()
            )
            pred_cov = pred_cov + torch.diag(transition_cov_positive) + jitter_matrix

            pred_obs = torch.matmul(observation_matrix.t(), pred_state)

            obs_cov = torch.matmul(
                torch.matmul(observation_matrix.t(), pred_cov), observation_matrix
            )
            obs_jitter = (
                torch.eye(len(self.maturities), dtype=torch.float64, device=self.device)
                * 1e-6
            )
            obs_cov = obs_cov + torch.diag(observation_cov_positive) + obs_jitter

            innovation = data_tensor[t] - pred_obs

            try:
                L = torch.linalg.cholesky(obs_cov)
                log_det = 2 * torch.sum(torch.log(torch.diag(L)))

                solved = torch.linalg.solve(obs_cov, innovation.unsqueeze(1))
                inno_prec_inno = torch.matmul(innovation.unsqueeze(0), solved).squeeze()

                contribution = -0.5 * (
                    log_det + inno_prec_inno + len(innovation) * np.log(2 * np.pi)
                )
                log_likelihood += contribution

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

    def fit(
        self,
        train_data,
        validation_split=0.1,
        validation_freq=25,
        early_stopping_patience=5,
    ):

        start_time = time.time()

        print(
            f"Ajustando modelo NNSS Attention com {self.n_factors} fatores, {self.n_blocks} blocos residuais e {self.num_heads} cabeças de atenção..."
        )

        n_val = int(len(train_data) * validation_split)
        if n_val > 0:
            val_data = train_data.iloc[-n_val:]
            actual_train_data = train_data.iloc[:-n_val]
            print(
                f"Usando {len(actual_train_data)} observações para treino e {len(val_data)} para validação"
            )
        else:
            actual_train_data = train_data
            val_data = None
            print(
                f"Modo sem validação: usando todas as {len(train_data)} observações para treino"
            )

        train_tensor = torch.tensor(
            actual_train_data.values, dtype=torch.float64, device=self.device
        )

        if val_data is not None:
            val_tensor = torch.tensor(
                val_data.values, dtype=torch.float64, device=self.device
            )

        train_mean = train_tensor.mean(dim=0)
        train_tensor_normalized = train_tensor - train_mean

        if val_data is not None:
            val_tensor_normalized = val_tensor - train_mean

        maturities_tensor = torch.tensor(
            np.array(self.maturities, dtype=float).reshape(-1, 1)
            / 12,
            dtype=torch.float64,
            device=self.device,
        )

        self.basis_learner = self._initialize_basis_learner(actual_train_data)

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
                {
                    "params": self.basis_learner.parameters(),
                    "lr": 5e-4,
                    "weight_decay": 5e-3,
                },
                {"params": [transition_matrix], "lr": 5e-4, "weight_decay": 5e-3},
                {"params": [transition_covariance_diag], "lr": 1e-4},
                {"params": [observation_covariance_diag], "lr": 5e-5},
            ]
        )

        best_val_loss = float("inf")
        best_epoch = 0
        no_improvement_count = 0
        best_state = None

        n_epochs = 1000
        train_losses = []
        val_losses = []

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            observation_matrix = self.basis_learner(maturities_tensor).t()

            jitter = 1e-6
            transition_cov_positive = torch.abs(transition_covariance_diag) + jitter
            observation_cov_positive = torch.abs(observation_covariance_diag) + jitter

            train_log_likelihood = self._compute_log_likelihood(
                train_tensor_normalized,
                transition_matrix,
                transition_cov_positive,
                observation_matrix,
                observation_cov_positive,
            )

            log_prior = self._ig_minnesota_prior(
                transition_cov_positive, transition_matrix
            )
            log_prior += self._nn_prior()

            train_loss = -(train_log_likelihood + log_prior)
            train_losses.append(train_loss.item())

            train_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.basis_learner.parameters(), max_norm=1.0
            )
            torch.nn.utils.clip_grad_norm_([transition_matrix], max_norm=1.0)

            optimizer.step()

            with torch.no_grad():
                transition_covariance_diag.copy_(torch.abs(transition_covariance_diag))
                observation_covariance_diag.copy_(
                    torch.abs(observation_covariance_diag)
                )

            if val_data is not None and (epoch + 1) % validation_freq == 0:
                with torch.no_grad():
                    val_log_likelihood = self._compute_log_likelihood(
                        val_tensor_normalized,
                        transition_matrix,
                        transition_cov_positive,
                        observation_matrix,
                        observation_cov_positive,
                    )

                    val_loss = -(val_log_likelihood + log_prior)
                    val_losses.append(val_loss.item())

                print(
                    f"Época {epoch+1}/{n_epochs}, Treino Loss: {train_loss.item():.4f}, Validação Loss: {val_loss.item():.4f}"
                )

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_epoch = epoch + 1
                    no_improvement_count = 0

                    best_state = {
                        "basis_learner": self.basis_learner.state_dict(),
                        "transition_matrix": transition_matrix.detach().clone(),
                        "transition_covariance_diag": transition_covariance_diag.detach().clone(),
                        "observation_covariance_diag": observation_covariance_diag.detach().clone(),
                    }

                    print(f"Nova melhor validação loss: {best_val_loss:.4f}")
                else:
                    no_improvement_count += 1
                    print(
                        f"Sem melhoria por {no_improvement_count} verificações (melhor: {best_val_loss:.4f} na época {best_epoch})"
                    )

                    if no_improvement_count >= early_stopping_patience:
                        print(
                            f"Early stopping após {early_stopping_patience} validações sem melhoria"
                        )
                        print(
                            f"Restaurando modelo de melhor validação da época {best_epoch}"
                        )

                        self.basis_learner.load_state_dict(best_state["basis_learner"])
                        transition_matrix.data.copy_(best_state["transition_matrix"])
                        transition_covariance_diag.data.copy_(
                            best_state["transition_covariance_diag"]
                        )
                        observation_covariance_diag.data.copy_(
                            best_state["observation_covariance_diag"]
                        )
                        break
            elif (epoch + 1) % 50 == 0:
                print(
                    f"Época {epoch+1}/{n_epochs}, Treino Loss: {train_loss.item():.4f}"
                )

        if best_state is not None and no_improvement_count < early_stopping_patience:
            print(
                f"Treinamento completo, restaurando melhor modelo da época {best_epoch}"
            )
            self.basis_learner.load_state_dict(best_state["basis_learner"])
            transition_matrix.data.copy_(best_state["transition_matrix"])
            transition_covariance_diag.data.copy_(
                best_state["transition_covariance_diag"]
            )
            observation_covariance_diag.data.copy_(
                best_state["observation_covariance_diag"]
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

        full_train_tensor = torch.tensor(
            actual_train_data.values, dtype=torch.float64, device=self.device
        )
        full_train_normalized = full_train_tensor - train_mean

        self.filtered_factors, _ = kf.filter(full_train_normalized.cpu().numpy())
        self.smoothed_factors, _ = kf.smooth(full_train_normalized.cpu().numpy())

        self.data_mean = train_mean.detach().cpu().numpy()

        self.train_loss_history = train_losses
        self.val_loss_history = val_losses if val_data is not None else None

        self.is_fitted = True
        self.training_time = time.time() - start_time

        print(f"Modelo NNSS Attention ajustado em {self.training_time:.2f} segundos")
        if best_state is not None:
            print(
                f"Melhor performance na validação: {best_val_loss:.4f} (época {best_epoch})"
            )

        if val_data is not None and len(val_losses) > 0:
            epochs = list(range(validation_freq, n_epochs + 1, validation_freq))[
                : len(val_losses)
            ]
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, val_losses, label="Validação")
            plt.plot(
                [e for i, e in enumerate(epochs) if i % (early_stopping_patience) == 0],
                [
                    train_losses[e - 1]
                    for i, e in enumerate(epochs)
                    if i % (early_stopping_patience) == 0
                ],
                "ro",
                label="Treino (amostra)",
            )
            plt.xlabel("Época")
            plt.ylabel("Loss")
            plt.title("Histórico de Treino e Validação")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{self.name}_training_history.png", dpi=300)
            plt.close()

        return self

    def predict(self, data, horizon):

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

        n_forecasts = len(data) - horizon
        if n_forecasts <= 0:
            raise ValueError(
                f"Dados insuficientes para horizonte {horizon}. Necessário pelo menos {horizon+1} pontos."
            )

        current_state = np.zeros(self.n_factors)
        current_cov = np.eye(self.n_factors) * 0.01

        jitter = 1e-6

        for i in range(n_forecasts):
            try:
                normalized_obs = data.iloc[i].values - self.data_mean
                current_state, current_cov = kf.update(
                    current_state, current_cov, normalized_obs
                )

                current_cov += np.eye(self.n_factors) * jitter

                future_states, future_covs = kf.forecast(
                    current_state, current_cov, horizon
                )

                yield_forecast = (
                    np.dot(self.observation_matrix.T, future_states[-1])
                    + self.data_mean
                )
                forecasts.append(yield_forecast)

            except Exception as e:
                print(f"Erro na previsão do ponto {i}: {str(e)}")
                if len(forecasts) > 0:
                    forecasts.append(forecasts[-1])
                else:
                    forecasts.append(self.data_mean)

                current_state = np.zeros(self.n_factors)
                current_cov = np.eye(self.n_factors) * 0.01

        return np.array(forecasts)

    def plot_attention_maps(self, output_dir=None):

        if not self.is_fitted:
            raise ValueError(
                "Modelo deve ser ajustado antes de plotar mapas de atenção"
            )

        maturities_array = np.array([float(m) for m in self.maturities])
        maturities_years = maturities_array / 12

        maturities_tensor = torch.tensor(
            maturities_years.reshape(-1, 1),
            dtype=torch.float64,
            device=self.device,
        )

        with torch.no_grad():
            _ = self.basis_learner(maturities_tensor)
            attention_weights = self.basis_learner.get_attention_weights()

        if attention_weights is None:
            print("Não foi possível obter pesos de atenção")
            return

        attention_numpy = attention_weights.cpu().numpy()

        row_sums = attention_numpy.sum(axis=-1)
        if not np.allclose(row_sums, 1.0, atol=1e-2):
            print("Aviso: Os pesos de atenção não somam aproximadamente 1 por linha")
            print(f"Média das somas por linha: {row_sums.mean():.4f}")

        num_heads = attention_numpy.shape[0]
        plt.figure(figsize=(15, 5 * num_heads))

        for head in range(num_heads):
            plt.subplot(num_heads, 1, head + 1)

            sns.heatmap(
                attention_numpy[head],
                cmap="RdBu_r",
                vmin=0.0,
                vmax=1.0,
                xticklabels=[f"{m:.1f}y" for m in maturities_years],
                yticklabels=[f"{m:.1f}y" for m in maturities_years],
                annot=True,
                fmt=".2f",
                annot_kws={"size": 6},
                cbar_kws={"label": "Peso de Atenção"},
                square=True,
            )

            plt.title(f"Mapa de Atenção - Cabeça {head+1}")
            plt.xlabel("Maturidade Consultada")
            plt.ylabel("Maturidade Consultante")

            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)

        plt.tight_layout()

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(
                output_path / f"{self.name}_attention_maps.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()

    def plot_factor_loadings(self, num_points=1000, output_dir=None):

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
        else:
            plt.show()

    def analyze_attention_patterns(self, output_dir=None):

        if not self.is_fitted:
            raise ValueError(
                "Modelo deve ser ajustado antes de analisar padrões de atenção"
            )

        maturities_array = np.array([float(m) for m in self.maturities])
        maturities_years = maturities_array / 12

        maturities_tensor = torch.tensor(
            maturities_years.reshape(-1, 1),
            dtype=torch.float64,
            device=self.device,
        )

        with torch.no_grad():
            _ = self.basis_learner(maturities_tensor)
            attention_weights = self.basis_learner.get_attention_weights()

        if attention_weights is None:
            print("Não foi possível obter pesos de atenção")
            return

        attention_numpy = attention_weights.cpu().numpy()

        mean_attention_received = np.mean(attention_numpy, axis=(0, 1))

        mean_attention_given = np.mean(attention_numpy, axis=(0, 2))

        plt.figure(figsize=(12, 10))

        plt.subplot(2, 1, 1)
        plt.bar(maturities_years, mean_attention_received)
        plt.xlabel("Maturidade (anos)")
        plt.ylabel("Atenção Média Recebida")
        plt.title("Quais maturidades são mais consultadas?")
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.bar(maturities_years, mean_attention_given)
        plt.xlabel("Maturidade (anos)")
        plt.ylabel("Atenção Média Dada")
        plt.title("Quais maturidades consultam mais?")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(
                output_path / f"{self.name}_attention_analysis.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()

    def save(self, output_dir):

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
        self.plot_attention_maps(output_dir=output_dir)

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


def train_and_evaluate_standalone(
    data_path="insira seu path",
    output_dir="insira seu path",
    train_test_split=0.8,
    horizons=[5, 20, 60, 120],
    use_gpu=False,
):

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

    split_idx = int(len(data) * train_test_split)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    print(f"Dados carregados: {len(data)} observações")
    print(f"Dados de treino: {len(train_data)} observações")
    print(f"Dados de teste: {len(test_data)} observações")
    print(f"Maturidades: {maturities}")

    model = NNSSAttentionModel(
        maturities=maturities,
        n_factors=4,
        n_hidden=64,
        n_blocks=2,
        num_heads=4,
        IG_a=0.1,
        IG_b=0.001,
        minnesota_lambda=0.5,
        minnesota_gamma=0.9,
        nn_prior_var=0.05,
        device=device,
    )

    model.fit(train_data)

    model.save(output_dir)

    model.analyze_attention_patterns(output_dir=output_dir)

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

    parser = argparse.ArgumentParser(
        description="Treinar e avaliar modelo NNSS Attention"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="insira seu path",
        help="Caminho para o arquivo de dados",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="insira seu path",
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
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="Número de cabeças de atenção",
    )

    args = parser.parse_args()

    if not Path(args.data_path).exists():
        print(f"ERRO: Arquivo de dados não encontrado em {args.data_path}")
        print("Verificando caminhos alternativos...")

        possible_paths = [
            "insira seu path",
            "insira seu path",
            "insira seu path",
            "insira seu path",
            "insira seu path",
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

    print("\nIniciando treinamento e avaliação do modelo NNSS Attention...")
    model, results = train_and_evaluate_standalone(
        data_path=args.data_path,
        output_dir=args.output_dir,
        train_test_split=args.train_test_split,
        horizons=args.horizons,
        use_gpu=args.use_gpu,
    )
    print("\nProcesso concluído!")
