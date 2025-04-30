import sys
from pathlib import Path
import argparse
import warnings
import math  

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
import torch.optim.lr_scheduler as lr_scheduler
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch.utils.data


torch.set_default_dtype(torch.float64)

from src.models.base_model import BaseYieldCurveModel


class PositionalEncoding(nn.Module):
    """
    Implementa a codificação posicional sinusoidal.
    Adaptado de https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model: Dimensão do modelo (embedding).
            dropout: Taxa de dropout.
            max_len: Comprimento máximo da sequência esperada.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.dtype = torch.get_default_dtype()

        position = torch.arange(max_len, dtype=self.dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=self.dtype)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model, dtype=self.dtype)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer(
            "pe", pe.permute(1, 0, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        Returns:
            Tensor com codificação posicional adicionada.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class YieldCurveTransformerEncoder(nn.Module):
    """
    Encoder Transformer para processar janelas de curvas de juros.
    """

    def __init__(
        self,
        n_maturities: int,
        window_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            n_maturities: Número de maturidades na entrada (dimensão da característica).
            window_size: Tamanho da janela temporal (comprimento da sequência).
            d_model: Dimensão interna do modelo Transformer.
            nhead: Número de cabeças na Multi-Head Attention.
            num_encoder_layers: Número de camadas no encoder.
            dim_feedforward: Dimensão da camada feed-forward interna.
            dropout: Taxa de dropout.
        """
        super().__init__()
        self.d_model = d_model
        self.dtype = torch.get_default_dtype()
        self.n_maturities = n_maturities

        self.input_projection = nn.Linear(n_maturities, d_model, dtype=self.dtype)

        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=window_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            dtype=self.dtype,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.final_norm = nn.LayerNorm(d_model, dtype=self.dtype)

    def forward(self, input_window: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do encoder.

        Args:
            input_window: Tensor da janela de entrada, shape [batch_size, window_size, n_maturities].

        Returns:
            Tensor com as representações processadas pelo Transformer,
            shape [batch_size, window_size, d_model].
        """
        if input_window.dtype != self.dtype:
            input_window = input_window.to(dtype=self.dtype)
        if input_window.device != next(self.parameters()).device:
            input_window = input_window.to(next(self.parameters()).device)

        batch_size, seq_len, n_mats = input_window.shape
        if n_mats != self.n_maturities:
            raise ValueError(
                f"Dimensão de maturidade da entrada ({n_mats}) "
                f"não confere com n_maturities ({self.n_maturities})"
            )

        x = self.input_projection(input_window)
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        transformer_output = self.transformer_encoder(x)
        transformer_output = self.final_norm(transformer_output)

        return transformer_output


class MaturityProcessor(nn.Module):
    """
    Processa o contexto temporal do Transformer e as maturidades para gerar
    as cargas fatoriais H_t.
    """

    def __init__(
        self,
        d_model: int,
        n_maturities: int,
        n_factors: int,
        n_hidden: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        intercept_term: bool = True,
    ):
        """
        Args:
            d_model: Dimensão do modelo Transformer (entrada de contexto).
            n_maturities: Número de maturidades.
            n_factors: Número de fatores latentes.
            n_hidden: Dimensão das camadas ocultas do MLP interno.
            n_layers: Número de camadas ocultas do MLP interno.
            dropout: Taxa de dropout.
            intercept_term: Se True, adiciona um termo de intercepto (coluna de 1s) a H_t.
        """
        super().__init__()
        self.d_model = d_model
        self.n_maturities = n_maturities
        self.n_factors = n_factors
        self.intercept_term = intercept_term
        self.dtype = torch.get_default_dtype()

        self.maturity_embedding = nn.Linear(1, d_model, dtype=self.dtype)

        layers = []
        input_dim = d_model
        for _ in range(n_layers):
            layers.append(nn.Linear(input_dim, n_hidden, dtype=self.dtype))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_dim = n_hidden
        self.mlp = nn.Sequential(*layers)

        output_dim = n_factors - 1 if intercept_term else n_factors
        self.output_projection = nn.Linear(n_hidden, output_dim, dtype=self.dtype)

    def forward(
        self, transformer_context: torch.Tensor, maturities_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Gera as cargas fatoriais H_t.

        Args:
            transformer_context: Saída do Transformer para o último passo de tempo,
                                 shape [batch_size, d_model].
            maturities_tensor: Tensor com as maturidades (normalizadas ou não),
                               shape [n_maturities, 1].

        Returns:
            Tensor das cargas fatoriais H_t, shape [batch_size, n_maturities, n_factors].
        """
        batch_size = transformer_context.shape[0]

        if maturities_tensor.device != transformer_context.device:
            maturities_tensor = maturities_tensor.to(transformer_context.device)
        if maturities_tensor.dtype != self.dtype:
            maturities_tensor = maturities_tensor.to(dtype=self.dtype)
        if (
            maturities_tensor.shape[0] != self.n_maturities
            or maturities_tensor.shape[1] != 1
        ):
            raise ValueError(
                f"Shape do maturities_tensor ({maturities_tensor.shape}) "
                f"não confere com n_maturities ({self.n_maturities})"
            )

        maturity_embedded = self.maturity_embedding(maturities_tensor)

        context_expanded = transformer_context.unsqueeze(1).expand(
            -1, self.n_maturities, -1
        )
        maturity_expanded = maturity_embedded.unsqueeze(0).expand(batch_size, -1, -1)

        combined_representation = context_expanded + maturity_expanded
        processed_representation = self.mlp(combined_representation)
        factor_loadings = self.output_projection(processed_representation)

        if self.intercept_term:
            intercept = torch.ones(
                (batch_size, self.n_maturities, 1),
                dtype=self.dtype,
                device=transformer_context.device,
            )
            H_t = torch.cat((factor_loadings, intercept), dim=-1)
        else:
            H_t = factor_loadings

        return H_t

class FactorTransition(nn.Module):
    """
    Aprende a transição dos fatores x_t -> x_{t+h} de forma não linear,
    condicionada no contexto e no horizonte.
    """

    def __init__(
        self,
        n_factors: int,
        d_model: int,
        horizon_embedding_dim: int,
        possible_horizons: list[int],
        mlp_hidden_dim: int,
        mlp_layers: int,
        dropout: float,
    ):
        """
        Args:
            n_factors: Número de fatores latentes.
            d_model: Dimensão do modelo Transformer (contexto).
            horizon_embedding_dim: Dimensão do embedding para o horizonte.
            possible_horizons: Lista de possíveis horizontes de previsão (e.g., [5, 20, 60]).
            mlp_hidden_dim: Dimensão oculta do MLP de transição.
            mlp_layers: Número de camadas ocultas no MLP de transição.
            dropout: Taxa de dropout no MLP de transição.
        """
        super().__init__()
        self.n_factors = n_factors
        self.d_model = d_model
        self.possible_horizons = sorted(list(set(possible_horizons)))
        self.horizon_map = {h: i for i, h in enumerate(self.possible_horizons)}
        self.num_horizons = len(self.possible_horizons)
        self.dtype = torch.get_default_dtype()

        self.horizon_embedding = nn.Embedding(
            self.num_horizons, horizon_embedding_dim, dtype=self.dtype
        )

        mlp_input_dim = n_factors + d_model + horizon_embedding_dim
        layers = []
        current_dim = mlp_input_dim
        for _ in range(mlp_layers):
            layers.append(nn.Linear(current_dim, mlp_hidden_dim, dtype=self.dtype))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            current_dim = mlp_hidden_dim
        layers.append(nn.Linear(current_dim, n_factors, dtype=self.dtype))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, x_t: torch.Tensor, context: torch.Tensor, horizon_int: int
    ) -> torch.Tensor:
        """
        Calcula a previsão dos fatores x_{t+h}.

        Args:
            x_t: Fatores no tempo t, shape [batch_size, n_factors].
            context: Contexto do Transformer no tempo t, shape [batch_size, d_model].
            horizon_int: Horizonte de previsão (inteiro).

        Returns:
            Previsão dos fatores x_{t+h}, shape [batch_size, n_factors].
        """
        batch_size = x_t.shape[0]
        device = x_t.device

        try:
            horizon_idx = self.horizon_map[horizon_int]
        except KeyError:
            raise ValueError(
                f"Horizon {horizon_int} not in possible_horizons used for training: {self.possible_horizons}"
            )
        horizon_indices = torch.tensor(
            [horizon_idx] * batch_size, dtype=torch.long, device=device
        )

        h_emb = self.horizon_embedding(
            horizon_indices
        )

        mlp_input = torch.cat([x_t, context, h_emb], dim=1)

        x_pred = self.mlp(mlp_input)

        return x_pred


class NNSSTransformerModel(nn.Module, BaseYieldCurveModel):
    """
    Modelo NNSS baseado em Transformer para previsão da curva de juros,
    com mecanismo de transição de fatores aprendido.
    """

    def __init__(
        self,
        maturities: np.ndarray | list,
        n_factors: int,
        window_size: int,
        possible_horizons: list[int],
        intercept_term: bool = True,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 128,
        transformer_dropout: float = 0.25,
        maturity_processor_hidden: int = 64,
        maturity_processor_layers: int = 2,
        maturity_processor_dropout: float = 0.25,
        horizon_embedding_dim: int = 16,
        ft_mlp_hidden_dim: int = 32,
        ft_mlp_layers: int = 1,
        ft_dropout: float = 0.25,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 10,
        device: str | None = None,
        gradient_clipping: float | None = 1.0,
        l2_lambda: float = 1e-4,
    ):
        """
        Args:
            maturities: Maturidades da curva de juros.
            n_factors: Número de fatores latentes.
            window_size: Tamanho da janela temporal de entrada.
            possible_horizons: Lista de horizontes de previsão treinados (e.g., [5, 20, 60]).
            intercept_term: Se True, H_t terá um termo de intercepto.
            d_model: Dimensão interna do Transformer.
            nhead: Número de cabeças na Multi-Head Attention.
            num_encoder_layers: Número de camadas no encoder Transformer.
            dim_feedforward: Dimensão da camada feed-forward no Transformer.
            transformer_dropout: Taxa de dropout no Transformer.
            maturity_processor_hidden: Dimensão oculta no MLP do Maturity Processor.
            maturity_processor_layers: Número de camadas ocultas no MLP do Maturity Processor.
            maturity_processor_dropout: Taxa de dropout no Maturity Processor.
            horizon_embedding_dim: Dimensão do embedding para o horizonte no FactorTransition.
            ft_mlp_hidden_dim: Dimensão oculta no MLP do FactorTransition.
            ft_mlp_layers: Número de camadas ocultas no MLP do FactorTransition.
            ft_dropout: Taxa de dropout no FactorTransition.
            learning_rate: Taxa de aprendizado inicial.
            batch_size: Tamanho do lote para treinamento.
            epochs: Número máximo de épocas de treinamento.
            patience: Paciência para early stopping.
            device: Dispositivo ('cpu' ou 'cuda').
            gradient_clipping: Valor máximo para clipping de gradientes.
            l2_lambda: Coeficiente de regularização L2.
        """
        super().__init__()
        self.name = f"NNSS_Trans_FT_f{n_factors}_w{window_size}_d{d_model}_h{nhead}_l{num_encoder_layers}"
        self.maturities = (
            list(maturities) if isinstance(maturities, np.ndarray) else maturities
        )
        self.n_maturities = len(self.maturities)
        BaseYieldCurveModel.__init__(self, name=self.name, maturities=self.maturities)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.gradient_clipping = gradient_clipping
        self.l2_lambda = l2_lambda

        self.n_factors = n_factors
        self.window_size = window_size
        self.intercept_term = intercept_term
        self.d_model = d_model
        self.possible_horizons = possible_horizons

        self.dtype = torch.get_default_dtype()

        self.transformer_encoder = YieldCurveTransformerEncoder(
            n_maturities=self.n_maturities,
            window_size=window_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
        ).to(self.device, dtype=self.dtype)

        self.maturity_processor = MaturityProcessor(
            d_model=d_model,
            n_maturities=self.n_maturities,
            n_factors=n_factors,
            n_hidden=maturity_processor_hidden,
            n_layers=maturity_processor_layers,
            dropout=maturity_processor_dropout,
            intercept_term=intercept_term,
        ).to(self.device, dtype=self.dtype)

        self.factor_projection = nn.Linear(d_model, n_factors).to(
            self.device, dtype=self.dtype
        )

        self.factor_transition = FactorTransition(
            n_factors=n_factors,
            d_model=d_model,
            horizon_embedding_dim=horizon_embedding_dim,
            possible_horizons=possible_horizons,
            mlp_hidden_dim=ft_mlp_hidden_dim,
            mlp_layers=ft_mlp_layers,
            dropout=ft_dropout,
        ).to(self.device, dtype=self.dtype)

        self.hyperparams = {
            "maturities": self.maturities,
            "n_factors": n_factors,
            "window_size": window_size,
            "possible_horizons": possible_horizons,
            "n_maturities": self.n_maturities,
            "intercept_term": intercept_term,
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "dim_feedforward": dim_feedforward,
            "transformer_dropout": transformer_dropout,
            "maturity_processor_hidden": maturity_processor_hidden,
            "maturity_processor_layers": maturity_processor_layers,
            "maturity_processor_dropout": maturity_processor_dropout,
            "horizon_embedding_dim": horizon_embedding_dim,
            "ft_mlp_hidden_dim": ft_mlp_hidden_dim,
            "ft_mlp_layers": ft_mlp_layers,
            "ft_dropout": ft_dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "gradient_clipping": gradient_clipping,
            "l2_lambda": l2_lambda,
        }

        self.register_buffer(
            "data_mean",
            torch.zeros(self.n_maturities, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "data_std",
            torch.ones(self.n_maturities, device=self.device, dtype=self.dtype),
        )
        self.to(device=self.device, dtype=self.dtype)

    def forward(
        self,
        input_window: torch.Tensor,
        maturities_tensor: torch.Tensor,
        horizon: int,  # Recebe o horizonte como inteiro
    ) -> torch.Tensor:
        """
        Forward pass do modelo NNSSTransformer com transição aprendida.

        Args:
            input_window: Janela de entrada, shape [batch_size, window_size, n_maturities].
            maturities_tensor: Tensor com as maturidades, shape [n_maturities, 1].
            horizon: Horizonte de previsão (inteiro).

        Returns:
            Previsão da curva de juros no horizonte h, shape [batch_size, n_maturities].
        """
        transformer_output = self.transformer_encoder(input_window)

        last_time_step_output = transformer_output[:, -1, :]

        x_t = self.factor_projection(last_time_step_output)

        H_t = self.maturity_processor(last_time_step_output, maturities_tensor)

        x_pred = self.factor_transition(x_t, last_time_step_output, horizon)

        y_pred = torch.bmm(H_t, x_pred.unsqueeze(-1)).squeeze(-1)

        return y_pred


    def _normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Normaliza os dados usando a média e desvio padrão armazenados."""
        return (data - self.data_mean) / self.data_std

    def _denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Desnormaliza os dados usando a média e desvio padrão armazenados."""
        return data * self.data_std + self.data_mean

    def fit(
        self,
        train_sequences: list[dict[str, np.ndarray]],
        val_sequences: list[dict[str, np.ndarray]],
        maturities_tensor: torch.Tensor,
        checkpoint_dir: str | Path | None = None,
        checkpoint_freq: int = 10,
    ):
        """
        Treina o modelo NNSSTransformerModel.

        Args:
            train_sequences: Lista de sequências de treino.
            val_sequences: Lista de sequências de validação.
            maturities_tensor: Tensor de maturidades.
            checkpoint_dir: Diretório para salvar checkpoints.
            checkpoint_freq: Salvar checkpoint a cada X épocas.
        """
        start_time = time.time()
        self.train()

        checkpoint_path = None
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            print(f"Checkpoints will be saved to: {checkpoint_path}")

        all_train_yields = np.concatenate(
            [seq["target_yields"][np.newaxis, :] for seq in train_sequences]
            + [seq["input_window"] for seq in train_sequences],
            axis=0,
        )
        data_mean_np = np.mean(all_train_yields, axis=0)
        data_std_np = np.std(all_train_yields, axis=0)
        data_std_np[data_std_np < 1e-6] = 1.0

        self.data_mean.data = torch.tensor(
            data_mean_np, dtype=self.dtype, device=self.device
        )
        self.data_std.data = torch.tensor(
            data_std_np, dtype=self.dtype, device=self.device
        )

        maturities_tensor = maturities_tensor.to(device=self.device, dtype=self.dtype)

        train_dataset = YieldCurveDataset(train_sequences, self.device, self.dtype)
        val_dataset = YieldCurveDataset(val_sequences, self.device, self.dtype)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_lambda,
        )
        criterion = nn.MSELoss()

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=5, verbose=True, min_lr=1e-6
        )

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model_state = None
        best_optimizer_state = None
        best_epoch = -1

        history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}

        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self.train()
            total_train_loss = 0.0

            for batch in train_loader:
                input_window = batch["input_window"]
                target_yields = batch["target_yields"]
                horizon = batch["horizon"]

                optimizer.zero_grad()

                input_window_norm = self._normalize(input_window)
                target_yields_norm = self._normalize(target_yields)

                batch_loss_sum = 0
                unique_horizons = torch.unique(horizon)

                for h_int in unique_horizons:
                    h = h_int.item()
                    mask = horizon == h
                    input_window_h = input_window_norm[mask]
                    target_yields_h = target_yields_norm[mask]

                    if input_window_h.shape[0] == 0:
                        continue

                    y_pred_norm = self.forward(input_window_h, maturities_tensor, h)

                    loss = criterion(
                        y_pred_norm, target_yields_h
                    )
                    batch_loss_sum += (
                        loss * input_window_h.shape[0]
                    )

                if input_window.shape[0] > 0:
                    batch_loss_avg = batch_loss_sum / input_window.shape[0]
                else:
                    batch_loss_avg = torch.tensor(
                        0.0, device=self.device, requires_grad=True
                    )

                batch_loss_avg.backward()

                if self.gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), self.gradient_clipping
                    )

                optimizer.step()

                if input_window.shape[0] > 0:
                    total_train_loss += batch_loss_sum.item()

            avg_train_loss = (
                total_train_loss / len(train_dataset) if len(train_dataset) > 0 else 0.0
            )

            self.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_window = batch["input_window"]
                    target_yields = batch["target_yields"]
                    horizon = batch["horizon"]

                    input_window_norm = self._normalize(input_window)
                    target_yields_norm = self._normalize(target_yields)

                    batch_val_loss = 0
                    unique_horizons = torch.unique(horizon)
                    for h_int in unique_horizons:
                        h = h_int.item()
                        mask = horizon == h
                        input_window_h = input_window_norm[mask]
                        target_yields_h = target_yields_norm[mask]

                        if input_window_h.shape[0] == 0:
                            continue

                        y_pred_norm = self.forward(input_window_h, maturities_tensor, h)
                        loss = criterion(y_pred_norm, target_yields_h)
                        batch_val_loss += loss * input_window_h.shape[0]

                    if input_window.shape[0] > 0:
                        batch_val_loss /= input_window.shape[0]
                        total_val_loss += batch_val_loss.item() * input_window.shape[0]

            avg_val_loss = (
                total_val_loss / len(val_dataset)
                if len(val_dataset) > 0
                else float("inf")
            )
            epoch_duration = time.time() - epoch_start_time

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(avg_val_loss)

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["lr"].append(current_lr)

            print(
                f"Epoch [{epoch+1}/{self.epochs}] | Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.1e} | Duration: {epoch_duration:.2f}s"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_model_state = self.state_dict()
                best_optimizer_state = optimizer.state_dict()
                best_epoch = epoch
                print(
                    f"New best validation loss: {best_val_loss:.6f}. Saving best model state."
                )
                if checkpoint_path:
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": best_model_state,
                            "optimizer_state_dict": best_optimizer_state,
                            "best_val_loss": best_val_loss,
                            "hyperparams": self.hyperparams,
                        },
                        checkpoint_path / "best_model.pth",
                    )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(
                        f"Early stopping triggered after {epochs_no_improve} epochs without improvement."
                    )
                    break

            if checkpoint_path and (epoch + 1) % checkpoint_freq == 0:
                chkpt_file = checkpoint_path / f"checkpoint_epoch_{epoch+1}.pth"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": avg_val_loss,
                        "hyperparams": self.hyperparams,
                    },
                    chkpt_file,
                )
                print(f"Checkpoint saved to {chkpt_file}")

        if best_model_state:
            print(f"Loading best model state from epoch {best_epoch + 1}...")
            self.load_state_dict(best_model_state)

        self.training_time = time.time() - start_time
        print(f"Training finished. Total time: {self.training_time:.2f}s")
        print(f"Best Validation Loss: {best_val_loss:.6f} at epoch {best_epoch + 1}")

        return history

    def predict(
        self,
        input_window: np.ndarray | torch.Tensor,
        maturities_tensor: np.ndarray | torch.Tensor,
        horizon: int,
    ) -> np.ndarray:
        """
        Realiza a previsão para uma única janela de entrada.
        """
        self.eval()

        if isinstance(input_window, np.ndarray):
            input_window = torch.tensor(input_window, dtype=self.dtype)
        input_window = input_window.to(self.device)

        if isinstance(maturities_tensor, np.ndarray):
            maturities_tensor = torch.tensor(maturities_tensor, dtype=self.dtype)
        maturities_tensor = maturities_tensor.to(self.device)

        if input_window.dim() == 2:
            input_window = input_window.unsqueeze(0)
        elif input_window.dim() != 3 or input_window.shape[0] != 1:
            raise ValueError(
                "input_window must have shape [W, M] or [1, W, M] for single prediction."
            )

        with torch.no_grad():
            input_window_norm = self._normalize(input_window)
            y_pred_norm = self.forward(input_window_norm, maturities_tensor, horizon)
            y_pred = self._denormalize(y_pred_norm)

        return y_pred.squeeze(0).cpu().numpy()

    def save(self, file_path: str | Path):
        """Salva o modelo treinado e os metadados necessários."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            "hyperparams": self.hyperparams,
            "model_state_dict": self.state_dict(),
            "training_time": getattr(self, "training_time", None),
        }
        torch.save(save_data, file_path)
        print(f"Model saved to {file_path}")

    @classmethod
    def load(cls, file_path: str | Path, device: str | None = None):
        """Carrega um modelo treinado."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found at {file_path}")

        load_device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint = torch.load(file_path, map_location="cpu")

        hyperparams = checkpoint["hyperparams"]
        dtype = checkpoint.get("dtype", torch.get_default_dtype())

        hyperparams["device"] = load_device
        model = cls(**hyperparams)

        model.load_state_dict(checkpoint["model_state_dict"])

        model.to(device=load_device)
        model.eval()
        print(f"Model loaded from {file_path} to device {load_device}")
        return model

    def plot_factor_loadings(
        self,
        maturities_tensor: torch.Tensor,
        input_window_sample: torch.Tensor | None = None,
        file_path: str | Path | None = None,
        title: str = "Learned Factor Loadings (H_t)",
    ):
        """
        Plota as cargas fatoriais H_t aprendidas pelo modelo para uma amostra de entrada.
        Nota: H_t não depende mais da matriz de transição, então o plot é o mesmo.
        """
        self.eval()
        maturities_tensor_dev = maturities_tensor.to(self.device, dtype=self.dtype)

        if input_window_sample is None:
            print(
                "Warning: input_window_sample not provided. Using zeros input for H_t generation."
            )
            input_window_sample_dev = torch.zeros(
                (1, self.window_size, self.n_maturities),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            if isinstance(input_window_sample, np.ndarray):
                input_window_sample = torch.tensor(
                    input_window_sample, dtype=self.dtype
                )
            input_window_sample_dev = input_window_sample.to(self.device)
            if input_window_sample_dev.dim() == 2:
                input_window_sample_dev = input_window_sample_dev.unsqueeze(0)
            if (
                input_window_sample_dev.shape[0] != 1
                or input_window_sample_dev.shape[1] != self.window_size
                or input_window_sample_dev.shape[2] != self.n_maturities
            ):
                raise ValueError(
                    f"input_window_sample must have shape [1, {self.window_size}, {self.n_maturities}]"
                )

        with torch.no_grad():
            input_window_norm = self._normalize(input_window_sample_dev)
            transformer_output = self.transformer_encoder(input_window_norm)
            last_time_step_output = transformer_output[:, -1, :]
            H_t = self.maturity_processor(last_time_step_output, maturities_tensor_dev)

        H_t_np = H_t.squeeze(0).cpu().numpy()
        maturities_np = maturities_tensor.cpu().numpy().flatten()

        plt.figure(figsize=(10, 6))
        for i in range(self.n_factors):
            label = f"Factor {i+1}"
            if self.intercept_term and i == self.n_factors - 1:
                label += " (Intercept)"
            plt.plot(maturities_np, H_t_np[:, i], label=label)

        plt.xlabel("Maturity (Units from data)")
        plt.ylabel("Factor Loading")
        plt.title(title)
        plt.legend()
        plt.grid(True)

        if file_path:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(file_path)
            print(f"Factor loadings plot saved to {file_path}")
            plt.close()
        else:
            plt.show()

class YieldCurveDataset(torch.utils.data.Dataset):
    def __init__(self, sequences: list[dict], device, dtype):
        self.sequences = sequences
        self.device = device
        self.dtype = dtype

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            "input_window": torch.tensor(
                seq["input_window"], dtype=self.dtype, device=self.device
            ),
            "target_yields": torch.tensor(
                seq["target_yields"], dtype=self.dtype, device=self.device
            ),
            "horizon": torch.tensor(
                seq["horizon"], dtype=torch.long, device=self.device
            ),
        }


def train_and_evaluate_standalone(args):
    """
    Função principal para treinar e avaliar o modelo NNSSTransformerModel standalone.
    """
    print("Starting Standalone Training and Evaluation for NNSSTransformerModel")
    print(f"Arguments: {vars(args)}")

    device = (
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    if not args.sequences_dir:
        raise ValueError("Argument --sequences_dir must be provided.")

    base_sequences_dir = Path(args.sequences_dir)
    sequences_subdir = base_sequences_dir / "sequences"

    if not sequences_subdir.is_dir():
        raise FileNotFoundError(
            f"Subdirectory 'sequences' not found inside: {base_sequences_dir}"
        )

    try:
        maturities_path_subdir = sequences_subdir / "maturities.npy"
        maturities_path_base = base_sequences_dir / "maturities.npy"
        maturities_path_txt = sequences_subdir / "maturities.txt"

        if maturities_path_subdir.exists():
            maturities = np.load(maturities_path_subdir)
            print(f"Loaded maturities from {maturities_path_subdir}")
        elif maturities_path_base.exists():
            maturities = np.load(maturities_path_base)
            print(f"Loaded maturities from {maturities_path_base}")
        elif maturities_path_txt.exists():
            maturities = np.loadtxt(maturities_path_txt, delimiter=",")
            print(f"Loaded maturities from {maturities_path_txt}")
        else:
            raise FileNotFoundError(
                f"maturities.npy or maturities.txt not found in {sequences_subdir} or {base_sequences_dir}"
            )

        n_maturities = len(maturities)
        maturities_tensor_cpu = torch.tensor(
            maturities, dtype=torch.get_default_dtype()
        ).unsqueeze(-1)
        print(f"Maturities (n={n_maturities}): {maturities}")

    except Exception as e:
        print(f"Error loading maturities file: {e}")
        sys.exit(1)

    train_sequences_list = []
    val_sequences_list = []
    test_data_by_horizon = {}

    print(
        f"\nLoading sequences for window_size={args.window_size} and horizons={args.horizons}..."
    )
    all_horizons_found = True
    possible_horizons_list = [int(h) for h in args.horizons]

    for horizon in possible_horizons_list:
        print(f"  Loading horizon h={horizon}...")
        try:
            fname_base = f"seq_w{args.window_size}_h{horizon}"
            X_train_path = sequences_subdir / f"{fname_base}_X_train.pt"
            y_train_path = sequences_subdir / f"{fname_base}_y_train.pt"
            X_val_path = sequences_subdir / f"{fname_base}_X_val.pt"
            y_val_path = sequences_subdir / f"{fname_base}_y_val.pt"
            X_test_path = sequences_subdir / f"{fname_base}_X_test.pt"
            y_test_path = sequences_subdir / f"{fname_base}_y_test.pt"

            X_train = torch.load(X_train_path, map_location="cpu")
            y_train = torch.load(y_train_path, map_location="cpu")
            X_val = torch.load(X_val_path, map_location="cpu")
            y_val = torch.load(y_val_path, map_location="cpu")
            X_test = torch.load(X_test_path, map_location="cpu")
            y_test = torch.load(y_test_path, map_location="cpu")

            default_dtype = torch.get_default_dtype()
            X_train = X_train.to(dtype=default_dtype)
            y_train = y_train.to(dtype=default_dtype)
            X_val = X_val.to(dtype=default_dtype)
            y_val = y_val.to(dtype=default_dtype)
            X_test = X_test.to(dtype=default_dtype)
            y_test = y_test.to(dtype=default_dtype)

            for i in range(X_train.shape[0]):
                train_sequences_list.append(
                    {
                        "input_window": X_train[i].numpy(),
                        "target_yields": y_train[i].numpy(),
                        "horizon": horizon,
                    }
                )
            for i in range(X_val.shape[0]):
                val_sequences_list.append(
                    {
                        "input_window": X_val[i].numpy(),
                        "target_yields": y_val[i].numpy(),
                        "horizon": horizon,
                    }
                )

            test_data_by_horizon[horizon] = {
                "X": X_test,
                "y": y_test,
            }
            print(f"    Loaded train/val/test for h={horizon}.")

        except FileNotFoundError as e:
            print(
                f"    ERROR: File not found for h={horizon}, w={args.window_size}: {e}"
            )
            print(
                f"    Skipping horizon {horizon}. Ensure files generated with correct parameters exist."
            )
            all_horizons_found = False
        except Exception as e:
            print(f"    ERROR loading data for h={horizon}: {e}")
            all_horizons_found = False

    if not train_sequences_list or not val_sequences_list:
        print(
            "\nError: No valid training or validation sequences were loaded for the specified horizons/window size."
        )
        sys.exit(1)

    print(
        f"\nTotal sequences loaded: {len(train_sequences_list)} train, {len(val_sequences_list)} val."
    )
    print(f"Test sequences loaded for horizons: {list(test_data_by_horizon.keys())}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        model_dir_name = f"nnsstf_f{args.n_factors}_w{args.window_size}_d{args.d_model}_h{args.nhead}_l{args.num_encoder_layers}_{args.model_name}"
        output_dir = Path("./results") / model_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory set to: {output_dir}")

    checkpoint_dir_path = None
    if args.checkpoint_dir:
        checkpoint_dir_path = output_dir / "checkpoints"
        checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoints enabled, will be saved to: {checkpoint_dir_path}")
    else:
        print("Checkpoints disabled.")

    model = NNSSTransformerModel(
        maturities=maturities,
        n_factors=args.n_factors,
        window_size=args.window_size,
        possible_horizons=possible_horizons_list,
        intercept_term=not args.no_intercept,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        transformer_dropout=args.transformer_dropout,
        maturity_processor_hidden=args.mp_hidden,
        maturity_processor_layers=args.mp_layers,
        maturity_processor_dropout=args.mp_dropout,
        horizon_embedding_dim=args.horizon_emb_dim,
        ft_mlp_hidden_dim=args.ft_hidden,
        ft_mlp_layers=args.ft_layers,
        ft_dropout=args.ft_dropout,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        device=device,
        gradient_clipping=args.gradient_clipping,
        l2_lambda=args.l2_lambda,
    )

    print("\n--- Starting Training ---")
    history = model.fit(
        train_sequences_list,
        val_sequences_list,
        maturities_tensor_cpu,
        checkpoint_dir=checkpoint_dir_path,
        checkpoint_freq=args.checkpoint_freq,
    )
    print("--- Training Finished ---")

    print("\n--- Starting Evaluation on Test Set ---")
    test_predictions = {}
    test_targets = {}
    test_metrics_detailed = {}
    test_factors_loadings = {}
    basis_point_factor = 100

    if not test_data_by_horizon:
        print("Warning: No test data loaded for any horizon. Skipping evaluation.")
    else:
        model.eval()
        with torch.no_grad():
            maturities_tensor_dev = maturities_tensor_cpu.to(model.device)
            for horizon_int in test_data_by_horizon.keys():
                horizon = int(horizon_int)
                print(f"  Evaluating horizon h={horizon}...")
                test_data = test_data_by_horizon[horizon]
                X_test_h = test_data["X"].to(model.device)
                y_test_h = test_data["y"].to(model.device)

                horizon_predictions_list = []
                horizon_targets_list = []
                horizon_xt_list = []
                horizon_xpred_list = []
                horizon_ht_list = []

                for i in range(X_test_h.shape[0]):
                    single_window = X_test_h[i].unsqueeze(0)
                    target_yields = y_test_h[i]

                    input_window_norm = model._normalize(single_window)
                    transformer_output = model.transformer_encoder(
                        input_window_norm
                    )
                    last_time_step_output = transformer_output[:, -1, :]
                    x_t = model.factor_projection(last_time_step_output)
                    H_t = model.maturity_processor(
                        last_time_step_output, maturities_tensor_dev
                    )
                    x_pred = model.factor_transition(
                        x_t, last_time_step_output, horizon
                    )
                    y_pred_norm = torch.bmm(H_t, x_pred.unsqueeze(-1)).squeeze(
                        -1
                    )
                    y_pred = model._denormalize(y_pred_norm)

                    horizon_predictions_list.append(y_pred.squeeze(0).cpu().numpy())
                    horizon_targets_list.append(target_yields.cpu().numpy())
                    horizon_xt_list.append(x_t.squeeze(0).cpu().numpy())
                    horizon_xpred_list.append(x_pred.squeeze(0).cpu().numpy())
                    horizon_ht_list.append(H_t.squeeze(0).cpu().numpy())

                if not horizon_predictions_list:
                    print(
                        f"    No predictions generated for horizon {horizon}. Skipping."
                    )
                    continue

                horizon_predictions_np = np.stack(horizon_predictions_list)
                horizon_targets_np = np.stack(horizon_targets_list)
                horizon_xt_np = np.stack(horizon_xt_list)
                horizon_xpred_np = np.stack(horizon_xpred_list)
                horizon_ht_np = np.stack(horizon_ht_list)

                test_predictions[horizon] = horizon_predictions_np
                test_targets[horizon] = horizon_targets_np

                test_factors_loadings[f"h{horizon}_xt"] = horizon_xt_np
                test_factors_loadings[f"h{horizon}_xpred"] = horizon_xpred_np
                test_factors_loadings[f"h{horizon}_ht"] = horizon_ht_np

                diff_bps = (
                    horizon_predictions_np - horizon_targets_np
                ) * basis_point_factor
                rmse_per_maturity_bps = np.sqrt(np.mean(diff_bps**2, axis=0))
                mae_per_maturity_bps = np.mean(np.abs(diff_bps), axis=0)

                test_metrics_detailed[horizon] = {
                    "maturities": model.maturities,
                    "rmse_bps": rmse_per_maturity_bps,
                    "mae_bps": mae_per_maturity_bps,
                }

                avg_rmse_bps = np.mean(rmse_per_maturity_bps)
                avg_mae_bps = np.mean(mae_per_maturity_bps)

                print(
                    f"    Horizon {horizon}: Avg Test RMSE = {avg_rmse_bps:.2f} bps, Avg Test MAE = {avg_mae_bps:.2f} bps"
                )

    print(f"\nSaving results to {output_dir}")

    if test_metrics_detailed:
        metrics_rows = []
        for horizon, metrics in test_metrics_detailed.items():
            for i, mat in enumerate(metrics["maturities"]):
                metrics_rows.append(
                    {
                        "Horizon": horizon,
                        "Maturity": mat,
                        "RMSE_bps": metrics["rmse_bps"][i],
                        "MAE_bps": metrics["mae_bps"][i],
                    }
                )
        detailed_metrics_df = pd.DataFrame(metrics_rows)
        detailed_metrics_df.to_csv(
            output_dir / "test_metrics_detailed_bps.csv", index=False
        )
        print("Detailed test metrics (per maturity, in BPS) saved.")

        avg_metrics_df = (
            detailed_metrics_df.groupby("Horizon")[["RMSE_bps", "MAE_bps"]]
            .mean()
            .reset_index()
        )
        avg_metrics_df.to_csv(output_dir / "test_metrics_average_bps.csv", index=False)
        print("Average test metrics (per horizon, in BPS) saved.")

    else:
        print("No test metrics to save.")

    if args.save_predictions and test_predictions:
        test_predictions_str_keys = {str(k): v for k, v in test_predictions.items()}
        test_targets_str_keys = {str(k): v for k, v in test_targets.items()}
        np.savez(output_dir / "test_predictions.npz", **test_predictions_str_keys)
        np.savez(output_dir / "test_targets.npz", **test_targets_str_keys)
        print("Test predictions and targets saved.")
    elif args.save_predictions:
        print("Save predictions enabled, but no test predictions to save.")

    if test_factors_loadings:
        np.savez(output_dir / "test_factors_loadings.npz", **test_factors_loadings)
        print("Test factors (x_t, x_pred) and loadings (H_t) saved.")
    else:
        print("No test factors/loadings to save.")

    model.save(output_dir / "trained_model.pth")

    hyperparams_path = output_dir / "hyperparameters.json"
    import json

    with open(hyperparams_path, "w") as f:
        final_hyperparams = model.hyperparams.copy()
        final_hyperparams["command_line_args"] = vars(args)
        final_hyperparams["sequences_dir"] = str(base_sequences_dir)
        final_hyperparams["output_dir"] = str(output_dir)
        final_hyperparams["model_name"] = args.model_name
        final_hyperparams["device_used"] = model.device
        final_hyperparams["training_time_seconds"] = getattr(
            model, "training_time", None
        )
        final_hyperparams["best_val_loss"] = (
            history.get("val_loss", [None])[-1] if history else None
        )

        if "command_line_args" in final_hyperparams:
            del final_hyperparams["command_line_args"]["device"]
            for key, value in final_hyperparams["command_line_args"].items():
                if isinstance(value, Path):
                    final_hyperparams["command_line_args"][key] = str(value)

        json.dump(
            final_hyperparams,
            f,
            indent=4,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x),
        )
    print(f"Hyperparameters saved to {hyperparams_path}")

    if args.plot_loadings:
        sample_seq_dict = None
        loaded_test_sequences_for_plot = []
        if test_data_by_horizon:
            first_h_with_test = list(test_data_by_horizon.keys())[0]
            X_test_sample = test_data_by_horizon[first_h_with_test]["X"]
            if X_test_sample.shape[0] > 0:
                loaded_test_sequences_for_plot.append(
                    {"input_window": X_test_sample[0].numpy()}
                )

        if loaded_test_sequences_for_plot:
            sample_seq_dict = loaded_test_sequences_for_plot[0]
        elif val_sequences_list:
            print(
                "Warning: No test sequences loaded for plotting. Using validation sample."
            )
            sample_seq_dict = val_sequences_list[0]
        elif train_sequences_list:
            print(
                "Warning: No test/val sequences loaded for plotting. Using training sample."
            )
            sample_seq_dict = train_sequences_list[0]

        if sample_seq_dict:
            input_sample = torch.tensor(
                sample_seq_dict["input_window"], dtype=model.dtype
            )
            try:
                model.plot_factor_loadings(
                    maturities_tensor=maturities_tensor_cpu,
                    input_window_sample=input_sample,
                    file_path=output_dir / "factor_loadings.png",
                )
            except Exception as e:
                print(f"Error plotting factor loadings: {e}")
        else:
            print("Cannot plot factor loadings: No sequences loaded.")
    elif args.plot_loadings:
        print("Plot loadings enabled, but cannot plot: No sequences loaded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and Evaluate NNSSTransformerModel with Factor Transition"
    )

    parser.add_argument(
        "--sequences_dir",
        type=str,
        required=True,
        help="Directory containing pre-generated sequences (parent of 'sequences' subdir). Example: 'insira o path aqui'",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        required=True,
        help="List of prediction horizons (e.g., 5 20 60).",
    )
    parser.add_argument(
        "--window_size", type=int, required=True, help="Input window size."
    )
    parser.add_argument(
        "--n_factors", type=int, default=3, help="Number of latent factors."
    )
    parser.add_argument(
        "--no_intercept",
        action="store_true",
        help="Do not include intercept term in Maturity Processor.",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=32,
        help="Transformer internal dimension (embedding size).",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=2,
        help="Number of attention heads (must divide d_model).",
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=2,
        help="Number of Transformer encoder layers.",
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=64,
        help="Dimension of the feedforward network model in Transformer.",
    )
    parser.add_argument(
        "--transformer_dropout",
        type=float,
        default=0.25,
        help="Dropout rate in Transformer.",
    )
    parser.add_argument(
        "--mp_hidden",
        type=int,
        default=32,
        help="Maturity Processor MLP hidden dimension.",
    )
    parser.add_argument(
        "--mp_layers",
        type=int,
        default=1,
        help="Number of hidden layers in Maturity Processor MLP.",
    )
    parser.add_argument(
        "--mp_dropout",
        type=float,
        default=0.25,
        help="Dropout rate in Maturity Processor MLP.",
    )

    parser.add_argument(
        "--horizon_emb_dim",
        type=int,
        default=16,
        help="Dimension for the horizon embedding vector.",
    )
    parser.add_argument(
        "--ft_hidden",
        type=int,
        default=32,
        help="Hidden dimension in the Factor Transition MLP.",
    )
    parser.add_argument(
        "--ft_layers",
        type=int,
        default=1,
        help="Number of hidden layers in the Factor Transition MLP.",
    )
    parser.add_argument(
        "--ft_dropout",
        type=float,
        default=0.25,
        help="Dropout rate in the Factor Transition MLP.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Training batch size."
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate.")
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Patience for early stopping.",
    )
    parser.add_argument(
        "--gradient_clipping",
        type=float,
        default=1.0,
        help="Gradient clipping value (norm).",
    )
    parser.add_argument(
        "--l2_lambda",
        type=float,
        default=1e-4,
        help="L2 regularization (weight decay) coefficient.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cpu' or 'cuda'). Auto-detects if None.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save all results and the model. If None, a directory name is generated based on parameters under 'insira o path aqui'.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="run",
        help="Identifier added to the default output directory name if output_dir is None.",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save test predictions and targets to NPZ files.",
    )
    parser.add_argument(
        "--plot_loadings",
        action="store_true",
        help="Generate and save a plot of learned factor loadings using a sample input.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        action="store_true",
        help="Enable saving checkpoints. If enabled, checkpoints are saved in a 'checkpoints' subdirectory within the main output_dir.",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=10,
        help="Frequency (in epochs) to save checkpoints, if checkpoint saving is enabled.",
    )

    args = parser.parse_args()

    if args.d_model % args.nhead != 0:
        parser.error(
            f"--d_model ({args.d_model}) must be divisible by --nhead ({args.nhead})"
        )
    if args.window_size <= 0:
        parser.error("--window_size must be positive.")
    if args.horizon_emb_dim <= 0:
        parser.error("--horizon_emb_dim must be positive.")
    if args.ft_hidden <= 0:
        parser.error("--ft_hidden must be positive.")
    if args.ft_layers < 0:
        parser.error("--ft_layers cannot be negative.")
    if args.mp_hidden <= 0:
        parser.error("--mp_hidden must be positive.")
    if args.mp_layers < 0:
        parser.error("--mp_layers cannot be negative.")
    if args.num_encoder_layers <= 0:
        parser.error("--num_encoder_layers must be positive.")
    if args.dim_feedforward <= 0:
        parser.error("--dim_feedforward must be positive.")
    if args.n_factors <= 0:
        parser.error("--n_factors must be positive.")
    if args.epochs <= 0:
        parser.error("--epochs must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive.")
    if args.lr <= 0:
        parser.error("--lr must be positive.")
    if args.patience < 0:
        parser.error("--patience cannot be negative.")
    if args.gradient_clipping is not None and args.gradient_clipping <= 0:
        parser.error("--gradient_clipping must be positive if specified.")
    if args.l2_lambda < 0:
        parser.error("--l2_lambda cannot be negative.")
    if args.checkpoint_freq <= 0:
        parser.error("--checkpoint_freq must be positive.")
    if args.checkpoint_dir and args.output_dir is None:
        print(
            "Warning: Checkpoints enabled but output_dir is not specified. Checkpoints will be saved relative to the automatically generated output directory."
        )

    train_and_evaluate_standalone(args)

