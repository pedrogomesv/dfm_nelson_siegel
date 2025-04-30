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
import torch.utils.data

torch.set_default_dtype(torch.float64)

from src.models.base_model import BaseYieldCurveModel


class MaturityGraphNN(nn.Module):
    """
    Mecanismo de grafo para processar interdependências entre maturidades usando
    propagação de mensagens em um grafo completamente conectado.
    """

    def __init__(
        self,
        dim,
        n_message_passing_layers=3,
        edge_weight_type="log_distance",
        aggregation_type="weighted_mean",
        update_type="gru",
        dropout=0.5,
    ):
        super().__init__()
        self.dim = dim
        self.n_message_passing_layers = n_message_passing_layers
        self.edge_weight_type = edge_weight_type
        self.aggregation_type = aggregation_type
        self.update_type = update_type
        self.dtype = torch.get_default_dtype()

        self.message_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        dim * 2, dim, dtype=self.dtype
                    ), 
                    nn.LayerNorm(dim, dtype=self.dtype),
                    nn.GELU(),
                    nn.Linear(dim, dim, dtype=self.dtype),
                )
                for _ in range(n_message_passing_layers)
            ]
        )

        if update_type == "gru":
            self.node_update_fns = nn.ModuleList(
                [nn.GRUCell(dim, dim) for _ in range(n_message_passing_layers)]
            )
        else:  
            self.node_update_fns = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(
                            dim * 2, dim, dtype=self.dtype
                        ), 
                        nn.LayerNorm(dim, dtype=self.dtype),
                        nn.GELU(),
                        nn.Linear(dim, dim, dtype=self.dtype),
                    )
                    for _ in range(n_message_passing_layers)
                ]
            )

        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(dim, dtype=self.dtype)
                for _ in range(n_message_passing_layers)
            ]
        )

        self.layer_norms = nn.ModuleList(
            [
                nn.LayerNorm(dim, dtype=self.dtype)
                for _ in range(n_message_passing_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        self.edge_weights = None
        self.node_features = None

    def _compute_edge_weights(self, seq_len, maturities=None):
        """
        Calcula os pesos das arestas com base no tipo de peso especificado.

        Args:
            seq_len: Número de nós (maturidades)
            maturities: Tensor de maturidades em anos (opcional)

        Returns:
            Tensor de pesos das arestas de forma [seq_len, seq_len]
        """
        weights = torch.eye(seq_len, device=self.get_device())

        if self.edge_weight_type == "uniform":
            weights = torch.ones((seq_len, seq_len), device=self.get_device())

        elif self.edge_weight_type == "log_distance" and maturities is not None:
            for i in range(seq_len):
                for j in range(seq_len):
                    if i != j:
                        log_dist = torch.abs(
                            torch.log(maturities[i]) - torch.log(maturities[j])
                        )
                        weights[i, j] = 1.0 / (1.0 + log_dist)

        elif self.edge_weight_type == "inverse_distance" and maturities is not None:
            for i in range(seq_len):
                for j in range(seq_len):
                    if i != j:
                        dist = torch.abs(maturities[i] - maturities[j])
                        weights[i, j] = 1.0 / (1.0 + dist)

        return weights

    def get_device(self):
        """Retorna o dispositivo do primeiro parâmetro do módulo"""
        return next(self.parameters()).device

    def forward(self, x, maturities=None):
        """
        Forward pass do mecanismo de grafo.

        Args:
            x: Tensor de forma [batch_size, seq_len, dim]
            maturities: Tensor de maturidades em anos (opcional)

        Returns:
            Tensor processado com propagação de mensagens de forma [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        residual = x

        if maturities is not None:
            self.edge_weights = self._compute_edge_weights(
                seq_len, maturities.squeeze()
            )
        else:
            self.edge_weights = self._compute_edge_weights(seq_len)

        node_features = x.clone()
        self.node_features = []

        for layer in range(self.n_message_passing_layers):
            self.node_features.append(node_features.detach().clone())

            messages = torch.zeros_like(node_features)
            for i in range(seq_len):
                for j in range(seq_len):
                    if i != j:
                        node_pair = torch.cat(
                            [node_features[:, i], node_features[:, j]], dim=-1
                        )

                        message = self.message_projections[layer](node_pair)

                        message = message * self.edge_weights[i, j]

                        messages[:, i] += message

            
            if self.aggregation_type == "weighted_mean":
                weight_sum = (
                    self.edge_weights.sum(dim=1) - 1.0
                )
                weight_sum = weight_sum.unsqueeze(0).unsqueeze(-1).expand_as(messages)

                weight_sum = torch.clamp(weight_sum, min=1e-6)

                aggregated_messages = messages / weight_sum
            else:
                aggregated_messages = messages

            aggregated_messages = self.dropout(aggregated_messages)

            updated_features = torch.zeros_like(node_features)
            for i in range(seq_len):
                if self.update_type == "gru":
                    h_i = self.node_update_fns[layer](
                        aggregated_messages[:, i], node_features[:, i]
                    )
                    updated_features[:, i] = h_i
                else:  
                    combined = torch.cat(
                        [node_features[:, i], aggregated_messages[:, i]], dim=-1
                    )

                    updated_features[:, i] = self.node_update_fns[layer](combined)

            node_features = self.norms[layer](updated_features)

            if layer == self.n_message_passing_layers - 1:
                node_features = node_features + residual

        self.node_features.append(node_features.detach().clone())

        return node_features

    def get_edge_weights(self):
        """Retorna os pesos das arestas do último forward pass."""
        return self.edge_weights

    def get_node_features(self):
        """Retorna as características dos nós do último forward pass."""
        return self.node_features


class ResidualBlock(nn.Module):
    """
    Bloco residual com normalização e ativação GELU.
    """

    def __init__(self, dim):
        super().__init__()
        self.dtype = torch.get_default_dtype()
        self.block = nn.Sequential(
            nn.Linear(dim, dim, dtype=self.dtype),
            nn.LayerNorm(dim, dtype=self.dtype),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(dim, dim, dtype=self.dtype),
            nn.LayerNorm(dim, dtype=self.dtype),
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(x + self.block(x))


class WindowAwareGraphBasisLearner(nn.Module):
    """
    Versão window-aware do GraphBasisLearner que integra informações da janela temporal.
    Esta classe processa uma janela de observações passadas para gerar cargas fatoriais dependentes
    do contexto temporal.
    """

    def __init__(
        self,
        n_maturities,  
        n_factors=4,
        n_hidden=64,
        n_blocks=2,
        window_size=20,
        gru_hidden_size=32,
        gru_num_layers=1,
        gru_dropout=0.2,
        n_message_passing_layers=3,
        edge_weight_type="log_distance",
        aggregation_type="weighted_mean",
        update_type="gru",
        intercept_term=True,
    ):
        super().__init__()

        self.n_factors = n_factors
        self.intercept_term = intercept_term
        self.dtype = torch.get_default_dtype()
        self.n_hidden = n_hidden
        self.edge_weight_type = edge_weight_type
        self.window_size = window_size
        self.gru_hidden_size = gru_hidden_size
        self.n_maturities = n_maturities  

        self.window_processor = nn.GRU(
            input_size=n_maturities,  
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=gru_dropout if gru_num_layers > 1 else 0,
        )

        self.input_embedding = nn.Sequential(
            nn.Linear(1, n_hidden, dtype=self.dtype),
            nn.LayerNorm(n_hidden, dtype=self.dtype),
            nn.GELU(),
        )

        self.graph_nn = MaturityGraphNN(
            dim=n_hidden
            + gru_hidden_size,  
            n_message_passing_layers=n_message_passing_layers,
            edge_weight_type=edge_weight_type,
            aggregation_type=aggregation_type,
            update_type=update_type,
        )

        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(n_hidden + gru_hidden_size) for _ in range(n_blocks)]
        )

        output_size = n_factors - 1 if intercept_term else n_factors

        self.output_layer = nn.Sequential(
            nn.Linear(n_hidden + gru_hidden_size, output_size, dtype=self.dtype),
            nn.Sigmoid(),  
        )

    def forward(self, x_maturity, input_window):
        """
        Forward pass da rede com processamento conjunto via grafo, considerando a janela temporal.

        Args:
            x_maturity: Tensor de maturidades [n_maturities, 1]
            input_window: Tensor da janela de entrada [batch_size, window_size, n_maturities]

        Returns:
            Tensor de cargas fatoriais de forma [batch_size, n_maturities, n_factors]
        """
        if x_maturity.dtype != self.dtype:
            x_maturity = x_maturity.to(dtype=self.dtype)
        if input_window.dtype != self.dtype:
            input_window = input_window.to(dtype=self.dtype)

        batch_size = input_window.size(0)
        n_maturities = x_maturity.size(0)

        _, window_context = self.window_processor(input_window)
        window_context = window_context[-1]  

        maturity_embedding = self.input_embedding(
            x_maturity
        )  
        window_context_expanded = window_context.unsqueeze(1).expand(
            -1, n_maturities, -1
        )

        maturity_embedding_expanded = maturity_embedding.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        combined_representation = torch.cat(
            [maturity_embedding_expanded, window_context_expanded], dim=2
        )

        graph_processed = self.graph_nn(combined_representation, x_maturity)

        x_res = graph_processed
        for block in self.residual_blocks:
            x_res = block(x_res)

        basis_fns = self.output_layer(x_res)  

        if self.intercept_term:
            intercept = torch.ones(
                batch_size, n_maturities, 1, device=x_maturity.device, dtype=self.dtype
            )
            return torch.cat(
                [intercept, basis_fns], dim=2
            )  

        return basis_fns  

    def get_edge_weights(self):
        """
        Retorna os pesos das arestas do último forward pass.
        """
        if hasattr(self.graph_nn, "edge_weights"):
            return self.graph_nn.get_edge_weights()
        return None

    def get_node_features(self):
        """
        Retorna as características dos nós do último forward pass.
        """
        if hasattr(self.graph_nn, "node_features"):
            return self.graph_nn.get_node_features()
        return None


class FactorStateEstimator(nn.Module):
    """
    Estimador do estado dos fatores latentes a partir de uma janela de observações passadas.
    Esta classe processa uma janela temporal de curvas de juros para estimar o estado atual dos
    fatores latentes subjacentes à dinâmica da curva.
    """

    def __init__(
        self,
        n_maturities,
        n_factors,
        window_size,
        factor_gru_hidden_size=32,
        factor_gru_num_layers=2,
        factor_gru_dropout=0.3,
        use_attention=False,
    ):
        """
        Inicializa o estimador de estados dos fatores.

        Args:
            n_maturities: Número de maturidades na curva de juros.
            n_factors: Número de fatores latentes a serem estimados.
            window_size: Tamanho da janela de observações passadas.
            factor_gru_hidden_size: Dimensão do estado oculto do GRU.
            factor_gru_num_layers: Número de camadas do GRU.
            factor_gru_dropout: Taxa de dropout para o GRU (aplicado entre camadas).
            use_attention: Se deve usar um mecanismo de atenção sobre a sequência.
        """
        super().__init__()

        self.n_maturities = n_maturities
        self.n_factors = n_factors
        self.window_size = window_size
        self.factor_gru_hidden_size = factor_gru_hidden_size
        self.use_attention = use_attention
        self.dtype = torch.get_default_dtype()

        self.factor_gru = nn.GRU(
            input_size=n_maturities,
            hidden_size=factor_gru_hidden_size,
            num_layers=factor_gru_num_layers,
            batch_first=True,
            dropout=factor_gru_dropout if factor_gru_num_layers > 1 else 0,
            bidirectional=False,  
        )

        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(factor_gru_hidden_size, window_size, dtype=self.dtype),
                nn.Tanh(),
                nn.Softmax(dim=1),
            )

        self.projection_layers = nn.Sequential(
            nn.Linear(factor_gru_hidden_size, factor_gru_hidden_size, dtype=self.dtype),
            nn.LayerNorm(factor_gru_hidden_size, dtype=self.dtype),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(factor_gru_hidden_size, n_factors, dtype=self.dtype),
        )

    def forward(self, input_window):
        """
        Processa uma janela de curvas de juros para estimar o estado atual dos fatores.

        Args:
            input_window: Tensor de forma [batch_size, window_size, n_maturities]
                         representando a janela de observações passadas.

        Returns:
            x_t: Tensor de forma [batch_size, n_factors] representando o estado estimado
                dos fatores latentes no tempo t (fim da janela).
        """
        if input_window.dtype != self.dtype:
            input_window = input_window.to(dtype=self.dtype)

        batch_size = input_window.size(0)

        output, hidden = self.factor_gru(input_window)


        if self.use_attention:
            attention_weights = self.attention(hidden[-1])  

            attention_weights = attention_weights.unsqueeze(
                2
            )  

            weighted_output = (
                output * attention_weights
            )  

            context = weighted_output.sum(dim=1)  
        else:
            context = hidden[-1]  

        x_t = self.projection_layers(context)  

        return x_t


class NNSSGraphModel(BaseYieldCurveModel):
    """
    Modelo NNSS baseado em grafos para previsão da curva de rendimentos.
    """

    def __init__(
        self,
        maturities,
        n_factors=4,
        n_hidden=64,
        n_blocks=2,
        window_size=20,
        gru_hidden_size=32,
        factor_gru_hidden_size=32,
        n_message_passing_layers=3,
        edge_weight_type="inverse_distance",
        aggregation_type="weighted_mean",
        update_type="gru",
        weight_decay=0.01,
        device="cpu",
    ):
        """
        Inicializar o modelo NNSS baseado em grafos com processamento de janelas temporais.

        Args:
            maturities: Lista de maturidades para modelar.
            n_factors: Número de fatores latentes.
            n_hidden: Dimensão oculta para processamento de maturidades.
            n_blocks: Número de blocos residuais.
            window_size: Tamanho da janela de observações históricas.
            gru_hidden_size: Dimensão oculta do GRU para o WindowAwareGraphBasisLearner.
            factor_gru_hidden_size: Dimensão oculta do GRU para o FactorStateEstimator.
            n_message_passing_layers: Número de camadas de propagação de mensagens no grafo.
            edge_weight_type: Tipo de peso das arestas ('uniform', 'log_distance', 'inverse_distance').
            aggregation_type: Tipo de agregação no grafo ('sum', 'weighted_mean').
            update_type: Tipo de atualização nos nós do grafo ('gru', 'mlp').
            weight_decay: Regularização L2 para os parâmetros do modelo.
            device: Dispositivo de processamento ('cpu' ou 'cuda').
        """
        name = f"NNSS_Graph_Seq_{n_factors}_w{window_size}"
        super().__init__(name, maturities)

        self.n_factors = n_factors
        self.n_hidden = n_hidden
        self.n_blocks = n_blocks
        self.window_size = window_size
        self.gru_hidden_size = gru_hidden_size
        self.factor_gru_hidden_size = factor_gru_hidden_size
        self.n_message_passing_layers = n_message_passing_layers
        self.edge_weight_type = edge_weight_type
        self.aggregation_type = aggregation_type
        self.update_type = update_type
        self.weight_decay = weight_decay
        self.dtype = torch.get_default_dtype()
        self.n_maturities = len(maturities)

        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.basis_learner = WindowAwareGraphBasisLearner(
            n_maturities=self.n_maturities,  
            n_factors=n_factors,
            n_hidden=n_hidden,
            n_blocks=n_blocks,
            window_size=window_size,
            gru_hidden_size=gru_hidden_size,
            n_message_passing_layers=n_message_passing_layers,
            edge_weight_type=edge_weight_type,
            aggregation_type=aggregation_type,
            update_type=update_type,
        ).to(self.device)

        self.factor_estimator = FactorStateEstimator(
            n_maturities=len(maturities),
            n_factors=n_factors,
            window_size=window_size,
            factor_gru_hidden_size=factor_gru_hidden_size,
        ).to(self.device)

        self.transition_matrix = nn.Parameter(torch.eye(n_factors, dtype=self.dtype))

        self.hyperparams = {
            "n_blocks": n_blocks,
            "n_message_passing_layers": n_message_passing_layers,
            "edge_weight_type": edge_weight_type,
            "aggregation_type": aggregation_type,
            "update_type": update_type,
            "window_size": window_size,
            "gru_hidden_size": gru_hidden_size,
            "factor_gru_hidden_size": factor_gru_hidden_size,
            "weight_decay": weight_decay,
        }

        self.data_mean = None
        self.data_std = None

        self.train_loss_history = None
        self.val_loss_history = None

    def forward(self, input_window, maturities_tensor, horizon=1):
        """
        Forward pass do modelo para previsão da curva de juros a partir de uma janela de observações.

        Args:
            input_window: Tensor de forma [batch_size, window_size, n_maturities]
                         representando a janela de observações passadas.
            maturities_tensor: Tensor de forma [n_maturities, 1] com os valores de maturidade.
            horizon: Horizonte de previsão (número de passos no futuro).

        Returns:
            y_pred: Tensor de forma [batch_size, n_maturities] com as previsões da curva
                   para horizon passos à frente.
        """
        if input_window.device != self.device:
            input_window = input_window.to(self.device)
        if maturities_tensor.device != self.device:
            maturities_tensor = maturities_tensor.to(self.device)

        if input_window.dtype != self.dtype:
            input_window = input_window.to(dtype=self.dtype)
        if maturities_tensor.dtype != self.dtype:
            maturities_tensor = maturities_tensor.to(dtype=self.dtype)

        batch_size = input_window.size(0)

        H_t = self.basis_learner(maturities_tensor, input_window)

        x_t = self.factor_estimator(input_window)

        x_pred = x_t.unsqueeze(-1)

        batch_transition = self.transition_matrix.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        for _ in range(horizon):
            x_pred = torch.bmm(batch_transition, x_pred)

        y_pred = torch.bmm(H_t, x_pred).squeeze(-1)

        return y_pred

    def fit(
        self,
        train_data,
        window_size=None,
        horizon=1,
        validation_split=0.2,
        batch_size=32,
        learning_rate=1e-3,
        validation_freq=1,
        early_stopping_patience=10,
        n_epochs=1000,
        normalize_data=True,
    ):
        """
        Ajustar o modelo NNSS Graph com processamento de janelas aos dados de treinamento.

        Args:
            train_data: DataFrame com dados de treinamento.
            window_size: Tamanho da janela de observações (se None, usa o valor definido na inicialização).
            horizon: Horizonte de previsão em passos.
            validation_split: Proporção dos dados para validação (entre 0 e 1).
            batch_size: Tamanho do lote para treinamento.
            learning_rate: Taxa de aprendizado para o otimizador.
            validation_freq: Frequência (em épocas) para avaliação no conjunto de validação.
            early_stopping_patience: Número de validações sem melhoria antes de parar.
            n_epochs: Número máximo de épocas de treinamento.
            normalize_data: Se deve normalizar os dados antes do treinamento.

        Returns:
            self: O modelo treinado.
        """
        start_time = time.time()

        if window_size is None:
            window_size = self.window_size
        else:
            self.window_size = window_size

        print(f"Treinando modelo com window_size={window_size}, horizon={horizon}")

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

        if normalize_data:
            self.data_mean = actual_train_data.mean().values
            self.data_std = actual_train_data.std().values

            self.data_std = np.where(self.data_std < 1e-8, 1.0, self.data_std)

            train_data_normalized = (
                actual_train_data.values - self.data_mean
            ) / self.data_std
        if val_data is not None:
            val_data_normalized = (val_data.values - self.data_mean) / self.data_std
        else:
            self.data_mean = np.zeros(actual_train_data.shape[1])
            self.data_std = np.ones(actual_train_data.shape[1])
            train_data_normalized = actual_train_data.values
            if val_data is not None:
                val_data_normalized = val_data.values

        train_windows, train_targets = self._create_sequential_windows(
            train_data_normalized, window_size, horizon
        )

        if val_data is not None:
            val_windows, val_targets = self._create_sequential_windows(
                val_data_normalized, window_size, horizon
            )

        maturities_array = np.array([float(m) for m in self.maturities])
        maturities_years = maturities_array / 12  

        maturities_tensor = torch.tensor(
            maturities_years.reshape(-1, 1), device=self.device, dtype=self.dtype
        )

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(train_windows, dtype=self.dtype, device=self.device),
            torch.tensor(train_targets, dtype=self.dtype, device=self.device),
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        if val_data is not None:
            val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(val_windows, dtype=self.dtype, device=self.device),
                torch.tensor(val_targets, dtype=self.dtype, device=self.device),
            )

            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )

        optimizer = torch.optim.Adam(
            [
                {"params": self.basis_learner.parameters()},
                {"params": self.factor_estimator.parameters()},
                {"params": self.transition_matrix},
            ],
            lr=learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=early_stopping_patience // 2,
            verbose=True,
            min_lr=1e-6,
        )

        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_epoch = 0
        no_improvement_count = 0
        best_state = None

        train_losses = []
        val_losses = []

        for epoch in range(n_epochs):
            self.basis_learner.train()
            self.factor_estimator.train()
            epoch_loss = 0.0

            for batch_windows, batch_targets in train_loader:
                optimizer.zero_grad()  

                y_pred = self.forward(batch_windows, maturities_tensor, horizon)

                loss = criterion(y_pred, batch_targets)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.basis_learner.parameters(), max_norm=1.0
                )
                torch.nn.utils.clip_grad_norm_(
                    self.factor_estimator.parameters(), max_norm=1.0
                )
                torch.nn.utils.clip_grad_norm_([self.transition_matrix], max_norm=1.0)

                optimizer.step()

                epoch_loss += loss.item() * batch_windows.size(0)

            epoch_loss /= len(train_dataset)
            train_losses.append(epoch_loss)

            if val_data is not None and (epoch + 1) % validation_freq == 0:
                self.basis_learner.eval()
                self.factor_estimator.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_windows, batch_targets in val_loader:
                        y_pred = self.forward(batch_windows, maturities_tensor, horizon)

                        loss = criterion(y_pred, batch_targets)
                        val_loss += loss.item() * batch_windows.size(0)

                val_loss /= len(val_dataset)
                val_losses.append(val_loss)

                scheduler.step(val_loss)

                print(
                    f"Época {epoch+1}/{n_epochs}, "
                    f"Treino Loss: {epoch_loss:.6f}, "
                    f"Validação Loss: {val_loss:.6f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    no_improvement_count = 0

                    best_state = {
                        "basis_learner": self.basis_learner.state_dict(),
                        "factor_estimator": self.factor_estimator.state_dict(),
                        "transition_matrix": self.transition_matrix.detach().clone(),
                    }

                    print(f"  Nova melhor validação loss: {best_val_loss:.6f}")
                else:
                    no_improvement_count += 1
                    print(
                        f"  Sem melhoria por {no_improvement_count} verificações "
                        f"(melhor: {best_val_loss:.6f} na época {best_epoch})"
                    )

                    if no_improvement_count >= early_stopping_patience:
                        print(
                            f"Early stopping após {early_stopping_patience} "
                            f"validações sem melhoria"
                        )
                        print(
                            f"Restaurando modelo de melhor validação da época {best_epoch}"
                        )

                        self.basis_learner.load_state_dict(best_state["basis_learner"])
                        self.factor_estimator.load_state_dict(
                            best_state["factor_estimator"]
                        )
                        self.transition_matrix.data.copy_(
                            best_state["transition_matrix"]
                        )
                        break
            elif (epoch + 1) % 10 == 0:
                print(f"Época {epoch+1}/{n_epochs}, Treino Loss: {epoch_loss:.6f}")

        if (
            val_data is not None
            and best_state is not None
            and no_improvement_count < early_stopping_patience
        ):
            print(
                f"Treinamento completo, restaurando melhor modelo da época {best_epoch}"
            )
            self.basis_learner.load_state_dict(best_state["basis_learner"])
            self.factor_estimator.load_state_dict(best_state["factor_estimator"])
            self.transition_matrix.data.copy_(best_state["transition_matrix"])

        self.train_loss_history = train_losses
        self.val_loss_history = val_losses if val_data is not None else None

        self.is_fitted = True
        self.training_time = time.time() - start_time

        print(f"Modelo ajustado em {self.training_time:.2f} segundos")
        if val_data is not None:
            print(
                f"Melhor performance na validação: {best_val_loss:.6f} (época {best_epoch})"
            )

        if val_data is not None and len(val_losses) > 0:
            epochs = list(range(validation_freq, n_epochs + 1, validation_freq))[
                : len(val_losses)
            ]
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, val_losses, label="Validação")

            sampled_train_epochs = np.arange(0, n_epochs, validation_freq)[
                : len(val_losses)
            ]
            sampled_train_losses = [train_losses[i] for i in sampled_train_epochs]
            plt.plot(epochs, sampled_train_losses, label="Treino")

            plt.xlabel("Época")
            plt.ylabel("Loss")
            plt.title(
                f"Histórico de Treino e Validação (window_size={window_size}, horizon={horizon})"
            )
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(
                f"{self.name}_w{window_size}_h{horizon}_training_history.png", dpi=300
            )
            plt.close()

        return self

    def _create_sequential_windows(self, data, window_size, horizon):
        """
        Criar janelas sequenciais a partir de dados temporais para modelos sequenciais.

        Args:
            data: Array numpy com as curvas de juros.
            window_size: Tamanho da janela (número de observações passadas).
            horizon: Horizonte de previsão (número de observações no futuro).

        Returns:
            X: Array de janelas de forma [n_samples, window_size, n_maturities]
            Y: Array de alvos de forma [n_samples, n_maturities]
        """
        n_samples = len(data) - window_size - horizon + 1
        n_maturities = data.shape[1]

        if n_samples <= 0:
            raise ValueError(
                f"Dados insuficientes para window_size={window_size} e horizon={horizon}"
            )

        X = np.zeros((n_samples, window_size, n_maturities))
        Y = np.zeros((n_samples, n_maturities))

        for i in range(n_samples):
            X[i] = data[i : i + window_size]
            Y[i] = data[
                i + window_size + horizon - 1
            ]  

        return X, Y

    def predict(self, data, horizon=1, window_size=None):
        """
        Gerar previsões para o horizonte dado usando a abordagem baseada em janelas.

        Args:
            data: DataFrame com dados históricos para previsão.
            horizon: Horizonte de previsão em passos.
            window_size: Tamanho da janela de observações (se None, usa o valor definido na inicialização).

        Returns:
            numpy.ndarray: Array com previsões para cada ponto do tempo (format rolling)
        """
        if not self.is_fitted:
            raise ValueError("Modelo deve ser ajustado antes da previsão")

        if window_size is None:
            window_size = self.window_size

        if len(data) < window_size:
            raise ValueError(
                f"Dados insuficientes para window_size={window_size}. "
                f"Necessário pelo menos {window_size} pontos."
            )

        maturities_array = np.array([float(m) for m in self.maturities])
        maturities_years = maturities_array / 12  

        maturities_tensor = torch.tensor(
            maturities_years.reshape(-1, 1), device=self.device, dtype=self.dtype
        )

        self.basis_learner.eval()
        self.factor_estimator.eval()

        if self.data_mean is not None and self.data_std is not None:
            data_normalized = (data.values - self.data_mean) / self.data_std
        else:
            data_normalized = data.values

        n_forecasts = len(data) - window_size + 1

        forecasts = np.zeros((n_forecasts, len(self.maturities)))

        with torch.no_grad():
            for i in range(n_forecasts):
                current_window = data_normalized[i : i + window_size]

                window_tensor = torch.tensor(
                    current_window.reshape(1, window_size, -1),
                    device=self.device,
                    dtype=self.dtype,
                )

                pred_model_scale = self.forward(
                    window_tensor, maturities_tensor, horizon
                )

                if self.data_mean is not None and self.data_std is not None:
                    std_tensor = torch.tensor(
                        self.data_std, device=self.device, dtype=self.dtype
                    )
                    mean_tensor = torch.tensor(
                        self.data_mean, device=self.device, dtype=self.dtype
                    )
                    pred = pred_model_scale * std_tensor + mean_tensor
            else:
                pred = pred_model_scale  

            forecasts[i] = pred.cpu().numpy()

        return forecasts

    def predict_multiple_horizons(self, data, horizons, window_size=None):
        """
        Gerar previsões para múltiplos horizontes.

        Args:
            data: DataFrame com dados históricos para previsão.
            horizons: Lista de horizontes de previsão.
            window_size: Tamanho da janela de observações (se None, usa o valor definido na inicialização).

        Returns:
            dict: Dicionário com previsões para cada horizonte.
        """
        if not self.is_fitted:
            raise ValueError("Modelo deve ser ajustado antes da previsão")

        if not isinstance(horizons, (list, tuple, np.ndarray)):
            horizons = [horizons]

        forecasts_dict = {}

        for h in horizons:
            print(f"Gerando previsões para horizonte {h}...")
            forecasts_dict[h] = self.predict(data, horizon=h, window_size=window_size)

        return forecasts_dict

    def plot_graph_structure(self, window_example=None, output_dir=None):
        """
        Visualizar a estrutura do grafo com as conexões entre maturidades.

        Args:
            window_example: Exemplo de janela para usar na visualização. Se None, cria uma janela sintética.
            output_dir: Diretório para salvar a visualização. Se None, mostra o gráfico.

        Returns:
            None
        """
        maturities_array = np.array([float(m) for m in self.maturities])
        maturities_years = maturities_array / 12  
        maturities_tensor = torch.tensor(
            maturities_years.reshape(-1, 1), device=self.device, dtype=self.dtype
        )

        if not self.is_fitted:
            raise ValueError(
                "Modelo deve ser ajustado antes de plotar a estrutura do grafo"
            )


        if window_example is None:
            window_example = torch.zeros(
                (1, self.window_size, len(self.maturities)),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            if not isinstance(window_example, torch.Tensor):
                window_example = torch.tensor(
                    window_example, dtype=self.dtype, device=self.device
                )

            if window_example.dim() == 2:
                window_example = window_example.unsqueeze(0)

            if window_example.shape[1] != self.window_size or window_example.shape[
                2
            ] != len(self.maturities):
                raise ValueError(
                    f"Janela de exemplo deve ter forma [batch, {self.window_size}, {len(self.maturities)}], "
                    f"mas tem forma {window_example.shape}"
                )

        with torch.no_grad():
            _ = self.basis_learner(maturities_tensor, window_example)
            edge_weights = self.basis_learner.get_edge_weights()

        if edge_weights is None:
            print("Não foi possível obter pesos das arestas")
            return

        edge_weights_np = edge_weights.cpu().numpy()

        plt.figure(figsize=(12, 10))

        import networkx as nx

        G = nx.Graph()

        for i, maturity in enumerate(maturities_years):
            G.add_node(i, maturity=float(maturity))

        for i in range(len(self.maturities)):
            for j in range(i + 1, len(self.maturities)):
                weight = edge_weights_np[i, j]
                if weight > 0.2:  
                    G.add_edge(i, j, weight=weight)

        pos = {}
        for i, maturity in enumerate(maturities_years):
            angle = 2 * np.pi * i / len(maturities_years)
            pos[i] = (np.sin(angle), np.cos(angle))

        node_colors = [float(G.nodes[i]["maturity"]) for i in G.nodes()]

        edge_weights = [G[u][v]["weight"] * 3 for u, v in G.edges()]

        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            node_size=500,
            node_color=node_colors,
            cmap=plt.cm.viridis,
            alpha=0.8,
        )
        edges = nx.draw_networkx_edges(
            G,
            pos,
            width=edge_weights,
            alpha=0.5,
            edge_color=edge_weights,
            edge_cmap=plt.cm.Blues,
        )

        labels = {i: f"{maturities_years[i]:.1f}y" for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight="bold")

        plt.colorbar(nodes, label="Maturidade (anos)")
        plt.axis("off")
        plt.title("Estrutura do Grafo de Maturidades")

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(
                output_path / f"{self.name}_graph_structure.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            edge_weights_np,
            cmap="Blues",
            xticklabels=[f"{m:.1f}y" for m in maturities_years],
            yticklabels=[f"{m:.1f}y" for m in maturities_years],
            annot=True,
            fmt=".2f",
            annot_kws={"size": 6},
            cbar_kws={"label": "Força da Conexão"},
            square=True,
        )
        plt.title("Matriz de Adjacência do Grafo de Maturidades")
        plt.xlabel("Maturidade")
        plt.ylabel("Maturidade")

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        if output_dir:
            plt.savefig(
                output_path / f"{self.name}_adjacency_matrix.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()

    def plot_factor_loadings(
        self, num_points=1000, window_example=None, output_dir=None
    ):
        """
        Plotar as cargas fatoriais aprendidas para diferentes maturidades.

        Args:
            num_points: Número de pontos para plotar ao longo do eixo de maturidades.
            window_example: Exemplo de janela para usar na visualização. Se None, cria uma janela sintética.
            output_dir: Diretório para salvar a visualização. Se None, mostra o gráfico.

        Returns:
            None
        """
        maturities_float = [float(m) for m in self.maturities]

        extended_maturities = np.linspace(
            min(maturities_float) / 12, 10, num_points  
        )

        maturities_tensor = torch.tensor(
            extended_maturities.reshape(-1, 1), dtype=self.dtype, device=self.device
        )

        if not self.is_fitted:
            raise ValueError("Modelo deve ser ajustado antes de plotar")

        if window_example is None:
            window_example = torch.zeros(
                (1, self.window_size, len(self.maturities)),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            if not isinstance(window_example, torch.Tensor):
                window_example = torch.tensor(
                    window_example, dtype=self.dtype, device=self.device
                )

            if window_example.dim() == 2:
                window_example = window_example.unsqueeze(0)

            if window_example.shape[1] != self.window_size or window_example.shape[
                2
            ] != len(self.maturities):
                raise ValueError(
                    f"Janela de exemplo deve ter forma [batch, {self.window_size}, {len(self.maturities)}], "
                    f"mas tem forma {window_example.shape}"
                )

        with torch.no_grad():
            H_t = self.basis_learner(maturities_tensor, window_example)
            loadings = H_t[0].detach().cpu().numpy()

        plt.figure(figsize=(12, 8))

        factor_names = ["Level", "Slope", "Curvature 1", "Curvature 2"]
        if self.n_factors > len(factor_names):
            for i in range(len(factor_names), self.n_factors):
                factor_names.append(f"Factor {i+1}")

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

    def analyze_node_importance(self, window_example=None, output_dir=None):
        """
        Analisa a importância de cada nó (maturidade) no grafo.

        Args:
            window_example: Exemplo de janela para usar na análise. Se None, cria uma janela sintética.
            output_dir: Diretório para salvar a visualização. Se None, mostra o gráfico.

        Returns:
            pandas.DataFrame: DataFrame com medidas de centralidade para cada maturidade.
        """
        maturities_array = np.array([float(m) for m in self.maturities])
        maturities_years = maturities_array / 12
        maturities_tensor = torch.tensor(
            maturities_years.reshape(-1, 1), device=self.device, dtype=self.dtype
        )

        if not self.is_fitted:
            raise ValueError(
                "Modelo deve ser ajustado antes de analisar importância dos nós"
            )

        with torch.no_grad():
            _ = self.basis_learner(maturities_tensor, window_example)
            edge_weights = self.basis_learner.get_edge_weights()

        if edge_weights is None:
            print("Não foi possível obter pesos das arestas")
            return

        edge_weights_np = edge_weights.cpu().numpy()

        import networkx as nx

        G = nx.Graph()

        for i, maturity in enumerate(maturities_years):
            G.add_node(i, maturity=float(maturity))

        for i in range(len(self.maturities)):
            for j in range(i + 1, len(self.maturities)):
                weight = edge_weights_np[i, j]
                if weight > 0.1:  
                    G.add_edge(i, j, weight=weight)

        degree_centrality = nx.degree_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality_numpy(G)

        centrality_df = pd.DataFrame(
            {
                "Maturidade": [f"{m:.1f}y" for m in maturities_years],
                "Grau": list(degree_centrality.values()),
                "Proximidade": list(closeness_centrality.values()),
                "Intermediação": list(betweenness_centrality.values()),
                "Autovetor": list(eigenvector_centrality.values()),
            }
        )

        for col in ["Grau", "Proximidade", "Intermediação", "Autovetor"]:
            max_val = centrality_df[col].max()
            if max_val > 0:
                centrality_df[col] = centrality_df[col] / max_val

        plt.figure(figsize=(14, 10))

        cols = ["Grau", "Proximidade", "Intermediação", "Autovetor"]
        for i, col in enumerate(cols):
            plt.subplot(2, 2, i + 1)
            sns.barplot(x="Maturidade", y=col, data=centrality_df)
            plt.title(f"Centralidade de {col}")
            plt.xticks(rotation=45)
            plt.tight_layout()

        plt.suptitle("Análise de Centralidade dos Nós (Maturidades)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(
                output_path / f"{self.name}_node_centrality.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            centrality_df.to_csv(
                output_path / f"{self.name}_node_centrality.csv", index=False
            )
        else:
            plt.show()

        top_nodes = {}
        for col in cols:
            top_idx = centrality_df[col].nlargest(3).index.tolist()
            top_nodes[col] = centrality_df.iloc[top_idx]["Maturidade"].tolist()

        print("\nNós (Maturidades) Mais Importantes:")
        for measure, nodes in top_nodes.items():
            print(f"Baseado em {measure}: {', '.join(nodes)}")

        return centrality_df

    def analyze_edge_strengths(self, window_example=None, output_dir=None):
        """
        Analisa a força das conexões entre diferentes maturidades.

        Args:
            window_example: Exemplo de janela para usar na análise. Se None, cria uma janela sintética.
            output_dir: Diretório para salvar a visualização. Se None, mostra o gráfico.

        Returns:
            pandas.DataFrame: DataFrame com informações sobre a força das conexões.
        """
        maturities_array = np.array([float(m) for m in self.maturities])
        maturities_years = maturities_array / 12
        maturities_tensor = torch.tensor(
            maturities_years.reshape(-1, 1), device=self.device, dtype=self.dtype
        )

        if not self.is_fitted:
            raise ValueError(
                "Modelo deve ser ajustado antes de analisar força das conexões"
            )

        with torch.no_grad():
            _ = self.basis_learner(maturities_tensor, window_example)
            edge_weights = self.basis_learner.get_edge_weights()

        if edge_weights is None:
            print("Não foi possível obter pesos das arestas")
            return

        edge_weights_np = edge_weights.cpu().numpy()

        connections = []
        for i in range(len(self.maturities)):
            for j in range(i + 1, len(self.maturities)):
                connections.append(
                    {
                        "De": f"{maturities_years[i]:.1f}y",
                        "Para": f"{maturities_years[j]:.1f}y",
                        "Força": edge_weights_np[i, j],
                        "Distância": abs(maturities_years[i] - maturities_years[j]),
                    }
                )

        connections_df = pd.DataFrame(connections)
        connections_df = connections_df.sort_values("Força", ascending=False)

        plt.figure(figsize=(12, 6))
        top_n = min(20, len(connections_df))
        sns.barplot(
            x="Força",
            y="De → Para",
            data=connections_df.head(top_n).assign(
                **{"De → Para": lambda x: x["De"] + " → " + x["Para"]}
            ),
        )
        plt.title(f"Top {top_n} Conexões Mais Fortes Entre Maturidades")
        plt.tight_layout()

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(
                output_path / f"{self.name}_edge_strengths.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="Distância", y="Força", data=connections_df, alpha=0.7)
        plt.title("Relação entre Distância e Força das Conexões")
        plt.xlabel("Distância entre Maturidades (anos)")
        plt.ylabel("Força da Conexão")

        from scipy import stats

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            connections_df["Distância"], connections_df["Força"]
        )
        x = np.array(
            [min(connections_df["Distância"]), max(connections_df["Distância"])]
        )
        y = intercept + slope * x
        plt.plot(x, y, "r", label=f"R² = {r_value**2:.3f}")
        plt.legend()

        if output_dir:
            plt.savefig(
                output_path / f"{self.name}_distance_strength_relation.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            connections_df.to_csv(
                output_path / f"{self.name}_edge_strengths.csv", index=False
            )
        else:
            plt.show()

        print("\nAnálise de Conexões:")
        print(
            f"Conexão mais forte: {connections_df.iloc[0]['De']} → {connections_df.iloc[0]['Para']} (Força: {connections_df.iloc[0]['Força']:.3f})"
        )
        print(
            f"Correlação entre distância e força da conexão: {r_value:.3f} (p-value: {p_value:.3f})"
        )

        return connections_df

    def extract_clusters(self, n_clusters=3, window_example=None, output_dir=None):
        """
        Identifica clusters de maturidades com comportamento similar.

        Args:
            n_clusters: Número de clusters a serem identificados.
            window_example: Exemplo de janela para usar na análise. Se None, cria uma janela sintética.
            output_dir: Diretório para salvar a visualização. Se None, mostra o gráfico.

        Returns:
            pandas.DataFrame: DataFrame com informações sobre os clusters.
        """
        maturities_array = np.array([float(m) for m in self.maturities])
        maturities_years = maturities_array / 12
        maturities_tensor = torch.tensor(
            maturities_years.reshape(-1, 1), device=self.device, dtype=self.dtype
        )

        if not self.is_fitted:
            raise ValueError("Modelo deve ser ajustado antes de extrair clusters")

        with torch.no_grad():
            _ = self.basis_learner(maturities_tensor, window_example)
            edge_weights = self.basis_learner.get_edge_weights()
            node_features = self.basis_learner.get_node_features()

        if edge_weights is None or node_features is None:
            print("Não foi possível obter informações do grafo")
            return

        final_features = (
            node_features[-1][0].cpu().numpy()
        )  

        from sklearn.cluster import SpectralClustering, KMeans

        edge_weights_np = edge_weights.cpu().numpy()

        n_clusters = min(n_clusters, len(self.maturities) // 2)

        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42,
        ).fit(edge_weights_np)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(final_features)

        clusters_df = pd.DataFrame(
            {
                "Maturidade": [f"{m:.1f}y" for m in maturities_years],
                "Cluster_Spectral": clustering.labels_,
                "Cluster_KMeans": kmeans.labels_,
            }
        )

        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(final_features)

        plot_df = pd.DataFrame(
            {
                "Maturidade": [f"{m:.1f}y" for m in maturities_years],
                "PC1": features_2d[:, 0],
                "PC2": features_2d[:, 1],
                "Cluster_Spectral": clustering.labels_,
                "Cluster_KMeans": kmeans.labels_,
                "Maturity_Value": maturities_years,
            }
        )

        plt.figure(figsize=(15, 7))

        plt.subplot(1, 2, 1)
        scatter = sns.scatterplot(
            x="PC1",
            y="PC2",
            hue="Cluster_Spectral",
            size="Maturity_Value",
            palette="viridis",
            data=plot_df,
            legend="brief",
        )

        for i, row in plot_df.iterrows():
            plt.text(row["PC1"], row["PC2"], row["Maturidade"], fontsize=8)

        plt.title("Clusters de Maturidades (Spectral Clustering)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.subplot(1, 2, 2)
        scatter = sns.scatterplot(
            x="PC1",
            y="PC2",
            hue="Cluster_KMeans",
            size="Maturity_Value",
            palette="viridis",
            data=plot_df,
            legend="brief",
        )

        for i, row in plot_df.iterrows():
            plt.text(row["PC1"], row["PC2"], row["Maturidade"], fontsize=8)

        plt.title("Clusters de Maturidades (K-means)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(
                output_path / f"{self.name}_maturity_clusters.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            clusters_df.to_csv(output_path / f"{self.name}_clusters.csv", index=False)
        else:
            plt.show()

        print("\nClusters de Maturidades:")
        for cluster in range(n_clusters):
            spec_maturities = clusters_df[clusters_df["Cluster_Spectral"] == cluster][
                "Maturidade"
            ].tolist()
            kmeans_maturities = clusters_df[clusters_df["Cluster_KMeans"] == cluster][
                "Maturidade"
            ].tolist()

            print(f"Cluster {cluster+1}:")
            print(f"  Spectral: {', '.join(spec_maturities)}")
            print(f"  K-means: {', '.join(kmeans_maturities)}")

        return clusters_df

    def save(self, output_dir):
        """
        Salvar os parâmetros do modelo.

        Args:
            output_dir: Diretório para salvar os parâmetros e visualizações.
        """
        super().save(output_dir)

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        np.savez(
            output_path / f"{self.name}_model_params.npz",
            transition_matrix=self.transition_matrix.detach().cpu().numpy(),
            data_mean=self.data_mean if self.data_mean is not None else np.array([]),
            data_std=self.data_std if self.data_std is not None else np.array([]),
        )

        torch.save(
            self.basis_learner.state_dict(),
            output_path / f"{self.name}_basis_learner.pt",
        )

        torch.save(
            self.factor_estimator.state_dict(),
            output_path / f"{self.name}_factor_estimator.pt",
        )

        with open(output_path / f"{self.name}_hyperparams.json", "w") as f:
            import json

            params = {
                "n_factors": self.n_factors,
                "n_hidden": self.n_hidden,
                "n_blocks": self.n_blocks,
                "window_size": self.window_size,
                "gru_hidden_size": self.gru_hidden_size,
                "factor_gru_hidden_size": self.factor_gru_hidden_size,
                "n_message_passing_layers": self.n_message_passing_layers,
                "edge_weight_type": self.edge_weight_type,
                "aggregation_type": self.aggregation_type,
                "update_type": self.update_type,
                "weight_decay": self.weight_decay,
            }
            json.dump(params, f, indent=4)

        if self.train_loss_history is not None:
            np.save(
                output_path / f"{self.name}_train_loss_history.npy",
                np.array(self.train_loss_history),
            )

        if self.val_loss_history is not None:
            np.save(
                output_path / f"{self.name}_val_loss_history.npy",
                np.array(self.val_loss_history),
            )

        window_example = torch.zeros(
            (1, self.window_size, len(self.maturities)),
            dtype=self.dtype,
            device=self.device,
        )

        try:
            self.plot_factor_loadings(
                window_example=window_example, output_dir=output_dir
            )
            self.plot_graph_structure(
                window_example=window_example, output_dir=output_dir
            )
            self.analyze_node_importance(
                window_example=window_example, output_dir=output_dir
            )
            self.analyze_edge_strengths(
                window_example=window_example, output_dir=output_dir
            )
            self.extract_clusters(window_example=window_example, output_dir=output_dir)

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                self.transition_matrix.detach().cpu().numpy(),
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                square=True,
            )
            plt.title(f"{self.name} - Matriz de Transição")
            plt.tight_layout()
            plt.savefig(
                output_path / f"{self.name}_transition_matrix.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        except Exception as e:
            print(f"AVISO: Erro ao gerar plotagens durante o save: {e}")


def train_and_evaluate_standalone(
    data_path="insira o path aqui",
    sequences_dir="insira o path aqui",
    output_dir="insira o path aqui",
    train_val_test_split=[
        0.7,
        0.15,
        0.15,
    ],  
    horizons=[5, 20, 60, 120],
    use_gpu=False,
    n_factors=4,
    n_hidden=64,
    n_blocks=2,
    window_size=20,
    gru_hidden_size=32,
    factor_gru_hidden_size=32,
    n_message_passing_layers=3,
    edge_weight_type="inverse_distance",
    aggregation_type="weighted_mean",
    update_type="gru",
    weight_decay=0.01,
    batch_size=32,
    learning_rate=1e-3,
    n_epochs=100,  
    early_stopping_patience=10,
    normalize_data=True,  
    use_pregenerated_sequences=True,
):
    """
    Função para treinar e avaliar o modelo NNSS Graph Sequencial de forma independente.

    Args:
        data_path: Caminho para o arquivo de dados original
        sequences_dir: Diretório contendo as sequências pré-geradas (base path)
        output_dir: Diretório para salvar resultados
        train_val_test_split: Proporção dos dados para treino, validação e teste (usado apenas se não usar sequências pré-geradas)
        horizons: Lista de horizontes de previsão
        use_gpu: Se deve usar GPU
        n_factors: Número de fatores latentes
        n_hidden: Dimensão oculta para processamento de maturidades
        n_blocks: Número de blocos residuais
        window_size: Tamanho da janela de observações históricas
        gru_hidden_size: Dimensão oculta do GRU para o WindowAwareGraphBasisLearner
        factor_gru_hidden_size: Dimensão oculta do GRU para o FactorStateEstimator
        n_message_passing_layers: Número de camadas de propagação de mensagens
        edge_weight_type: Tipo de peso das arestas ('uniform', 'log_distance', 'inverse_distance')
        aggregation_type: Tipo de agregação ('sum', 'weighted_mean')
        update_type: Tipo de função de atualização ('gru', 'mlp')
        weight_decay: Regularização L2 para os parâmetros do modelo
        batch_size: Tamanho do lote para treinamento
        learning_rate: Taxa de aprendizado para o otimizador
        n_epochs: Número máximo de épocas de treinamento
        early_stopping_patience: Número de validações sem melhoria antes de parar
        normalize_data: Se deve normalizar os dados antes do treinamento
        use_pregenerated_sequences: Se deve usar as sequências pré-geradas
    """
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print(f"Carregando dados originais de {data_path}")
    try:
        data = pd.read_csv(data_path, parse_dates=["Date"])
        data.set_index("Date", inplace=True)
    except Exception as e:
        print(f"Erro ao carregar dados originais: {str(e)}")
        return

    maturities = [int(col) for col in data.columns]
    n_maturities = len(maturities)

    evaluation_results = {}
    all_horizon_results = []

    maturities_array = np.array([float(m) for m in maturities])
    maturities_years = maturities_array / 12
    maturities_tensor = torch.tensor(
        maturities_years.reshape(-1, 1), device=device, dtype=torch.get_default_dtype()
    )

    data_mean_train = None
    data_std_train = None
    sequences_path = Path(sequences_dir) / "sequences"  

    if normalize_data:
        if use_pregenerated_sequences:
            print(f"Carregando média/std do treino de {sequences_path}...")
            try:
                mean_path = sequences_path / "train_mean.npy"
                std_path = sequences_path / "train_std.npy"
                data_mean_train = np.load(mean_path)
                data_std_train = np.load(std_path)
                print("  Média/Std carregados com sucesso.")
                if (
                    len(data_mean_train) != n_maturities
                    or len(data_std_train) != n_maturities
                ):
                    print(
                        f"ERRO: Incompatibilidade de dimensões entre maturidades ({n_maturities}) e estatísticas carregadas ({len(data_mean_train)}/{len(data_std_train)})."
                    )
                    print(
                        "Verifique se as sequências foram geradas com os dados corretos."
                    )
                    return  
            except FileNotFoundError:
                print(
                    f"ERRO: Arquivos train_mean.npy ou train_std.npy não encontrados em {sequences_path}."
                )
                print(
                    "Execute create_sequences.py para gerá-los ou desative a normalização."
                )
                normalize_data = False
            except Exception as e:
                print(f"ERRO ao carregar média/std: {e}. Desativando normalização.")
                normalize_data = False
        else:
            print("Calculando média/std do treino (modo não-sequencial)...")
            split_idx = int(
                len(data) * train_val_test_split[0]
            )
            train_data_orig = data.iloc[:split_idx]
            if len(train_data_orig) > 0:
                data_mean_train = train_data_orig.mean().values
                data_std_train = train_data_orig.std().values
                data_std_train = np.where(data_std_train < 1e-8, 1.0, data_std_train)
            else:
                print(
                    "AVISO: Split de treino resultou em 0 amostras. Desativando normalização."
                )
                normalize_data = False

    if not normalize_data:
        print("Normalização desativada. Usando dados brutos / previsões brutas.")
        data_mean_train = np.zeros(n_maturities)
        data_std_train = np.ones(n_maturities)

    for horizon in horizons:
        print(f"\n=== Treinando modelo para horizonte {horizon} ===")

        horizon_dir = output_path / f"horizon_{horizon}"
        horizon_dir.mkdir(exist_ok=True, parents=True)

        model_h = NNSSGraphModel(
            maturities=maturities,
            n_factors=n_factors,
            n_hidden=n_hidden,
            n_blocks=n_blocks,
            window_size=window_size,
            gru_hidden_size=gru_hidden_size,
            factor_gru_hidden_size=factor_gru_hidden_size,
            n_message_passing_layers=n_message_passing_layers,
            edge_weight_type=edge_weight_type,
            aggregation_type=aggregation_type,
            update_type=update_type,
            weight_decay=weight_decay,
            device=device,
        )

        model_h.data_mean = data_mean_train
        model_h.data_std = data_std_train

        if use_pregenerated_sequences:
            try:
                print(f"Carregando sequências de {sequences_path}...")
                X_train = torch.load(
                    sequences_path / f"seq_w{window_size}_h{horizon}_X_train.pt"
                ).to(device)
                y_train = torch.load(
                    sequences_path / f"seq_w{window_size}_h{horizon}_y_train.pt"
                ).to(device)

                X_val, y_val = None, None
                use_validation = True
                val_path_x = sequences_path / f"seq_w{window_size}_h{horizon}_X_val.pt"
                if val_path_x.exists():
                    X_val = torch.load(val_path_x).to(device)
                    y_val = torch.load(
                        sequences_path / f"seq_w{window_size}_h{horizon}_y_val.pt"
                    ).to(device)
                else:
                    print("AVISO: Sequências de validação não encontradas.")
                    use_validation = False

                X_test, y_test = None, None
                use_test = True
                test_path_x = (
                    sequences_path / f"seq_w{window_size}_h{horizon}_X_test.pt"
                )
                if test_path_x.exists():
                    X_test = torch.load(test_path_x).to(device)
                    y_test = torch.load(
                        sequences_path / f"seq_w{window_size}_h{horizon}_y_test.pt"
                    ).to(device)
                else:
                    print("AVISO: Sequências de teste não encontradas.")
                    use_test = False

            except FileNotFoundError as e:
                print(
                    f"ERRO ao carregar arquivos de sequência: {e}. Pulando horizonte {horizon}."
                )
                continue

            if normalize_data:
                print("Normalizando sequências carregadas...")
                mean_tensor = torch.tensor(
                    data_mean_train, dtype=torch.get_default_dtype(), device=device
                )
                std_tensor = torch.tensor(
                    data_std_train, dtype=torch.get_default_dtype(), device=device
                )

                X_train = (X_train - mean_tensor) / std_tensor
                y_train = (y_train - mean_tensor) / std_tensor
                if use_validation:
                    X_val = (X_val - mean_tensor) / std_tensor
                    y_val = (y_val - mean_tensor) / std_tensor
            else:
                print(
                    "Usando sequências carregadas como estão (assumindo brutas ou já normalizadas)."
                )

            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = None
            if use_validation:
                val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
                val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False
                )

            print(f"Iniciando treinamento para horizonte {horizon}...")
            optimizer = torch.optim.Adam(
                [
                    {"params": model_h.basis_learner.parameters()},
                    {"params": model_h.factor_estimator.parameters()},
                    {"params": model_h.transition_matrix},
                ],
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=early_stopping_patience // 2,
                verbose=True,
                min_lr=1e-6,
            )
            criterion = nn.MSELoss()
            best_val_loss = float("inf")
            best_epoch = 0
            no_improvement_count = 0
            best_state = None
            train_losses = []
            val_losses = []

            for epoch in range(n_epochs):
                model_h.basis_learner.train()
                model_h.factor_estimator.train()
                epoch_loss = 0.0
                for batch_windows, batch_targets in train_loader:
                    optimizer.zero_grad()
                    y_pred = model_h.forward(batch_windows, maturities_tensor, horizon)
                    loss = criterion(
                        y_pred, batch_targets
                    )  
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model_h.basis_learner.parameters(), max_norm=1.0
                    )
                    torch.nn.utils.clip_grad_norm_(
                        model_h.factor_estimator.parameters(), max_norm=1.0
                    )
                    torch.nn.utils.clip_grad_norm_(
                        [model_h.transition_matrix], max_norm=1.0
                    )
                    optimizer.step()
                    epoch_loss += loss.item() * batch_windows.size(0)
                epoch_loss /= len(train_dataset)
                train_losses.append(epoch_loss)

                if use_validation:
                    model_h.basis_learner.eval()
                    model_h.factor_estimator.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for batch_windows, batch_targets in val_loader:
                            y_pred = model_h.forward(
                                batch_windows, maturities_tensor, horizon
                            )
                            loss = criterion(y_pred, batch_targets)
                            val_loss += loss.item() * batch_windows.size(0)
                    val_loss /= len(val_dataset)
                    val_losses.append(val_loss)
                    scheduler.step(val_loss)
                    print(
                        f"Época {epoch+1}/{n_epochs}, Treino Loss: {epoch_loss:.6f}, Validação Loss: {val_loss:.6f}"
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch + 1
                        no_improvement_count = 0
                        best_state = {
                            "basis_learner": model_h.basis_learner.state_dict(),
                            "factor_estimator": model_h.factor_estimator.state_dict(),
                            "transition_matrix": model_h.transition_matrix.detach().clone(),
                        }
                        print(f"  Nova melhor validação loss: {best_val_loss:.6f}")
                    else:
                        no_improvement_count += 1
                        print(
                            f"  Sem melhoria por {no_improvement_count} verificações (melhor: {best_val_loss:.6f} na época {best_epoch})"
                        )
                        if no_improvement_count >= early_stopping_patience:
                            print(
                                f"Early stopping após {early_stopping_patience} validações sem melhoria"
                            )
                            if (
                                best_state
                            ):  
                                print(
                                    f"Restaurando modelo de melhor validação da época {best_epoch}"
                                )
                                model_h.basis_learner.load_state_dict(
                                    best_state["basis_learner"]
                                )
                                model_h.factor_estimator.load_state_dict(
                                    best_state["factor_estimator"]
                                )
                                model_h.transition_matrix.data.copy_(
                                    best_state["transition_matrix"]
                                )
                            else:
                                print("AVISO: Early stopping sem melhor estado salvo.")
                            break  
                elif (epoch + 1) % 10 == 0:
                    print(f"Época {epoch+1}/{n_epochs}, Treino Loss: {epoch_loss:.6f}")

            if (
                use_validation
                and best_state is not None
                and no_improvement_count < early_stopping_patience
            ):
                print(
                    f"Treinamento completo, restaurando melhor modelo da época {best_epoch}"
                )
                model_h.basis_learner.load_state_dict(best_state["basis_learner"])
                model_h.factor_estimator.load_state_dict(best_state["factor_estimator"])
                model_h.transition_matrix.data.copy_(best_state["transition_matrix"])

            model_h.train_loss_history = train_losses
            model_h.val_loss_history = val_losses if use_validation else None
            model_h.is_fitted = True

            model_h.save(str(horizon_dir))

            if use_test:
                print(
                    f"Avaliando modelo no conjunto de teste para horizonte {horizon}..."
                )
                actual_values = (
                    y_test.cpu().numpy()
                )  

                if normalize_data:
                    X_test_input = (X_test - mean_tensor) / std_tensor
                else:
                    X_test_input = X_test  

                y_test_input = (
                    (y_test - mean_tensor) / std_tensor if normalize_data else y_test
                )
                test_dataset = torch.utils.data.TensorDataset(
                    X_test_input, y_test_input
                )
                test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False
                )

                model_h.basis_learner.eval()
                model_h.factor_estimator.eval()
                all_predictions = []

                with torch.no_grad():
                    for (
                        batch_windows,
                        _,
                    ) in test_loader:  
                        y_pred_model_scale = model_h.forward(
                            batch_windows, maturities_tensor, horizon
                        )
                        all_predictions.append(y_pred_model_scale.cpu().numpy())

                predicted_values_model_scale = np.vstack(all_predictions)

                if normalize_data:
                    predicted_values = (
                        predicted_values_model_scale * data_std_train + data_mean_train
                    )
                    actual_values_final = actual_values  
                else:
                    predicted_values = (
                        predicted_values_model_scale  
                    )
                    actual_values_final = actual_values  

                basis_point_factor = 100
                rmse_per_maturity_bps = (
                    np.sqrt(
                        np.mean((actual_values_final - predicted_values) ** 2, axis=0)
                    )
                    * basis_point_factor
                )
                mae_per_maturity_bps = (
                    np.mean(np.abs(actual_values_final - predicted_values), axis=0)
                    * basis_point_factor
                )
                avg_rmse_bps = np.mean(rmse_per_maturity_bps)
                avg_mae_bps = np.mean(mae_per_maturity_bps)

                np.save(
                    horizon_dir / f"{model_h.name}_forecasts_h{horizon}.npy",
                    predicted_values,
                )
                np.save(
                    horizon_dir / f"{model_h.name}_actuals_h{horizon}.npy",
                    actual_values_final,
                )

                results_df = pd.DataFrame(
                    {
                        "maturity": maturities + ["avg"],
                        "horizon": [horizon] * (n_maturities + 1),
                        "rmse_bps": np.append(rmse_per_maturity_bps, avg_rmse_bps),
                        "mae_bps": np.append(mae_per_maturity_bps, avg_mae_bps),
                    }
                )
                results_df.to_csv(
                    horizon_dir / "evaluation_results_bps.csv", index=False
                )
                all_horizon_results.append(results_df)
                print(
                    f"Horizonte {horizon} - RMSE médio: {avg_rmse_bps:.2f} bps, MAE médio: {avg_mae_bps:.2f} bps"
                )

                n_samples = min(100, len(predicted_values))
                indices = np.linspace(
                    0, len(predicted_values) - 1, n_samples, dtype=int
                )
                plot_maturities = [
                    maturities[0],
                    maturities[len(maturities) // 2],
                    maturities[-1],
                ]
                plt.figure(figsize=(15, 6))
                for i, mat in enumerate(plot_maturities):
                    idx = maturities.index(mat)
                    plt.subplot(1, 3, i + 1)
                    plt.plot(
                        range(n_samples),
                        actual_values_final[indices, idx] * 100,
                        "b-",
                        label="Atual (%)",
                    )
                    plt.plot(
                        range(n_samples),
                        predicted_values[indices, idx] * 100,
                        "r--",
                        label="Previsto (%)",
                    )
                    plt.title(f"Maturidade {mat}")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    horizon_dir / f"predictions_h{horizon}_desnorm.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

        else:
            split_idx = int(len(data) * train_val_test_split[0])
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]
            print(
                f"Usando {len(train_data)} observações para treino e {len(test_data)} para teste"
            )

            model_h.fit(
                train_data=train_data,
                window_size=window_size,
                horizon=horizon,
                batch_size=batch_size,
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                early_stopping_patience=early_stopping_patience,
                normalize_data=normalize_data,      
            )
            model_h.save(str(horizon_dir))

            print(f"Avaliando modelo no conjunto de teste para horizonte {horizon}...")
            if (
                len(test_data) < window_size
            ):  
                print(
                    f"AVISO: Conjunto de teste muito pequeno ({len(test_data)} vs {window_size}). Pulando avaliação."
                )
                continue

            predictions = model_h.predict(
                test_data, horizon=horizon, window_size=window_size
            )

            n_forecasts = len(predictions)
            actual_indices = [i + window_size + horizon - 1 for i in range(n_forecasts)]
            valid_actual_indices = [
                idx for idx in actual_indices if idx < len(test_data)
            ]
            valid_predictions = predictions[: len(valid_actual_indices)]
            actual_values = test_data.iloc[valid_actual_indices].values

            if len(valid_predictions) != len(actual_values):
                print(
                    f"AVISO: Disparidade inesperada no tamanho de previsões ({len(valid_predictions)}) e alvos ({len(actual_values)})."
                )
                continue  

            if len(actual_values) == 0:
                print(
                    f"AVISO: Não há alvos válidos no conjunto de teste para horizonte {horizon}. Pulando avaliação."
                )
                continue

            basis_point_factor = 100
            rmse_per_maturity_bps = (
                np.sqrt(np.mean((actual_values - valid_predictions) ** 2, axis=0))
                * basis_point_factor
            )
            mae_per_maturity_bps = (
                np.mean(np.abs(actual_values - valid_predictions), axis=0)
                * basis_point_factor
            )
            avg_rmse_bps = np.mean(rmse_per_maturity_bps)
            avg_mae_bps = np.mean(mae_per_maturity_bps)

            np.save(
                horizon_dir / f"{model_h.name}_forecasts_h{horizon}.npy",
                valid_predictions,
            )
            np.save(
                horizon_dir / f"{model_h.name}_actuals_h{horizon}.npy",
                actual_values,
            )

            results_df = pd.DataFrame(
                {
                    "maturity": maturities + ["avg"],
                    "horizon": [horizon] * (n_maturities + 1),
                    "rmse_bps": np.append(rmse_per_maturity_bps, avg_rmse_bps),
                    "mae_bps": np.append(mae_per_maturity_bps, avg_mae_bps),
                }
            )
            results_df.to_csv(horizon_dir / "evaluation_results_bps.csv", index=False)
            all_horizon_results.append(results_df)
            print(
                f"Horizonte {horizon} - RMSE médio: {avg_rmse_bps:.2f} bps, MAE médio: {avg_mae_bps:.2f} bps"
            )

            test_indices = test_data.index[
                valid_actual_indices
            ]  
            n_samples = min(100, len(valid_predictions))
            plot_indices_range = np.linspace(
                0, len(valid_predictions) - 1, n_samples, dtype=int
            )
            plot_maturities = [
                maturities[0],
                maturities[len(maturities) // 2],
                maturities[-1],
            ]
            plt.figure(figsize=(15, 6))
            for i, mat in enumerate(plot_maturities):
                idx = maturities.index(mat)
                plt.subplot(1, 3, i + 1)
                plt.plot(
                    test_indices, actual_values[:, idx] * 100, "b-", label="Atual (%)"
                )
                plt.plot(
                    test_indices,
                    valid_predictions[:, idx] * 100,
                    "r--",
                    label="Previsto (%)",
                )
                plt.title(f"Maturidade {mat}")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=30)  
            plt.tight_layout()
            plt.savefig(
                horizon_dir / f"predictions_h{horizon}_desnorm.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    if all_horizon_results:
        final_results_df = pd.concat(all_horizon_results, ignore_index=True)
        final_results_df.to_csv(
            output_path / "all_horizon_results_bps.csv", index=False
        )
        print("\nResultados consolidados salvos em all_horizon_results_bps.csv")

    print("\nProcesso de treinamento e avaliação concluído!")
    return (
        model_h,
        evaluation_results,
    )  


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Treinar e avaliar modelo NNSS Graph Sequencial"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="insira o path aqui",
        help="Caminho para o arquivo de dados original",
    )
    parser.add_argument(
        "--sequences_dir",
        type=str,
        default="insira o path aqui",
        help="Diretório base contendo o subdiretório 'sequences'",
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
        default=0.8,
        help="Proporção dos dados para treino (usado apenas se não usar sequências pré-geradas)",
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
        "--n_factors",
        type=int,
        default=4,
        help="Número de fatores latentes",
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=64,
        help="Dimensão oculta para processamento de maturidades",
    )
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=2,
        help="Número de blocos residuais",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=20,
        help="Tamanho da janela de observações históricas",
    )
    parser.add_argument(
        "--gru_hidden_size",
        type=int,
        default=32,
        help="Dimensão oculta do GRU para o WindowAwareGraphBasisLearner",
    )
    parser.add_argument(
        "--factor_gru_hidden_size",
        type=int,
        default=32,
        help="Dimensão oculta do GRU para o FactorStateEstimator",
    )
    parser.add_argument(
        "--n_message_passing_layers",
        type=int,
        default=3,
        help="Número de camadas de propagação de mensagens no grafo",
    )
    parser.add_argument(
        "--edge_weight_type",
        type=str,
        default="inverse_distance",
        choices=["uniform", "log_distance", "inverse_distance"],
        help="Tipo de peso das arestas do grafo",
    )
    parser.add_argument(
        "--aggregation_type",
        type=str,
        default="weighted_mean",
        choices=["sum", "weighted_mean"],
        help="Tipo de agregação das mensagens no grafo",
    )
    parser.add_argument(
        "--update_type",
        type=str,
        default="gru",
        choices=["gru", "mlp"],
        help="Tipo de função de atualização dos nós no grafo",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Regularização L2 para os parâmetros do modelo",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Tamanho do lote para treinamento",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Taxa de aprendizado para o otimizador",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=100,  
        help="Número máximo de épocas de treinamento",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Número de validações sem melhoria antes de parar",
    )
    parser.add_argument(
        "--normalize_data",
        action="store_true",
        default=True,
        help="Se deve normalizar os dados antes do treinamento",
    )
    parser.add_argument(
        "--use_pregenerated_sequences",
        action="store_true",
        default=True,
        help="Se deve usar as sequências pré-geradas",
    )

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

    if args.use_pregenerated_sequences and not Path(args.sequences_dir).exists():
        print(
            f"AVISO: Diretório base de sequências não encontrado em {args.sequences_dir}"
        )
        print("Verificando caminhos alternativos...")

        possible_dirs = [
            "insira o path aqui",
            "insira o path aqui",
            "insira o path aqui",
            "insira o path aqui",
            "insira o path aqui",
        ]

        base_dir_found = False
        for dir_path in possible_dirs:
            if Path(dir_path).exists():
                args.sequences_dir = dir_path
                print(f"Diretório base de sequências encontrado em {dir_path}")
                base_dir_found = True
                break

        if not base_dir_found:
            print(
                "AVISO: Diretório base de sequências não encontrado em nenhum local padrão."
            )
            print("Desativando o uso de sequências pré-geradas.")
            args.use_pregenerated_sequences = False
        elif not (Path(args.sequences_dir) / "sequences").exists():
            print(
                f"AVISO: Subdiretório 'sequences' não encontrado dentro de {args.sequences_dir}"
            )
            print("Desativando o uso de sequências pré-geradas.")
            args.use_pregenerated_sequences = False

    print("\nIniciando treinamento e avaliação do modelo NNSS Graph Sequencial...")
    if args.use_pregenerated_sequences:
        print(
            f"Usando sequências pré-geradas de {Path(args.sequences_dir) / 'sequences'}"
        )
    else:
        print("Usando dados brutos (sem sequências pré-geradas)")

    model, results = train_and_evaluate_standalone(
        data_path=args.data_path,
        sequences_dir=args.sequences_dir,  
        output_dir=args.output_dir,
        train_val_test_split=args.train_test_split,
        horizons=args.horizons,
        use_gpu=args.use_gpu,
        n_factors=args.n_factors,
        n_hidden=args.n_hidden,
        n_blocks=args.n_blocks,
        window_size=args.window_size,
        gru_hidden_size=args.gru_hidden_size,
        factor_gru_hidden_size=args.factor_gru_hidden_size,
        n_message_passing_layers=args.n_message_passing_layers,
        edge_weight_type=args.edge_weight_type,
        aggregation_type=args.aggregation_type,
        update_type=args.update_type,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs,
        early_stopping_patience=args.early_stopping_patience,
        normalize_data=args.normalize_data,
        use_pregenerated_sequences=args.use_pregenerated_sequences,
    )
    print("\nProcesso concluído!")
