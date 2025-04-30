import numpy as np
import pandas as pd
import time
from pathlib import Path
from src.models.base_model import BaseYieldCurveModel

class RandomWalkModel(BaseYieldCurveModel):
    """
    Modelo Random Walk para previsão da curva de juros.
    
    Este modelo simplemente assume que a melhor previsão para os juros futuros
    é o juro atual (previsão sem alteração).
    """
    
    def __init__(self, maturities):
        """
        Inicializa o modelo Random Walk.
        
        Args:
            maturities: Lista de maturidades
        """
        super().__init__("RandomWalk", maturities)
    
    def fit(self, train_data):
        """
        'Ajusta' o modelo Random Walk (nenhum ajuste real necessário).
        
        Args:
            train_data: Dados de treino (não utilizado para RW)
        
        Returns:
            self
        """
        start_time = time.time()
        
        self.is_fitted = True
        self.training_time = time.time() - start_time
        
        print(f"Random Walk model 'fitted' in {self.training_time:.4f} seconds")
        
        return self
    
    def predict(self, data, horizon):
        """
        Gera previsões usando o modelo Random Walk.
        
        Args:
            data: Dados de entrada
            horizon: Horizonte de previsão
            
        Returns:
            Array de previsões
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        forecasts = []
        
        for i in range(len(data) - horizon):
            forecast = data.iloc[i].values
            forecasts.append(forecast)
        
        return np.array(forecasts)